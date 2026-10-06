// harness/nslm-mova-bench.m - speed and memory footprint of one MoVA configuration on one pre-tokenized prompt.
//
//   nslm-mova-bench --model DIR [--res out/res] --ids FILE [--bucket NAME] [--decode 256]
//                   [--repeats 5] [--timing]
//
// --ids: int32 prompt ids (tools/mova_export.py bench).  One sequence slot, KV sized for prompt + decode; a warm-up
// run, then --repeats timed runs of greedy decoding of exactly --decode tokens (eng_generate ignores stop ids).
//   TTFT          prompt in -> first new token (prefill + the first decode step)
//   prefill tok/s n_prompt / TTFT
//   decode tok/s  (decode - 1) / (first new token -> last)
// Memory: engine accounting (weights, stream table, KV, scratch), MTLDevice.currentAllocatedSize, phys_footprint
// (current, peak), RSS.  Machine state before and after; other processes' CPU and GPU busy.
// --timing: one extra prefill + 64-token decode with per-kernel-group GPU timing (eng_mova_timing).
#import <Foundation/Foundation.h>
#include <math.h>

#include "common.h"
#include "mova_ext.h"
#include "telemetry.h"

static int flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], name)) return 1;
    return 0;
}
static void stats(const char* key, double* v, int n) {
    qsort(v, (size_t) n, sizeof(double), cmp_double);
    const double med = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    printf("%s_median=%.4f\n%s_min=%.4f\n%s_max=%.4f\n%s_spread_pct=%.2f\n", key, med, key, v[0], key, v[n - 1], key,
           med > 0 ? 100.0 * (v[n - 1] - v[0]) / med : 0);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        setvbuf(stdout, NULL, _IOLBF, 0);
        const int reps = atoi(opt(argc, argv, "--repeats", "5")), ndec = atoi(opt(argc, argv, "--decode", "256"));
        NSData* d = [NSData dataWithContentsOfFile:@(opt(argc, argv, "--ids", ""))];
        if (!d.length) { fprintf(stderr, "--ids FILE (int32 prompt ids) required\n"); return 2; }
        const int n = (int) (d.length / 4);
        const int32_t* p = (const int32_t*) d.bytes;
        EngOpts o;
        memset(&o, 0, sizeof o);
        o.model_dir = opt(argc, argv, "--model", NULL);
        o.resource_dir = opt(argc, argv, "--res", "out/res");
        o.max_seqs = 1;
        o.kv_tokens = n + ndec + 64;
        MachineState ms0 = machine_state();
        print_machine_state("start_", ms0);
        char err[512] = "";
        const double tl = now_s();
        Eng* e = eng_open(&o, err, sizeof err);
        if (!e) { fprintf(stderr, "eng_open: %s\n", err); return 2; }
        printf("engine=%s\nbucket=%s\nn_prompt=%d\ndecode_tokens=%d\nrepeats=%d\nload_s=%.2f\n", eng_describe(e),
               opt(argc, argv, "--bucket", "?"), n, ndec, reps, now_s() - tl);
        int32_t* out = (int32_t*) malloc(sizeof(int32_t) * (size_t) ndec);
        double ttft[64], pre[64], dec[64];
        int seq = 0;
        if (eng_prefill(e, 0, p, n) || eng_generate(e, &seq, 1, ndec, ENG_MODE_AR, out, NULL)) { fprintf(stderr, "warm-up failed\n"); return 3; }
        const double cpu0 = host_cpu_busy_s(), self0 = self_cpu_s();
        TelemMark tm = telem_mark();
        const long long g0 = telem_self_gpu_ns();
        const double t_start = now_s();
        uint32_t h = 2166136261u;
        for (int r = 0; r < reps && r < 64; ++r) {
            const double t0 = now_s();
            if (eng_prefill(e, 0, p, n)) return 3;
            if (eng_generate(e, &seq, 1, 1, ENG_MODE_AR, out, NULL)) return 3;
            const double t1 = now_s();
            if (eng_generate(e, &seq, 1, ndec - 1, ENG_MODE_AR, out + 1, NULL)) return 3;
            const double t2 = now_s();
            ttft[r] = t1 - t0;
            pre[r] = n / (t1 - t0);
            dec[r] = (ndec - 1) / (t2 - t1);
            if (r == 0) for (int k = 0; k < ndec; ++k) h = (h ^ (uint32_t) out[k]) * 16777619u;
        }
        const double wall = now_s() - t_start;
        TelemResult tr = telem_since(tm);
        const long long g1 = telem_self_gpu_ns();
        const double other = ((host_cpu_busy_s() - cpu0) - (self_cpu_s() - self0)) / wall;
        stats("ttft_s", ttft, reps);
        stats("prefill_tok_s", pre, reps);
        stats("decode_tok_s", dec, reps);
        printf("output_fnv=%08x\n", h);
        EngMem m;
        eng_mem(e, &m);
        const ProcMem pm = proc_mem(getpid());
        printf("weights_mb=%.2f\nlut_mb=%.3f\nweights_resident_mb=%.2f\nkv_mb=%.2f\nscratch_mb=%.2f\ngpu_allocated_mb=%.2f\n",
               m.weights / 1e6, m.lut / 1e6, (m.weights + m.lut) / 1e6, m.kv / 1e6, m.scratch / 1e6, m.gpu_allocated / 1e6);
        printf("phys_footprint_mb=%.2f\nphys_footprint_peak_mb=%.2f\nrss_mb=%.2f\n", pm.phys_footprint_mb * 1.048576,
               pm.phys_footprint_peak_mb * 1.048576, pm.rss_mb * 1.048576);
        printf("timed_wall_s=%.2f\nother_cpu_cores=%.2f\ngpu_busy_pct=%.2f\nself_gpu_pct=%.2f\n", wall, other, tr.gpu_busy_pct,
               g0 >= 0 && g1 >= 0 ? 100.0 * (double) (g1 - g0) / 1e9 / wall : -1);
        if (flag(argc, argv, "--timing")) {   // per-kernel-group GPU time of a 64-token decode after this prompt
            const char* names[MOVA_TG_N] = {"embed_norm", "attn_proj", "values", "attention", "router", "experts", "shared_dense", "head"};
            eng_mova_timing(e, 1);   // the prompt's prefill, then a 64-token decode
            const double tp0 = now_s();
            if (eng_prefill(e, 0, p, n)) return 3;
            const double tp1 = now_s();
            {
                double sec[MOVA_TG_N], tot = 0;
                eng_mova_timing_read(e, sec);
                for (int i = 0; i < MOVA_TG_N; ++i) tot += sec[i];
                for (int i = 0; i < MOVA_TG_N; ++i) printf("prefill_s_%s=%.4f\n", names[i], sec[i]);
                printf("prefill_s_gpu_total=%.4f\nprefill_s_wall_timing_mode=%.4f\n", tot, tp1 - tp0);
            }
            eng_mova_timing(e, 0);
            eng_mova_timing(e, 1);
            const double tt0 = now_s();
            if (eng_generate(e, &seq, 1, 64, ENG_MODE_AR, out, NULL)) return 3;
            const double tt1 = now_s();
            eng_mova_timing(e, 0);
            double sec[MOVA_TG_N], tot = 0;
            eng_mova_timing_read(e, sec);
            for (int i = 0; i < MOVA_TG_N; ++i) tot += sec[i];
            for (int i = 0; i < MOVA_TG_N; ++i) printf("decode_ms_per_token_%s=%.4f\n", names[i], 1e3 * sec[i] / 64);
            printf("decode_ms_per_token_gpu_total=%.4f\ndecode_ms_per_token_wall_timing_mode=%.4f\n", 1e3 * tot / 64, 1e3 * (tt1 - tt0) / 64);
        }
        print_machine_state("end_", machine_state());
        printf("status=ok\n");
        eng_close(e);
        return 0;
    }
}
