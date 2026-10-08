// harness/nslm-mova-bench.c - speed and memory footprint of one MoVA configuration on one pre-tokenized prompt.
//
//   nslm-mova-bench --model DIR [--res out/res] (--ids FILE | --text FILE --ctx N) [--bucket NAME] [--decode 256]
//                   [--repeats 5] [--warmup 1] [--kv bf16|q8] [--timing]
//
// --ids: int32 prompt ids (tools/mova_export.py bench).  --text FILE --ctx N builds the same matched-bench prompt
// here: BOS + the text's tokens + one chat question, exactly N tokens.  One sequence slot, KV sized for prompt +
// decode; a warm-up run, then --repeats timed runs of greedy decoding of exactly --decode tokens (eng_generate ignores
// stop ids).
//   TTFT          prompt in -> first new token (prefill + the first decode step)
//   prefill tok/s n_prompt / TTFT
//   decode tok/s  (decode - 1) / (first new token -> last)
// Memory: engine accounting (weights, stream table, KV, scratch, staging), GPU memory in use, process footprint.
// Machine state before and after; other processes' CPU and the GPU's busy share.
// --warmup 0 skips the warm-up run (long prompts); --kv q8 runs the 8-bit KV cache.
// --timing: one extra prefill + 64-token decode with per-kernel-group GPU timing (eng_mova_timing).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine_api.h"
#include "mova_ext.h"
#include "platform.h"
#include "telemetry.h"
#include "tokenizer.h"

static void stats(const char* key, double* v, int n) {
    qsort(v, (size_t) n, sizeof(double), cmp_double);
    const double med = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    printf("%s_median=%.4f\n%s_min=%.4f\n%s_max=%.4f\n%s_spread_pct=%.2f\n", key, med, key, v[0], key, v[n - 1], key,
           med > 0 ? 100.0 * (v[n - 1] - v[0]) / med : 0);
}

// The matched-bench prompt (tools/mova_export.py cmd_bench): BOS + filler[:n - 1 - len(q)] + q.
static int32_t* bench_prompt(const char* model, const char* text_path, int n, char* err, int errlen) {
    char path[2048];
    snprintf(path, sizeof path, "%s/tokenizer.json", model);
    Tok* t = tok_open(path, err, errlen);
    if (!t) return NULL;
    size_t len = 0;
    char* text = plat_slurp(text_path, &len);
    if (!text) { snprintf(err, (size_t) errlen, "cannot read %s", text_path); tok_close(t); return NULL; }
    const char* qs = "<|ifm|im_start|>user\nWrite a short paragraph about the history of the bicycle.<|ifm|im_end|>"
                     "<|ifm|im_start|>assistant\n<ifm|think>\n";
    int32_t q[256];
    const int nq = tok_encode(t, qs, 0, q, 256);
    const int cap = (int) len + 16;
    int32_t* f = (int32_t*) malloc(sizeof(int32_t) * (size_t) cap);
    const int nf = tok_encode(t, text, 0, f, cap);
    free(text);
    const int bos = tok_bos(t) >= 0 ? tok_bos(t) : 0;
    tok_close(t);
    if (nq < 0 || nf < 0 || nf < n - 1 - nq) { snprintf(err, (size_t) errlen, "%s: %d tokens, %d needed", text_path, nf, n - 1 - nq); free(f); return NULL; }
    int32_t* ids = (int32_t*) malloc(sizeof(int32_t) * (size_t) n);
    ids[0] = bos;
    memcpy(ids + 1, f, sizeof(int32_t) * (size_t) (n - 1 - nq));
    memcpy(ids + n - nq, q, sizeof(int32_t) * (size_t) nq);
    free(f);
    return ids;
}

int main(int argc, char** argv) {
    plat_init(&argc, &argv);
    setvbuf(stdout, NULL, _IONBF, 0);   // every line out at once (a crash loses nothing)
    const int reps = atoi(opt(argc, argv, "--repeats", "5")), ndec = atoi(opt(argc, argv, "--decode", "256"));
    const char* model = opt(argc, argv, "--model", NULL);
    char err[512] = "";
    int n = 0;
    int32_t* p = NULL;
    if (!model) { fprintf(stderr, "--model DIR required\n"); return 2; }
    if (opt(argc, argv, "--ids", NULL)) {
        size_t len = 0;
        p = (int32_t*) plat_slurp(opt(argc, argv, "--ids", ""), &len);
        n = (int) (len / 4);
    } else if (opt(argc, argv, "--text", NULL)) {
        n = atoi(opt(argc, argv, "--ctx", "1024"));
        p = bench_prompt(model, opt(argc, argv, "--text", ""), n, err, sizeof err);
        if (!p) { fprintf(stderr, "prompt: %s\n", err); return 2; }
    }
    if (!p || n < 2) { fprintf(stderr, "--ids FILE (int32 prompt ids) or --text FILE --ctx N required\n"); return 2; }
    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = model;
    o.resource_dir = opt(argc, argv, "--res", "out/res");
    o.max_seqs = 1;
    o.kv_format = !strcmp(opt(argc, argv, "--kv", "bf16"), "q8") ? ENG_KV_Q8 : ENG_KV_BF16;
    o.kv_tokens = n + ndec + 64;
    MachineState ms0 = machine_state();
    print_machine_state("start_", ms0);
    const double tl = now_s();
    Eng* e = eng_open(&o, err, sizeof err);
    if (!e) { fprintf(stderr, "eng_open: %s\n", err); return 2; }
    printf("engine=%s\nbucket=%s\nn_prompt=%d\ndecode_tokens=%d\nrepeats=%d\nload_s=%.2f\n", eng_describe(e),
           opt(argc, argv, "--bucket", "?"), n, ndec, reps, now_s() - tl);
    fflush(stdout);
    int32_t* out = (int32_t*) malloc(sizeof(int32_t) * (size_t) ndec);
    double ttft[64], pre[64], dec[64];
    int seq = 0;
    if (atoi(opt(argc, argv, "--warmup", "1")) && (eng_prefill(e, 0, p, n) || eng_generate(e, &seq, 1, ndec, ENG_MODE_AR, out, NULL))) {
        fprintf(stderr, "warm-up failed\n");
        return 3;
    }
    const double cpu0 = host_cpu_busy_s(), self0 = self_cpu_s();
    TelemMark tm = telem_mark();
    const long long g0 = telem_self_gpu_ns();
    const double t_start = now_s();
    uint32_t h = 2166136261u;
    int nr = reps < 64 ? reps : 64;
    for (int r = 0; r < nr; ++r) {
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
    stats("ttft_s", ttft, nr);
    stats("prefill_tok_s", pre, nr);
    stats("decode_tok_s", dec, nr);
    printf("output_fnv=%08x\n", h);
    EngMem m;
    eng_mem(e, &m);
    const ProcMem pm = proc_mem();
    printf("weights_mb=%.2f\nlut_mb=%.3f\nweights_resident_mb=%.2f\nkv_mb=%.2f\nscratch_mb=%.2f\nstaging_mb=%.2f\ngpu_allocated_mb=%.2f\n",
           m.weights / 1e6, m.lut / 1e6, (m.weights + m.lut) / 1e6, m.kv / 1e6, m.scratch / 1e6, m.staging / 1e6, m.gpu_allocated / 1e6);
    printf("phys_footprint_mb=%.2f\nphys_footprint_peak_mb=%.2f\nrss_mb=%.2f\n", pm.phys_footprint_mb * 1.048576,
           pm.phys_footprint_peak_mb * 1.048576, pm.rss_mb * 1.048576);
    printf("timed_wall_s=%.2f\nother_cpu_cores=%.2f\ngpu_busy_pct=%.2f\nself_gpu_pct=%.2f\n", wall, other, tr.gpu_busy_pct,
           g0 >= 0 && g1 >= 0 ? 100.0 * (double) (g1 - g0) / 1e9 / wall : -1);
    if (opt_flag(argc, argv, "--timing")) {   // per-kernel-group GPU time of the prefill and a 64-token decode
        const char* names[MOVA_TG_N] = {"embed_norm", "attn_proj", "values", "attention", "router", "experts", "shared_dense", "head"};
        eng_mova_timing(e, 1);
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
