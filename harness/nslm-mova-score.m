// harness/nslm-mova-score.m - KL(P_ref || P_ours) of the MoVA engine against the MLX BF16 reference on held-out
// windows; with --routes, also the agreement of our expert routing with the reference's.
//
//   nslm-mova-score --model DIR [--res out/res] --windows PRE --ref PRE [--routes] [--max-windows N]
//                   [--name NAME] [--out FILE.kv]
//
// Windows (WINDOWS.ids/.refnll/.reftop) and reference log-probs (REF.f16.npy) come from tools/mova_export.py and
// tools/mova_score.py.  Each window is BOS + 2047 tokens (CTX); rows FIRST..CTX-2 (1024..2046) predict tokens
// 1025..2047.  Per position: the exact full-vocabulary KL(P_ref || P_ours) with P_ref renormalised from the stored f16
// log-probs (as tools/mova_score.py computes it), the NLL of both, and top-1 agreement.  --routes: per (row, sparse
// layer), our MLP top-8 and value top-4 against the reference's (WINDOWS.routes_*); a disagreement is a near-tie when
// the reference's k-th / (k+1)-th selection gap (WINDOWS.gap_*) is < 1e-3; each is logged.
#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#include "engine_api.h"
#include "mova_ext.h"

#define CTX 2048
#define FIRST 1024

static const char* opt(int argc, char** argv, const char* name, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static int has(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], name)) return 1;
    return 0;
}
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static const void* map_file(const char* path, size_t* n) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    fstat(fd, &st);
    void* m = mmap(NULL, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    *n = (size_t) st.st_size;
    return m == MAP_FAILED ? NULL : m;
}
static inline float h2f(uint16_t h) { _Float16 f; memcpy(&f, &h, 2); return (float) f; }
static int cmp_int(const void* a, const void* b) { return *(const int32_t*) a - *(const int32_t*) b; }

int main(int argc, char** argv) {
    @autoreleasepool {
        setvbuf(stdout, NULL, _IOLBF, 0);
        const char* wpre = opt(argc, argv, "--windows", NULL), *rpre = opt(argc, argv, "--ref", NULL);
        EngOpts o;
        memset(&o, 0, sizeof o);
        o.model_dir = opt(argc, argv, "--model", NULL);
        o.resource_dir = opt(argc, argv, "--res", "out/res");
        o.max_seqs = 1;
        o.kv_tokens = CTX;
        if (!o.model_dir || !wpre || !rpre) { fprintf(stderr, "usage: nslm-mova-score --model DIR --windows PRE --ref PRE [--routes]\n"); return 2; }
        char path[2048];
        size_t nids = 0, nref = 0, nnll = 0, ntop = 0;
        snprintf(path, sizeof path, "%s.ids", wpre);
        const int32_t* wins = map_file(path, &nids);
        snprintf(path, sizeof path, "%s.refnll", wpre);
        const float* refnll = map_file(path, &nnll);
        snprintf(path, sizeof path, "%s.reftop", wpre);
        const int32_t* reftop = map_file(path, &ntop);
        snprintf(path, sizeof path, "%s.f16.npy", rpre);
        const uint8_t* refmap = map_file(path, &nref);
        if (!wins || !refnll || !reftop || !refmap) { fprintf(stderr, "missing inputs (%s, %s)\n", wpre, rpre); return 2; }
        const uint16_t hl = (uint16_t) (refmap[8] | refmap[9] << 8);
        const uint16_t* ref = (const uint16_t*) (refmap + 10 + hl);
        int nw = (int) (nids / 4 / CTX);
        const int maxw = atoi(opt(argc, argv, "--max-windows", "0"));
        if (maxw > 0 && maxw < nw) nw = maxw;
        char err[512] = "";
        const double t0 = now_s();
        Eng* e = eng_open(&o, err, sizeof err);
        if (!e) { fprintf(stderr, "eng_open: %s\n", err); return 1; }
        const int V = eng_vocab(e), np = CTX - 1 - FIRST;
        printf("%s; %d windows, %d scored positions each; load %.1f s\n", eng_describe(e), nw, np, now_s() - t0);
        const int routes = has(argc, argv, "--routes");
        const int NS = 45, KM = 8, KV = 4, EM = 100, EV = 64;
        const int32_t* mlx_mlp = NULL, *mlx_val = NULL;
        const float* gap_mlp = NULL, *gap_val = NULL;
        if (routes) {
            size_t n1, n2, n3, n4;
            snprintf(path, sizeof path, "%s.routes_mlp", wpre); mlx_mlp = map_file(path, &n1);
            snprintf(path, sizeof path, "%s.routes_val", wpre); mlx_val = map_file(path, &n2);
            snprintf(path, sizeof path, "%s.gap_mlp", wpre); gap_mlp = map_file(path, &n3);
            snprintf(path, sizeof path, "%s.gap_val", wpre); gap_val = map_file(path, &n4);
            if (!mlx_mlp || !mlx_val || !gap_mlp || !gap_val) { fprintf(stderr, "--routes: missing MLX route files\n"); return 2; }
            if ((int) (n1 / 4 / ((size_t) (CTX - 1) * NS * KM)) < nw) { fprintf(stderr, "--routes: MLX routes cover fewer windows\n"); return 2; }
        }
        const size_t npos = (size_t) nw * np;
        double* kl = (double*) calloc(npos, sizeof(double));
        double* nllc = (double*) calloc(npos, sizeof(double));
        int* same = (int*) calloc(npos, sizeof(int));
        float* logits = (float*) malloc((size_t) np * V * 4);
        int32_t* rm = routes ? (int32_t*) malloc((size_t) (CTX - 1) * NS * KM * 4) : NULL;
        int32_t* rv = routes ? (int32_t*) malloc((size_t) (CTX - 1) * NS * KV * 4) : NULL;
        int64_t agree_m = 0, agree_v = 0, tot = 0, tie_m = 0, tie_v = 0, nontie_m = 0, nontie_v = 0;
        FILE* rlog = NULL;
        if (routes) {
            snprintf(path, sizeof path, "%s.route_disagreements.tsv", opt(argc, argv, "--out", "score.kv"));
            rlog = fopen(path, "w");
            if (rlog) fprintf(rlog, "window\trow\tlayer\tkind\tmlx\tours\tmlx_gap\tnear_tie\n");
        }
        for (int w = 0; w < nw; ++w) {
            const int32_t* ids = wins + (size_t) w * CTX;
            if (routes) eng_mova_routes(e, 1, CTX - 1);
            const double tw = now_s();
            if (eng_score(e, 0, ids, FIRST + 1, np, logits)) { fprintf(stderr, "eng_score failed\n"); return 1; }
            dispatch_apply((size_t) np, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t j) {
                const size_t pos = (size_t) w * np + j;
                const float* lc = logits + j * (size_t) V;
                const uint16_t* lr = ref + pos * (size_t) V;
                double mc = -1e300, mr = -1e300;
                int top = 0;
                for (int v = 0; v < V; ++v) {
                    if (lc[v] > mc) { mc = lc[v]; top = v; }
                    const double r = h2f(lr[v]);
                    if (r > mr) mr = r;
                }
                double sc = 0, sr = 0;
                for (int v = 0; v < V; ++v) { sc += exp(lc[v] - mc); sr += exp(h2f(lr[v]) - mr); }
                const double lzc = mc + log(sc), lzr = mr + log(sr);
                double k = 0;
                for (int v = 0; v < V; ++v) {
                    const double r = h2f(lr[v]) - lzr;
                    k += exp(r) * (r - (lc[v] - lzc));
                }
                kl[pos] = k;
                nllc[pos] = -(lc[ids[FIRST + 1 + j]] - lzc);
                same[pos] = top == reftop[pos];
            });
            if (routes) {
                eng_mova_routes_read(e, CTX - 1, rm, rv, NULL, NULL);
                eng_mova_routes(e, 0, 0);
                for (int r = 0; r < CTX - 1; ++r)
                    for (int s = 0; s < NS; ++s) {
                        int32_t* om = rm + ((size_t) r * NS + s) * KM, *ov = rv + ((size_t) r * NS + s) * KV;
                        qsort(om, KM, 4, cmp_int);
                        qsort(ov, KV, 4, cmp_int);
                        const size_t mi = (((size_t) w * (CTX - 1) + r) * NS + s);
                        const int32_t* xm = mlx_mlp + mi * KM, *xv = mlx_val + mi * KV;
                        const int okm = !memcmp(om, xm, KM * 4), okv = !memcmp(ov, xv, KV * 4);
                        agree_m += okm; agree_v += okv; ++tot;
                        if (!okm) {
                            const int tie = gap_mlp[mi] < 1e-3f;
                            tie_m += tie; nontie_m += !tie;
                            if (rlog) fprintf(rlog, "%d\t%d\t%d\tmlp\t%d,%d,%d,%d,%d,%d,%d,%d\t%d,%d,%d,%d,%d,%d,%d,%d\t%.3g\t%d\n", w, r, s + 3,
                                              xm[0], xm[1], xm[2], xm[3], xm[4], xm[5], xm[6], xm[7], om[0], om[1], om[2], om[3], om[4], om[5], om[6], om[7],
                                              gap_mlp[mi], tie);
                        }
                        if (!okv) {
                            const int tie = gap_val[mi] < 1e-3f;
                            tie_v += tie; nontie_v += !tie;
                            if (rlog) fprintf(rlog, "%d\t%d\t%d\tval\t%d,%d,%d,%d\t%d,%d,%d,%d\t%.3g\t%d\n", w, r, s + 3, xv[0], xv[1], xv[2], xv[3],
                                              ov[0], ov[1], ov[2], ov[3], gap_val[mi], tie);
                        }
                    }
            }
            double kw = 0;
            for (int j = 0; j < np; ++j) kw += kl[(size_t) w * np + j];
            printf("  window %d/%d  %.1f s  kld %.5f%s\n", w + 1, nw, now_s() - tw, kw / np,
                   routes ? [NSString stringWithFormat:@"  routes agree mlp %.4f val %.4f", (double) agree_m / tot, (double) agree_v / tot].UTF8String : "");
        }
        double km = 0, kv2 = 0, nr = 0, nc = 0, sm = 0;
        for (size_t i = 0; i < npos; ++i) { km += kl[i]; kv2 += kl[i] * kl[i]; nr += refnll[i]; nc += nllc[i]; sm += same[i]; }
        km /= (double) npos;
        const double se = sqrt((kv2 / (double) npos - km * km) / (double) (npos - 1));
        char kvbuf[4096];
        int off = snprintf(kvbuf, sizeof kvbuf,
                           "name=%s\nengine=%s\nwindows=%d\npositions=%zu\nkld_mean=%.7f\nkld_se=%.7f\nnll_ref=%.6f\nnll_cand=%.6f\nppl_ratio=%.6f\nsame_top1_pct=%.3f\n",
                           opt(argc, argv, "--name", "c"), eng_describe(e), nw, npos, km, se, nr / (double) npos, nc / (double) npos,
                           exp((nc - nr) / (double) npos), 100.0 * sm / (double) npos);
        if (routes)
            off += snprintf(kvbuf + off, sizeof kvbuf - (size_t) off,
                            "route_positions=%lld\nroute_agree_mlp_pct=%.4f\nroute_agree_val_pct=%.4f\nroute_disagree_mlp_neartie=%lld\n"
                            "route_disagree_mlp_other=%lld\nroute_disagree_val_neartie=%lld\nroute_disagree_val_other=%lld\n",
                            (long long) tot, 100.0 * agree_m / tot, 100.0 * agree_v / tot, (long long) tie_m, (long long) nontie_m, (long long) tie_v,
                            (long long) nontie_v);
        printf("%s", kvbuf);
        const char* outp = opt(argc, argv, "--out", NULL);
        if (outp) { FILE* f = fopen(outp, "w"); if (f) { fputs(kvbuf, f); fclose(f); } }
        if (rlog) fclose(rlog);
        (void) EM; (void) EV;
        eng_close(e);
        return 0;
    }
}
