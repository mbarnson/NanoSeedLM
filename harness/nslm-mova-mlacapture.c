// harness/nslm-mova-mlacapture.c - calibration statistics of an MLA model's projection inputs, for the activation-
// weighted seed search over the MLA projections (nslm-moe --scope mla --act FILE).
//
//   nslm-mova-mlacapture --model DIR --text CALIB.txt --out FILE [--ctx 2048] [--max-windows N] [--res out/res] [--xtx DIR]
//
// The text is tokenized whole and cut into windows of BOS + (ctx - 1) tokens (as tools/mova_capture.py); each window is
// prefilled with the engine's MLA capture on (mova_ext.h: prompts attend in latent space), which sums, per layer and
// column, the squares of the attention input, the value experts' output (MoVA layers), the heads' queries and the
// latent outputs over every computed row.
//
// FILE ("NSLMMLA1"): int32 n_layer, d, kvd, qd, n_head; int64 rows; then per layer: int32 r, double x[d], v[kvd],
// q[qd], o[n_head * r].
// --xtx DIR: also each input's X^T X (capture mode 2, for nslm-moe --scope mla --xtx), DIR/L<l>_<site>.xtx for site x
// (kv_a_x, k_rope_proj), v (kv_a_v; MoVA layers), q (q_rope_mix, q_lat; per head), o (v_up; per head): "NSLMXTX2",
// int32 dim, int32 blocks, int64 rows, float sums[blocks][dim][dim].
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "engine_api.h"
#include "mova_cfg.h"
#include "mova_ext.h"
#include "platform.h"
#include "tokenizer.h"

int main(int argc, char** argv) {
    plat_init(&argc, &argv);
    setvbuf(stdout, NULL, _IONBF, 0);
    const char* model = opt(argc, argv, "--model", NULL), *text = opt(argc, argv, "--text", NULL), *out = opt(argc, argv, "--out", NULL);
    const int ctx = atoi(opt(argc, argv, "--ctx", "2048")), maxw = atoi(opt(argc, argv, "--max-windows", "0"));
    if (!model || !text || !out || ctx < 16) {
        fprintf(stderr, "usage: nslm-mova-mlacapture --model DIR --text CALIB.txt --out FILE [--ctx 2048] [--max-windows N]\n");
        return 2;
    }
    char err[512], path[2048];
    MovaCfg c;
    if (mova_cfg_load(&c, model, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    if (!c.mla) { fprintf(stderr, "%s: not an MLA model\n", model); return 1; }
    snprintf(path, sizeof path, "%s/tokenizer.json", model);
    Tok* t = tok_open(path, err, sizeof err);
    if (!t) { fprintf(stderr, "%s\n", err); return 1; }
    size_t len = 0;
    char* s = plat_slurp(text, &len);
    if (!s) { fprintf(stderr, "cannot read %s\n", text); return 1; }
    int32_t* ids = (int32_t*) malloc(sizeof(int32_t) * (len + 16));
    const int n = tok_encode(t, s, 0, ids, (int) len + 16);
    const int bos = tok_bos(t) >= 0 ? tok_bos(t) : 0;
    free(s);
    tok_close(t);
    int nw = n < 0 ? 0 : n / (ctx - 1);
    if (maxw > 0 && nw > maxw) nw = maxw;
    printf("%s: %d tokens, %d windows of BOS + %d\n", text, n, nw, ctx - 1);
    if (nw < 1) return 1;
    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = model;
    o.resource_dir = opt(argc, argv, "--res", "out/res");
    o.max_seqs = 1;
    o.kv_tokens = ctx + 64;
    Eng* e = eng_open(&o, err, sizeof err);
    if (!e) { fprintf(stderr, "eng_open: %s\n", err); return 1; }
    const char* xdir = opt(argc, argv, "--xtx", NULL);
    if (eng_mla_capture(e, xdir ? 2 : 1)) { fprintf(stderr, "this engine has no MLA capture%s\n", xdir ? " with X^T X" : ""); return 1; }
    int32_t* w = (int32_t*) malloc(sizeof(int32_t) * (size_t) ctx);
    const double t0 = now_s();
    for (int i = 0; i < nw; ++i) {
        w[0] = bos;
        memcpy(w + 1, ids + (size_t) i * (ctx - 1), sizeof(int32_t) * (size_t) (ctx - 1));
        if (eng_prefill(e, 0, w, ctx)) { fprintf(stderr, "window %d: prefill failed\n", i); return 1; }
        if (i % 10 == 0 || i == nw - 1) printf("  window %d/%d  %.0f s\n", i + 1, nw, now_s() - t0);
    }
    const int kvd = c.n_kv * c.head_dim, qd = c.n_head * c.head_dim;
    char tmp[2100];
    snprintf(tmp, sizeof tmp, "%s.tmp", out);
    FILE* f = fopen(tmp, "wb");
    const int32_t hd[5] = {c.n_layer, c.d, kvd, qd, c.n_head};
    int64_t rows = 0;
    eng_mla_capture_read(e, 0, NULL, NULL, NULL, NULL, &rows);
    int ok = f && fwrite("NSLMMLA1", 1, 8, f) == 8 && fwrite(hd, 4, 5, f) == 5 && fwrite(&rows, 8, 1, f) == 1;
    double* x = (double*) malloc(sizeof(double) * (size_t) (c.d + kvd + qd + c.n_head * 1024));
    for (int l = 0; l < c.n_layer && ok; ++l) {
        const int32_t r = c.mla_rank[l];
        const size_t nn = (size_t) (c.d + kvd + qd + c.n_head * r);
        ok = !eng_mla_capture_read(e, l, x, x + c.d, x + c.d + kvd, x + c.d + kvd + qd, NULL) && fwrite(&r, 4, 1, f) == 1 &&
             fwrite(x, sizeof(double), nn, f) == nn;
    }
    if (f && fclose(f)) ok = 0;
    if (!ok || rename(tmp, out)) { fprintf(stderr, "cannot write %s\n", out); remove(tmp); return 1; }
    if (xdir) {
        mkdir(xdir, 0755);
        float* h = (float*) malloc(sizeof(float) * (size_t) c.n_head * 1024 * 1024);
        for (int l = 0; l < c.n_layer && h; ++l)
            for (int site = 0; site < 4; ++site) {
                if (site == 1 && l < c.first_sparse) continue;   // no value experts
                const int32_t dm[2] = {site == 0 ? c.d : site == 1 ? kvd : site == 2 ? c.head_dim : c.mla_rank[l], site < 2 ? 1 : c.n_head};
                const size_t nn = (size_t) dm[0] * dm[0] * dm[1];
                snprintf(path, sizeof path, "%s/L%d_%c.xtx", xdir, l, "xvqo"[site]);
                snprintf(tmp, sizeof tmp, "%s.tmp", path);
                FILE* g = fopen(tmp, "wb");
                int gk = g && !eng_mla_capture_xtx(e, l, site, h) && fwrite("NSLMXTX2", 1, 8, g) == 8 && fwrite(dm, 4, 2, g) == 2 &&
                         fwrite(&rows, 8, 1, g) == 1 && fwrite(h, sizeof(float), nn, g) == nn;
                if (g && fclose(g)) gk = 0;
                if (!gk || rename(tmp, path)) { fprintf(stderr, "cannot write %s\n", path); remove(tmp); return 1; }
            }
        free(h);
        printf("wrote %s/L*_{x,v,q,o}.xtx\n", xdir);
    }
    printf("wrote %s: %lld rows, %d layers (%.0f s)\n", out, (long long) rows, c.n_layer, now_s() - t0);
    eng_close(e);
    free(x); free(w); free(ids);
    return 0;
}
