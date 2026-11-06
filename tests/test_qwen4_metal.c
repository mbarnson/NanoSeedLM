// tests/test_qwen4_metal.c - the qwen4 Metal engine (engine/qwen4_gpu.m) on tests/data/qwen4_tiny against mlx-vlm's
// float32 logits, with test_qwen4_engine.c's limits: matvec rows (within BF16 rounding; any split of up to 8 rows, bit for
// bit), one GEMM row of 48, the same logits twice (the QSA selection in block order), and a prompt of 199 rows in one
// forward or three, bit for bit. Then tests/data/qwen4_tiny_attn (strong attention) against nslm/qwen4_ref.h, with
// test_qwen4_engine.c's bounds, also with the decode rows in key splits. Bit for bit across row splits is what MTP's
// verify forwards need: drafts then change no logits (plan gate MB, tests/test_qwen4_mtp.c). Last, the tiny model with
// trellis experts (L 12, K 3 banks, as test_qwen4_engine.c's tcq_check) against the C reference: a prompt of 20 rows
// (the trellis GEMM) and 8 decode rows (the gathers); then with the MTP head's experts as a bank too (its drafts, gate MB).
#define _DARWIN_C_SOURCE   // setenv
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "format.h"
#include "moe.h"
#include "qwen4_eng.h"
#include "qwen4_ref.h"
#include "trellis.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static int argmax(const float* v, int n) {
    int a = 0;
    for (int i = 1; i < n; ++i) if (v[i] > v[a]) a = i;
    return a;
}

static int copy_file(const char* from, const char* to) {
    FILE* a = fopen(from, "rb"), *b = a ? fopen(to, "wb") : NULL;
    char buf[1 << 16];
    size_t got;
    while (a && b && (got = fread(buf, 1, sizeof buf, a)) > 0) fwrite(buf, 1, got, b);
    if (a) fclose(a);
    return b && !fclose(b) ? 0 : -1;
}
// The tiny model with every layer's experts as trellis banks (L 12, K 3, T = a row; random weights encoded by the C
// encoder), as test_qwen4_engine.c's tcq_fixture: the banks replace the folder's affine experts. head: also the MTP head's
// experts, as the bank L{n_layer} (ns_add_banks' head rule)
static int tcq_fixture(const char* dir, int head, char* err, int errlen) {
    char a[1024], b[1024];
    mkdir(dir, 0755);
    snprintf(b, sizeof b, "%s/config.json", dir);
    snprintf(a, sizeof a, "%s/model.safetensors", dir);
    Qwen4Cfg c;
    char m[1024];
    snprintf(m, sizeof m, "%s/model-mtp.safetensors", dir);   // the MTP head (mtp.*)
    if (copy_file("tests/data/qwen4_tiny/config.json", b) || copy_file("tests/data/qwen4_tiny/model.safetensors", a) ||
        copy_file("tests/data/qwen4_tiny/model-mtp.safetensors", m) || qwen4_cfg_load(&c, dir, err, errlen)) {
        if (!err[0]) snprintf(err, (size_t) errlen, "cannot copy the tiny model to %s", dir);
        return -1;
    }
    const char* proj[3] = {"gate_proj", "up_proj", "down_proj"};
    uint32_t r = 777;
    for (int l = 0; l < c.n_layer + (head != 0); ++l)
        for (int p = 0; p < 3; ++p) {
            const int R = p == 2 ? c.d : c.ff_exp, C = p == 2 ? c.ff_exp : c.d, E = c.n_exp, T = C;
            const size_t nw = nslm_tq_row_words(C, T, 12, 3);
            MoeBlkHeader h;
            memset(&h, 0, sizeof h);
            memcpy(h.magic, "NSLMBLKT", 8);
            h.rows = (uint32_t) R; h.cols = (uint32_t) C; h.n_experts = (uint32_t) E; h.n_seeds = 12; h.n_exp = 3; h.refit = (uint32_t) T;
            uint32_t* words = (uint32_t*) calloc(nw * (size_t) E * R, 4);
            uint16_t* sc = (uint16_t*) calloc((size_t) E * R, 2);
            float* w = (float*) malloc(sizeof(float) * (size_t) C), *rel = (float*) calloc(2 * (size_t) E, 4);
            NsTqScratch z;
            nslm_tq_scratch_init(&z, T, 12, 3);
            for (int k = 0; k < E * R; ++k) {
                for (int j = 0; j < C; ++j) { r = r * 1664525u + 1013904223u; w[j] = ((float) (r >> 8) / 16777216.0f - 0.5f) * 0.5f; }
                nslm_tq_encode_row(w, NULL, C, T, &z, words + nw * (size_t) k, sc + k, NULL);
            }
            nslm_tq_scratch_free(&z);
            snprintf(a, sizeof a, "%s/L%d_%s.blkt", dir, l, proj[p]);
            FILE* f = fopen(a, "wb");
            const int ok = f && fwrite(&h, sizeof h, 1, f) == 1 && fwrite(rel, 4, 2 * (size_t) E, f) == 2 * (size_t) E &&
                           fwrite(words, 4, nw * E * R, f) == nw * E * R && fwrite(sc, 2, (size_t) E * R, f) == (size_t) E * R;
            if (f) fclose(f);
            free(words); free(sc); free(w); free(rel);
            if (!ok) { snprintf(err, (size_t) errlen, "cannot write %s", a); return -1; }
        }
    return 0;
}
// Trellis experts (NS_TCQ banks) in the Metal engine against the C reference, with tcq_check's limits
// Q4EOpts.resource_dir reaches the engine (nslm-serve's ../res beside the binary, --res) when NSLM_RES is unset: a folder
// without the metallibs fails with its path in the error, and out/res by its absolute path opens from another directory.
static void res_check(void) {
    char err[512] = "", cwd[1024], res[1100], model[1100];
    const char* env = getenv("NSLM_RES");
    char* saved = env ? strdup(env) : NULL;
    unsetenv("NSLM_RES");
    if (!getcwd(cwd, sizeof cwd)) { CHECK(0, "getcwd"); free(saved); return; }
    snprintf(res, sizeof res, "%s/%s", cwd, saved ? saved : "out/res");
    snprintf(model, sizeof model, "%s/tests/data/qwen4_tiny", cwd);
    if (saved && saved[0] == '/') snprintf(res, sizeof res, "%s", saved);
    Q4EOpts o;
    memset(&o, 0, sizeof o);
    o.max_pos = 64;
    o.resource_dir = "/nonexistent-nslm-res";
    Q4Eng* e = q4e_open(model, &o, err, sizeof err);
    CHECK(!e && strstr(err, "/nonexistent-nslm-res/kernels_moe.metallib"), "resource_dir without metallibs: %s", e ? "opened" : err);
    if (e) q4e_close(e);
    o.resource_dir = res;
    CHECK(!chdir("/tmp"), "chdir /tmp");
    err[0] = 0;
    e = q4e_open(model, &o, err, sizeof err);
    CHECK(e != NULL, "resource_dir %s from /tmp: %s", res, err);
    if (e) q4e_close(e);
    CHECK(!chdir(cwd), "chdir back");
    if (saved) setenv("NSLM_RES", saved, 1);
    free(saved);
    printf("resource_dir: a folder without the metallibs fails (%s), out/res by its absolute path opens from /tmp\n",
           "named in the error");
}
static void tcq_check(void) {
    char dir[512], err[512] = "";
    snprintf(dir, sizeof dir, "%s/nslm_qwen4_tiny_tcq_metal", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (tcq_fixture(dir, 0, err, sizeof err)) { CHECK(0, "trellis fixture: %s", err); return; }
    Qwen4Ref* r = qwen4_ref_open(dir, err, sizeof err);
    if (r) qwen4_ref_kv_bf16(r, 1);   // the Metal engine's BF16 KV cache (plan K)
    CHECK(r != NULL, "trellis fixture, reference: %s", err);
    if (!r) return;
    enum { N = 28, NP = 20 };
    const int V = qwen4_ref_cfg(r)->vocab;
    int32_t ids[N];
    for (int i = 0; i < N; ++i) ids[i] = (i * 37 + 11) % V;
    float* ref = (float*) malloc(sizeof(float) * (size_t) N * V), *lg = (float*) malloc(sizeof(float) * (size_t) N * V);
    CHECK(qwen4_ref_forward(r, ids, N, ref) == 0, "trellis fixture, reference forward");
    double m = 1e-30;
    for (size_t i = 0; i < (size_t) N * V; ++i) m = fmax(m, fabs((double) ref[i]));
    Q4EOpts o;
    memset(&o, 0, sizeof o);
    o.max_pos = 64;
    Q4Eng* e = q4e_open(dir, &o, err, sizeof err);
    CHECK(e != NULL, "open the trellis fixture: %s", err);
    if (e) {
        int rc = q4e_forward(e, ids, NP, lg);
        for (int t = NP; !rc && t < N; ++t) rc = q4e_forward(e, ids + t, 1, lg + (size_t) t * V);
        CHECK(rc == 0, "trellis fixture, forward: %s", q4e_error(e));
        double d = 0;
        int am = 0;
        for (int t = 0; t < N && !rc; ++t) {
            for (int v = 0; v < V; ++v) d = fmax(d, fabs((double) lg[(size_t) t * V + v] - ref[(size_t) t * V + v]) / m);
            am += argmax(lg + (size_t) t * V, V) != argmax(ref + (size_t) t * V, V);
        }
        printf("trellis experts (L 12, K 3): logits %.1e of max |logit| from the C reference, argmax differs in %d of %d rows\n", d, am, N);
        CHECK(!rc && d < 3e-2 && am <= N / 16, "trellis experts: %.1e from the reference, %d argmax rows", d, am);
        q4e_close(e);
    }
    free(ref); free(lg);
    qwen4_ref_close(r);
}

// The MTP head with trellis experts (the bank L{n_layer}, ns_add_banks' head rule): the head opens; greedy decode with 1 to
// 3 drafts a step gives the tokens of decode without drafts (plan gate MB, as test_qwen4_mtp's decode); and the head runs
// its bank (draft 1 differs from the same fixture's without the head bank, as the C reference's does)
static void tcq_head_check(void) {
    char dir[512], err[512] = "";
    snprintf(dir, sizeof dir, "%s/nslm_qwen4_tiny_tcq_head_metal", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (tcq_fixture(dir, 1, err, sizeof err)) { CHECK(0, "trellis head fixture: %s", err); return; }
    Q4EOpts o;
    memset(&o, 0, sizeof o);
    o.max_pos = 64;
    o.mtp = 3;
    Q4Eng* e = q4e_open(dir, &o, err, sizeof err);
    CHECK(e != NULL, "open the trellis head fixture: %s", err);
    if (!e) return;
    const int V = q4e_cfg(e)->vocab;
    enum { N = 12, NT = 24 };
    int32_t ids[N], plain[NT], spec[NT];
    for (int i = 0; i < N; ++i) ids[i] = (i * 41 + 7) % V;
    float* lg = (float*) malloc(sizeof(float) * (size_t) V * 4);
    for (int k = 0; k <= 3; ++k) {
        int32_t* out = k ? spec : plain;
        q4e_reset(e);
        int rc = q4e_forward(e, ids, N - 1, NULL) || q4e_forward(e, ids + N - 1, 1, lg), got = 0, acc = 0;
        while (!rc && got < NT) {
            int32_t row[4];
            row[0] = argmax(lg, V);
            if (!k) { out[got++] = row[0]; rc = q4e_forward(e, row, 1, lg); continue; }
            int m = 1;
            rc = q4e_draft(e, k, row + 1) || q4e_forward_keep(e, row, k + 1, lg);
            while (!rc && m <= k && argmax(lg + (size_t) (m - 1) * V, V) == row[m]) ++m;
            if (!rc) rc = q4e_accept(e, m);
            for (int i = 0; i < m && got < NT; ++i) out[got++] = row[i];
            acc += m - 1;
            memmove(lg, lg + (size_t) (m - 1) * V, sizeof(float) * (size_t) V);
        }
        CHECK(rc == 0, "trellis head, %d drafts: %s", k, q4e_error(e));
        if (k) {
            int same = 0;
            while (same < NT && spec[same] == plain[same]) ++same;
            printf("trellis head, %d drafts a step: %d accepted, the first %d of %d tokens as without drafts\n", k, acc, same, NT);
            CHECK(same == NT, "trellis head: %d drafts change token %d", k, same);
        }
    }
    {   // the head runs its bank: against the same fixture without the head bank (the same main model, so the same bonus),
        // draft 1 differs. The prompt (i 85 + 7) mod 512 is one where the C reference's draft 1 differs (171 against 511).
        char d0[512];
        snprintf(d0, sizeof d0, "%s/nslm_qwen4_tiny_tcq_metal", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");   // tcq_check's
        Q4Eng* a = q4e_open(d0, &o, err, sizeof err);
        int32_t p[N], da[3] = {0}, dt[3] = {0};
        for (int i = 0; i < N; ++i) p[i] = (i * 85 + 7) % V;
        float* la = (float*) malloc(sizeof(float) * (size_t) V * N), *lt = (float*) malloc(sizeof(float) * (size_t) V * N);
        q4e_reset(e);
        const int ok = a && !q4e_forward(a, p, N, la) && !q4e_draft(a, 3, da) && !q4e_forward(e, p, N, lt) && !q4e_draft(e, 3, dt);
        const int ba = argmax(la + (size_t) (N - 1) * V, V), bt = argmax(lt + (size_t) (N - 1) * V, V);
        printf("trellis head: bonus %d, drafts %d %d %d; without the head bank: bonus %d, drafts %d %d %d\n", bt, dt[0], dt[1], dt[2], ba, da[0],
               da[1], da[2]);
        CHECK(ok && ba == bt && da[0] != dt[0], "the head bank: bonus %d / %d, draft 1 %d / %d (%s)", bt, ba, dt[0], da[0], a ? q4e_error(a) : err);
        q4e_close(a);
        free(la); free(lt);
    }
    free(lg);
    q4e_close(e);
}

int main(void) {
    char err[512] = "";
    StFile st;
    if (st_open_file(&st, "tests/data/qwen4_tiny_golden.safetensors", err, sizeof err)) { printf("FAIL: %s\n", err); return 1; }
    const StEntry* ei = st_find(&st, "ids");
    const StEntry* el = st_find(&st, "logits");
    const int n = (int) ei->shape[0], V = (int) el->shape[1];
    const int32_t* ids = (const int32_t*) st_data(&st, ei);
    const float* want = (const float*) st_data(&st, el);
    Q4EOpts o;   // zeroed, so a new field needs no change here
    memset(&o, 0, sizeof o);
    o.max_pos = 256;
    Q4Eng* e = q4e_open("tests/data/qwen4_tiny", &o, err, sizeof err);
    if (!e) { printf("FAIL: open: %s\n", err); return 1; }
    float* got = (float*) malloc(sizeof(float) * (size_t) n * V);
    float* twice = (float*) malloc(sizeof(float) * (size_t) n * V);
    float* again = (float*) malloc(sizeof(float) * (size_t) n * V);
    float* big = (float*) malloc(sizeof(float) * (size_t) n * V);
    const int rows[8] = {7, 1, 1, 8, 8, 8, 8, 7};
    for (int i = 0, at = 0; i < 8; at += rows[i++]) CHECK(q4e_forward(e, ids + at, rows[i], got + (size_t) at * V) == 0, "forward: %s", q4e_error(e));
    CHECK(q4e_pos(e) == n, "position %d", q4e_pos(e));
    q4e_reset(e);
    for (int i = 0, at = 0; i < 8; at += rows[i++]) CHECK(q4e_forward(e, ids + at, rows[i], twice + (size_t) at * V) == 0, "again: %s", q4e_error(e));
    int ndiff = 0;
    for (size_t j = 0; j < (size_t) n * V; ++j) ndiff += got[j] != twice[j];
    printf("the same rows twice: %d logits differ\n", ndiff);
    CHECK(ndiff == 0, "the logits depend on the run (QSA selection order?)");
    q4e_reset(e);
    for (int at = 0; at < n; at += 8) CHECK(q4e_forward(e, ids + at, n - at < 8 ? n - at : 8, again + (size_t) at * V) == 0, "rows of 8: %s", q4e_error(e));
    q4e_reset(e);
    CHECK(q4e_forward(e, ids, n, big) == 0, "one row of %d: %s", n, q4e_error(e));
    double worst = 0, scale = 0, same = 0, wbig = 0;
    int argmax_diff = 0, argmax_big = 0;
    for (int t = 0; t < n; ++t) {
        for (int v = 0; v < V; ++v) {
            const size_t i = (size_t) t * V + v;
            worst = fmax(worst, fabs((double) got[i] - want[i]));
            scale = fmax(scale, fabs((double) want[i]));
            same = fmax(same, fabs((double) got[i] - again[i]));
            wbig = fmax(wbig, fabs((double) big[i] - want[i]));
        }
        const int b = argmax(want + (size_t) t * V, V);
        argmax_diff += argmax(got + (size_t) t * V, V) != b;
        argmax_big += argmax(big + (size_t) t * V, V) != b;
    }
    int first = -1;   // the first row whose logits depend on the split
    for (int t = 0; t < n && first < 0; ++t)
        for (int v = 0; v < V; ++v) if (got[(size_t) t * V + v] != again[(size_t) t * V + v]) { first = t; break; }
    printf("engine logits: max |Metal - mlx-vlm| %.2e of max |logit| %.3f, argmax differs in %d of %d rows; rows of 8 vs uneven %.1e"
           " (from row %d)\n", worst, scale, argmax_diff, n, same, first);
    printf("one GEMM row of %d: max |Metal - mlx-vlm| %.2e, argmax differs in %d rows\n", n, wbig, argmax_big);
    CHECK(worst < 3e-2 * scale && argmax_diff <= n / 16, "logits differ from mlx-vlm beyond the BF16 matvec rounding");
    // the GEMM rounds every projection's inputs to BF16 too, on top of the BF16 outputs: twice the matvec bound
    CHECK(wbig < 6e-2 * scale && argmax_big <= n / 16, "GEMM-row logits differ from mlx-vlm beyond the BF16 rounding");
    // the matvecs run without fast math, so one token's row sums in the same order for any row count
    CHECK(same == 0, "the row split changes the logits (from row %d)", first);
    q4e_close(e);
    {   // qwen4_tiny_attn (tools/qwen4_tiny_attn.py): attention strong enough that zero keys move the C reference's
        // logits by about 1.7 of 1.7. The engine against the reference, in decode rows and in one prompt forward.
        Qwen4Ref* r = qwen4_ref_open("tests/data/qwen4_tiny_attn", err, sizeof err);
        if (r) qwen4_ref_kv_bf16(r, 1);   // the Metal engine's BF16 KV cache (plan K)
        CHECK(r != NULL, "reference: %s", err);
        Q4Eng* ea = r ? q4e_open("tests/data/qwen4_tiny_attn", &o, err, sizeof err) : NULL;
        CHECK(!r || ea != NULL, "open qwen4_tiny_attn: %s", err);
        float* ref = (float*) malloc(sizeof(float) * (size_t) n * V);
        float* sr8 = (float*) malloc(sizeof(float) * (size_t) n * V);   // in decode rows
        float* s1 = (float*) malloc(sizeof(float) * (size_t) n * V);    // in one forward
        if (r && ea && qwen4_ref_forward(r, ids, n, ref) == 0) {
            for (int i = 0, at = 0; i < 8; at += rows[i++]) CHECK(q4e_forward(ea, ids + at, rows[i], sr8 + (size_t) at * V) == 0, "attn rows: %s", q4e_error(ea));
            q4e_reset(ea);
            CHECK(q4e_forward(ea, ids, n, s1) == 0, "attn, one row of %d: %s", n, q4e_error(ea));
            // per row: a near-tie of two QSA blocks can flip under BF16 rounding, and strong attention then moves that
            // row a lot. So at most n / 16 rows may pass the bound, as for the argmax.
            double sr = 0, wr = 0, wg = 0;
            for (size_t i = 0; i < (size_t) n * V; ++i) sr = fmax(sr, fabs((double) ref[i]));
            int ar = 0, ag = 0, br = 0, bg = 0;
            for (int t = 0; t < n; ++t) {
                double er = 0, eg = 0;
                for (int v = 0; v < V; ++v) {
                    const size_t i = (size_t) t * V + v;
                    er = fmax(er, fabs((double) sr8[i] - ref[i]));
                    eg = fmax(eg, fabs((double) s1[i] - ref[i]));
                }
                const int a0 = argmax(ref + (size_t) t * V, V);
                ar += a0 != argmax(sr8 + (size_t) t * V, V);
                ag += a0 != argmax(s1 + (size_t) t * V, V);
                br += er > 3e-2 * sr;
                bg += eg > 6e-2 * sr;
                wr = fmax(wr, er);
                wg = fmax(wg, eg);
            }
            printf("strong attention: max |Metal - C reference| %.2e (rows), %.2e (one GEMM row) of max |logit| %.3f; past the bound in %d, %d of"
                   " %d rows; argmax differs in %d, %d\n", wr, wg, sr, br, bg, n, ar, ag);
            CHECK(br <= n / 16 && ar <= n / 16, "strong-attention logits differ from the C reference beyond the BF16 matvec rounding");
            CHECK(bg <= n / 16 && ag <= n / 16, "strong-attention GEMM-row logits differ from the C reference beyond the BF16 rounding");
            // The long-context decode path, forced: rows of 5 keys or more in k_q4_attend_g's splits of 10 keys, merged by
            // k_q4_attend_comb. Against the reference, and rows of 1 against uneven rows. This model's lists of at most 20
            // keys fall into the same 8-key tiles for any split count, so the rule that a row's kernel and split count
            // are its own (not its forward's) is checked on P3 at 510 to 770 keys instead (docs/log.md, 2026-11-05)
            setenv("Q4E_SPLIT_MIN", "5", 1);
            setenv("Q4E_SPLIT_KEYS", "10", 1);
            Q4Eng* es = q4e_open("tests/data/qwen4_tiny_attn", &o, err, sizeof err);
            unsetenv("Q4E_SPLIT_MIN");
            unsetenv("Q4E_SPLIT_KEYS");
            CHECK(es != NULL, "open with splits: %s", err);
            float* s8 = (float*) malloc(sizeof(float) * (size_t) n * V);
            for (int i = 0, at = 0; es && i < 8; at += rows[i++]) CHECK(q4e_forward(es, ids + at, rows[i], s8 + (size_t) at * V) == 0, "split rows: %s", q4e_error(es));
            if (es) q4e_reset(es);
            for (int t = 0; es && t < n; ++t) CHECK(q4e_forward(es, ids + t, 1, s1 + (size_t) t * V) == 0, "split, one row: %s", q4e_error(es));
            int bs = 0, nd = 0;
            for (int t = 0; t < n; ++t) {
                double er = 0;
                for (int v = 0; v < V; ++v) {
                    const size_t i = (size_t) t * V + v;
                    er = fmax(er, fabs((double) s1[i] - ref[i]));
                    nd += s1[i] != s8[i];
                }
                bs += er > 3e-2 * sr;
            }
            printf("strong attention in key splits: past the bound in %d of %d rows; rows of 1 vs uneven: %d logits differ\n", bs, n, nd);
            CHECK(bs <= n / 16, "split-attention logits differ from the C reference beyond the BF16 matvec rounding");
            CHECK(nd == 0, "split attention: a row's logits depend on the rows beside it");
            free(s8);
            q4e_close(es);
        } else CHECK(!r || !ea, "reference forward: %s", r ? qwen4_ref_error(r) : "");
        free(ref); free(sr8); free(s1);
        q4e_close(ea);
        qwen4_ref_close(r);
    }
    {   // a prompt of more than Q4E_LOGIT_ROWS rows in one forward, or in three: the next token's logits bit for bit
        const int L = 199;
        int32_t* lid = (int32_t*) malloc(sizeof(int32_t) * (L + 1));
        for (int i = 0; i <= L; ++i) lid[i] = ids[i % n];
        float* one = (float*) malloc(sizeof(float) * (size_t) V);
        float* three = (float*) malloc(sizeof(float) * (size_t) V);
        e = q4e_open("tests/data/qwen4_tiny", &o, err, sizeof err);
        CHECK(e != NULL, "open: %s", err);
        if (e) {
            CHECK(q4e_forward(e, lid, L, NULL) == 0 && q4e_forward(e, lid + L, 1, one) == 0, "one forward of %d rows: %s", L, q4e_error(e));
            q4e_reset(e);
            CHECK(q4e_forward(e, lid, 64, NULL) == 0 && q4e_forward(e, lid + 64, 64, NULL) == 0 && q4e_forward(e, lid + 128, L - 128, NULL) == 0 &&
                  q4e_forward(e, lid + L, 1, three) == 0, "three forwards: %s", q4e_error(e));
            int diff = 0;
            for (int v = 0; v < V; ++v) diff += one[v] != three[v];
            printf("a prompt of %d rows in one forward or three: %d logits differ\n", L, diff);
            CHECK(diff == 0, "the prompt's row split changes the next token's logits");
            q4e_close(e);
        }
        free(lid); free(one); free(three);
    }
    free(got); free(twice); free(again); free(big);
    st_close_file(&st);
    tcq_check();
    tcq_head_check();
    res_check();
    printf("test_qwen4_metal: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
