// tests/kernel_backend_cuda.c - tests/kernel_backend.h on the CUDA kernels (engine/kernels_cuda.h), as the CUDA engine
// launches them.  Each call uploads its inputs, runs on the default stream, copies the outputs back and frees.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernel_backend.h"
#include "kernels_cuda.h"
#include "lfsr.h"

static uint32_t *g_stab24, *g_stab32;   // the seed tables: lfsr_stream24 (SEED4) and lfsr_stream32 (SEED4P4)
static void* g_live[64];                 // this call's device allocations
static int g_nlive;

static void* dev(const void* p, size_t n) {   // a device copy of n bytes (zeros when p is NULL)
    void* d = NULL;
    if (cudaMalloc(&d, n ? n : 16) != cudaSuccess) return NULL;
    if (p) cudaMemcpy(d, p, n, cudaMemcpyHostToDevice);
    else cudaMemset(d, 0, n ? n : 16);
    g_live[g_nlive++] = d;
    return d;
}
static int done(const char* what, void* host, const void* d, size_t n) {   // sync, copy back, free the call's buffers
    cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess && host) e = cudaMemcpy(host, d, n, cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaGetLastError();
    for (int i = 0; i < g_nlive; ++i) cudaFree(g_live[i]);
    g_nlive = 0;
    if (e != cudaSuccess) { printf("FAIL: %s: %s\n", what, cudaGetErrorString(e)); return -1; }
    return 0;
}
// an output before the last one: copied back now (0), or the call's buffers freed and -1
static int fetch(const char* what, void* host, const void* d, size_t n) {
    if (cudaDeviceSynchronize() == cudaSuccess && cudaMemcpy(host, d, n, cudaMemcpyDeviceToHost) == cudaSuccess) return 0;
    done(what, NULL, NULL, 0);
    printf("FAIL: %s: %s\n", what, cudaGetErrorString(cudaGetLastError()));
    return -1;
}

int kt_open(char* err, int errlen) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { snprintf(err, (size_t) errlen, "no CUDA device"); return -1; }
    uint32_t* g = (uint32_t*) malloc(65536 * 4);
    for (uint32_t s = 0; s < 65536; ++s) g[s] = lfsr_stream24((uint16_t) s);
    cudaMalloc((void**) &g_stab24, 65536 * 4);
    cudaMemcpy(g_stab24, g, 65536 * 4, cudaMemcpyHostToDevice);
    for (uint32_t s = 0; s < 65536; ++s) g[s] = lfsr_stream32((uint16_t) s);
    cudaMalloc((void**) &g_stab32, 65536 * 4);
    cudaMemcpy(g_stab32, g, 65536 * 4, cudaMemcpyHostToDevice);
    free(g);
    if (cudaGetLastError() != cudaSuccess) { snprintf(err, (size_t) errlen, "CUDA init failed"); return -1; }
    return 0;
}
void kt_close(void) {
    cudaFree(g_stab24);
    cudaFree(g_stab32);
}
const char* kt_name(void) { return "CUDA"; }
int kt_mm_tile(void) { return MMT_BN; }
int kt_attn_key_tile(void) { return FA_BK; }
int kt_seed_gemm_exact(int on) {
    kc_seed_gemm_f32(on);
    return 1;
}

// The weight's slices (device addresses in freshly uploaded streams) and its seed table.
static void slices(const KtWeight* w, WSlice* out, const uint32_t** G) {
    const uint8_t* W = (const uint8_t*) dev(w->w, w->wn);
    const uint8_t* S = (const uint8_t*) dev(w->s, w->sn);
    const uint8_t* B = (const uint8_t*) dev(w->b, w->bn);
    const uint8_t* E = (const uint8_t*) dev(w->e, w->en);
    const size_t rk = (size_t) w->R * w->K;
    for (int s = 0; s < w->S; ++s) {
        WSlice* o = &out[s];
        memset(o, 0, sizeof *o);
        switch (w->fmt) {
        case MF_BF16: o->p[0] = W + s * rk * 2; break;
        case MF_Q8: case MF_Q4:
            o->p[0] = W + s * rk / (w->fmt == MF_Q8 ? 1 : 2);
            o->p[1] = S + s * rk / 64 * 2;
            o->p[2] = B + s * rk / 64 * 2;
            break;
        default:   // seeds
            o->p[0] = W + s * rk / 8 * 2;
            o->p[1] = S + s * rk / 8 * 2;
            if (w->fmt == MF_SEED4P4) o->p[3] = E + s * rk / 16;
            o->eb = ((const int32_t*) w->b)[s];
        }
    }
    *G = w->fmt == MF_SEED4P4 ? g_stab32 : g_stab24;
}

int kt_mv(const KtWeight* w, int T, int add, const float* x, float* y) {
    WSlice ws[8];
    const uint32_t* G;
    slices(w, ws, &G);
    const size_t yn = sizeof(float) * (size_t) T * w->R;
    float* dy = (float*) dev(y, yn);
    kc_mv(0, w->fmt, ws[0], w->K, w->R, (const float*) dev(x, sizeof(float) * (size_t) T * w->K), w->K, dy, w->R, T, add, G);
    return done("kc_mv", y, dy, yn);
}
int kt_mv_sel(const KtWeight* w, int P, int xdiv, const int32_t* sel, const float* x, float* y) {
    WSlice ws[8];
    const uint32_t* G;
    slices(w, ws, &G);
    const size_t yn = sizeof(float) * (size_t) P * w->R;
    float* dy = (float*) dev(NULL, yn);
    kc_mv_sel(0, w->fmt, (const WSlice*) dev(ws, sizeof(WSlice) * (size_t) w->S), w->K, w->R,
              (const float*) dev(x, sizeof(float) * (size_t) ((P + xdiv - 1) / xdiv) * w->K), w->K, dy, w->R,
              (const int32_t*) dev(sel, 4 * (size_t) P), P, xdiv, G);
    return done("kc_mv_sel", y, dy, yn);
}
int kt_mv_gu(const KtWeight* w, int P, int xdiv, const int32_t* sel, const float* x, float* a) {
    WSlice ws[8];
    const uint32_t* G;
    slices(w, ws, &G);
    const size_t yn = sizeof(float) * (size_t) P * w->R;
    float* da = (float*) dev(NULL, yn);
    const WSlice* d = (const WSlice*) dev(ws, sizeof(WSlice) * (size_t) w->S);
    kc_mv_gu(0, w->fmt, d, d + 1, w->K, w->R, (const float*) dev(x, sizeof(float) * (size_t) ((P + xdiv - 1) / xdiv) * w->K),
             w->K, da, w->R, (const int32_t*) dev(sel, 4 * (size_t) P), P, xdiv, G);
    return done("kc_mv_gu", a, da, yn);
}
int kt_mm(const KtWeight* w, int T, const float* x, float* y) {
    WSlice ws[8];
    const uint32_t* G;
    slices(w, ws, &G);
    const size_t yn = sizeof(float) * (size_t) T * w->R;
    float* dy = (float*) dev(y, yn);
    kc_mm(0, w->fmt, ws[0], w->K, w->R, (const float*) dev(x, sizeof(float) * (size_t) T * w->K), w->K, dy, w->R, T, 1, G);
    return done("kc_mm", y, dy, yn);
}
int kt_mm_grouped(const KtWeight* w, int P, int xdiv, const int32_t* perm, const MmTile* tiles, int nt, const float* x,
                  float* y) {
    WSlice ws[8];
    const uint32_t* G;
    slices(w, ws, &G);
    const size_t yn = sizeof(float) * (size_t) P * w->R;
    float* dy = (float*) dev(NULL, yn);
    kc_mm_grouped(0, w->fmt, (const WSlice*) dev(ws, sizeof(WSlice) * (size_t) w->S), w->K, w->R,
                  (const float*) dev(x, sizeof(float) * (size_t) ((P + xdiv - 1) / xdiv) * w->K), w->K, dy, w->R, xdiv,
                  (const int32_t*) dev(perm, 4 * (size_t) P), (const MmTile*) dev(tiles, sizeof(MmTile) * (size_t) nt), nt, G);
    return done("kc_mm_grouped", y, dy, yn);
}

int kt_gnorm(int d, float eps, const float* x, const uint16_t* w, float* y, int T) {
    const size_t n = sizeof(float) * (size_t) T * d;
    float* dy = (float*) dev(NULL, n);
    kc_gnorm(0, d, eps, (const float*) dev(x, n), (const uint16_t*) dev(w, 2 * (size_t) d), dy, T);
    return done("kc_gnorm", y, dy, n);
}
int kt_router(RouterArgs a, const uint16_t* W, const uint16_t* bias, const float* x, int T, int32_t* ind, float* wt) {
    int32_t* di = (int32_t*) dev(NULL, 4 * (size_t) T * a.top_k);
    float* dw = (float*) dev(NULL, 4 * (size_t) T * a.top_k);
    kc_router(0, a, (const uint16_t*) dev(W, 2 * (size_t) a.n * a.d), (const uint16_t*) dev(bias, 2 * (size_t) a.n),
              (const float*) dev(x, 4 * (size_t) T * a.d), (float*) dev(NULL, 4 * (size_t) T * a.n),
              (float*) dev(NULL, 4 * (size_t) T * a.n), di, dw, T);
    if (fetch("kc_router", ind, di, 4 * (size_t) T * a.top_k)) return -1;
    return done("kc_router", wt, dw, 4 * (size_t) T * a.top_k);
}
int kt_router_topk(RouterArgs a, const float* score, const float* sel, int T, int32_t* ind, float* wt) {
    int32_t* di = (int32_t*) dev(NULL, 4 * (size_t) T * a.top_k);
    float* dw = (float*) dev(NULL, 4 * (size_t) T * a.top_k);
    kc_router_topk(0, a, (const float*) dev(score, 4 * (size_t) T * a.n), (const float*) dev(sel, 4 * (size_t) T * a.n), di,
                   dw, T);
    if (fetch("kc_router_topk", ind, di, 4 * (size_t) T * a.top_k)) return -1;
    return done("kc_router_topk", wt, dw, 4 * (size_t) T * a.top_k);
}
int kt_swiglu(const float* g, const float* u, float* a, int n) {
    float* da = (float*) dev(NULL, 4 * (size_t) n);
    kc_swiglu(0, (const float*) dev(g, 4 * (size_t) n), (const float*) dev(u, 4 * (size_t) n), da, n);
    return done("kc_swiglu", a, da, 4 * (size_t) n);
}
int kt_combine(const float* D, const float* w, const float* shared, float* x, float* v, int d, int k, int T) {
    const float* dD = (const float*) dev(D, 4 * (size_t) T * k * d);
    const float* dw = (const float*) dev(w, 4 * (size_t) T * k);
    float* dx = (float*) dev(x, 4 * (size_t) T * d);
    float* dv = (float*) dev(NULL, 4 * (size_t) T * d);
    kc_moe_combine(0, dD, dw, (const float*) dev(shared, 4 * (size_t) T * d), dx, d, k, T);
    kc_vcombine(0, dD, dw, dv, d, k, T);
    if (fetch("kc_moe_combine", x, dx, 4 * (size_t) T * d)) return -1;
    return done("kc_vcombine", v, dv, 4 * (size_t) T * d);
}

static KvView bf16_kv(void* K, void* V) {   // every position in segment a
    KvView kv;
    memset(&kv, 0, sizeof kv);
    kv.a.k = K;
    kv.a.v = V;
    kv.b = kv.a;
    kv.nv = 1 << 30;
    kv.fmt = KV_BF16;
    return kv;
}
static int attn(AttnArgs a, float* dq, KvView kv, const RowInfo* dri, const float* g, float* o, int T, const char* what) {
    const size_t qn = 4 * (size_t) T * a.n_head * 128;
    float* dout = (float*) dev(NULL, qn);
    float* part = (float*) dev(NULL, 4 * (size_t) T * a.n_head * a.n_splits * 130);
    kc_attn(0, a, dq, kv, dri, part, (const float*) dev(g, qn), dout, T);
    return done(what, o, dout, qn);
}
int kt_rope_attn(AttnArgs a, float* q, const float* k, const float* v, uint16_t* Kc, uint16_t* Vc, int npos,
                 const RowInfo* ri, const float* inv, const float* g, float* o, int T) {
    const size_t qn = 4 * (size_t) T * a.n_head * 128, kn = 4 * (size_t) T * a.n_kv * 128, cn = 2 * (size_t) npos * a.n_kv * 128;
    float* dq = (float*) dev(q, qn);
    void *dK = dev(Kc, cn), *dV = dev(Vc, cn);
    const KvView kv = bf16_kv(dK, dV);
    const RowInfo* dri = (const RowInfo*) dev(ri, sizeof(RowInfo) * (size_t) T);
    kc_rope_kv(0, dq, (const float*) dev(k, kn), (const float*) dev(v, kn), kv, dri, (const float*) dev(inv, 4 * 64),
               a.n_head, a.n_kv, T);
    if (fetch("kc_rope_kv", q, dq, qn) || fetch("kc_rope_kv", Kc, dK, cn) || fetch("kc_rope_kv", Vc, dV, cn)) return -1;
    return attn(a, dq, kv, dri, g, o, T, "kc_rope_kv + kc_attn");
}
int kt_attn(AttnArgs a, const float* q, const uint16_t* Kc, const uint16_t* Vc, int npos, const RowInfo* ri,
            const float* g, float* o, int T) {
    const size_t qn = 4 * (size_t) T * a.n_head * 128, cn = 2 * (size_t) npos * a.n_kv * 128;
    return attn(a, (float*) dev(q, qn), bf16_kv(dev(Kc, cn), dev(Vc, cn)), (const RowInfo*) dev(ri, sizeof(RowInfo) * (size_t) T),
                g, o, T, "kc_attn");
}
int kt_attn_prefill(AttnArgs a, const float* q, const uint16_t* Kc, const uint16_t* Vc, int npos, const RowInfo* ri,
                    const float* g, float* o, int T) {
    const size_t qn = 4 * (size_t) T * a.n_head * 128, cn = 2 * (size_t) npos * a.n_kv * 128;
    float* dout = (float*) dev(NULL, qn);
    kc_attn_prefill(0, a, (const float*) dev(q, qn), bf16_kv(dev(Kc, cn), dev(Vc, cn)),
                    (const RowInfo*) dev(ri, sizeof(RowInfo) * (size_t) T), (const float*) dev(g, qn), dout, T);
    return done("kc_attn_prefill", o, dout, qn);
}
static KvView q8_kv(const KtKvQ8* kv, int npos, int nkv, void** dk, void** dv, float** dks, float** dvs) {   // segment a
    const size_t cn = (size_t) npos * nkv * 128, sn = 4 * (size_t) npos * nkv;
    *dk = dev(kv->k, cn); *dv = dev(kv->v, cn);
    *dks = (float*) dev(kv->ks, sn); *dvs = (float*) dev(kv->vs, sn);
    KvView v;
    memset(&v, 0, sizeof v);
    v.a.k = *dk; v.a.v = *dv; v.a.ks = *dks; v.a.vs = *dvs;
    v.b = v.a;
    v.nv = 1 << 30;
    v.fmt = KV_Q8;
    return v;
}
int kt_rope_kv_q8(AttnArgs a, float* q, const float* k, const float* v, KtKvQ8 kv, int npos, const RowInfo* ri,
                  const float* inv, int T) {
    const size_t qn = 4 * (size_t) T * a.n_head * 128, kn = 4 * (size_t) T * a.n_kv * 128;
    const size_t cn = (size_t) npos * a.n_kv * 128, sn = 4 * (size_t) npos * a.n_kv;
    void *dk, *dv;
    float *dks, *dvs;
    const KvView view = q8_kv(&kv, npos, a.n_kv, &dk, &dv, &dks, &dvs);
    float* dq = (float*) dev(q, qn);
    kc_rope_kv(0, dq, (const float*) dev(k, kn), (const float*) dev(v, kn), view, (const RowInfo*) dev(ri, sizeof(RowInfo) * (size_t) T),
               (const float*) dev(inv, 4 * 64), a.n_head, a.n_kv, T);
    if (fetch("kc_rope_kv (Q8)", q, dq, qn) || fetch("kc_rope_kv (Q8)", kv.k, dk, cn) || fetch("kc_rope_kv (Q8)", kv.v, dv, cn) ||
        fetch("kc_rope_kv (Q8)", kv.ks, dks, sn))
        return -1;
    return done("kc_rope_kv (Q8)", kv.vs, dvs, sn);
}
int kt_attn_q8(AttnArgs a, const float* q, KtKvQ8 kv, int npos, const RowInfo* ri, const float* g, float* o, int T) {
    void *dk, *dv;
    float *dks, *dvs;
    const KvView view = q8_kv(&kv, npos, a.n_kv, &dk, &dv, &dks, &dvs);
    return attn(a, (float*) dev(q, 4 * (size_t) T * a.n_head * 128), view, (const RowInfo*) dev(ri, sizeof(RowInfo) * (size_t) T), g,
                o, T, "kc_attn (Q8)");
}
int kt_attn_prefill_q8(AttnArgs a, const float* q, KtKvQ8 kv, int npos, const RowInfo* ri, const float* g, float* o, int T) {
    const size_t qn = 4 * (size_t) T * a.n_head * 128;
    void *dk, *dv;
    float *dks, *dvs;
    const KvView view = q8_kv(&kv, npos, a.n_kv, &dk, &dv, &dks, &dvs);
    float* dout = (float*) dev(NULL, qn);
    kc_attn_prefill(0, a, (const float*) dev(q, qn), view, (const RowInfo*) dev(ri, sizeof(RowInfo) * (size_t) T),
                    (const float*) dev(g, qn), dout, T);
    return done("kc_attn_prefill (Q8)", o, dout, qn);
}
// ---- MLA ---------------------------------------------------------------------------------------------------------------
int kt_heads_mv(int H, int O, int I, const uint16_t* W, const float* x, int xs, int hs, const float* g, float* y, int T) {
    const size_t xn = 4 * ((size_t) (T - 1) * xs + (size_t) (H - 1) * hs + I), yn = 4 * (size_t) T * H * O;
    const HmvArgs a = {H, O, I, xs, hs, g != NULL, {0}};
    float* dy = (float*) dev(NULL, yn);
    kc_heads_mv(0, a, (const uint16_t*) dev(W, 2 * (size_t) H * O * I), (const float*) dev(x, xn), g ? (const float*) dev(g, yn) : NULL,
                dy, T);
    return done("kc_heads_mv", y, dy, yn);
}
// The MLA cache (RoPE keys [npos][128], latents [npos][r]) split in two segments at npos / 2, as the engine's VRAM and
// host rows
static KvView mla_kv(void* K, void* V, int npos, int r) {
    KvView kv;
    memset(&kv, 0, sizeof kv);
    kv.nv = npos / 2;
    kv.a.k = K;
    kv.a.v = V;
    kv.b.k = (uint16_t*) K + (size_t) kv.nv * 128;
    kv.b.v = (uint16_t*) V + (size_t) kv.nv * r;
    kv.fmt = KV_BF16;
    return kv;
}
int kt_mla_rope(MlaArgs a, float* qr, const float* kr, const float* c, uint16_t* Kc, uint16_t* Vc, int npos,
                const RowInfo* ri, const float* inv, int T) {
    const size_t qn = 4 * (size_t) T * a.n_head * 128, kn = 2 * (size_t) npos * 128, vn = 2 * (size_t) npos * a.r;
    float* dq = (float*) dev(qr, qn);
    void *dK = dev(Kc, kn), *dV = dev(Vc, vn);
    kc_mla_rope(0, a, dq, (const float*) dev(kr, 4 * (size_t) T * 128), (const float*) dev(c, 4 * (size_t) T * a.r),
                mla_kv(dK, dV, npos, a.r), (const RowInfo*) dev(ri, sizeof(RowInfo) * (size_t) T), (const float*) dev(inv, 4 * 64), T);
    if (fetch("kc_mla_rope", qr, dq, qn) || fetch("kc_mla_rope", Kc, dK, kn)) return -1;
    return done("kc_mla_rope", Vc, dV, vn);
}
int kt_mla_attn(MlaArgs a, const float* ql, const float* qr, const uint16_t* Kc, const uint16_t* Vc, int npos,
                const RowInfo* ri, float* olat, int T) {
    const size_t on = 4 * (size_t) T * a.n_head * a.r;
    float* dout = (float*) dev(NULL, on);
    kc_mla_attn(0, a, (const float*) dev(ql, on), (const float*) dev(qr, 4 * (size_t) T * a.n_head * 128),
                mla_kv(dev(Kc, 2 * (size_t) npos * 128), dev(Vc, 2 * (size_t) npos * a.r), npos, a.r),
                (const RowInfo*) dev(ri, sizeof(RowInfo) * (size_t) T), (float*) dev(NULL, 4 * (size_t) T * a.n_head * a.n_splits * (a.r + 2)),
                dout, T);
    return done("kc_mla_attn", olat, dout, on);
}

int kt_embed(int fmt, const uint16_t* E, const uint32_t* q8, const uint16_t* s8, const uint16_t* b8, int V, int d,
             const int32_t* ids, int n, float* x) {
    WSlice w;
    memset(&w, 0, sizeof w);
    if (fmt == MF_Q8) {
        w.p[0] = dev(q8, (size_t) V * d);
        w.p[1] = dev(s8, 2 * (size_t) V * d / 64);
        w.p[2] = dev(b8, 2 * (size_t) V * d / 64);
    } else w.p[0] = dev(E, 2 * (size_t) V * d);
    float* dx = (float*) dev(NULL, 4 * (size_t) n * d);
    kc_embed(0, fmt, w, d, (const int32_t*) dev(ids, 4 * (size_t) n), dx, n);
    return done("kc_embed", x, dx, 4 * (size_t) n * d);
}
int kt_argmax(const float* logits, int V, int n, int32_t* out) {
    int32_t* d = (int32_t*) dev(NULL, 4 * (size_t) n);
    kc_argmax(0, (const float*) dev(logits, 4 * (size_t) V * n), d, V, n);
    return done("kc_argmax", out, d, 4 * (size_t) n);
}
