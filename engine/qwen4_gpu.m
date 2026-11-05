// engine/qwen4_gpu.m - the qwen4 (qwen4_exp) Metal engine (engine/qwen4_eng.h): the steps of nslm/qwen4_ref.h, as
// engine/qwen4_cuda.c runs them, on the matrix kernels of engine/kernels_moe.metal and the qwen4 kernels of
// engine/kernels_qwen4.metal.  Unified memory: every weight, the routed experts included, is a no-copy buffer over the
// folder's mapping (copied only when a stream is not page-aligned), so there is no expert cache, and Q4EOpts.cache_gb /
// stream_experts / vram_experts change nothing.  Each forward of up to Q4E_MAX_ROWS rows is one command buffer, except
// that prompt rows read their routed experts back to the host once a layer to group the pairs by expert.
// The MTP head (Q4EOpts.mtp) as engine/qwen4_cuda.c runs it: one row behind each forward (keys only), and q4e_draft's
// head rows in one command buffer.  A verify forward's rows also write the recurrent state after each row to snapshot
// buffers; q4e_accept swaps the kept row's snapshot in, so a failed draft costs no copy.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen4_eng.h"

#include "kernels_moe.metal"     // argument structs and constants only
#include "kernels_qwen4.metal"
#include "model_st.h"
#include "ple.h"
#include "qwen4_ops.h"
#include "rows.h"

// The default rows of one forward without logits (Q4EOpts.rows overrides it): the grouped expert GEMMs want rows (4,096
// rows: about 80 a routed expert, 2.5 GEMM tiles; 1,024: 20, under one)
#define MAXT_DEFAULT 4096
#define LOGT Q4E_LOGIT_ROWS

typedef struct {   // a tensor of the folder: its streams (nslm/model_st.h), format, shape and affine group
    id<MTLBuffer> b[4];
    uint64_t o[4];
    int fmt, slices, rows, cols, gs;
} MW;

typedef struct {
    MW hc_down[2], hc_up[2], hc_inj[2];
    id<MTLBuffer> hc_norm[2];
    MW qkv, z, b, a, out;                           // Gated DeltaNet
    id<MTLBuffer> conv_w, A_log, dt_bias, norm_w, conv, S;
    MW q, k, v, o, ix;                              // QSA attention
    id<MTLBuffer> qn, kn, iqn, ikn, K, V, raw, blk;
    MW gate, sg, sh_g, sh_u, sh_d, eg, eu, ed;      // MoE
    MW key, val;                                    // PLE
    id<MTLBuffer> nk, nq, nc, pw, hist;
    id<MTLBuffer> kconv[Q4E_KEEP_SLOTS], kS[Q4E_KEEP_SLOTS], khist[Q4E_KEEP_SLOTS];   // the state after verify rows 0 .. ndraft - 1 (MTP)
} Layer;
_Static_assert(Q4E_KEEP_SLOTS == Q4E_MAX_DRAFTS, "the kernels' snapshot buffers: one for each verify row but the last");

struct Q4Eng {
    Qwen4Cfg c;
    NsModel nm;
    NsRows rows;
    Qwen4Ngram ng[QWEN4_MAX_LAYERS];
    PleTable pt;
    const PleTable* table;   // the GGUF PLE table, or NULL (the folder's)
    Layer L[QWEN4_MAX_LAYERS];
    MW embed, head, mix_down, mix_up;
    id<MTLBuffer> mix_norm;
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> lib, lib4;
    id<MTLResidencySet> residency;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipes;
    NSMutableArray<id<MTLBuffer>>* buffers;
    id<MTLBuffer> stab, stab32, inv;   // the P = 3 and P = 4 seed stream tables; RoPE inverse frequencies
    int max_pos, pos, nb, nple, kq;    // context; QSA block slots; PLE layers; QSA blocks a row attends
    int maxt;                          // rows of one prompt forward (Q4EOpts.rows, or MAXT_DEFAULT)
    int32_t* ids;                      // the context (host), for the PLE n-grams
    id<MTLBuffer> x, H, normed, lo, up, inj, br, qp, kk, vv, o, ixo, sc, emb, keys, qs, vals, gated;
    id<MTLBuffer> lg, wt, A, Gb, Ub, Dx, sh, sgv, logits, y, cv, qkvb, zb, bb, ab;
    id<MTLBuffer> tok, ind, sel, done, nsel, zero, perm, tiles, apart;   // apart: k_q4_attend_g's split partials
    id<MTLCommandBuffer> cb;   // the forward being encoded
    id<MTLComputeCommandEncoder> enc;
    int prof;                  // Q4E_PROFILE=1: a command buffer per stage, its GPU time added to the stage; 2: decode rows only;
                               // 3: decode rows, a command buffer per dispatch, its GPU time added to the kernel (and specialisation);
                               // 4: the same for prompt rows only
    NSString* kname;           // Q4E_PROFILE=3: the pipeline of the next dispatch
    NSMutableDictionary<NSString*, NSNumber*>* kms, *kcalls;
    int prof_skip;             // this forward's rows are not counted (Q4E_PROFILE=2 and T > MV_MAXT)
    int no_split;              // Q4E_NO_SPLIT=1: k_mv for the few-row projections too (A / B)
    int no_attg;               // Q4E_NO_ATTG=1: prompt rows through k_q4_attend too (A / B)
    int no_splitk;             // Q4E_NO_SPLITK=1: long decode rows through k_q4_attend too (A / B)
    int smin, skeys, smax;     // decode rows of smin keys or more in k_q4_attend_g's splits of skeys, at most smax splits
                               // (Q4E_SPLIT_MIN, _KEYS, _MAX)
    int no_mm64;               // Q4E_NO_MM64=1: dense prompt rows through k_mm, not k_mm64 (A / B)
    // The MTP head (Q4EOpts.mtp). Hin: its input streams, row 0 the last main row's (the next head row's input), row
    // 1 + t main row t's of this forward; Hm: its streams; dtok: q4e_draft's bonus and drafts. lrow: the last row's logits
    // in logits (-1: none); keep: a q4e_forward_keep of nkeep rows
    Layer M;
    MW m_fe, m_fh, m_down, m_up;
    id<MTLBuffer> m_ne, m_nh, m_norm, Hin, Hm, ones, mlog, dtok;
    int mtp, ndraft, lrow, keep, nkeep;   // ndraft: Q4EOpts.mtp, the most drafts (and snapshot buffers a layer)
    double stage_ms[10];
    char err[256];
};

// the stages of a forward, for Q4E_PROFILE (as engine/qwen4_cuda.c)
enum { ST_EMBED, ST_PLE, ST_HC, ST_ATTN, ST_QSA, ST_ATTEND, ST_GDN, ST_ROUTE, ST_EXPERT, ST_HEAD, ST_N };
static const char* const kStage[ST_N] = {"embed", "ple", "hyper-connections", "attention: proj", "attention: QSA", "attention: attend",
                                         "gated deltanet", "router", "experts", "head"};

static int fail(Q4Eng* e, const char* fmt, ...) {
    if (!e->err[0]) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(e->err, sizeof e->err, fmt, ap);
        va_end(ap);
    }
    return -1;
}

// ---- buffers, pipelines, commands ------------------------------------------------------------------------------------

static id<MTLBuffer> dbuf(Q4Eng* e, const void* p, size_t n) {   // a shared buffer, zeroed or a copy of p
    id<MTLBuffer> b = [e->dev newBufferWithLength:(n ? (n + 255) & ~(size_t) 255 : 256) options:MTLResourceStorageModeShared];
    if (!b) { fail(e, "out of GPU memory (%zu bytes)", n); return nil; }
    if (p) memcpy(b.contents, p, n);
    else memset(b.contents, 0, b.length);
    [e->buffers addObject:b];
    return b;
}
#define F(n) dbuf(e, NULL, sizeof(float) * (size_t) (n))

static id<MTLComputePipelineState> pipe_g(Q4Eng* e, const char* name, int fmt, int T, int gs) {
    NSString* key = [NSString stringWithFormat:@"%s|%d|%d|%d", name, fmt, T, gs];
    id<MTLComputePipelineState> p = e->pipes[key];
    if (e->prof >= 3) e->kname = key;
    if (p) return p;
    const int q4 = !strncmp(name, "k_q4_", 5);
    NSError* err = nil;
    id<MTLFunction> fn;
    if (q4) fn = [e->lib4 newFunctionWithName:[NSString stringWithUTF8String:name]];
    else {
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        const short f = (short) fmt, t = (short) T, g = (short) gs;
        [cv setConstantValue:&f type:MTLDataTypeShort atIndex:0];
        [cv setConstantValue:&t type:MTLDataTypeShort atIndex:1];
        if (gs && gs != 64) [cv setConstantValue:&g type:MTLDataTypeShort atIndex:2];
        fn = [e->lib newFunctionWithName:[NSString stringWithUTF8String:name] constantValues:cv error:&err];
    }
    if (e->prof >= 3) e->kname = key;
    p = fn ? [e->dev newComputePipelineStateWithFunction:fn error:&err] : nil;
    if (!p) { fprintf(stderr, "qwen4 engine: kernel %s: %s\n", name, err ? err.localizedDescription.UTF8String : "not found"); abort(); }
    e->pipes[key] = p;
    return p;
}
static id<MTLComputePipelineState> pipe4(Q4Eng* e, const char* name) { return pipe_g(e, name, 0, 0, 0); }

static void cmd_begin(Q4Eng* e) {
    e->cb = [e->queue commandBufferWithUnretainedReferences];
    e->enc = [e->cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
}
static int cmd_wait(Q4Eng* e) {
    [e->enc endEncoding];
    [e->cb commit];
    [e->cb waitUntilCompleted];
    const int bad = e->cb.status == MTLCommandBufferStatusError;
    if (bad) fail(e, "GPU error: %s", e->cb.error.localizedDescription.UTF8String);
    e->enc = nil;
    e->cb = nil;
    return bad ? -1 : 0;
}
// Q4E_PROFILE: the work encoded since the last mark is stage's (its command buffer's GPU time)
static void mark(Q4Eng* e, int stage) {
    if (!e->prof || e->prof >= 3 || !e->cb) return;
    id<MTLCommandBuffer> cb = e->cb;
    if (cmd_wait(e)) return;
    if (!e->prof_skip) e->stage_ms[stage] += 1e3 * (cb.GPUEndTime - cb.GPUStartTime);
    cmd_begin(e);
}
static void B(Q4Eng* e, int i, id<MTLBuffer> b, uint64_t off) { [e->enc setBuffer:b offset:off atIndex:(NSUInteger) i]; }
static void A_(Q4Eng* e, const void* p, size_t n) { [e->enc setBytes:p length:n atIndex:0]; }
// Q4E_PROFILE=3: the dispatch just encoded runs alone; its GPU time goes to its pipeline's line
static void kprof(Q4Eng* e) {
    if (e->prof < 3 || e->prof_skip || !e->cb) return;
    id<MTLCommandBuffer> cb = e->cb;
    if (cmd_wait(e)) return;
    NSString* k = e->kname ? e->kname : @"?";
    e->kms[k] = @(e->kms[k].doubleValue + 1e3 * (cb.GPUEndTime - cb.GPUStartTime));
    e->kcalls[k] = @(e->kcalls[k].intValue + 1);
    cmd_begin(e);
}
static void tgs(Q4Eng* e, uint64_t x, uint64_t y, uint64_t tx) {
    [e->enc dispatchThreadgroups:MTLSizeMake(x, y, 1) threadsPerThreadgroup:MTLSizeMake(tx, 1, 1)];
    kprof(e);
}
static void thr(Q4Eng* e, uint64_t x, uint64_t y, uint64_t tx) {
    [e->enc dispatchThreads:MTLSizeMake(x, y, 1) threadsPerThreadgroup:MTLSizeMake(tx, 1, 1)];
    kprof(e);
}

// ---- weights ---------------------------------------------------------------------------------------------------------

#define NAME(buf, fmt) do { va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap); } while (0)

// The tensor named by fmt: page-aligned streams without a copy, the others copied
static void load_w(Q4Eng* e, MW* w, const char* fmt, ...) {
    char name[192];
    NAME(name, fmt);
    const NsTensor* t = ns_find(&e->nm, name);
    *w = (MW) {0};
    if (!t) { fail(e, "%s: missing", name); return; }
    if (t->enc == NS_SEED6P8) { fail(e, "%s: P = 8 seeds not supported", name); return; }
    if ((t->enc == NS_Q4 && t->group != 32 && t->group != 64) || (t->enc == NS_Q8 && t->group != 64)) {
        fail(e, "%s: affine group %d not supported", name, t->group);
        return;
    }
    w->fmt = t->enc; w->slices = t->slices; w->rows = t->rows; w->cols = t->cols;
    w->gs = t->enc == NS_Q4 || t->enc == NS_Q8 ? t->group : 0;
    for (int s = 0; s < 4; ++s) {
        const NsStream* st = &t->s[s];
        if (!st->len) continue;
        id<MTLBuffer> b = nil;
        if (ns_mappable(&e->nm, st)) {
            b = [e->dev newBufferWithBytesNoCopy:(void*) st->p length:(st->len + NS_PAGE - 1) & ~(uint64_t) (NS_PAGE - 1)
                                         options:MTLResourceStorageModeShared deallocator:nil];
            if (b) [e->buffers addObject:b];
        } else b = dbuf(e, st->p, st->len);
        if (!b) { fail(e, "%s: cannot map stream %d", name, s); return; }
        w->b[s] = b;
    }
}
// A BF16 vector of n values (any shape) as f32
static id<MTLBuffer> load_v(Q4Eng* e, int64_t n, const char* fmt, ...) {
    char name[192];
    NAME(name, fmt);
    const NsTensor* t = ns_find(&e->nm, name);
    if (!t || t->enc != NS_BF16 || (int64_t) t->slices * t->rows * t->cols != n) { fail(e, "%s: missing, or not BF16 with %lld values", name, (long long) n); return nil; }
    id<MTLBuffer> b = F(n);
    float* v = b ? (float*) b.contents : NULL;
    for (int64_t i = 0; v && i < n; ++i) { const uint32_t u = (uint32_t) ((const uint16_t*) t->s[0].p)[i] << 16; memcpy(&v[i], &u, 4); }
    return b;
}

// Layer L named lp (attention if attn): main layer l, or the MTP head's layer (l < 0: no PLE)
static void load_layer(Q4Eng* e, Layer* L, const char* lp, int attn, int l) {
    const Qwen4Cfg* c = &e->c;
    const int HD = c->hc * c->d, cd = 2 * c->lin_hk * c->lin_dk + c->lin_hv * c->lin_dv;
    for (int b = 0; b < 2; ++b) {
        const char* hc = b ? "mlp_hyper_connection" : "attn_hyper_connection";
        L->hc_norm[b] = load_v(e, HD, "%s.%s.hc_norm.weight", lp, hc);
        load_w(e, &L->hc_down[b], "%s.%s.input_mix_weight_down.weight", lp, hc);
        load_w(e, &L->hc_up[b], "%s.%s.input_mix_weight_up.weight", lp, hc);
        load_w(e, &L->hc_inj[b], "%s.%s.block_inject_weight.weight", lp, hc);
    }
    if (attn) {
        load_w(e, &L->q, "%s.self_attn.q_proj.weight", lp);
        load_w(e, &L->k, "%s.self_attn.k_proj.weight", lp);
        load_w(e, &L->v, "%s.self_attn.v_proj.weight", lp);
        load_w(e, &L->o, "%s.self_attn.o_proj.weight", lp);
        load_w(e, &L->ix, "%s.self_attn.indexer.index_qk_proj.weight", lp);
        L->qn = load_v(e, c->head_dim, "%s.self_attn.q_norm.weight", lp);
        L->kn = load_v(e, c->head_dim, "%s.self_attn.k_norm.weight", lp);
        L->iqn = load_v(e, c->ix_dim, "%s.self_attn.indexer.q_layernorm.weight", lp);
        L->ikn = load_v(e, c->ix_dim, "%s.self_attn.indexer.k_layernorm.weight", lp);
        L->K = F((size_t) e->max_pos * c->n_kv * c->head_dim);
        L->V = F((size_t) e->max_pos * c->n_kv * c->head_dim);
        L->raw = F((size_t) e->max_pos * c->ix_dim);
        L->blk = F((size_t) e->nb * c->ix_dim);
    } else {
        load_w(e, &L->qkv, "%s.linear_attn.in_proj_qkv.weight", lp);
        load_w(e, &L->z, "%s.linear_attn.in_proj_z.weight", lp);
        load_w(e, &L->b, "%s.linear_attn.in_proj_b.weight", lp);
        load_w(e, &L->a, "%s.linear_attn.in_proj_a.weight", lp);
        load_w(e, &L->out, "%s.linear_attn.out_proj.weight", lp);
        L->conv_w = load_v(e, (int64_t) cd * c->lin_conv, "%s.linear_attn.conv1d.weight", lp);
        L->A_log = load_v(e, c->lin_hv, "%s.linear_attn.A_log", lp);
        L->dt_bias = load_v(e, c->lin_hv, "%s.linear_attn.dt_bias", lp);
        L->norm_w = load_v(e, c->lin_dv, "%s.linear_attn.norm.weight", lp);
        L->conv = F(qwen4_gdn_conv_len(c));
        L->S = F(qwen4_gdn_state_len(c));
    }
    load_w(e, &L->gate, "%s.mlp.gate.weight", lp);
    load_w(e, &L->sg, "%s.mlp.shared_expert_gate.weight", lp);
    load_w(e, &L->sh_g, "%s.mlp.shared_expert.gate_proj.weight", lp);
    load_w(e, &L->sh_u, "%s.mlp.shared_expert.up_proj.weight", lp);
    load_w(e, &L->sh_d, "%s.mlp.shared_expert.down_proj.weight", lp);
    load_w(e, &L->eg, "%s.mlp.switch_mlp.gate_proj.weight", lp);
    load_w(e, &L->eu, "%s.mlp.switch_mlp.up_proj.weight", lp);
    load_w(e, &L->ed, "%s.mlp.switch_mlp.down_proj.weight", lp);
    if (!e->err[0] && (L->eg.slices != c->n_exp || L->eu.fmt != L->eg.fmt))
        fail(e, "%s: %d routed experts (want %d), or gate and up in different formats", lp, L->eg.slices, c->n_exp);
    if (l >= 0 && c->ple[l] >= 0) {
        load_w(e, &L->key, "%s.ple.key_proj.weight", lp);
        load_w(e, &L->val, "%s.ple.value_proj.weight", lp);
        L->nk = load_v(e, HD, "%s.ple.norm_key.weight", lp);
        L->nq = load_v(e, HD, "%s.ple.norm_query.weight", lp);
        L->nc = load_v(e, HD, "%s.ple.norm_conv.weight", lp);
        L->pw = load_v(e, (int64_t) HD * c->ple_conv, "%s.ple.conv1d.weight", lp);
        L->hist = F(qwen4_ple_hist_len(c));
    }
}

// ---- the matrix kernels (engine/kernels_moe.metal) ---------------------------------------------------------------------

static id<MTLBuffer> stab_for(Q4Eng* e, const MW* w) { return w->fmt == MF_SEED4P4 ? e->stab32 : e->stab; }
static void bind_w(Q4Eng* e, const MW* w, int first) {   // streams 0..2 at first .. first + 2
    for (int s = 0; s < 3; ++s) B(e, first + s, w->b[s] ? w->b[s] : w->b[0], w->b[s] ? w->o[s] : w->o[0]);
}
static void bind_codes(Q4Eng* e, const MW* w, int index) {   // SEED4P4's exponent codes (stream 3), else unused
    if (w->fmt == MF_SEED4P4 && w->b[3]) B(e, index, w->b[3], w->o[3]);
    else B(e, index, e->stab, 0);
}

// y[T][rows] = W x[T][cols] (row strides xs, ys; offsets xo, yo in bytes): the matvec for up to MV_MAXT rows, the GEMM above
static void mv_o(Q4Eng* e, const MW* w, id<MTLBuffer> X, uint64_t xo, int xs, id<MTLBuffer> Y, uint64_t yo, int ys, int T) {
    const int aff = w->fmt == MF_BF16 || w->fmt == MF_Q8 || w->fmt == MF_Q4;
    if (T <= MV_MAXT && aff && !e->no_split && w->rows * 8 < w->cols && w->cols % 8 == 0 && xs % 4 == 0 && xo % 16 == 0) {
        // few rows, long K (hyper-connection inject / down, DeltaNet a / b): a threadgroup per row (k_mv_split)
        const MvArgs a = {w->cols, w->rows, T, 1, xs, ys, 0, 0};
        [e->enc setComputePipelineState:pipe_g(e, "k_mv_split", w->fmt, T, w->gs)];
        A_(e, &a, sizeof a);
        bind_w(e, w, 1);
        B(e, 4, X, xo);
        B(e, 5, Y, yo);
        tgs(e, (uint64_t) w->rows, 1, MVS_TG);
        return;
    }
    if (T <= MV_MAXT) {
        const MvArgs a = {w->cols, w->rows, T, 1, xs, ys, 0, 0};
        [e->enc setComputePipelineState:pipe_g(e, "k_mv", w->fmt, T, w->gs)];
        A_(e, &a, sizeof a);
        bind_w(e, w, 1);
        B(e, 4, X, xo);
        B(e, 5, Y, yo);
        B(e, 6, e->tok, 0);   // unused (no selection)
        B(e, 7, stab_for(e, w), 0);
        bind_codes(e, w, 8);
        tgs(e, (uint64_t) (w->rows + MV_RPT_G(w->fmt, T, w->cols) - 1) / MV_RPT_G(w->fmt, T, w->cols), 1, 32 * MV_ROWS);
        return;
    }
    const MmArgs a = {w->cols, w->rows, T, xs, ys, 1, 0, 0};
    // k_mm64's 64 x 64 tiles: the same y, about 24% faster (its float4 input loads want 16-byte rows and offsets)
    const int m64 = !e->no_mm64 && xs % 4 == 0 && xo % 16 == 0 && w->cols % MM_BK == 0;
    [e->enc setComputePipelineState:pipe_g(e, m64 ? "k_mm64" : "k_mm", w->fmt, 1, w->gs)];
    A_(e, &a, sizeof a);
    bind_w(e, w, 1);
    B(e, 4, X, xo);
    B(e, 5, Y, yo);
    B(e, 6, e->tok, 0);
    B(e, 7, stab_for(e, w), 0);
    B(e, 8, e->tok, 0);
    bind_codes(e, w, 9);
    B(e, 10, Y, yo);   // the gate's (unused)
    if (m64) tgs(e, (uint64_t) (w->rows + MM2_BM - 1) / MM2_BM, (uint64_t) (T + MM2_BN - 1) / MM2_BN, 128);
    else tgs(e, (uint64_t) (w->rows + MM_BM - 1) / MM_BM, (uint64_t) (T + MM_BN - 1) / MM_BN, 128);
}
static void mv(Q4Eng* e, const MW* w, id<MTLBuffer> X, id<MTLBuffer> Y, int T) { mv_o(e, w, X, 0, w->cols, Y, 0, w->rows, T); }

// P pairs: y[p] = W[sel[p]] x[p / xdiv] (k_mv with a selection)
static void mv_sel(Q4Eng* e, const MW* w, id<MTLBuffer> X, int xs, id<MTLBuffer> Y, int ys, id<MTLBuffer> sel, int P, int xdiv) {
    const MvArgs a = {w->cols, w->rows, P, xdiv, xs, ys, 0, 0};
    [e->enc setComputePipelineState:pipe_g(e, "k_mv", w->fmt, 0, w->gs)];
    A_(e, &a, sizeof a);
    bind_w(e, w, 1);
    B(e, 4, X, 0);
    B(e, 5, Y, 0);
    B(e, 6, sel, 0);
    B(e, 7, stab_for(e, w), 0);
    bind_codes(e, w, 8);
    tgs(e, (uint64_t) (w->rows + MV_RPT_G(w->fmt, 0, w->cols) - 1) / MV_RPT_G(w->fmt, 0, w->cols), (uint64_t) P, 32 * MV_ROWS);
}
// P pairs: A[p] = silu(Wg[sel[p]] x) * (Wu[sel[p]] x), x = row p / xdiv (k_mv_gu)
static void mv_gu(Q4Eng* e, const MW* wg, const MW* wu, id<MTLBuffer> X, int xs, id<MTLBuffer> Out, int ys, id<MTLBuffer> sel, int P, int xdiv) {
    const MvArgs a = {wg->cols, wg->rows, P, xdiv, xs, ys, 0, 0};
    [e->enc setComputePipelineState:pipe_g(e, "k_mv_gu", wg->fmt, 0, wg->gs)];
    A_(e, &a, sizeof a);
    bind_w(e, wg, 1);
    B(e, 4, X, 0);
    B(e, 5, Out, 0);
    B(e, 6, sel, 0);
    B(e, 7, stab_for(e, wg), 0);
    bind_w(e, wu, 8);
    bind_codes(e, wg, 11);
    bind_codes(e, wu, 12);
    tgs(e, (uint64_t) (wg->rows + MV_ROWS - 1) / MV_ROWS, (uint64_t) P, 32 * MV_ROWS);
}
// Grouped GEMM over the tiles of bucket(): pair perm[i] reads input row perm[i] / xdiv and writes output row perm[i]
static void mm_grouped(Q4Eng* e, const MW* w, id<MTLBuffer> X, int xs, id<MTLBuffer> Y, int ys, int xdiv, int ntiles) {
    if (!ntiles) return;
    const MmArgs a = {w->cols, w->rows, 0, xs, ys, xdiv, 0, 0};
    const int m64 = !e->no_mm64;   // the tiles of bucket() hold MM2_BN pairs then (xs: the input row stride, a multiple of 4)
    [e->enc setComputePipelineState:pipe_g(e, m64 ? "k_mm64" : "k_mm", w->fmt, 0, w->gs)];
    A_(e, &a, sizeof a);
    bind_w(e, w, 1);
    B(e, 4, X, 0);
    B(e, 5, Y, 0);
    B(e, 6, e->perm, 0);
    B(e, 7, stab_for(e, w), 0);
    B(e, 8, e->tiles, 0);
    bind_codes(e, w, 9);
    B(e, 10, Y, 0);
    tgs(e, (uint64_t) (w->rows + (m64 ? MM2_BM : MM_BM) - 1) / (m64 ? MM2_BM : MM_BM), (uint64_t) ntiles, 128);
}
static void swiglu(Q4Eng* e, id<MTLBuffer> G, id<MTLBuffer> U, id<MTLBuffer> Out, int n) {
    const int32_t nn = n;
    [e->enc setComputePipelineState:pipe_g(e, "k_swiglu", 0, 0, 0)];
    B(e, 0, G, 0);
    B(e, 1, U, 0);
    B(e, 2, Out, 0);
    [e->enc setBytes:&nn length:4 atIndex:3];
    tgs(e, (uint64_t) (n + 255) / 256, 1, 256);
}
// The T x k routed pairs grouped by expert (ascending pair within an expert): perm[], tiles of <= MM_BN pairs
static int bucket(Q4Eng* e, int T, int cap) {   // tiles of at most cap pairs
    const int k = e->c.top_k, E = e->c.n_exp, P = T * k;
    const int32_t* inds = (const int32_t*) e->ind.contents;
    int* cnt = (int*) calloc((size_t) E + 1, sizeof(int));
    int* off = (int*) malloc(sizeof(int) * ((size_t) E + 1));
    for (int p = 0; p < P; ++p) cnt[inds[p]]++;
    off[0] = 0;
    for (int x = 0; x < E; ++x) off[x + 1] = off[x] + cnt[x];
    memcpy(cnt, off, sizeof(int) * (size_t) E);
    int32_t* perm = (int32_t*) e->perm.contents;
    for (int p = 0; p < P; ++p) perm[cnt[inds[p]]++] = p;
    MmTile* tl = (MmTile*) e->tiles.contents;
    int nt = 0;
    for (int x = 0; x < E; ++x)
        for (int s = off[x]; s < off[x + 1]; s += cap) tl[nt++] = (MmTile) {x, s, off[x + 1] - s < cap ? off[x + 1] - s : cap, 0};
    free(cnt);
    free(off);
    return nt;
}

// ---- the qwen4 kernels (engine/kernels_qwen4.metal) -------------------------------------------------------------------

static void norm1p_o(Q4Eng* e, id<MTLBuffer> X, uint64_t xo, int len, int g, id<MTLBuffer> w, id<MTLBuffer> Y, int T) {
    // few groups (decode rows): up to 8 threadgroups a group, each writing a slice; not in place (the others still read x)
    const int ng = T * (len / g), parts = X == Y || ng >= 32 ? 1 : ng >= 16 ? 2 : ng >= 8 ? 4 : 8;
    const Q4NormArgs a = {len, g, e->c.eps, parts};
    [e->enc setComputePipelineState:pipe4(e, "k_q4_norm1p")];
    A_(e, &a, sizeof a);
    B(e, 1, X, xo);
    B(e, 2, w, 0);
    B(e, 3, Y, 0);
    tgs(e, (uint64_t) ng * parts, 1, 256);
}
static void norm1p(Q4Eng* e, id<MTLBuffer> X, int len, int g, id<MTLBuffer> w, id<MTLBuffer> Y, int T) { norm1p_o(e, X, 0, len, g, w, Y, T); }
// RMSNorm (1 + w) + RoPE of nh heads of hd (stride hs) in rows of rs floats at byte offset xo, row t at pos0 + t * pstep
static void head_norm_rope(Q4Eng* e, id<MTLBuffer> X, uint64_t xo, int rs, int nh, int hs, int hd, id<MTLBuffer> w, int pos0, int pstep, int T) {
    const Q4RopeArgs a = {rs, hs, hd, e->c.rot_dim, pos0, pstep, e->c.eps, 0};
    [e->enc setComputePipelineState:pipe4(e, "k_q4_head_norm_rope")];
    A_(e, &a, sizeof a);
    B(e, 1, X, xo);
    B(e, 2, w, 0);
    B(e, 3, e->inv, 0);
    tgs(e, (uint64_t) nh, (uint64_t) T, 128);
}
// dst rows pos0 .. pos0 + T - 1 (width w) = the rows of src (stride ss, byte offset so)
static void store(Q4Eng* e, id<MTLBuffer> dst, int w, id<MTLBuffer> src, uint64_t so, int ss, int pos0, int T) {
    const Q4StoreArgs a = {w, ss, pos0, 0};
    [e->enc setComputePipelineState:pipe4(e, "k_q4_store")];
    A_(e, &a, sizeof a);
    B(e, 1, dst, 0);
    B(e, 2, src, so);
    thr(e, (uint64_t) w, (uint64_t) T, 256);
}

// The streams of H mixed into x; with inj, the weights that add the branch back (nslm/qwen4_ops.h)
static void hc_mix(Q4Eng* e, id<MTLBuffer> norm, const MW* down, const MW* upw, const MW* injw, int T) {
    const Qwen4Cfg* c = &e->c;
    const int HD = c->hc * c->d;
    const float ih = 1.0f / (float) c->hc;
    norm1p(e, e->H, HD, c->d, norm, e->normed, T);
    mv(e, down, e->normed, e->lo, T);
    const Q4HcArgs act = {c->hc, c->d, T * c->hc_rank, ih}, mix = {c->hc, c->d, 0, ih}, gate = {c->hc, c->d, T * c->hc, ih};
    [e->enc setComputePipelineState:pipe4(e, "k_q4_hc_act")];
    A_(e, &act, sizeof act);
    B(e, 1, e->lo, 0);
    thr(e, (uint64_t) T * c->hc_rank, 1, 256);
    mv(e, upw, e->lo, e->up, T);
    [e->enc setComputePipelineState:pipe4(e, "k_q4_hc_mix")];
    A_(e, &mix, sizeof mix);
    B(e, 1, e->normed, 0);
    B(e, 2, e->up, 0);
    B(e, 3, e->x, 0);
    thr(e, (uint64_t) c->d, (uint64_t) T, 256);
    if (injw) {
        mv(e, injw, e->normed, e->inj, T);
        [e->enc setComputePipelineState:pipe4(e, "k_q4_hc_gate")];
        A_(e, &gate, sizeof gate);
        B(e, 1, e->inj, 0);
        thr(e, (uint64_t) T * c->hc, 1, 256);
    }
}
static void hc_add(Q4Eng* e, id<MTLBuffer> H, id<MTLBuffer> inj, int T) {   // H[t][s] += inj[t][s] br[t]
    const Q4HcArgs a = {e->c.hc, e->c.d, 0, 1.0f / (float) e->c.hc};
    [e->enc setComputePipelineState:pipe4(e, "k_q4_hc_add")];
    A_(e, &a, sizeof a);
    B(e, 1, H, 0);
    B(e, 2, e->br, 0);
    B(e, 3, inj, 0);
    thr(e, (uint64_t) e->c.d, (uint64_t) T, 256);
}

// The snapshot buffers k[0 .. 2] at first .. first + 2 (a verify forward), else the state itself (unused)
static void bind_keep(Q4Eng* e, int first, id<MTLBuffer> __strong const* k, id<MTLBuffer> state) {
    for (int i = 0; i < Q4E_KEEP_SLOTS; ++i) B(e, first + i, e->keep && k[i] ? k[i] : state, 0);
}

// The PLE rows of rows [pos, pos + T) of each PLE layer into emb [ple index][maxt][ple_dim]: host work, before encoding
static int ple_rows(Q4Eng* e, int T) {
    const Qwen4Cfg* c = &e->c;
    float* emb = (float*) e->emb.contents;
    for (int l = 0; l < c->n_layer; ++l)
        for (int t = 0; c->ple[l] >= 0 && t < T; ++t)
            if (qwen4_ple_embed(&e->nm, &e->rows, e->table, c, &e->ng[c->ple[l]], l, e->ids, e->pos + t,
                                emb + ((size_t) c->ple[l] * e->maxt + t) * c->ple_dim))
                return fail(e, "layer %d: a PLE table shard is missing or does not hold a row", l);
    return 0;
}

static void ple(Q4Eng* e, int l, int T) {
    const Qwen4Cfg* c = &e->c;
    Layer* L = &e->L[l];
    const int HD = c->hc * c->d;
    const uint64_t eo = sizeof(float) * (uint64_t) c->ple[l] * e->maxt * c->ple_dim;
    mv_o(e, &L->key, e->emb, eo, c->ple_dim, e->keys, 0, HD, T);
    norm1p(e, e->keys, HD, c->d, L->nk, e->keys, T);
    mv_o(e, &L->val, e->emb, eo, c->ple_dim, e->vals, 0, c->d, T);
    norm1p(e, e->H, HD, c->d, L->nq, e->qs, T);
    const Q4PleGateArgs ga = {c->hc, c->d, {0, 0}};
    [e->enc setComputePipelineState:pipe4(e, "k_q4_ple_gate")];
    A_(e, &ga, sizeof ga);
    B(e, 1, e->keys, 0);
    B(e, 2, e->qs, 0);
    B(e, 3, e->vals, 0);
    B(e, 4, e->gated, 0);
    tgs(e, (uint64_t) c->hc, (uint64_t) T, 256);
    norm1p(e, e->gated, HD, c->d, L->nc, e->keys, T);   // keys now holds the conv input
    const Q4PleConvArgs ca = {HD, c->ple_conv, c->ngram, T, e->keep && L->khist[0] != nil, {0, 0, 0}};
    [e->enc setComputePipelineState:pipe4(e, "k_q4_ple_conv")];
    A_(e, &ca, sizeof ca);
    B(e, 1, L->pw, 0);
    B(e, 2, e->keys, 0);
    B(e, 3, e->gated, 0);
    B(e, 4, L->hist, 0);
    B(e, 5, e->H, 0);
    bind_keep(e, 6, L->khist, L->hist);
    thr(e, (uint64_t) HD, 1, 256);
}

// Rows at positions pos0 .. pos0 + T - 1. keys_only: the KV and QSA block-key stores, no output (the MTP head's rows
// one behind a forward). drow: the rows' first entry in done / nsel (q4e_draft's head row j)
static void attention(Q4Eng* e, Layer* L, int T, int pos0, int keys_only, int drow) {
    const Qwen4Cfg* c = &e->c;
    const int hd = c->head_dim, kvw = c->n_kv * hd, D = c->ix_dim, R = c->ix_ratio, iw = (c->ix_heads + 1) * D;
    if (!keys_only) mv(e, &L->q, e->x, e->qp, T);
    mv(e, &L->k, e->x, e->kk, T);
    mv(e, &L->v, e->x, e->vv, T);
    if (!keys_only) head_norm_rope(e, e->qp, 0, c->n_head * 2 * hd, c->n_head, 2 * hd, hd, L->qn, pos0, 1, T);
    head_norm_rope(e, e->kk, 0, kvw, c->n_kv, hd, hd, L->kn, pos0, 1, T);
    store(e, L->K, kvw, e->kk, 0, kvw, pos0, T);
    store(e, L->V, kvw, e->vv, 0, kvw, pos0, T);
    mv(e, &L->ix, e->x, e->ixo, T);
    store(e, L->raw, D, e->ixo, sizeof(float) * (uint64_t) c->ix_heads * D, iw, pos0, T);
    if (!keys_only) head_norm_rope(e, e->ixo, 0, iw, c->ix_heads, D, D, L->iqn, pos0, 1, T);
    mark(e, ST_ATTN);
    // the key blocks these rows complete: the mean of their raw keys, then norm + RoPE at the block's first position
    const int j0 = pos0 / R, j1 = (pos0 + T) / R;
    if (j1 > j0) {
        const Q4QsaMeanArgs ma = {D, R, {0, 0}};
        [e->enc setComputePipelineState:pipe4(e, "k_q4_qsa_mean")];
        A_(e, &ma, sizeof ma);
        B(e, 1, L->raw, sizeof(float) * (uint64_t) j0 * R * D);
        B(e, 2, L->blk, sizeof(float) * (uint64_t) j0 * D);
        tgs(e, (uint64_t) (j1 - j0), 1, 128);
        head_norm_rope(e, L->blk, sizeof(float) * (uint64_t) j0 * D, D, 1, D, D, L->ikn, j0 * R, R, j1 - j0);
    }
    if (keys_only) return;
    const uint64_t dro = sizeof(int32_t) * (uint64_t) drow;
    if (j1 > 0) {   // done / nsel are the host's (the forward's rows)
        const Q4QsaArgs qa = {c->ix_heads, D, iw, j1, e->kq, {0, 0, 0}};
        [e->enc setComputePipelineState:pipe4(e, "k_q4_qsa_scores")];
        A_(e, &qa, sizeof qa);
        B(e, 1, e->ixo, 0);
        B(e, 2, L->blk, 0);
        B(e, 3, e->done, dro);
        B(e, 4, e->sc, 0);
        tgs(e, (uint64_t) (j1 + 7) / 8, (uint64_t) T, 256);
        [e->enc setComputePipelineState:pipe4(e, "k_q4_qsa_topk")];
        A_(e, &qa, sizeof qa);
        B(e, 1, e->sc, 0);
        B(e, 2, e->done, dro);
        B(e, 3, e->sel, 0);
        tgs(e, (uint64_t) T, 1, 256);
    }
    mark(e, ST_QSA);
    // prompt rows: a KV group's heads as the rows of simdgroup matrices, each key tile loaded once for all of them.
    // Decode rows with long key lists (n >= smin): the same in key splits, merged by k_q4_attend_comb (k_q4_attend reloads
    // each key for each of the group's heads). Each decode row's kernel and splits follow from its own keys, so a verify
    // forward's rows are bit for bit the one-row forwards' (MTP).
    const int canG = c->n_head / c->n_kv <= 16 && hd % (8 * ATG_NQ) == 0;
    int nlong = 0, zmax = 1;
    for (int t = 0; t < T; ++t) {
        const int pos = pos0 + t, done = (pos + 1) / R, ns = done > e->kq ? e->kq : 0, tail = ns ? (pos + 1) / R * R : 0;
        const int n = ns * R + pos + 1 - tail, z = (n + e->skeys - 1) / e->skeys;
        if (n >= e->smin) { ++nlong; if (z > zmax) zmax = z < e->smax ? z : e->smax; }
    }
    const int split = T <= MV_MAXT && canG && nlong && !e->no_splitk, grouped = (T > MV_MAXT && canG && !e->no_attg) || split;
    const Q4AttArgs aa = {c->n_head, c->n_kv, hd, R, e->kq, pos0, 1.0f / sqrtf((float) hd), split ? zmax : 1, split ? e->smin : 0, e->skeys, e->smax, 0};
    for (int pass = 0; pass < 1 + (split && nlong < T); ++pass) {   // split rows: then k_q4_attend for the short rows
        const int g = grouped && !pass;
        [e->enc setComputePipelineState:pipe4(e, g ? "k_q4_attend_g" : "k_q4_attend")];
        A_(e, &aa, sizeof aa);
        B(e, 1, e->qp, 0);
        B(e, 2, L->K, 0);
        B(e, 3, L->V, 0);
        B(e, 4, e->sel, 0);
        B(e, 5, e->nsel, dro);
        B(e, 6, e->o, 0);
        B(e, 7, e->apart, 0);
        if (g) {
            [e->enc dispatchThreadgroups:MTLSizeMake((NSUInteger) c->n_kv, (NSUInteger) T, (NSUInteger) aa.nsplit)
                  threadsPerThreadgroup:MTLSizeMake(64 * ATG_NQ, 1, 1)];
            kprof(e);
        } else tgs(e, (uint64_t) c->n_head, (uint64_t) T, 256);
    }
    if (split && zmax > 1) {   // a row of one split is k_q4_attend_g's output already
        [e->enc setComputePipelineState:pipe4(e, "k_q4_attend_comb")];
        A_(e, &aa, sizeof aa);
        B(e, 1, e->qp, 0);
        B(e, 2, e->apart, 0);
        B(e, 3, e->o, 0);
        B(e, 4, e->nsel, dro);
        tgs(e, (uint64_t) c->n_head, (uint64_t) T, 256);
    }
    mark(e, ST_ATTEND);
    mv(e, &L->o, e->o, e->br, T);
}

static void gdn(Q4Eng* e, Layer* L, int T) {
    const Qwen4Cfg* c = &e->c;
    const int dk = c->lin_dk, dv = c->lin_dv, hv = c->lin_hv, cd = 2 * c->lin_hk * dk + hv * dv, vd = hv * dv;
    mv(e, &L->qkv, e->x, e->qkvb, T);
    mv(e, &L->z, e->x, e->zb, T);
    mv(e, &L->b, e->x, e->bb, T);
    mv(e, &L->a, e->x, e->ab, T);
    const Q4ConvArgs ca = {cd, c->lin_conv, T, e->keep && L->kconv[0] != nil};
    [e->enc setComputePipelineState:pipe4(e, "k_q4_gdn_conv")];
    A_(e, &ca, sizeof ca);
    B(e, 1, L->conv_w, 0);
    B(e, 2, e->qkvb, 0);
    B(e, 3, L->conv, 0);
    B(e, 4, e->cv, 0);
    bind_keep(e, 5, L->kconv, L->conv);
    thr(e, (uint64_t) cd, 1, 256);
    const Q4GdnArgs ga = {cd, c->lin_hk * dk, vd, hv, hv / c->lin_hk, dv, T, e->keep && L->kS[0] != nil};
    [e->enc setComputePipelineState:pipe4(e, dk == 32 ? "k_q4_gdn_step_32" : dk == 64 ? "k_q4_gdn_step_64" : "k_q4_gdn_step_128")];
    A_(e, &ga, sizeof ga);
    B(e, 1, e->cv, 0);
    B(e, 2, e->ab, 0);
    B(e, 3, e->bb, 0);
    B(e, 4, L->A_log, 0);
    B(e, 5, L->dt_bias, 0);
    B(e, 6, L->S, 0);
    B(e, 7, e->y, 0);
    bind_keep(e, 8, L->kS, L->S);
    tgs(e, (uint64_t) hv, (uint64_t) dv / 32, 32 * Q4_GDN_SPLIT(dk));
    const Q4GNormArgs na = {hv, dv, c->eps, 0};
    [e->enc setComputePipelineState:pipe4(e, "k_q4_gdn_norm")];
    A_(e, &na, sizeof na);
    B(e, 1, L->norm_w, 0);
    B(e, 2, e->zb, 0);
    B(e, 3, e->y, 0);
    tgs(e, (uint64_t) hv, (uint64_t) T, (uint64_t) dv);
    mv(e, &L->out, e->y, e->br, T);
}

static int moe(Q4Eng* e, Layer* L, int T) {
    const Qwen4Cfg* c = &e->c;
    const int k = c->top_k, P = T * k;
    mv(e, &L->gate, e->x, e->lg, T);
    const Q4RouteArgs ra = {c->n_exp, k, {0, 0}};
    [e->enc setComputePipelineState:pipe4(e, "k_q4_route")];
    A_(e, &ra, sizeof ra);
    B(e, 1, e->lg, 0);
    B(e, 2, e->ind, 0);
    B(e, 3, e->wt, 0);
    tgs(e, (uint64_t) T, 1, 256);
    mark(e, ST_ROUTE);
    if (T > MV_MAXT) {   // prompt rows: the pairs by expert (host), so each tile reads its expert's weights once
        if (cmd_wait(e)) return -1;
        cmd_begin(e);
        const int nt = bucket(e, T, e->no_mm64 ? MM_BN : MM2_BN);
        mm_grouped(e, &L->eg, e->x, c->d, e->Gb, c->ff_exp, k, nt);
        mm_grouped(e, &L->eu, e->x, c->d, e->Ub, c->ff_exp, k, nt);
        swiglu(e, e->Gb, e->Ub, e->A, P * c->ff_exp);
        mm_grouped(e, &L->ed, e->A, c->ff_exp, e->Dx, c->d, 1, nt);
        mv(e, &L->sh_g, e->x, e->Gb, T);
        mv(e, &L->sh_u, e->x, e->Ub, T);
        swiglu(e, e->Gb, e->Ub, e->A, T * c->ff_shared);
    } else {
        mv_gu(e, &L->eg, &L->eu, e->x, c->d, e->A, c->ff_exp, e->ind, P, k);
        mv_sel(e, &L->ed, e->A, c->ff_exp, e->Dx, c->d, e->ind, P, 1);
        if (L->sh_g.fmt == L->sh_u.fmt) mv_gu(e, &L->sh_g, &L->sh_u, e->x, c->d, e->A, c->ff_shared, e->zero, T, 1);
        else { mv(e, &L->sh_g, e->x, e->Gb, T); mv(e, &L->sh_u, e->x, e->Ub, T); swiglu(e, e->Gb, e->Ub, e->A, T * c->ff_shared); }
    }
    mv(e, &L->sh_d, e->A, e->sh, T);
    mv(e, &L->sg, e->x, e->sgv, T);
    const Q4CombineArgs ma = {c->d, k, {0, 0}};
    [e->enc setComputePipelineState:pipe4(e, "k_q4_moe_combine")];
    A_(e, &ma, sizeof ma);
    B(e, 1, e->Dx, 0);
    B(e, 2, e->wt, 0);
    B(e, 3, e->sh, 0);
    B(e, 4, e->sgv, 0);
    B(e, 5, e->br, 0);
    thr(e, (uint64_t) c->d, (uint64_t) T, 256);
    return 0;
}

// x[t] = the embedding of tokens tok[t] (byte offset to)
static void embed_x(Q4Eng* e, id<MTLBuffer> tok, uint64_t to, int T) {
    const int32_t d = e->c.d;
    const MW* w = &e->embed;
    [e->enc setComputePipelineState:pipe_g(e, "k_embed", w->fmt, 0, w->gs)];
    [e->enc setBytes:&d length:4 atIndex:0];
    B(e, 1, w->b[0], w->o[0]);
    B(e, 2, w->b[0], w->o[0]);
    B(e, 3, w->b[1] ? w->b[1] : w->b[0], w->b[1] ? w->o[1] : w->o[0]);
    B(e, 4, w->b[2] ? w->b[2] : w->b[0], w->b[2] ? w->o[2] : w->o[0]);
    B(e, 5, tok, to);
    B(e, 6, e->x, 0);
    B(e, 7, e->stab32, 0);
    B(e, 8, w->fmt == MF_SEED4P4 && w->b[3] ? w->b[3] : w->b[0], w->fmt == MF_SEED4P4 && w->b[3] ? w->o[3] : 0);
    thr(e, (uint64_t) d, (uint64_t) T, 256);
}
static void embed(Q4Eng* e, int T) {   // the forward's tokens into each stream of H
    const int32_t d = e->c.d;
    embed_x(e, e->tok, 0, T);
    const Q4HcArgs a = {e->c.hc, e->c.d, 0, 0};
    [e->enc setComputePipelineState:pipe4(e, "k_q4_tile")];
    A_(e, &a, sizeof a);
    B(e, 1, e->x, 0);
    B(e, 2, e->H, 0);
    thr(e, (uint64_t) d, (uint64_t) T, 256);
}

// ---- a layer, the MTP head ------------------------------------------------------------------------------------------

// Layer L on the streams H: main layer l, or the MTP head's (l < 0), rows at positions pos0 .. (drow: attention())
static int run_layer(Q4Eng* e, Layer* L, int l, int attn, int T, int pos0, int drow) {
    const Qwen4Cfg* c = &e->c;
    if (l >= 0 && c->ple[l] >= 0) ple(e, l, T);
    mark(e, ST_PLE);
    hc_mix(e, L->hc_norm[0], &L->hc_down[0], &L->hc_up[0], &L->hc_inj[0], T);
    mark(e, ST_HC);
    if (attn) attention(e, L, T, pos0, 0, drow);
    else gdn(e, L, T);
    mark(e, attn ? ST_ATTN : ST_GDN);
    hc_add(e, e->H, e->inj, T);
    hc_mix(e, L->hc_norm[1], &L->hc_down[1], &L->hc_up[1], &L->hc_inj[1], T);
    mark(e, ST_HC);
    if (moe(e, L, T)) return -1;
    hc_add(e, e->H, e->inj, T);
    mark(e, ST_EXPERT);
    return 0;
}

// The MTP head's input streams Hm[n]: fc_embedding(norm(embed(tok))) added to each stream of fc_hidden(norm(Hin)),
// tokens at byte offset to of tok, streams at byte offset ho of Hin
static void head_in(Q4Eng* e, id<MTLBuffer> tok, uint64_t to, id<MTLBuffer> Hin, uint64_t ho, int n) {
    const Qwen4Cfg* c = &e->c;
    const int HD = c->hc * c->d;
    embed_x(e, tok, to, n);
    norm1p(e, e->x, c->d, c->d, e->m_ne, e->x, n);
    mv(e, &e->m_fe, e->x, e->br, n);
    norm1p_o(e, Hin, ho, HD, HD, e->m_nh, e->normed, n);
    mv(e, &e->m_fh, e->normed, e->Hm, n * c->hc);   // a stream a row
    hc_add(e, e->Hm, e->ones, n);
}
// After a forward's T rows (in H, tokens in tok): the head's keys one row behind, head row t on token t + s and the
// streams of main row t - 1 + s (Hin row 0 for t + s = 0), at position pos - 1 + s + t (s = 1 at position 0, where no row
// comes before). Then Hin row 0 = main row T - 1, the next head row's input.
static void head_rows(Q4Eng* e, int T) {
    const Qwen4Cfg* c = &e->c;
    const int HD = c->hc * c->d, s = e->pos == 0, n = T - s;
    if (!e->mtp) return;
    if (n > 0) {
        if (T > 1) store(e, e->Hin, HD, e->H, 0, HD, 1, T - 1);
        head_in(e, e->tok, sizeof(int32_t) * (uint64_t) s, e->Hin, sizeof(float) * (uint64_t) s * HD, n);
        id<MTLBuffer> H = e->H;
        e->H = e->Hm;
        hc_mix(e, e->M.hc_norm[0], &e->M.hc_down[0], &e->M.hc_up[0], NULL, n);
        attention(e, &e->M, n, e->pos - 1 + s, 1, 0);
        e->H = H;
    }
    store(e, e->Hin, HD, e->H, sizeof(float) * (uint64_t) (T - 1) * HD, HD, 0, 1);
    mark(e, ST_HEAD);
}
static void argmax(Q4Eng* e, id<MTLBuffer> X, id<MTLBuffer> out, uint64_t oo) {   // out = the arg max of X[vocab] (k_argmax)
    const int32_t V = e->c.vocab;
    [e->enc setComputePipelineState:pipe_g(e, "k_argmax", 0, 0, 0)];
    B(e, 0, X, 0);
    B(e, 1, out, oo);
    [e->enc setBytes:&V length:4 atIndex:2];
    tgs(e, 1, 1, 1024);
}
// k head rows on dtok = {bonus, ...}: row j at pos - 1 + j on token dtok[j] and Hin row 0 (j = 0), then its own
// streams; draft j to dtok[1 + j]
static int draft_rows(Q4Eng* e, int k) {
    id<MTLBuffer> H = e->H;
    int rc = 0;
    for (int j = 0; j < k && !rc; ++j) {
        head_in(e, e->dtok, sizeof(int32_t) * (uint64_t) j, j ? e->Hm : e->Hin, 0, 1);
        e->H = e->Hm;
        rc = run_layer(e, &e->M, -1, 1, 1, e->pos - 1 + j, j);
        hc_mix(e, e->m_norm, &e->m_down, &e->m_up, NULL, 1);
        e->H = H;
        mv(e, &e->head, e->x, e->mlog, 1);
        argmax(e, e->mlog, e->dtok, sizeof(int32_t) * (uint64_t) (j + 1));
        mark(e, ST_HEAD);
    }
    return rc;
}

// ---- open / close ----------------------------------------------------------------------------------------------------

void q4e_cache_stats(Q4Eng* e, uint64_t* hits, uint64_t* misses, int* slots) {
    (void) e;
    *hits = 0;
    *misses = 0;
    *slots = 0;
}

static int check_shapes(Q4Eng* e) {   // what the kernels take (as tests/kernel_backend_metal.m checks them)
    const Qwen4Cfg* c = &e->c;
    if (c->n_exp > 1023 || c->top_k > 32) return fail(e, "%d experts, top %d: at most 1023 and 32", c->n_exp, c->top_k);
    if (c->head_dim % 32 || c->head_dim > 256) return fail(e, "head_dim %d not supported", c->head_dim);
    if (c->ix_dim > 1024 || c->rot_dim > c->ix_dim || c->rot_dim > c->head_dim || c->rot_dim % 2) return fail(e, "QSA / RoPE widths not supported");
    if (c->lin_conv < 1 || c->lin_conv > 8 || c->lin_dv % 32 || c->lin_dv > 1024 || (c->lin_dk != 32 && c->lin_dk != 64 && c->lin_dk != 128))
        return fail(e, "Gated DeltaNet head widths not supported");
    if ((c->ple_conv - 1) * c->ngram > 32) return fail(e, "PLE conv history above 32");
    return 0;
}

Q4Eng* q4e_open(const char* model_dir, const Q4EOpts* o, char* err, int errlen) {
    Q4Eng* e = (Q4Eng*) calloc(1, sizeof *e);
    if (qwen4_cfg_load(&e->c, model_dir, err, errlen) || ns_open(&e->nm, model_dir, err, errlen)) { free(e); return NULL; }
    const Qwen4Cfg* c = &e->c;
    e->max_pos = o && o->max_pos > 0 ? o->max_pos : 4096;
    e->nb = e->max_pos / c->ix_ratio + 1;
    e->maxt = o && o->rows > 0 ? o->rows : MAXT_DEFAULT;   // from LOGT to the context
    if (e->maxt > e->max_pos) e->maxt = e->max_pos;
    if (e->maxt < LOGT) e->maxt = LOGT;
    e->kq = c->ix_budget / c->ix_ratio;
    e->lrow = -1;
    e->dev = MTLCreateSystemDefaultDevice();
    e->queue = [e->dev newCommandQueue];
    e->pipes = [NSMutableDictionary new];
    e->buffers = [NSMutableArray new];
    const char* res = getenv("NSLM_RES") ? getenv("NSLM_RES") : "out/res";
    NSError* nerr = nil;
    e->lib = [e->dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithFormat:@"%s/kernels_moe.metallib", res]] error:&nerr];
    e->lib4 = [e->dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithFormat:@"%s/kernels_qwen4.metallib", res]] error:&nerr];
    if (!e->dev || !e->lib || !e->lib4) fail(e, "no Metal device, or no %s/kernels_moe.metallib and kernels_qwen4.metallib (NSLM_RES)", res);
    if (!e->err[0]) check_shapes(e);
    if (!e->err[0] && o && o->ple_gguf) {
        if (ple_open(&e->pt, o->ple_gguf, "per_layer_token_embd.weight", e->err, sizeof e->err)) {
            snprintf(err, (size_t) errlen, "%s", e->err);
            q4e_close(e);
            return NULL;
        }
        e->table = &e->pt;
    }
    ns_rows_init(&e->rows);
    e->stab = dbuf(e, e->rows.g24, sizeof e->rows.g24);
    e->stab32 = dbuf(e, e->rows.g32, sizeof e->rows.g32);
    {
        float inv[512] = {0};
        for (int i = 0; i < c->rot_dim / 2 && i < 512; ++i) inv[i] = (float) pow((double) c->rope_theta, -2.0 * i / c->rot_dim);
        e->inv = dbuf(e, inv, sizeof inv);
    }
    if (!e->err[0]) {
        load_w(e, &e->embed, "language_model.model.embed_tokens.weight");
        load_w(e, &e->head, "language_model.lm_head.weight");
        load_w(e, &e->mix_down, "language_model.model.hyper_connection_mixer.input_mix_weight_down.weight");
        load_w(e, &e->mix_up, "language_model.model.hyper_connection_mixer.input_mix_weight_up.weight");
        e->mix_norm = load_v(e, (int64_t) c->hc * c->d, "language_model.model.hyper_connection_mixer.hc_norm.weight");
        if (!e->err[0] && e->embed.fmt != MF_BF16 && e->embed.fmt != MF_Q8 && e->embed.fmt != MF_Q4 && e->embed.fmt != MF_SEED4P4)
            fail(e, "embedding format %d not supported", e->embed.fmt);
    }
    for (int l = 0; l < c->n_layer && !e->err[0]; ++l) {
        if (c->ple[l] >= 0) qwen4_ngram_init(c, c->ple[l], &e->ng[c->ple[l]]);
        if (c->ple[l] >= 0 && e->table && (c->ple[l] > 0 || e->table->rows != e->ng[0].rows))
            fail(e, "the PLE GGUF has %lld rows, the model's table %lld (and one PLE layer)", (long long) e->table->rows, (long long) e->ng[0].rows);
        char lp[48];
        snprintf(lp, sizeof lp, "language_model.model.layers.%d", l);
        load_layer(e, &e->L[l], lp, c->attn[l], l);
        if (c->ple[l] + 1 > e->nple) e->nple = c->ple[l] + 1;
    }
    if (o && o->mtp > 0 && !e->err[0] && !ns_find(&e->nm, "mtp.fc_embedding.weight")) fail(e, "no MTP head (mtp.*) in %s", model_dir);
    else if (o && o->mtp > 0 && !e->err[0]) {
        e->m_ne = load_v(e, c->d, "mtp.pre_fc_norm_embedding.weight");
        e->m_nh = load_v(e, (int64_t) c->hc * c->d, "mtp.pre_fc_norm_hidden.weight");
        load_w(e, &e->m_fe, "mtp.fc_embedding.weight");
        load_w(e, &e->m_fh, "mtp.fc_hidden.weight");
        load_w(e, &e->m_down, "mtp.hyper_connection_mixer.input_mix_weight_down.weight");
        load_w(e, &e->m_up, "mtp.hyper_connection_mixer.input_mix_weight_up.weight");
        e->m_norm = load_v(e, (int64_t) c->hc * c->d, "mtp.hyper_connection_mixer.hc_norm.weight");
        load_layer(e, &e->M, "mtp.layers.0", 1, -1);
        e->mtp = 1;
        e->ndraft = o->mtp < Q4E_MAX_DRAFTS ? o->mtp : Q4E_MAX_DRAFTS;
    }
    const size_t T = (size_t) e->maxt, HD = (size_t) c->hc * c->d, cd = (size_t) 2 * c->lin_hk * c->lin_dk + (size_t) c->lin_hv * c->lin_dv;
    const size_t vd = (size_t) c->lin_hv * c->lin_dv, iw = (size_t) (c->ix_heads + 1) * c->ix_dim, kf = (size_t) c->top_k * c->ff_exp;
    const size_t ka = kf > (size_t) c->ff_shared ? kf : (size_t) c->ff_shared;
    if (!e->err[0]) {
        e->x = F(T * c->d); e->H = F(T * HD); e->normed = F(T * HD); e->lo = F(T * c->hc_rank); e->up = F(T * HD); e->inj = F(T * c->hc);
        e->br = F(T * c->d); e->qp = F((T * c->n_head + 16) * 2 * c->head_dim); e->kk = F(T * c->n_kv * c->head_dim);   // qp: + 16 heads, k_q4_attend_g's padding
        e->vv = F(T * c->n_kv * c->head_dim); e->o = F(T * c->n_head * c->head_dim); e->ixo = F(T * iw); e->sc = F(T * (size_t) e->nb);
        e->emb = F(T * c->ple_dim * (size_t) (e->nple ? e->nple : 1)); e->keys = F(T * HD); e->qs = F(T * HD); e->vals = F(T * c->d);
        e->gated = F(T * HD); e->lg = F(T * c->n_exp); e->wt = F(T * c->top_k); e->A = F(T * ka); e->Gb = F(T * ka); e->Ub = F(T * ka);
        e->Dx = F(T * c->top_k * c->d); e->sh = F(T * c->d); e->sgv = F(T); e->logits = F((size_t) LOGT * c->vocab); e->y = F(T * vd);
        e->cv = F(T * cd); e->qkvb = F(T * cd); e->zb = F(T * vd); e->bb = F(T * c->lin_hv); e->ab = F(T * c->lin_hv);
        e->tok = dbuf(e, NULL, 4 * T); e->ind = dbuf(e, NULL, 4 * T * c->top_k); e->sel = dbuf(e, NULL, 4 * T * (size_t) (e->kq ? e->kq : 1));
        e->done = dbuf(e, NULL, 4 * T); e->nsel = dbuf(e, NULL, 4 * T); e->zero = dbuf(e, NULL, 4 * T);
        e->perm = dbuf(e, NULL, 4 * T * c->top_k);
        e->apart = F((size_t) MV_MAXT * c->n_head * 32 * (c->head_dim + 2)); e->tiles = dbuf(e, NULL, sizeof(MmTile) * (T * c->top_k / MM_BN + (size_t) c->n_exp + 1));
    }
    if (e->mtp && !e->err[0]) {
        e->Hin = F(T * HD); e->Hm = F(T * HD); e->ones = F(T * c->hc); e->mlog = F(c->vocab);
        e->dtok = dbuf(e, NULL, sizeof(int32_t) * (Q4E_MAX_DRAFTS + 1));
        for (size_t i = 0; e->ones && i < T * c->hc; ++i) ((float*) e->ones.contents)[i] = 1.0f;
        for (int l = 0; l < c->n_layer; ++l)
            for (int i = 0; i < e->ndraft; ++i) {   // a verify of ndraft + 1 rows keeps the state after its first ndraft
                Layer* L = &e->L[l];
                if (!c->attn[l]) { L->kconv[i] = F(qwen4_gdn_conv_len(c)); L->kS[i] = F(qwen4_gdn_state_len(c)); }
                if (c->ple[l] >= 0) L->khist[i] = F(qwen4_ple_hist_len(c));
            }
    }
    e->ids = (int32_t*) malloc(sizeof(int32_t) * (size_t) e->max_pos);
    e->prof = getenv("Q4E_PROFILE") ? atoi(getenv("Q4E_PROFILE")) : 0;
    e->kms = [NSMutableDictionary new];
    e->kcalls = [NSMutableDictionary new];
    e->no_split = getenv("Q4E_NO_SPLIT") != NULL;
    e->no_attg = getenv("Q4E_NO_ATTG") != NULL;
    e->no_splitk = getenv("Q4E_NO_SPLITK") != NULL;
    e->no_mm64 = getenv("Q4E_NO_MM64") != NULL;
    e->smin = getenv("Q4E_SPLIT_MIN") ? atoi(getenv("Q4E_SPLIT_MIN")) : 1;   // every decode row: k_q4_attend_g (70 -> 16 us at 100 keys)
    // splits of 32 keys, at most 32: one split walks its tiles serially (a 100-key row: 86 us in one, 39 in four). With
    // the QSA top-k fixed, 32 splits beat 16 at 1K to 20K tokens of context (47.7 / 45.3 / 45.2 against 47.0 / 43.5 / 44.4)
    e->skeys = getenv("Q4E_SPLIT_KEYS") ? atoi(getenv("Q4E_SPLIT_KEYS")) : 32;
    e->smax = getenv("Q4E_SPLIT_MAX") ? atoi(getenv("Q4E_SPLIT_MAX")) : ATG_MAX_SPLIT;
    if (e->smin < 1) e->smin = 1;
    if (e->skeys < 1) e->skeys = 1;
    if (e->smax < 1 || e->smax > ATG_MAX_SPLIT) e->smax = ATG_MAX_SPLIT;
    if (@available(macOS 15.0, *)) {   // the weights and buffers resident on the GPU
        MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
        e->residency = e->err[0] ? nil : [e->dev newResidencySetWithDescriptor:rd error:nil];
        if (e->residency) {
            [e->queue addResidencySet:e->residency];
            for (id<MTLBuffer> b in e->buffers) [e->residency addAllocation:b];
            [e->residency commit];
            [e->residency requestResidency];
        }
    }
    if (e->err[0]) { snprintf(err, (size_t) errlen, "%s", e->err); q4e_close(e); return NULL; }
    return e;
}

void q4e_close(Q4Eng* e) {
    if (!e) return;
    if (e->prof >= 3) {
        double total = 0;
        for (NSString* k in e->kms) total += e->kms[k].doubleValue;
        fprintf(stderr, "Q4E_PROFILE=%d: %.0f ms of %s GPU time (a command buffer per dispatch; kernel|format|T|group)\n", e->prof, total,
                e->prof == 4 ? "prompt" : "decode");
        NSArray* keys = [e->kms keysSortedByValueUsingComparator:^NSComparisonResult(NSNumber* a, NSNumber* b) { return [b compare:a]; }];
        for (NSString* k in keys)
            fprintf(stderr, "  %-32s %8d calls %9.1f ms %7.2f us a call %5.1f%%\n", k.UTF8String, e->kcalls[k].intValue, e->kms[k].doubleValue,
                    1e3 * e->kms[k].doubleValue / e->kcalls[k].intValue, 100 * e->kms[k].doubleValue / (total > 0 ? total : 1));
    } else if (e->prof) {
        double total = 0;
        for (int i = 0; i < ST_N; ++i) total += e->stage_ms[i];
        fprintf(stderr, "Q4E_PROFILE: %.0f ms of GPU time (a command buffer per stage)\n", total);
        for (int i = 0; i < ST_N; ++i)
            if (e->stage_ms[i] > 0) fprintf(stderr, "  %-18s %9.1f ms  %5.1f%%\n", kStage[i], e->stage_ms[i], 100 * e->stage_ms[i] / (total > 0 ? total : 1));
    }
    if (e->residency) { [e->residency endResidency]; [e->queue removeResidencySet:e->residency]; }
    e->residency = nil;
    e->pipes = nil; e->buffers = nil; e->lib = nil; e->lib4 = nil; e->queue = nil; e->dev = nil;
    for (int l = 0; l < QWEN4_MAX_LAYERS; ++l) e->L[l] = (Layer) {0};
    e->embed = e->head = e->mix_down = e->mix_up = (MW) {0};
    e->M = (Layer) {0};
    e->m_fe = e->m_fh = e->m_down = e->m_up = (MW) {0};
    e->m_ne = e->m_nh = e->m_norm = e->Hin = e->Hm = e->ones = e->mlog = e->dtok = nil;
    e->mix_norm = e->stab = e->stab32 = e->inv = nil;
    e->x = e->H = e->normed = e->lo = e->up = e->inj = e->br = e->qp = e->kk = e->vv = e->o = e->ixo = e->sc = e->emb = nil;
    e->keys = e->qs = e->vals = e->gated = e->lg = e->wt = e->A = e->Gb = e->Ub = e->Dx = e->sh = e->sgv = e->logits = nil;
    e->y = e->cv = e->qkvb = e->zb = e->bb = e->ab = e->tok = e->ind = e->sel = e->done = e->nsel = e->zero = e->perm = e->tiles = e->apart = nil;
    free(e->ids);
    if (e->table) ple_close(&e->pt);
    ns_close(&e->nm);
    free(e);
}
const Qwen4Cfg* q4e_cfg(const Q4Eng* e) { return &e->c; }
int q4e_pos(const Q4Eng* e) { return e->pos; }
const char* q4e_error(const Q4Eng* e) { return e->err; }

void q4e_reset(Q4Eng* e) {
    const Qwen4Cfg* c = &e->c;
    for (int l = 0; l < c->n_layer; ++l) {
        Layer* L = &e->L[l];
        if (L->conv) memset(L->conv.contents, 0, L->conv.length);
        if (L->S) memset(L->S.contents, 0, L->S.length);
        if (L->hist) memset(L->hist.contents, 0, L->hist.length);
    }
    e->pos = 0;
    e->lrow = -1;
    e->nkeep = 0;
}

// ---- the forward of up to maxt rows ----------------------------------------------------------------------------------

static int forward_rows(Q4Eng* e, const int32_t* ids, int T, float* logits) {
    const Qwen4Cfg* c = &e->c;
    memcpy(e->tok.contents, ids, sizeof(int32_t) * (size_t) T);
    int32_t* done = (int32_t*) e->done.contents;
    int32_t* nsel = (int32_t*) e->nsel.contents;
    for (int t = 0; t < T; ++t) {   // the QSA blocks each row sees, and whether it selects (kq_q4_qsa_rows on CUDA)
        done[t] = (e->pos + t + 1) / c->ix_ratio;
        nsel[t] = done[t] > e->kq ? e->kq : 0;
    }
    if (ple_rows(e, T)) return -1;
    e->prof_skip = e->prof == 4 ? T <= MV_MAXT : e->prof >= 2 && T > MV_MAXT;
    cmd_begin(e);
    embed(e, T);
    mark(e, ST_EMBED);
    for (int l = 0; l < c->n_layer; ++l)
        if (run_layer(e, &e->L[l], l, c->attn[l], T, e->pos, 0)) return -1;
    if (logits) {   // prompt rows without logits skip the mixer and the head
        hc_mix(e, e->mix_norm, &e->mix_down, &e->mix_up, NULL, T);
        mv(e, &e->head, e->x, e->logits, T);
        mark(e, ST_HEAD);
    }
    head_rows(e, T);
    id<MTLCommandBuffer> last = e->cb;
    if (cmd_wait(e)) return -1;
    if (e->prof && !e->prof_skip) e->stage_ms[ST_HEAD] += 1e3 * (last.GPUEndTime - last.GPUStartTime);
    if (logits) memcpy(logits, e->logits.contents, sizeof(float) * (size_t) T * c->vocab);
    e->pos += T;
    e->lrow = logits ? T - 1 : -1;
    return 0;
}

int q4e_forward(Q4Eng* e, const int32_t* ids, int n, float* logits) {
    e->err[0] = 0;
    if (e->pos + n > e->max_pos) return fail(e, "the context is full (%d positions)", e->max_pos);
    for (int i = 0; i < n; ++i) if (ids[i] < 0 || ids[i] >= e->embed.rows) return fail(e, "token %d out of range", ids[i]);
    memcpy(e->ids + e->pos, ids, sizeof(int32_t) * (size_t) n);
    for (int l = 0; l < e->c.n_layer; ++l)   // all PLE rows of the call at once: the OS reads them in parallel, not row by row
        for (int t = 0; e->c.ple[l] >= 0 && t < n; ++t) qwen4_ple_prefetch(&e->nm, e->table, &e->c, &e->ng[e->c.ple[l]], l, e->ids, e->pos + t);
    const int step = logits ? LOGT : e->maxt;
    for (int i = 0; i < n; i += step) {
        const int T = n - i < step ? n - i : step;
        if (forward_rows(e, ids + i, T, logits ? logits + (size_t) i * e->c.vocab : NULL)) return -1;
    }
    return 0;
}

// ---- MTP: drafts, and verify forwards that keep each row's state -----------------------------------------------------

int q4e_draft(Q4Eng* e, int k, int32_t* drafts) {
    const Qwen4Cfg* c = &e->c;
    e->err[0] = 0;
    if (!e->mtp || e->lrow < 0 || k < 1 || k > e->ndraft || e->pos + k > e->max_pos)
        return fail(e, "q4e_draft: %s", !e->mtp ? "no MTP head" : e->lrow < 0 ? "no logits of the last row" : "k out of range");
    const float* l = (const float*) e->logits.contents + (size_t) e->lrow * c->vocab;
    int32_t* dt = (int32_t*) e->dtok.contents;
    dt[0] = 0;   // the bonus: the last row's arg max (ties: the lowest id)
    for (int v = 1; v < c->vocab; ++v) if (l[v] > l[dt[0]]) dt[0] = v;
    int32_t* done = (int32_t*) e->done.contents;
    int32_t* nsel = (int32_t*) e->nsel.contents;
    for (int j = 0; j < k; ++j) {   // head row j at pos - 1 + j
        done[j] = (e->pos + j) / c->ix_ratio;
        nsel[j] = done[j] > e->kq ? e->kq : 0;
    }
    e->prof_skip = 0;
    cmd_begin(e);
    const int rc = draft_rows(e, k);
    id<MTLCommandBuffer> last = e->cb;
    if (cmd_wait(e) || rc) return -1;
    if (e->prof) e->stage_ms[ST_HEAD] += 1e3 * (last.GPUEndTime - last.GPUStartTime);
    memcpy(drafts, dt + 1, sizeof(int32_t) * (size_t) k);
    return 0;
}

int q4e_forward_keep(Q4Eng* e, const int32_t* ids, int n, float* logits) {
    e->err[0] = 0;
    e->nkeep = 0;
    if (!e->mtp || n < 1 || n > e->ndraft + 1) return fail(e, "q4e_forward_keep: %s", e->mtp ? "more rows than Q4EOpts.mtp + 1" : "no MTP head");
    e->keep = 1;
    const int rc = q4e_forward(e, ids, n, logits);
    e->keep = 0;
    if (!rc) e->nkeep = n;
    return rc;
}

int q4e_accept(Q4Eng* e, int m) {
    const Qwen4Cfg* c = &e->c;
    const int n = e->nkeep;
    e->err[0] = 0;
    e->nkeep = 0;
    if (m < 1 || m > n) return fail(e, "q4e_accept: %d of %d kept rows", m, n);
    if (m == n) return 0;
    for (int l = 0; l < c->n_layer; ++l) {   // row m - 1's state: its snapshot buffers become the state
        Layer* L = &e->L[l];
        id<MTLBuffer> t;
        if (L->kS[0]) {
            t = L->conv; L->conv = L->kconv[m - 1]; L->kconv[m - 1] = t;
            t = L->S; L->S = L->kS[m - 1]; L->kS[m - 1] = t;
        }
        if (L->khist[0]) { t = L->hist; L->hist = L->khist[m - 1]; L->khist[m - 1] = t; }
    }
    const size_t HD = (size_t) c->hc * c->d;   // the head's next input: main row m - 1's streams
    memcpy(e->Hin.contents, (const float*) e->H.contents + (size_t) (m - 1) * HD, sizeof(float) * HD);
    if (e->lrow >= 0) e->lrow = m - 1;
    e->pos -= n - m;
    return 0;
}
