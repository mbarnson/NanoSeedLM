// nslm/trellis.metal - the trellis Viterbi of nslm/trellis.h on Metal (host side: nslm/trellis_metal.m, API: trellis_gpu.h).
// One threadgroup a block of n <= T weights, one pass of nslm_tq_encode_row's three: the least sum_t h_t (w_t - a q(s_t))^2
// over the windows s_t = s_{t-1} >> K | b << (L - K), in natural labels as nslm/lib_trellis.c. The predecessors of s are
// the group g = s & (Q - 1): states g << K | j, contiguous. Each thread holds SPT consecutive states (whole groups), so the
// minimum m[g] (the lowest j on a tie) is in the thread. The minima go through threadgroup memory (at most TQ_CH floats a
// round); a state reads its m[s & (Q - 1)] and adds (h d) d with d = w - a q(s), as the C code: a separate multiply and
// add (this file is compiled with -fno-fast-math, so nothing is contracted). Back pointers: K bits a group a step, packed
// in words per thread. k_tq_trace then walks back from the last state of least cost (the lowest state on a tie).
#include <metal_stdlib>
using namespace metal;

#define TQ_CH 8192   // minima in threadgroup memory a round (32 KB)

struct TqVitArgs {   // = nslm/trellis_metal.m
    int L, K, T, nblk;
    int nt, spt;          // threads a block, states a thread (SPT / 2^K groups: their K-bit back pointers fit one word)
    uint back_stride;     // words of one block's back pointers: T * nt
};
struct TqBlk {           // = nslm/trellis_metal.m
    uint w0;              // the block's first weight in w
    int h0;               // its first importance in h, or -1 (all 1)
    int n;                // weights in the block
    float a;              // the scale of this pass
};

static inline int tq_code(uint s) {   // nslm_tq_code
    const uint x = s * 34038481u + 76625530u;
    return int((x & 255u) + (x >> 8 & 255u) + (x >> 16 & 255u) + (x >> 24)) - 510;
}
// Threadgroup slot of minimum x of a round: the low 5 bits XOR the thread's run, so a simdgroup's reads hit 32 banks
static inline uint tq_swz(uint x, uint sh) { return x ^ ((x >> sh) & 31u); }

template <int SPT, int KB>
static void tq_vit(constant TqVitArgs& A, device const float* w, device const float* hv, device const TqBlk* blk,
                   device uint* back, device uint* fin, threadgroup float* tm, uint b, uint i, uint lane, uint sg) {
    constexpr int H = 1 << KB, GPT = SPT / H;   // groups a thread
    static_assert(GPT * KB <= 32, "a thread's back pointers of a step fit one word");
    constexpr uint sh = SPT >= 32 ? uint(SPT >= 256 ? 8 : SPT >= 128 ? 7 : SPT >= 64 ? 6 : 5) : 5u;
    const int L = A.L;
    const uint N = 1u << L, Q = N >> KB;
    const TqBlk B = blk[b];
    const float a = B.a;
    device const float* wb = w + B.w0;
    const bool hw = B.h0 >= 0;
    device const float* hb = hv + (hw ? B.h0 : 0);
    const uint s0 = i * SPT;
    short q[SPT];   // the codes q(s) of the thread's states
#pragma unroll
    for (int k = 0; k < SPT; ++k) q[k] = short(tq_code(s0 + uint(k)));
    device uint* bk = back + ulong(b) * A.back_stride;
    const uint g0 = i * GPT, base = s0 & (Q - 1u);
    const uint CH = Q < TQ_CH ? Q : TQ_CH;
    const int rounds = int(Q / CH);
    // A state's cost is never stored: cost_t(s) = m_t[s & (Q - 1)] + (h_t d) d with d = w_t - a q(s) (cost_0: (h_0 d) d), and
    // the thread's states are whole groups, so each step reduces its costs at once into the next minima m_{t+1}[g] (the
    // lowest j on a tie) and their back pointers. Only the GPT minima live from step to step.
    float m[GPT];
    float bc = 0;   // the last step: the least cost and its state (the lowest on a tie)
    uint bs = s0;
    {
        const float w0 = wb[0], h0 = hw ? hb[0] : 1.0f;
        const bool last = B.n == 1;
        uint jw = 0;
#pragma unroll
        for (int gg = 0; gg < GPT; ++gg) {
            float mv = 0;
            uint jb = 0;
#pragma unroll
            for (int j = 0; j < H; ++j) {
                const int k = gg * H + j;
                const float ac = a * float(q[k]);
                const float d = w0 - ac;
                const float hd = h0 * d;
                const float cv = hd * d;
                if (last) { if (k == 0 || cv < bc) { bc = cv; bs = s0 + uint(k); } }
                else if (j == 0 || cv < mv) { mv = cv; jb = uint(j); }
            }
            m[gg] = mv;
            jw |= jb << uint(gg * KB);
        }
        if (!last) bk[ulong(1) * uint(A.nt) + i] = jw;
    }
    // steps 1 .. n - 2: the next minima; the last step after the loop, so the loop body holds one form
    for (int t = 1; t < B.n - 1; ++t) {
        const float wt = wb[t], ht = hw ? hb[t] : 1.0f;
        float mn[GPT];
        uint jw = 0;
        for (int r = 0; r < rounds; ++r) {
            const uint lo = uint(r) * CH;
            if (g0 >= lo && g0 < lo + CH)
#pragma unroll
                for (int gg = 0; gg < GPT; ++gg) tm[tq_swz(g0 - lo + uint(gg), sh)] = m[gg];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (base >= lo && base < lo + CH)
#pragma unroll
                for (int gg = 0; gg < GPT; ++gg) {
                    float mv = 0;
                    uint jb = 0;
#pragma unroll
                    for (int j = 0; j < H; ++j) {
                        const int k = gg * H + j;
                        const float mk = tm[tq_swz(base - lo + uint(k), sh)];
                        const float ac = a * float(q[k]);
                        const float d = wt - ac;
                        const float hd = ht * d;
                        const float e = hd * d;
                        const float cv = mk + e;
                        if (j == 0 || cv < mv) { mv = cv; jb = uint(j); }
                    }
                    mn[gg] = mv;
                    jw |= jb << uint(gg * KB);
                }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
#pragma unroll
        for (int gg = 0; gg < GPT; ++gg) m[gg] = mn[gg];
        bk[ulong(t + 1) * uint(A.nt) + i] = jw;
    }
    if (B.n > 1) {   // the last step: the least cost over all states
        const int t = B.n - 1;
        const float wt = wb[t], ht = hw ? hb[t] : 1.0f;
        for (int r = 0; r < rounds; ++r) {
            const uint lo = uint(r) * CH;
            if (g0 >= lo && g0 < lo + CH)
                for (int gg = 0; gg < GPT; ++gg) tm[tq_swz(g0 - lo + uint(gg), sh)] = m[gg];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (base >= lo && base < lo + CH)
                for (int k = 0; k < SPT; ++k) {
                    const float mk = tm[tq_swz(base - lo + uint(k), sh)];
                    const float ac = a * float(q[k]);
                    const float d = wt - ac;
                    const float hd = ht * d;
                    const float e = hd * d;
                    const float cv = mk + e;
                    if (k == 0 || cv < bc) { bc = cv; bs = s0 + uint(k); }
                }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    const uint width = A.nt < 32 ? uint(A.nt) : 32u;   // the simdgroup's lanes (a block of few states: under 32 threads)
    for (ushort off = 16; off > 0; off >>= 1) {
        const float oc = simd_shuffle_down(bc, off);
        const uint os = simd_shuffle_down(bs, off);
        if (lane + off < width && (oc < bc || (oc == bc && os < bs))) { bc = oc; bs = os; }
    }
    threadgroup uint* ts = (threadgroup uint*) (tm + 64);
    if (lane == 0) { tm[sg] = bc; ts[sg] = bs; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i == 0) {
        const int nsg = (A.nt + 31) / 32;
        for (int x = 1; x < nsg; ++x)
            if (tm[x] < bc || (tm[x] == bc && ts[x] < bs)) { bc = tm[x]; bs = ts[x]; }
        fin[b] = bs;
    }
}

#define TQ_VIT(SPT, KB)                                                                                                       \
    kernel void k_tq_vit_##SPT##_##KB(constant TqVitArgs& A [[buffer(0)]], device const float* w [[buffer(1)]],               \
                                     device const float* h [[buffer(2)]], device const TqBlk* blk [[buffer(3)]],              \
                                     device uint* back [[buffer(4)]], device uint* fin [[buffer(5)]],                         \
                                     uint b [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]],         \
                                     uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {    \
        threadgroup float tm[TQ_CH];                                                                                         \
        tq_vit<SPT, KB>(A, w, h, blk, back, fin, tm, b, i, lane, sg);                                                          \
    }
// (states a thread, K): SPT = max(2^K, 2^L / 1024), L <= 16
TQ_VIT(2, 1) TQ_VIT(4, 1) TQ_VIT(8, 1) TQ_VIT(16, 1) TQ_VIT(32, 1) TQ_VIT(64, 1)
TQ_VIT(4, 2) TQ_VIT(8, 2) TQ_VIT(16, 2) TQ_VIT(32, 2) TQ_VIT(64, 2)
TQ_VIT(8, 3) TQ_VIT(16, 3) TQ_VIT(32, 3) TQ_VIT(64, 3)
TQ_VIT(16, 4) TQ_VIT(32, 4) TQ_VIT(64, 4)
TQ_VIT(32, 5) TQ_VIT(64, 5)
TQ_VIT(64, 6) TQ_VIT(128, 7) TQ_VIT(256, 8)

// One thread a block: the path of windows from the last state back, as nslm/lib_trellis.c's viterbi
kernel void k_tq_trace(constant TqVitArgs& A [[buffer(0)]], device const TqBlk* blk [[buffer(3)]], device const uint* back [[buffer(4)]],
                       device const uint* fin [[buffer(5)]], device ushort* path [[buffer(6)]], uint b [[thread_position_in_grid]]) {
    if (int(b) >= A.nblk) return;
    const int L = A.L, K = A.K, H = 1 << K, GPT = A.spt / H;
    const uint Q = (1u << L) >> K;
    device const uint* bk = back + ulong(b) * A.back_stride;
    device ushort* pb = path + ulong(b) * uint(A.T);
    uint s = fin[b];
    for (int t = blk[b].n - 1; t > 0; --t) {
        pb[t] = ushort(s);
        const uint g = s & (Q - 1u), i = g / uint(GPT), gg = g % uint(GPT);
        s = g << K | ((bk[ulong(t) * uint(A.nt) + i] >> (gg * uint(K))) & uint(H - 1));
    }
    pb[0] = ushort(s);
}
