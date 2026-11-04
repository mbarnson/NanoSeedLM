// nslm/lib_rows.c - one weight row of a model-folder tensor as f32 (nslm/rows.h).
#include "rows.h"

#include <string.h>

#include "format.h"
#include "lfsr.h"
#include "search4.h"

static float bf2f(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }
static float bfr(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

void ns_rows_init(NsRows* r) {
    for (uint32_t s = 0; s < 65536; ++s) { r->g24[s] = lfsr_stream24((uint16_t) s); r->g32[s] = lfsr_stream32((uint16_t) s); }
}

void ns_row_f32(const NsRows* r, const NsTensor* t, int slice, int row, int round, float* out) {
    const int K = t->cols;
    const uint64_t R = (uint64_t) t->rows, base = ((uint64_t) slice * R + (uint64_t) row) * (uint64_t) K;
    switch (t->enc) {
    case NS_BF16: {
        const uint16_t* w = (const uint16_t*) t->s[0].p + base;
        for (int c = 0; c < K; ++c) out[c] = bf2f(w[c]);
        break;
    }
    case NS_Q8:
    case NS_Q4: {
        const uint32_t* q = (const uint32_t*) t->s[0].p;
        const uint16_t* S = (const uint16_t*) t->s[1].p;
        const uint16_t* B = (const uint16_t*) t->s[2].p;
        for (int c = 0; c < K; ++c) {
            const uint64_t i = base + (uint64_t) c;
            const float s = bf2f(S[i / (uint64_t) t->group]), b = bf2f(B[i / (uint64_t) t->group]);
            const uint32_t v = t->enc == NS_Q8 ? (q[i / 4] >> (8 * (i % 4))) & 255u : (q[i / 8] >> (4 * (i % 8))) & 15u;
            out[c] = round ? bfr(s * (float) v + b) : s * (float) v + b;
        }
        break;
    }
    case NS_SEED4P4:
    case NS_SEED4: {
        const int p4 = t->enc == NS_SEED4P4, P = p4 ? 4 : 3, nbk = K / 8;
        const uint16_t* seeds = (const uint16_t*) t->s[0].p;
        const uint16_t* cw = (const uint16_t*) t->s[1].p;
        int32_t eb;
        memcpy(&eb, t->s[2].p + 4 * (size_t) slice, 4);
        for (int j = 0; j < nbk; ++j) {
            const uint64_t k = base / 8 + (uint64_t) j;
            const uint32_t s = seeds[k], w = cw[k];
            const uint64_t st = (uint64_t) s | ((uint64_t) (p4 ? r->g32[s] : r->g24[s]) << 16);
            int e, q[4];
            if (p4) {
                e = eb + ((t->s[3].p[k >> 1] >> ((k & 1) * 4)) & 15);
                for (int p = 0; p < 4; ++p) q[p] = nslm4_q((uint16_t) w, p);
            } else {
                e = eb + (int) (w & 15u);
                for (int p = 0; p < 3; ++p) q[p] = nslm_q((uint16_t) w, p);
            }
            const float sc = nslm_scale(e);
            for (int cc = 0; cc < 8; ++cc) {
                int32_t isum = 0;
                for (int p = 0; p < P; ++p) isum += ((int32_t) ((st >> (cc * P + p + 1)) & 0xFFFFu) - 32768) * q[p];
                out[j * 8 + cc] = (float) isum * sc;
            }
        }
        break;
    }
    }
}

void ns_row_prefetch(const NsTensor* t, int slice, int row) {
    const uint64_t K = (uint64_t) t->cols, base = ((uint64_t) slice * (uint64_t) t->rows + (uint64_t) row) * K;
    if (t->enc == NS_BF16) st_willneed(t->s[0].p + 2 * base, 2 * K);
    if (t->enc != NS_Q8 && t->enc != NS_Q4) return;
    const uint64_t g = (uint64_t) t->group, g0 = base / g, gn = (base + K - 1) / g - g0 + 1;
    st_willneed(t->s[0].p + base * (t->enc == NS_Q8 ? 8 : 4) / 8, K * (t->enc == NS_Q8 ? 8 : 4) / 8);
    st_willneed(t->s[1].p + 2 * g0, 2 * gn);
    st_willneed(t->s[2].p + 2 * g0, 2 * gn);
}
