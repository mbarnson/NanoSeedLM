// tests/test_lfsr.c - the LFSR spec (nslm/lfsr.h): period, taps, seed semantics, word-parallel and state-cache
// generators vs scalar on every seed, block pack/unpack, decode exactness.  Exit 0 = pass.
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "lfsr.h"

static LfsrCache g_cache;

int main(void) {
    // period: from state 1 every non-zero state is visited exactly once before returning to 1
    {
        static uint8_t seen[65536];
        uint16_t s = 1;
        int n = 0;
        do { assert(!seen[s]); seen[s] = 1; s = lfsr_step(s); ++n; } while (s != 1);
        assert(n == LFSR_PERIOD && !seen[0]);
        printf("period %d: ok\n", n);
    }
    // taps (0, 1, 3, 12), shift right, new bit at 15: hand-computed states after seed 1
    {
        const uint16_t want[6] = {0x8000, 0x4000, 0x2000, 0x1000, 0x8800, 0x4400};
        uint16_t got[6];
        lfsr_states(1, 6, got);
        for (int i = 0; i < 6; ++i) assert(got[i] == want[i]);
        printf("taps: ok\n");
    }
    // paper semantics: V(s) starts at the state AFTER the seed
    {
        uint16_t st[24];
        lfsr_states(0xACE1, 24, st);
        assert(st[0] == lfsr_step(0xACE1) && st[0] != 0xACE1);
        printf("seed semantics: ok\n");
    }
    // word-parallel and state cache == scalar, all seeds, 24 and 48 states
    lfsr_cache_build(&g_cache);
    {
        uint64_t h = 1469598103934665603ull;
        for (uint32_t s = 1; s <= 65535; ++s) {
            uint16_t a[48], b[48], c[48];
            lfsr_states((uint16_t) s, 48, a);
            lfsr_states_wordpar((uint16_t) s, 48, b);
            lfsr_states_cache(&g_cache, (uint16_t) s, 48, c);
            for (int k = 0; k < 48; ++k) {
                if (a[k] != b[k] || a[k] != c[k]) { printf("seed %u state %d: %04x %04x %04x\n", s, k, a[k], b[k], c[k]); return 1; }
            }
            uint16_t b24[24];
            lfsr_states_wordpar((uint16_t) s, 24, b24);
            for (int k = 0; k < 24; ++k) { assert(b24[k] == a[k]); h = (h ^ a[k]) * 1099511628211ull; }
        }
        // golden: FNV-1a over the 24 states of every seed (locks the definition; any change is a format break)
        printf("all 65535 seeds: word-parallel == cache == scalar (24 and 48 states); fnv %016llx\n", (unsigned long long) h);
        assert(h == 0x740e86e513d34463ull);
    }
    // pack / unpack and int4 sign extension
    for (int e = 0; e < 16; ++e)
        for (int q0 = -8; q0 < 8; ++q0)
            for (int q1 = -8; q1 < 8; ++q1) {
                const int q[3] = {q0, q1, -q0 - 1};
                const uint16_t n = nslm_pack(e, q);
                assert(nslm_ecode(n) == e && nslm_q(n, 0) == q0 && nslm_q(n, 1) == q1 && nslm_q(n, 2) == -q0 - 1);
            }
    printf("pack/unpack: ok\n");
    // decode formula: fl(isum * fl(R32 * 2^e)) == fl(fl(isum * R32) * 2^e) (power-of-two scaling is exact), and the
    // BF16 result is the round-to-nearest-even of the exact real value except where f32 rounding moves a tie
    {
        srand(7);
        int ties = 0;
        for (int it = 0; it < 2000000; ++it) {
            const int32_t isum = (rand() % 1572863) - 786431;
            const int e = -30 + rand() % 31;
            const float a = (float) isum * nslm_scale(e);
            const float b = ldexpf((float) isum * NSLM_R32, e);
            assert(a == b);
            const double exact = (double) isum / 32767.0 * ldexp(1.0, e);
            const float rb = nslm_bf2f(nslm_f2bf(a));
            const double ulp = ldexp(1.0, ilogb(exact) - 7);
            if (fabs(rb - exact) > 0.5 * ulp * (1 + 1e-6)) ++ties;
        }
        printf("decode exactness: ok (%d of 2000000 bf16 roundings differ from exact-real RNE by double rounding)\n", ties);
        assert(ties < 2000);
    }
    printf("test_lfsr: PASS\n");
    return 0;
}
