// nslm/rng.h - splitmix64 (Steele, Lea and Flood, 2014). The state is a 64-bit counter that steps by the golden ratio
// 2^64 / phi (odd, so it visits every value). The output mixes the counter with two xor-shift and odd-multiply rounds,
// so each output bit depends on each counter bit. It is fast and has no bad seeds, but it is not cryptographic.
#pragma once
#include <stdint.h>

#define NS_GOLDEN64 0x9E3779B97F4A7C15ull
static inline uint64_t ns_mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static inline uint64_t ns_splitmix64(uint64_t* s) { return ns_mix64(*s += NS_GOLDEN64); }   // the next output of stream *s
static inline double ns_unit(uint64_t* s) { return (double) (ns_splitmix64(s) >> 11) * (1.0 / 9007199254740992.0); }   // [0, 1)
