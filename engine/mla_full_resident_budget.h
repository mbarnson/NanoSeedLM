#pragma once
#include <stdint.h>

// Forecast only after all other GPU buffers exist. The actual allocator still
// decides placement. Parts are the four full-KV and three expanded-workspace
// allocations, before the allocator's individual256-byte rounding.
static int mla_full_resident_budget(uint64_t free_bytes, uint64_t reserve,
                                    uint64_t expert_min, const uint64_t kv[4],
                                    const uint64_t workspace[3], uint64_t cap,
                                    int explicit_kv_limit, uint64_t allowed_rows) {
    if (!kv || !workspace || !cap || (explicit_kv_limit && allowed_rows < cap)) return 0;
    if (reserve > free_bytes) return 0;
    free_bytes -= reserve;
    if (expert_min > free_bytes) return 0;
    free_bytes -= expert_min;
    for (int i = 0; i < 7; ++i) {
        const uint64_t bytes = i < 4 ? kv[i] : workspace[i - 4];
        if (bytes > UINT64_MAX - 255) return 0;
        const uint64_t aligned = (bytes + 255) & ~(uint64_t) 255;
        if (aligned > free_bytes) return 0;
        free_bytes -= aligned;
    }
    return 1;
}
