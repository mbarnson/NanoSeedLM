#pragma once
#include <stdint.h>
// Sharing is a compute-only optimization: existing allocations are retained.
static int mla_resident_head_policy(int enabled, int forecast, int keys, uint64_t cap, uint64_t resident) {
    return enabled && forecast && keys > 0 && cap > 0 && resident == cap;
}
