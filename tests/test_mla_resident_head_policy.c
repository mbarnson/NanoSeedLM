#include <stdio.h>
#include "mla_resident_head_policy.h"
int main(void) {
    int checks = 0;
    for (int enabled = 0; enabled <= 1; ++enabled)
    for (int forecast = 0; forecast <= 1; ++forecast)
    for (int keys = 0; keys <= 1; ++keys)
    for (int full = 0; full <= 1; ++full) {
        const int got = mla_resident_head_policy(enabled, forecast, keys ? 16384 : 0, 40960, full ? 40960 : 8192);
        if (got != (enabled && forecast && keys && full)) return 1;
        ++checks;
    }
    if (mla_resident_head_policy(1, 1, 16384, 0, 0) ||
        mla_resident_head_policy(1, 1, -1, 40960, 40960) ||
        mla_resident_head_policy(1, 1, 8192, 131072, 88721) ||
        mla_resident_head_policy(1, 1, 16384, 40960, 40961)) return 1;
    checks += 4;
    printf("resident head policy: passed%dchecks\n", checks);
    return 0;
}
