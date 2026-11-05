// nslm/cpu_seed4_none.c - the CPU seed kernel's calls in a build without Clang vectors (MSVC, GCC): cpu_seed4_on() is 0
#include <stdlib.h>

#include "cpu_seed4.h"

int cpu_seed4_on(void) { return 0; }
uint32_t* cpu_seed4_table(void) { return NULL; }
void cpu_seed_xt(const float* x, int ntok, int kb, float* xt, float* xs) { (void) x; (void) ntok; (void) kb; (void) xt; (void) xs; abort(); }
void cpu_seed3_rows(const uint16_t* seeds, const uint16_t* nibs, int bias, const uint32_t* table, const float* xt,
                    const float* xs, int ntok, int kb, int r0, int r1, float* y, int ys) {
    (void) seeds; (void) nibs; (void) bias; (void) table; (void) xt; (void) xs; (void) ntok; (void) kb; (void) r0; (void) r1; (void) y; (void) ys;
    abort();
}
