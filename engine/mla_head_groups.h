#pragma once
#include <stdio.h>
#include <string.h>
#include "model_st.h"

// Share expanded K/V only when every q_lat and v_up stream is identical within groups of4.
// Unsupported source layouts or any untied head keep the existing32-head layout.
static int mla_head_groups(const NsModel* model, int H, int layers) {
    if (H < 4 || H % 4 || layers < 1) return 1;
    for (int layer = 0; layer < layers; ++layer) for (int kind = 0; kind < 2; ++kind) {
        char name[128];
        snprintf(name, sizeof name, "model.layers.%d.self_attn.mla.%s", layer, kind ? "v_up" : "q_lat");
        const NsTensor* tensor = ns_find(model, name);
        if (!tensor) {
            strcat(name, ".weight");
            tensor = ns_find(model, name);
        }
        if (!tensor || tensor->slices != H || tensor->rows < 1 || tensor->cols < 1 ||
            (tensor->enc != NS_BF16 && tensor->enc != NS_Q8 && tensor->enc != NS_Q4 && tensor->enc != NS_SEED4P4)) return 1;
        for (int stream = 0; stream < 4; ++stream) {
            const uint64_t step = ns_stream_len(tensor->enc, 1, tensor->rows, tensor->cols, stream);
            if (!step) continue;
            if (!tensor->s[stream].p || tensor->s[stream].len < step * (uint64_t) H) return 1;
            for (int h = 0; h < H; h += 4) for (int j = 1; j < 4; ++j)
                if (memcmp(tensor->s[stream].p + step * (uint64_t) h,
                           tensor->s[stream].p + step * (uint64_t) (h + j), (size_t) step)) return 1;
        }
    }
    return 4;
}
