// harness/nslm-mova-routes.c - router choices of every forwarded row: the prompt, then a greedy generation.
//
//   nslm-mova-routes --model DIR [--res out/res] (--text FILE --ctx N | --chat "prompt") [--decode 256] --out FILE
//
// FILE: int32 records, one per row: [n_sparse][top_k] MLP expert ids, then [n_sparse][top_kv] value expert ids.
// The header line on stdout gives rows, prompt rows, n_sparse, top_k, top_kv.  For expert-cache studies.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine_api.h"
#include "mova_cfg.h"
#include "mova_ext.h"
#include "platform.h"
#include "tokenizer.h"

int main(int argc, char** argv) {
    plat_init(&argc, &argv);
    const char* model = opt(argc, argv, "--model", NULL), *out = opt(argc, argv, "--out", NULL);
    const char* text = opt(argc, argv, "--text", NULL), *chat = opt(argc, argv, "--chat", NULL);
    const int ndec = atoi(opt(argc, argv, "--decode", "256"));
    if (!model || !out || (!text && !chat)) {
        fprintf(stderr, "usage: nslm-mova-routes --model DIR (--text FILE --ctx N | --chat PROMPT) [--decode 256] --out FILE\n");
        return 2;
    }
    char err[512] = "", path[2048];
    snprintf(path, sizeof path, "%s/tokenizer.json", model);
    Tok* tok = tok_open(path, err, sizeof err);
    if (!tok) { fprintf(stderr, "tokenizer: %s\n", err); return 1; }
    int n = 0, cap = 1 << 20;
    int32_t* ids = (int32_t*) malloc(sizeof(int32_t) * (size_t) cap);
    if (text) {
        char* t = plat_slurp(text, NULL);
        if (!t) { fprintf(stderr, "cannot read %s\n", text); return 1; }
        n = tok_encode(tok, t, 1, ids, cap);
        free(t);
        const int ctx = atoi(opt(argc, argv, "--ctx", "1024"));
        if (n > ctx) n = ctx;
    } else {
        char* buf = (char*) malloc(strlen(chat) + 256);
        sprintf(buf, "<|ifm|begin_of_text|><|ifm|im_start|>user\n%s<|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think>\n", chat);
        n = tok_encode(tok, buf, 0, ids, cap);
        free(buf);
    }
    MovaCfg c;
    if (mova_cfg_load(&c, model, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    const int ns = c.n_layer - c.first_sparse, rows = n - 1 + ndec;
    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = model;
    o.resource_dir = opt(argc, argv, "--res", "out/res");
    o.max_seqs = 1;
    o.kv_tokens = n + ndec + 64;
    Eng* e = eng_open(&o, err, sizeof err);
    if (!e) { fprintf(stderr, "eng_open: %s\n", err); return 1; }
    eng_mova_routes(e, 1, rows);
    int32_t* gen = (int32_t*) malloc(sizeof(int32_t) * (size_t) (ndec + 1));
    const int seq = 0;
    if (eng_prefill(e, 0, ids, n) || eng_generate(e, &seq, 1, ndec, ENG_MODE_AR, gen, NULL)) { fprintf(stderr, "engine failed\n"); return 1; }
    int32_t* m = (int32_t*) malloc(sizeof(int32_t) * (size_t) rows * ns * c.top_k);
    int32_t* v = (int32_t*) malloc(sizeof(int32_t) * (size_t) rows * ns * c.top_kv);
    if (eng_mova_routes_read(e, rows, m, v, NULL, NULL)) { fprintf(stderr, "routes\n"); return 1; }
    FILE* f = fopen(out, "wb");
    for (int r = 0; r < rows; ++r) {
        fwrite(m + (size_t) r * ns * c.top_k, 4, (size_t) ns * c.top_k, f);
        fwrite(v + (size_t) r * ns * c.top_kv, 4, (size_t) ns * c.top_kv, f);
    }
    fclose(f);
    printf("rows=%d prompt_rows=%d n_sparse=%d top_k=%d top_kv=%d\n", rows, n - 1, ns, c.top_k, c.top_kv);
    if (chat) {
        char* s = tok_decode(tok, gen, ndec, NULL);
        fprintf(stderr, "%s\n", s);
        free(s);
    }
    eng_close(e);
    tok_close(tok);
    return 0;
}
