// harness/nslm-qwen4-gen.c - greedy generation through the Qwen3.8 CUDA engine (engine/qwen4_eng.h), with timings.
//   nslm-qwen4-gen --model DIR --prompt TEXT [--max-tokens 128] [--ctx 4096] [--raw] [--vram-experts]   (--raw: no chat template)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"
#include "qwen4_eng.h"
#include "tokenizer.h"

int main(int argc, char** argv) {
    plat_init(&argc, &argv);
    const char *model = NULL, *prompt = NULL;
    int max_tokens = 128, ctx = 4096, raw = 0, vram = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--raw")) raw = 1;
        else if (!strcmp(argv[i], "--vram-experts")) vram = 1;
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (!model || !prompt) { fprintf(stderr, "usage: nslm-qwen4-gen --model DIR --prompt TEXT [--max-tokens N] [--ctx N] [--raw]\n"); return 2; }
    char err[512], path[2048];
    snprintf(path, sizeof path, "%s/tokenizer.json", model);
    Tok* tok = tok_open(path, err, sizeof err);
    if (!tok) { fprintf(stderr, "%s\n", err); return 1; }
    const size_t tl = strlen(prompt) + 128;
    char* text = (char*) malloc(tl);
    if (raw) snprintf(text, tl, "%s", prompt);
    else snprintf(text, tl, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
    int32_t* ids = (int32_t*) malloc(sizeof(int32_t) * (size_t) ctx);
    const int n = tok_encode(tok, text, 0, ids, ctx);
    if (n < 1 || n + max_tokens > ctx) { fprintf(stderr, "prompt of %d tokens does not fit --ctx %d\n", n, ctx); return 1; }
    double t0 = now_s();
    Q4Eng* e = q4e_open(model, ctx, vram, err, sizeof err);
    if (!e) { fprintf(stderr, "%s\n", err); return 1; }
    const Qwen4Cfg* c = q4e_cfg(e);
    fprintf(stderr, "loaded in %.1f s; prompt %d tokens\n", now_s() - t0, n);
    float* logits = (float*) malloc(sizeof(float) * (size_t) Q4E_MAX_ROWS * c->vocab);
    t0 = now_s();
    int rc = 0;
    for (int i = 0; i < n && !rc; i += Q4E_MAX_ROWS) {   // only the last row's logits are needed; rows of up to 8
        const int T = n - i < Q4E_MAX_ROWS ? n - i : Q4E_MAX_ROWS;
        rc = q4e_forward(e, ids + i, T, logits);
        if (!rc && i + T == n) memmove(logits, logits + (size_t) (T - 1) * c->vocab, sizeof(float) * (size_t) c->vocab);
    }
    const double tp = now_s() - t0;
    const int32_t im_end = tok_encode(tok, "<|im_end|>", 0, ids + n, 1) == 1 ? ids[n] : -1;
    int gen = 0;
    t0 = now_s();
    while (!rc && gen < max_tokens) {
        int32_t best = 0;
        for (int v = 1; v < c->vocab; ++v) if (logits[v] > logits[best]) best = v;
        if (best == c->eos || best == im_end) break;
        int valid = 1;
        char* s = tok_decode(tok, &best, 1, &valid);
        fputs(s, stdout);
        fflush(stdout);
        free(s);
        ++gen;
        rc = q4e_forward(e, &best, 1, logits);
    }
    const double td = now_s() - t0;
    if (rc) fprintf(stderr, "\n%s\n", q4e_error(e));
    fprintf(stderr, "\nprompt %d tokens in %.2f s (%.1f tokens/s); %d tokens in %.2f s (%.2f tokens/s)\n", n, tp, n / (tp > 0 ? tp : 1), gen,
            td, gen / (td > 0 ? td : 1));
    q4e_close(e);
    tok_close(tok);
    free(text); free(ids); free(logits);
    return rc ? 1 : 0;
}
