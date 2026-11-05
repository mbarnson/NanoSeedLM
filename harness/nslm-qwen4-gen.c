// harness/nslm-qwen4-gen.c - greedy generation through the qwen4 CUDA engine, with timings. The options are in
// docs/guides/run-and-serve.md (Qwen3.8-Flash-Next on CUDA).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat_template.h"
#include "platform.h"
#include "qwen4_eng.h"
#include "tokenizer.h"

static int32_t argmax(const float* l, int n) { int32_t b = 0; for (int v = 1; v < n; ++v) if (l[v] > l[b]) b = v; return b; }

int main(int argc, char** argv) {
    plat_init(&argc, &argv);
    const char *model = NULL, *prompt = NULL, *ple = NULL, *effort = NULL;
    int max_tokens = 128, ctx = 4096, raw = 0, vram = 0, no_think = 0;
    double cache_gb = 0;
    int stream = 0, rows = 0, drafts = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--raw")) raw = 1;
        else if (!strcmp(argv[i], "--effort") && i + 1 < argc) effort = argv[++i];
        else if (!strcmp(argv[i], "--no-think")) no_think = 1;
        else if (!strcmp(argv[i], "--vram-experts")) vram = 1;
        else if (!strcmp(argv[i], "--ple-gguf") && i + 1 < argc) ple = argv[++i];
        else if (!strcmp(argv[i], "--expert-cache-gb") && i + 1 < argc) cache_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--stream-experts")) stream = 1;
        else if (!strcmp(argv[i], "--prompt-rows") && i + 1 < argc) rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--drafts") && i + 1 < argc) drafts = atoi(argv[++i]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (!model || !prompt) { fprintf(stderr, "usage: nslm-qwen4-gen --model DIR --prompt TEXT [--max-tokens N] [--ctx N] [--raw | --effort xhigh|medium|low\n"
                                         "  | --no-think] [--vram-experts] [--ple-gguf FILE] [--expert-cache-gb N [--stream-experts]] [--prompt-rows N] [--drafts 1-3]\n"); return 2; }
    if (drafts < 0 || drafts > Q4E_MAX_DRAFTS) { fprintf(stderr, "--drafts: 0 to %d\n", Q4E_MAX_DRAFTS); return 2; }
    char err[512], path[2048];
    snprintf(path, sizeof path, "%s/tokenizer.json", model);
    Tok* tok = tok_open(path, err, sizeof err);
    if (!tok) { fprintf(stderr, "%s\n", err); return 1; }
    Json* msgs = json_new(J_ARR), *m = json_new(J_OBJ);   // one user turn through the model's chat template
    json_set(m, "role", json_str("user"));
    json_set(m, "content", json_str(prompt));
    json_push(msgs, m);
    CtOpts co = {NULL, NULL, effort, 1, no_think, 0};
    char* text = raw ? (char*) malloc(strlen(prompt) + 1) : ct_render_qwen(msgs, NULL, &co, err, sizeof err);
    json_free(msgs);
    if (!text) { fprintf(stderr, "chat template: %s\n", err); return 2; }
    if (raw) strcpy(text, prompt);
    int32_t* ids = (int32_t*) malloc(sizeof(int32_t) * (size_t) ctx);
    const int n = tok_encode(tok, text, 0, ids, ctx);
    if (n < 1 || n + max_tokens > ctx) { fprintf(stderr, "prompt of %d tokens does not fit --ctx %d\n", n, ctx); return 1; }
    double t0 = now_s();
    const Q4EOpts o = {ctx, vram, ple, cache_gb, stream, rows, drafts > 0};
    Q4Eng* e = q4e_open(model, &o, err, sizeof err);
    if (!e) { fprintf(stderr, "%s\n", err); return 1; }
    const Qwen4Cfg* c = q4e_cfg(e);
    fprintf(stderr, "loaded in %.1f s; prompt %d tokens\n", now_s() - t0, n);
    float* logits = (float*) malloc(sizeof(float) * (size_t) c->vocab * (Q4E_MAX_DRAFTS + 1));
    t0 = now_s();
    int rc = 0;
    if (n > 1) rc = q4e_forward(e, ids, n - 1, NULL);   // the prompt; the last token's logits start the decode
    if (!rc) rc = q4e_forward(e, ids + n - 1, 1, logits);
    const double tp = now_s() - t0;
    const int32_t im_end = tok_encode(tok, "<|im_end|>", 0, ids + n, 1) == 1 ? ids[n] : -1;
    int gen = 0, steps = 0;
    t0 = now_s();
    while (!rc && gen < max_tokens) {
        int32_t row[Q4E_MAX_DRAFTS + 1];
        int m = 1;
        row[0] = argmax(logits, c->vocab);   // the bonus: the last row's next token
        if (row[0] == c->eos || row[0] == im_end) break;
        if (drafts) {   // the bonus and the drafts in one forward: rows 0 .. m - 1 hold
            rc = q4e_draft(e, drafts, row + 1) || q4e_forward_keep(e, row, drafts + 1, logits);
            while (!rc && m <= drafts && argmax(logits + (size_t) (m - 1) * c->vocab, c->vocab) == row[m] && row[m] != c->eos && row[m] != im_end) ++m;
            rc = rc || q4e_accept(e, m);
            memmove(logits, logits + (size_t) (m - 1) * c->vocab, sizeof(float) * (size_t) c->vocab);
        } else rc = q4e_forward(e, row, 1, logits);
        for (int i = 0; i < m && gen < max_tokens; ++i, ++gen) {
            int valid = 1;
            char* s = tok_decode(tok, row + i, 1, &valid);
            fputs(s, stdout);
            fflush(stdout);
            free(s);
        }
        ++steps;
    }
    const double td = now_s() - t0;
    if (rc) fprintf(stderr, "\n%s\n", *q4e_error(e) ? q4e_error(e) : "this engine has no MTP drafts");
    fprintf(stderr, "\nprompt %d tokens in %.2f s (%.1f tokens/s); %d tokens in %.2f s (%.2f tokens/s)\n", n, tp, n / (tp > 0 ? tp : 1), gen,
            td, gen / (td > 0 ? td : 1));
    if (drafts) fprintf(stderr, "%d drafts a forward: %d forwards, %.2f tokens a forward\n", drafts, steps, gen / (steps > 0 ? (double) steps : 1));
    uint64_t hits, misses;
    int slots;
    q4e_cache_stats(e, &hits, &misses, &slots);
    if (slots) fprintf(stderr, "expert cache: %d slots, %llu hits, %llu misses\n", slots, (unsigned long long) hits, (unsigned long long) misses);
    q4e_close(e);
    tok_close(tok);
    free(text); free(ids); free(logits);
    return rc ? 1 : 0;
}
