// harness/nslm-chat.c - one chat turn with K2-Horizon-MoVA or Qwen3.8-Flash-Next, streamed to stdout (usage() lists options).
// Tokens come from harness/sample.h's MLX sampler (temp 0: the arg max). With Q4E_DRAFTS = K (1 to 3), Qwen3.8-Flash-Next
// decodes K + 1 tokens a call of eng_generate_pick, whose forwards verify up to K MTP drafts: the sampler's distribution.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat_model.h"
#include "sample.h"
#include "engine_api.h"
#include "json.h"
#include "platform.h"
#include "qwen4_ext.h"
#include "rng.h"
#include "tokenizer.h"

typedef struct { Sampler s; const ChatModel* cm; } Pick;   // a stop token is a stop (-1)
static int32_t chat_pick(void* ctx, const float* l, int V, int32_t draft, const EngDraftQ* q) {
    Pick* p = (Pick*) ctx;
    const int32_t t = sample_pick(&p->s, l, V, draft, q);
    return cm_stop(p->cm, t) ? -1 : t;
}

static int usage(void) {
    fprintf(stderr, "usage: nslm-chat --model DIR [--res out/res] [--max-tokens 512] [--temp 0] [--seed N] [--system TEXT]\n"
                    "                 [--effort high|medium|low (K2), xhigh|medium|low (Qwen)] [--no-think (Qwen)]\n"
                    "                 [--kv bf16|q8|fp8|fp4] [--ctx N] [--print-ids]\n"
                    "                 (\"prompt text\" | --prompt-file FILE)\n");
    return 2;
}

// Length of the longest prefix of b[0..n) that does not end inside a UTF-8 sequence.
static size_t utf8_complete(const unsigned char* b, size_t n) {
    for (size_t back = 1; back <= 4 && back <= n; ++back) {
        const unsigned char c = b[n - back];
        if ((c & 0xC0) == 0x80) continue;   // continuation byte
        const size_t need = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        return need > back ? n - back : n;
    }
    return n;
}

int main(int argc, char** argv) {
    plat_init(&argc, &argv);
    const char *model = NULL, *res = "out/res", *system = NULL, *effort = NULL, *prompt = NULL;
    int max_tokens = 512, print_ids = 0, kvf = 0, ctx = 0, no_think = 0;
    double temp = 0;
    uint64_t seed = 0;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        const int has_val = i + 1 < argc;
        if (!strcmp(a, "--print-ids")) print_ids = 1;
        else if (!strcmp(a, "--no-think")) no_think = 1;
        else if (!strcmp(a, "--model") && has_val) model = argv[++i];
        else if (!strcmp(a, "--res") && has_val) res = argv[++i];
        else if (!strcmp(a, "--max-tokens") && has_val) max_tokens = atoi(argv[++i]);
        else if (!strcmp(a, "--temp") && has_val) temp = atof(argv[++i]);
        else if (!strcmp(a, "--seed") && has_val) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--system") && has_val) system = argv[++i];
        else if (!strcmp(a, "--effort") && has_val) effort = argv[++i];
        else if (!strcmp(a, "--kv") && has_val) {
            if ((kvf = eng_kv_parse(argv[++i])) < 0) { fprintf(stderr, "nslm-chat: --kv: bf16, q8, fp8 or fp4\n"); return 2; }
        }
        else if (!strcmp(a, "--ctx") && has_val) ctx = atoi(argv[++i]);
        else if (!strcmp(a, "--prompt-file") && has_val) {
            prompt = plat_slurp(argv[++i], NULL);
            if (!prompt) { fprintf(stderr, "cannot read %s\n", argv[i]); return 2; }
        }
        else if (a[0] == '-' && a[1] == '-') { fprintf(stderr, "unknown or incomplete option %s\n", a); return usage(); }
        else if (!prompt) prompt = a;
        else return usage();
    }
    if (!model || !prompt || max_tokens < 1) return usage();

    char err[512] = "";
    char tpath[2048];
    snprintf(tpath, sizeof tpath, "%s/tokenizer.json", model);
    Tok* tok = tok_open(tpath, err, sizeof err);
    if (!tok) { fprintf(stderr, "tokenizer: %s\n", err); return 1; }
    ChatModel cm;
    if (cm_open(&cm, model, tok, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    Json* msgs = json_new(J_ARR);
    for (int r = !system; r < 2; ++r) {
        Json* m = json_new(J_OBJ);
        json_set(m, "role", json_str(r ? "user" : "system"));
        json_set(m, "content", json_str(r ? prompt : system));
        json_push(msgs, m);
    }
    const CtOpts co = {NULL, NULL, cm_effort(&cm, effort), 1, no_think, 0};
    char* text = cm_render(&cm, msgs, NULL, &co, err, sizeof err);
    json_free(msgs);
    if (!text) { fprintf(stderr, "template: %s\n", err); return 2; }
    int cap = 4096, n;
    int32_t* ids = NULL;
    for (;;) {
        ids = (int32_t*) realloc(ids, sizeof(int32_t) * (size_t) cap);
        n = tok_encode(tok, text, 0, ids, cap);
        if (n < 0) { fprintf(stderr, "tokenizer: cannot encode the prompt\n"); return 1; }
        if (n <= cap) break;
        cap = n;
    }
    free(text);
    if (print_ids) {
        for (int i = 0; i < n; ++i) fprintf(stderr, "%s%d", i ? "," : "prompt ids: ", ids[i]);
        fprintf(stderr, "\n");
    }

    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = model;
    o.resource_dir = res;
    o.max_seqs = 1;
    o.kv_tokens = ctx > n + max_tokens ? ctx : n + max_tokens + 64;
    o.kv_format = kvf;
    const double tl = now_s();
    Eng* e = eng_open(&o, err, sizeof err);
    if (!e) { fprintf(stderr, "eng_open: %s\n", err); return 1; }
    fprintf(stderr, "%s\nloaded in %.1f s\n", eng_describe(e), now_s() - tl);
    const int V = eng_vocab(e);
    float* lg = (float*) malloc(sizeof(float) * (size_t) V);
    Pick pk = {{temp, 1, 0, 0, 1, seed}, &cm};   // the MLX sampler
    const char* dk = getenv("Q4E_DRAFTS");
    int32_t got[ENG_MAX_DRAFTS + 1];   // a call's tokens
    const int drafts = !dk || atoi(dk) < 1 || eng_generate_pick(e, 0, 0, chat_pick, &pk, got, NULL) ? 0 :
                       atoi(dk) < ENG_MAX_DRAFTS ? atoi(dk) : ENG_MAX_DRAFTS;
    if (drafts) eng_draft_sampling(e, (float) temp, ns_mix64(seed));   // drafts sampled at temp, from their own stream
    EngStats gs;
    memset(&gs, 0, sizeof gs);

    unsigned char* pend = NULL;   // decoded bytes not yet written (an incomplete UTF-8 tail)
    size_t npend = 0;
    int ngen = 0;
    const char* finish = "length";
    const double t0 = now_s();
    double t1 = t0;
    if (eng_prefill(e, 0, ids, n)) { fprintf(stderr, "prefill failed\n"); return 1; }
    while (ngen < max_tokens) {
        const int left = max_tokens - ngen, want = !drafts || !ngen ? 1 : left < drafts + 1 ? left : drafts + 1;   // TTFT: 1
        int keep;
        if (drafts) keep = eng_generate_pick(e, 0, want, chat_pick, &pk, got, &gs);
        else if (eng_step(e, 0, lg)) keep = -1;
        else keep = (got[0] = chat_pick(&pk, lg, V, -1, NULL)) < 0 ? 0 : eng_push(e, 0, got[0]) ? -1 : 1;
        if (keep < 0) { fprintf(stderr, "decode failed\n"); return 1; }
        if (ngen == 0) t1 = now_s();
        for (int i = 0; i < keep; ++i) {
            char* piece = tok_decode(tok, got + i, 1, NULL);
            const size_t k = strlen(piece);
            pend = (unsigned char*) realloc(pend, npend + k + 1);
            memcpy(pend + npend, piece, k);
            npend += k;
            free(piece);
        }
        ngen += keep;
        const size_t ok = utf8_complete(pend, npend);
        fwrite(pend, 1, ok, stdout);
        fflush(stdout);
        memmove(pend, pend + ok, npend - ok);
        npend -= ok;
        if (keep < want) { finish = "stop"; break; }
    }
    const double t2 = now_s();
    if (npend) fwrite(pend, 1, npend, stdout);
    printf("\n");
    fflush(stdout);
    fprintf(stderr, "[%s] prompt %d tokens, TTFT %.2f s, prefill %.1f tok/s; generated %d tokens, decode %.1f tok/s\n",
            finish, n, t1 - t0, n / (t1 - t0), ngen, ngen > 1 ? (ngen - 1) / (t2 - t1) : 0.0);
    if (drafts)
        fprintf(stderr, "%d MTP draft%s a forward: %lld forwards, %.2f tokens a forward, %lld of %lld drafts accepted\n", drafts, drafts > 1 ? "s" : "",
                (long long) gs.forwards, gs.forwards ? (double) gs.tokens / (double) gs.forwards : 0.0, (long long) gs.accepted, (long long) gs.proposals);
    free(pend); free(lg); free(ids);
    eng_close(e);
    tok_close(tok);
    return 0;
}
