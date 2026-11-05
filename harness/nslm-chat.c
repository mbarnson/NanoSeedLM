// harness/nslm-chat.c - one chat turn with K2-Horizon-MoVA or Qwen3.8-Flash-Next, streamed to stdout (usage() lists options).
// temp > 0 is the MLX sampler: p ~ exp((l - l_max) / T) in f64, u = (splitmix64 seeded with --seed >> 11) 2^-53 sum, the
// first index whose cumulative sum exceeds u. temp 0: the arg max, ties to the lowest id. With Q4E_DRAFTS = K (1 to 3,
// Qwen3.8-Flash-Next's MTP drafts) temp 0 decodes the tokens after the first through eng_generate, K + 1 at a call: the
// same tokens, in forwards that verify up to K drafts each (eng_generate drafts across its calls, so short calls cost
// nothing). A call's tokens after a stop (at most K) are not shown; the engine keeps them, since the turn ends there (on
// Qwen3.8-Flash-Next an eng_rewind would compute the whole context again).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat_model.h"
#include "engine_api.h"
#include "json.h"
#include "platform.h"
#include "tokenizer.h"

static uint64_t splitmix(uint64_t* s) {
    *s += 0x9E3779B97F4A7C15ull;
    uint64_t z = *s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
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
    double* cum = (double*) malloc(sizeof(double) * (size_t) V);
    uint64_t rng = seed;
    const char* dk = getenv("Q4E_DRAFTS");
    const int drafts = temp > 0 || !dk || atoi(dk) < 1 ? 0 : atoi(dk) < 3 ? atoi(dk) : 3, seq0 = 0;
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
        int32_t got[4];   // this round's tokens: one by eng_step, or a chunk of eng_generate's (at most 3 + 1)
        int ng = 1, stepped = !drafts || !ngen;
        if (stepped) {
            if (eng_step(e, 0, lg)) { fprintf(stderr, "step failed\n"); return 1; }
            int b = 0;
            float x = lg[0];
            for (int v = 1; v < V; ++v) if (lg[v] > x) { x = lg[v]; b = v; }   // the running max: about half the time of lg[b]
            int pick = b;
            if (temp > 0) {
                double c = 0;
                for (int v = 0; v < V; ++v) { c += exp(((double) lg[v] - (double) lg[b]) / temp); cum[v] = c; }
                const double u = (double) (splitmix(&rng) >> 11) * (1.0 / 9007199254740992.0) * c;
                int lo = 0, hi = V;   // first index with cum > u
                while (lo < hi) { const int mid = (lo + hi) / 2; if (cum[mid] > u) hi = mid; else lo = mid + 1; }
                pick = lo < V ? lo : b;
            }
            got[0] = pick;
        } else {
            ng = max_tokens - ngen < drafts + 1 ? max_tokens - ngen : drafts + 1;
            if (eng_generate(e, &seq0, 1, ng, ENG_MODE_DEFAULT, got, &gs)) { fprintf(stderr, "generate failed\n"); return 1; }
        }
        if (ngen == 0) t1 = now_s();
        int keep = 0;
        while (keep < ng && !cm_stop(&cm, got[keep])) ++keep;
        if (stepped && keep) eng_push(e, 0, got[0]);
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
        if (keep < ng) { finish = "stop"; break; }
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
    free(pend); free(lg); free(cum); free(ids);
    eng_close(e);
    tok_close(tok);
    return 0;
}
