// harness/nslm-chat.m - single-turn chat with K2-Horizon-MoVA through the engine, streaming the reply to stdout.
//
//   nslm-chat --model DIR [--res out/res] [--max-tokens 512] [--temp 0] [--seed N]
//             [--system TEXT] [--effort high|medium|low] [--print-ids] "prompt text"
//
// The prompt is MoVA's chat_template.jinja rendered for an optional system message and one user message, with the
// generation prompt (reasoning effort high = <ifm|think>, the template's default).  temp 0 = greedy (arg max, ties to
// the lowest id); temp > 0 = the MLX reference sampler's pick(): p ~ exp((l - l_max) / T) in double, splitmix64 seeded
// with --seed, u = (z >> 11) 2^-53 * sum, the first index whose cumulative sum exceeds u.  Stops at
// <|ifm|endoftext|> (1) or <|ifm|im_end|> (250019).  Timing goes to stderr.
#import <Foundation/Foundation.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "engine_api.h"
#include "tokenizer.h"

#define TOK_EOT 1
#define TOK_IM_END 250019

static uint64_t splitmix(uint64_t* s) {
    *s += 0x9E3779B97F4A7C15ull;
    uint64_t z = *s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static int usage(void) {
    fprintf(stderr, "usage: nslm-chat --model DIR [--res out/res] [--max-tokens 512] [--temp 0]\n"
                    "                 [--seed N] [--system TEXT] [--effort high|medium|low] [--print-ids] \"prompt text\"\n");
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
    @autoreleasepool {
        const char* model = NULL, *res = "out/res", *system = NULL, *effort = "high", *prompt = NULL;
        int max_tokens = 512, print_ids = 0;
        double temp = 0;
        uint64_t seed = 0;
        for (int i = 1; i < argc; ++i) {
            const char* a = argv[i];
            const int has_val = i + 1 < argc;
            if (!strcmp(a, "--print-ids")) print_ids = 1;
            else if (!strcmp(a, "--model") && has_val) model = argv[++i];
            else if (!strcmp(a, "--res") && has_val) res = argv[++i];
            else if (!strcmp(a, "--max-tokens") && has_val) max_tokens = atoi(argv[++i]);
            else if (!strcmp(a, "--temp") && has_val) temp = atof(argv[++i]);
            else if (!strcmp(a, "--seed") && has_val) seed = strtoull(argv[++i], NULL, 10);
            else if (!strcmp(a, "--system") && has_val) system = argv[++i];
            else if (!strcmp(a, "--effort") && has_val) effort = argv[++i];
            else if (a[0] == '-' && a[1] == '-') { fprintf(stderr, "unknown or incomplete option %s\n", a); return usage(); }
            else if (!prompt) prompt = a;
            else return usage();
        }
        if (!model || !prompt || max_tokens < 1) return usage();
        const char* think = !strcmp(effort, "high") ? "<ifm|think>" : !strcmp(effort, "medium") ? "<ifm|think_fast>"
                          : !strcmp(effort, "low") ? "<ifm|think_faster>" : NULL;
        if (!think) { fprintf(stderr, "--effort must be high, medium or low\n"); return 2; }

        NSMutableString* text = [NSMutableString stringWithString:@"<|ifm|begin_of_text|>"];
        if (system) [text appendFormat:@"<|ifm|im_start|>system\n%@<|ifm|im_end|>", @(system)];
        [text appendFormat:@"<|ifm|im_start|>user\n%@<|ifm|im_end|><|ifm|im_start|>assistant\n%s\n", @(prompt), think];

        char err[512] = "";
        char tpath[2048];
        snprintf(tpath, sizeof tpath, "%s/tokenizer.json", model);
        Tok* tok = tok_open(tpath, err, sizeof err);
        if (!tok) { fprintf(stderr, "tokenizer: %s\n", err); return 1; }
        int cap = 4096, n;
        int32_t* ids = NULL;
        for (;;) {
            ids = (int32_t*) realloc(ids, sizeof(int32_t) * (size_t) cap);
            n = tok_encode(tok, text.UTF8String, 0, ids, cap);
            if (n < 0) { fprintf(stderr, "tokenizer: cannot encode the prompt\n"); return 1; }
            if (n <= cap) break;
            cap = n;
        }
        if (print_ids) {
            for (int i = 0; i < n; ++i) fprintf(stderr, "%s%d", i ? "," : "prompt ids: ", ids[i]);
            fprintf(stderr, "\n");
        }

        EngOpts o;
        memset(&o, 0, sizeof o);
        o.model_dir = model;
        o.resource_dir = res;
        o.max_seqs = 1;
        o.kv_tokens = n + max_tokens + 64;
        const double tl = now_s();
        Eng* e = eng_open(&o, err, sizeof err);
        if (!e) { fprintf(stderr, "eng_open: %s\n", err); return 1; }
        fprintf(stderr, "%s\nloaded in %.1f s\n", eng_describe(e), now_s() - tl);
        const int V = eng_vocab(e);
        float* lg = (float*) malloc(sizeof(float) * (size_t) V);
        double* cum = (double*) malloc(sizeof(double) * (size_t) V);
        uint64_t rng = seed;

        unsigned char* pend = NULL;   // decoded bytes not yet written (an incomplete UTF-8 tail)
        size_t npend = 0;
        int ngen = 0;
        const char* finish = "length";
        const double t0 = now_s();
        double t1 = t0;
        if (eng_prefill(e, 0, ids, n)) { fprintf(stderr, "prefill failed\n"); return 1; }
        for (int s = 0; s < max_tokens; ++s) {
            if (eng_step(e, 0, lg)) { fprintf(stderr, "step failed\n"); return 1; }
            int b = 0;
            for (int v = 1; v < V; ++v) if (lg[v] > lg[b]) b = v;
            int pick = b;
            if (temp > 0) {
                double c = 0;
                for (int v = 0; v < V; ++v) { c += exp(((double) lg[v] - (double) lg[b]) / temp); cum[v] = c; }
                const double u = (double) (splitmix(&rng) >> 11) * (1.0 / 9007199254740992.0) * c;
                int lo = 0, hi = V;   // first index with cum > u
                while (lo < hi) { const int mid = (lo + hi) / 2; if (cum[mid] > u) hi = mid; else lo = mid + 1; }
                pick = lo < V ? lo : b;
            }
            if (s == 0) t1 = now_s();
            if (pick == TOK_EOT || pick == TOK_IM_END) { finish = "stop"; break; }
            eng_push(e, 0, pick);
            ++ngen;
            char* piece = tok_decode(tok, &pick, 1, NULL);
            const size_t k = strlen(piece);
            pend = (unsigned char*) realloc(pend, npend + k + 1);
            memcpy(pend + npend, piece, k);
            npend += k;
            free(piece);
            const size_t ok = utf8_complete(pend, npend);
            fwrite(pend, 1, ok, stdout);
            fflush(stdout);
            memmove(pend, pend + ok, npend - ok);
            npend -= ok;
        }
        const double t2 = now_s();
        if (npend) fwrite(pend, 1, npend, stdout);
        printf("\n");
        fflush(stdout);
        fprintf(stderr, "[%s] prompt %d tokens, TTFT %.2f s, prefill %.1f tok/s; generated %d tokens, decode %.1f tok/s\n",
                finish, n, t1 - t0, n / (t1 - t0), ngen, ngen > 1 ? (ngen - 1) / (t2 - t1) : 0.0);
        free(pend); free(lg); free(cum); free(ids);
        eng_close(e);
        tok_close(tok);
        return 0;
    }
}
