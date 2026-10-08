// harness/nslm-mova-gen.c - greedy and/or sampled generations of a prompt suite through the MoVA engine.
//
//   nslm-mova-gen --model DIR [--res out/res] --prompts PRE --out DIR [--only id,id]
//                 [--modes greedy,sampled]
//
// PRE.ids (int32) / PRE.index from tools/mova_export.py prompts (harness/suite.h).
// Greedy: arg max of the BF16-valued logits, ties to the lowest id (MLX's argmax).  Sampled: exactly the MLX reference
// sampler's pick() - p ~ exp((l - l_max) / T) in double, splitmix64 seeded with the prompt's seed,
// u = (z >> 11) * 2^-53 * sum, the first index whose cumulative sum exceeds u.  Both stop at <|ifm|endoftext|> (1),
// <|ifm|im_end|> (250019) or max_tokens.  Writes OUT/<id>.<mode>.json: tokens, the top-2 logit margin of every step,
// finish reason, timing.  Existing outputs are skipped (resumable).
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "engine_api.h"
#include "json.h"
#include "platform.h"
#include "suite.h"

static uint64_t splitmix(uint64_t* s) {
    *s += 0x9E3779B97F4A7C15ull;
    uint64_t z = *s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

int main(int argc, char** argv) {
    plat_init(&argc, &argv);
    const char* pre = opt(argc, argv, "--prompts", NULL), *outd = opt(argc, argv, "--out", NULL);
    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = opt(argc, argv, "--model", NULL);
    o.resource_dir = opt(argc, argv, "--res", "out/res");
    o.max_seqs = 1;
    o.kv_tokens = 20480;
    if (!o.model_dir || !pre || !outd) { fprintf(stderr, "usage: nslm-mova-gen --model DIR --prompts PRE --out DIR\n"); return 2; }
    mkdir(outd, 0755);
    Suite su;
    if (suite_load(&su, pre)) { fprintf(stderr, "missing %s.ids / .index\n", pre); return 2; }
    const char* only = opt(argc, argv, "--only", NULL);
    char err[512] = "";
    Eng* e = eng_open(&o, err, sizeof err);
    if (!e) { fprintf(stderr, "eng_open: %s\n", err); return 1; }
    const int V = eng_vocab(e);
    float* lg = (float*) malloc((size_t) V * 4);
    double* cum = (double*) malloc((size_t) V * 8);
    printf("%s\n", eng_describe(e));
    char* modes = strdup(opt(argc, argv, "--modes", "greedy"));
    for (int pi = 0; pi < su.n; ++pi) {
        const SuitePrompt* q = &su.p[pi];
        if (!suite_selected(only, q->id)) continue;
        char mbuf[256];
        snprintf(mbuf, sizeof mbuf, "%s", modes);
        for (char* mode = strtok(mbuf, ","); mode; mode = strtok(NULL, ",")) {
            const int greedy = !strcmp(mode, "greedy");
            char path[2048];
            snprintf(path, sizeof path, "%s/%s.%s.json", outd, q->id, mode);
            FILE* ex = fopen(path, "rb");
            if (ex) { fclose(ex); continue; }
            uint64_t rng = q->seed;
            const double t0 = now_s();
            if (eng_prefill(e, 0, q->ids, q->n)) { fprintf(stderr, "%s: prefill failed\n", q->id); return 1; }
            const double t1 = now_s();
            Json* toks = json_new(J_ARR), *margins = json_new(J_ARR);
            const char* finish = "length";
            for (int s = 0; s < q->max_tokens; ++s) {
                if (eng_step(e, 0, lg)) { fprintf(stderr, "%s: step failed\n", q->id); return 1; }
                int b = 0, b2 = -1;
                for (int v = 1; v < V; ++v) {
                    if (lg[v] > lg[b]) { b2 = b; b = v; }
                    else if (b2 < 0 || lg[v] > lg[b2]) b2 = v;
                }
                int pick = b;
                if (!greedy) {
                    double c = 0;
                    for (int v = 0; v < V; ++v) { c += exp(((double) lg[v] - (double) lg[b]) / q->temperature); cum[v] = c; }
                    const double u = (double) (splitmix(&rng) >> 11) * (1.0 / 9007199254740992.0) * c;
                    int lo = 0, hi = V;   // first index with cum > u (numpy searchsorted, side right)
                    while (lo < hi) { const int mid = (lo + hi) / 2; if (cum[mid] > u) hi = mid; else lo = mid + 1; }
                    pick = lo < V ? lo : b;
                }
                if (pick == 1 || pick == 250019) { finish = "stop"; break; }
                Json* t = json_new(J_INT);
                t->i = pick;
                json_push(toks, t);
                Json* mg = json_new(J_NUM);
                mg->d = (double) (lg[b] - lg[b2]);
                json_push(margins, mg);
                eng_push(e, 0, pick);
            }
            const double t2 = now_s();
            const int ng = toks->n;
            Json* rec = json_new(J_OBJ);
            json_set(rec, "id", json_str(q->id));
            json_set(rec, "mode", json_str(mode));
            json_set(rec, "engine", json_str(eng_describe(e)));
            Json* x = json_new(J_INT); x->i = q->n; json_set(rec, "prompt_tokens", x);
            x = json_new(J_INT); x->i = ng; json_set(rec, "completion_tokens", x);
            json_set(rec, "finish_reason", json_str(finish));
            json_set(rec, "tokens", toks);
            json_set(rec, "margins", margins);
            x = json_new(J_NUM); x->d = t1 - t0; json_set(rec, "prefill_s", x);
            x = json_new(J_NUM); x->d = t2 - t1; json_set(rec, "decode_s", x);
            x = json_new(J_NUM); x->d = ng / (t2 - t1); json_set(rec, "decode_tok_s", x);
            if (greedy) { x = json_new(J_INT); x->i = 0; json_set(rec, "temperature", x); json_set(rec, "seed", json_new(J_NULL)); }
            else {
                x = json_new(J_NUM); x->d = q->temperature; json_set(rec, "temperature", x);
                x = json_new(J_INT); x->i = (int64_t) q->seed; json_set(rec, "seed", x);
            }
            char* js = json_dumps(rec, 1);
            json_free(rec);
            char tmp[2100];
            snprintf(tmp, sizeof tmp, "%s.tmp", path);
            FILE* f = fopen(tmp, "wb");
            int ok = f && fputs(js, f) >= 0;
            if (f && fclose(f)) ok = 0;
            if (!ok || rename(tmp, path)) { fprintf(stderr, "cannot write %s: %s\n", path, strerror(errno)); remove(tmp); }
            free(js);
            printf("%-8s %-7s prompt %5d  gen %5d  %-6s %6.1fs %5.1f tok/s\n", q->id, mode, q->n, ng, finish, t2 - t1, ng / (t2 - t1));
            fflush(stdout);
        }
    }
    free(modes); free(lg); free(cum);
    suite_free(&su);
    eng_close(e);
    return 0;
}
