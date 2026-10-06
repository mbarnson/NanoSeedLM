// harness/nslm-mova-gen.m - greedy and/or sampled generations of a prompt suite through the MoVA engine.
//
//   nslm-mova-gen --model DIR [--res out/res] --prompts PRE --out DIR [--only id,id]
//                 [--modes greedy,sampled]
//
// PRE.ids (int32) / PRE.index from tools/mova_export.py prompts (MoVA's template, BOS + the suite's chat text); index
// line: "id n_tokens max_tokens [temperature seed]".
// Greedy: arg max of the BF16-valued logits, ties to the lowest id (MLX's argmax).  Sampled: exactly the MLX reference
// sampler's pick() - p ~ exp((l - l_max) / T) in double, splitmix64 seeded with the prompt's seed,
// u = (z >> 11) * 2^-53 * sum, the first index whose cumulative sum exceeds u.  Both stop at <|ifm|endoftext|> (1),
// <|ifm|im_end|> (250019) or max_tokens.  Writes OUT/<id>.<mode>.json: tokens, the top-2 logit margin of every step,
// finish reason, timing.  Existing outputs are skipped (resumable).
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "engine_api.h"

static const char* opt(int argc, char** argv, const char* name, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static uint64_t splitmix(uint64_t* s) {
    *s += 0x9E3779B97F4A7C15ull;
    uint64_t z = *s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char** argv) {
    @autoreleasepool {
        setvbuf(stdout, NULL, _IOLBF, 0);
        const char* pre = opt(argc, argv, "--prompts", NULL), *outd = opt(argc, argv, "--out", NULL);
        EngOpts o;
        memset(&o, 0, sizeof o);
        o.model_dir = opt(argc, argv, "--model", NULL);
        o.resource_dir = opt(argc, argv, "--res", "out/res");
        o.max_seqs = 1;
        o.kv_tokens = 20480;
        if (!o.model_dir || !pre || !outd) { fprintf(stderr, "usage: nslm-mova-gen --model DIR --prompts PRE --out DIR\n"); return 2; }
        [[NSFileManager defaultManager] createDirectoryAtPath:@(outd) withIntermediateDirectories:YES attributes:nil error:nil];
        NSData* idsd = [NSData dataWithContentsOfFile:[NSString stringWithFormat:@"%s.ids", pre]];
        NSString* idx = [NSString stringWithContentsOfFile:[NSString stringWithFormat:@"%s.index", pre] encoding:NSUTF8StringEncoding error:nil];
        if (!idsd || !idx) { fprintf(stderr, "missing %s.ids / .index\n", pre); return 2; }
        const int32_t* all = (const int32_t*) idsd.bytes;
        NSArray* only = opt(argc, argv, "--only", NULL) ? [@(opt(argc, argv, "--only", "")) componentsSeparatedByString:@","] : nil;
        char err[512] = "";
        Eng* e = eng_open(&o, err, sizeof err);
        if (!e) { fprintf(stderr, "eng_open: %s\n", err); return 1; }
        const int V = eng_vocab(e);
        float* lg = (float*) malloc((size_t) V * 4);
        double* cum = (double*) malloc((size_t) V * 8);
        printf("%s\n", eng_describe(e));
        size_t off = 0;
        for (NSString* line in [idx componentsSeparatedByString:@"\n"]) {
            NSArray* f = [line componentsSeparatedByString:@" "];
            if (f.count < 3) continue;
            NSString* pid = f[0];
            const int n = [f[1] intValue], maxt = [f[2] intValue];
            const int32_t* ids = all + off;
            off += (size_t) n;
            if (only && ![only containsObject:pid]) continue;
            const double temp = f.count >= 5 ? [f[3] doubleValue] : 0.7;
            const uint64_t seed0 = f.count >= 5 ? (uint64_t) [f[4] longLongValue] : 0;
            for (NSString* mode in [@(opt(argc, argv, "--modes", "greedy")) componentsSeparatedByString:@","]) {
            const int greedy = [mode isEqualToString:@"greedy"];
            NSString* path = [NSString stringWithFormat:@"%s/%@.%@.json", outd, pid, mode];
            if ([[NSFileManager defaultManager] fileExistsAtPath:path]) continue;
            uint64_t rng = seed0;
            const double t0 = now_s();
            if (eng_prefill(e, 0, ids, n)) { fprintf(stderr, "%s: prefill failed\n", pid.UTF8String); return 1; }
            const double t1 = now_s();
            NSMutableArray* toks = [NSMutableArray array], *margins = [NSMutableArray array];
            NSString* finish = @"length";
            for (int s = 0; s < maxt; ++s) {
                if (eng_step(e, 0, lg)) { fprintf(stderr, "%s: step failed\n", pid.UTF8String); return 1; }
                int b = 0, b2 = -1;
                for (int v = 1; v < V; ++v) {
                    if (lg[v] > lg[b]) { b2 = b; b = v; }
                    else if (b2 < 0 || lg[v] > lg[b2]) b2 = v;
                }
                int pick = b;
                if (!greedy) {
                    double c = 0;
                    for (int v = 0; v < V; ++v) { c += exp(((double) lg[v] - (double) lg[b]) / temp); cum[v] = c; }
                    const double u = (double) (splitmix(&rng) >> 11) * (1.0 / 9007199254740992.0) * c;
                    int lo = 0, hi = V;   // first index with cum > u (numpy searchsorted, side right)
                    while (lo < hi) { const int mid = (lo + hi) / 2; if (cum[mid] > u) hi = mid; else lo = mid + 1; }
                    pick = lo < V ? lo : b;
                }
                if (pick == 1 || pick == 250019) { finish = @"stop"; break; }
                [toks addObject:@(pick)];
                [margins addObject:@((double) (lg[b] - lg[b2]))];
                eng_push(e, 0, pick);
            }
            const double t2 = now_s();
            NSDictionary* rec = @{@"id": pid, @"mode": mode, @"engine": @(eng_describe(e)), @"prompt_tokens": @(n),
                                  @"completion_tokens": @(toks.count), @"finish_reason": finish, @"tokens": toks, @"margins": margins,
                                  @"prefill_s": @(t1 - t0), @"decode_s": @(t2 - t1), @"decode_tok_s": @(toks.count / (t2 - t1)),
                                  @"temperature": greedy ? @0 : @(temp), @"seed": greedy ? [NSNull null] : @(seed0)};
            [[NSJSONSerialization dataWithJSONObject:rec options:0 error:nil] writeToFile:path atomically:YES];
            printf("%-8s %-7s prompt %5d  gen %5lu  %-6s %6.1fs %5.1f tok/s\n", pid.UTF8String, mode.UTF8String, n, (unsigned long) toks.count,
                   finish.UTF8String, t2 - t1, toks.count / (t2 - t1));
            }
        }
        eng_close(e);
        return 0;
    }
}
