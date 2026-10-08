// tests/test_tokenizer.c - harness/tokenizer.c against the Hugging Face tokenizer: the ids of every case in
// tests/data/tokenizer_golden.json (tools/tokenizer_golden.py), and decoding the ids back to the (NFC) text.
//
//   test_tokenizer [MODEL_DIR]      (or MOVA_DIR; skipped without a tokenizer.json)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "platform.h"
#include "tokenizer.h"

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : getenv("MOVA_DIR");
    char path[2048], err[512] = "";
    if (!dir) { printf("SKIP: no model folder (MOVA_DIR)\n"); return 77; }
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);
    Tok* t = tok_open(path, err, sizeof err);
    if (!t) { printf("SKIP: %s\n", err); return 77; }
    size_t len = 0;
    char* js = plat_slurp("tests/data/tokenizer_golden.json", &len);
    Json* g = js ? json_parse(js, len, err, sizeof err) : NULL;
    if (!g || g->t != J_ARR) { printf("FAIL: tests/data/tokenizer_golden.json: %s\n", err); return 1; }
    int fails = 0, nids = 0;
    int32_t* ids = (int32_t*) malloc(sizeof(int32_t) * 65536);
    for (int c = 0; c < g->n; ++c) {
        const char* text = json_gets(g->v[c], "text");
        const Json* want = json_get(g->v[c], "ids");
        const int n = tok_encode(t, text, 0, ids, 65536);
        int same = n == want->n;
        for (int i = 0; same && i < n; ++i) same = ids[i] == (int32_t) want->v[i]->i;
        nids += want->n;
        if (!same) {
            ++fails;
            printf("FAIL case %d (%d ids, want %d): \"%.60s\"\n  got ", c, n, want->n, text);
            for (int i = 0; i < n && i < 24; ++i) printf("%d ", ids[i]);
            printf("\n  want ");
            for (int i = 0; i < want->n && i < 24; ++i) printf("%lld ", (long long) want->v[i]->i);
            printf("\n");
        }
        // decoding gives the text back, except where NFC changed it (the decomposed-accent case)
        int valid = 0;
        char* back = tok_decode(t, ids, n < 0 ? 0 : n, &valid);
        if (!strstr(text, "decomposed") && strcmp(back, text)) { ++fails; printf("FAIL case %d: decode differs: \"%.60s\"\n", c, back); }
        free(back);
    }
    printf("tokenizer: %d cases, %d ids: %s\n", g->n, nids, fails ? "FAIL" : "PASS");
    json_free(g);
    free(js);
    free(ids);
    tok_close(t);
    return fails != 0;
}
