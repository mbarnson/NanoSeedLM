// harness/tokenizer.c - byte-level BPE from an HF tokenizer.json (see tokenizer.h), in C on ICU's C API.
// Pipeline (as tokenizer.m was): added tokens are split out first (leftmost, longest match, never normalized); each
// remaining segment is NFC-normalized, split by the pre-tokenizer regex (ICU, the same engine and pattern string as
// NSRegularExpression), mapped byte-to-unicode (GPT-2 byte level), and merged by BPE rank.  No BOS is added by the
// tokenizer itself.
//
// ICU: Windows 10 1903+ ships it (icu.h, icu.lib); macOS ships the same C ABI in libicucore without headers, so the
// few functions used here are declared below for it.
#include "tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>

#include "json.h"

#if defined(_WIN32)
#include <icu.h>
#elif !defined(__APPLE__)   // Linux: ICU's headers map the names to the library's versioned symbols (u_strFromUTF8_74)
#include <unicode/unorm2.h>
#include <unicode/uregex.h>
#include <unicode/ustring.h>
#else
typedef uint16_t UChar;
typedef int UErrorCode;
typedef struct URegularExpression URegularExpression;
typedef struct UNormalizer2 UNormalizer2;
typedef struct { int32_t line, offset; UChar preContext[16], postContext[16]; } UParseError;
#define U_ZERO_ERROR 0
#define U_FAILURE(x) ((x) > U_ZERO_ERROR)
#define U_BUFFER_OVERFLOW_ERROR 15
UChar* u_strFromUTF8(UChar* dest, int32_t cap, int32_t* len, const char* src, int32_t srclen, UErrorCode* e);
char* u_strToUTF8(char* dest, int32_t cap, int32_t* len, const UChar* src, int32_t srclen, UErrorCode* e);
const UNormalizer2* unorm2_getNFCInstance(UErrorCode* e);
int32_t unorm2_normalize(const UNormalizer2* n, const UChar* src, int32_t len, UChar* dest, int32_t cap, UErrorCode* e);
URegularExpression* uregex_open(const UChar* pat, int32_t len, uint32_t flags, UParseError* pe, UErrorCode* e);
void uregex_close(URegularExpression* r);
URegularExpression* uregex_clone(const URegularExpression* r, UErrorCode* e);
void uregex_setStackLimit(URegularExpression* r, int32_t limit, UErrorCode* e);
void uregex_setText(URegularExpression* r, const UChar* text, int32_t len, UErrorCode* e);
int8_t uregex_findNext(URegularExpression* r, UErrorCode* e);   // UBool
int32_t uregex_start(URegularExpression* r, int32_t group, UErrorCode* e);
int32_t uregex_end(URegularExpression* r, int32_t group, UErrorCode* e);
#endif

// ---- string -> int hash table (open addressing, FNV-1a) -------------------------------------------------------------

typedef struct {
    char** key;
    int32_t* val;
    uint32_t cap, n;
} Map;
static uint32_t fnv(const char* s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) h = (h ^ (uint8_t) s[i]) * 16777619u;
    return h;
}
static void map_init(Map* m, uint32_t want) {
    m->cap = 1024;
    while (m->cap < want * 2) m->cap *= 2;
    m->key = (char**) calloc(m->cap, sizeof(char*));
    m->val = (int32_t*) calloc(m->cap, sizeof(int32_t));
    m->n = 0;
}
static void map_free(Map* m) {
    if (!m->key) return;
    for (uint32_t i = 0; i < m->cap; ++i) free(m->key[i]);
    free(m->key);
    free(m->val);
    memset(m, 0, sizeof *m);
}
static int32_t* map_slot(Map* m, const char* k, size_t n, int insert) {
    uint32_t i = fnv(k, n) & (m->cap - 1);
    for (;; i = (i + 1) & (m->cap - 1)) {
        if (!m->key[i]) break;
        if (strlen(m->key[i]) == n && !memcmp(m->key[i], k, n)) return &m->val[i];
    }
    if (!insert) return NULL;
    if (2 * (m->n + 1) > m->cap) {   // grow
        Map g;
        map_init(&g, m->cap);
        for (uint32_t j = 0; j < m->cap; ++j)
            if (m->key[j]) {
                uint32_t q = fnv(m->key[j], strlen(m->key[j])) & (g.cap - 1);
                while (g.key[q]) q = (q + 1) & (g.cap - 1);
                g.key[q] = m->key[j];
                g.val[q] = m->val[j];
            }
        g.n = m->n;
        free(m->key);
        free(m->val);
        *m = g;
        return map_slot(m, k, n, 1);
    }
    m->key[i] = (char*) malloc(n + 1);
    memcpy(m->key[i], k, n);
    m->key[i][n] = 0;
    m->n++;
    return &m->val[i];
}
static int map_get(const Map* m, const char* k, size_t n, int32_t* v) {
    int32_t* p = map_slot((Map*) m, k, n, 0);
    if (!p) return 0;
    *v = *p;
    return 1;
}

// (id a, id b) -> merge rank and merged id
typedef struct {
    uint64_t* key;   // (a + 1) << 32 | (b + 1); 0 = empty
    int32_t* rank;
    int32_t* id;
    uint32_t cap;
} PairMap;
static uint32_t pm_hash(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdull; k ^= k >> 33; return (uint32_t) k; }
static void pm_put(PairMap* m, uint64_t k, int32_t rank, int32_t id) {
    uint32_t i = pm_hash(k) & (m->cap - 1);
    while (m->key[i] && m->key[i] != k) i = (i + 1) & (m->cap - 1);
    if (m->key[i]) return;   // a repeated merge keeps its first (lowest) rank
    m->key[i] = k;
    m->rank[i] = rank;
    m->id[i] = id;
}
static int pm_get(const PairMap* m, uint64_t k, int32_t* rank, int32_t* id) {
    uint32_t i = pm_hash(k) & (m->cap - 1);
    while (m->key[i]) {
        if (m->key[i] == k) { *rank = m->rank[i]; *id = m->id[i]; return 1; }
        i = (i + 1) & (m->cap - 1);
    }
    return 0;
}

// ---- a minimal JSON scanner for tokenizer.json ----------------------------------------------------------------------

typedef struct {
    const char* p;
    const char* e;
} Sc;
static void ws(Sc* s) { while (s->p < s->e && (*s->p == ' ' || *s->p == '\n' || *s->p == '\r' || *s->p == '\t')) ++s->p; }
static int expect(Sc* s, char c) {
    ws(s);
    if (s->p < s->e && *s->p == c) { ++s->p; return 1; }
    return 0;
}
static void put_utf8(Buf* b, uint32_t cp) {
    char o[4];
    int n;
    if (cp < 0x80) { o[0] = (char) cp; n = 1; }
    else if (cp < 0x800) { o[0] = (char) (0xC0 | (cp >> 6)); o[1] = (char) (0x80 | (cp & 63)); n = 2; }
    else if (cp < 0x10000) { o[0] = (char) (0xE0 | (cp >> 12)); o[1] = (char) (0x80 | ((cp >> 6) & 63)); o[2] = (char) (0x80 | (cp & 63)); n = 3; }
    else { o[0] = (char) (0xF0 | (cp >> 18)); o[1] = (char) (0x80 | ((cp >> 12) & 63)); o[2] = (char) (0x80 | ((cp >> 6) & 63)); o[3] = (char) (0x80 | (cp & 63)); n = 4; }
    buf_put(b, o, (size_t) n);
}
static int hex4(const char* p, uint32_t* v) {
    *v = 0;
    for (int i = 0; i < 4; ++i) {
        const char c = p[i];
        *v <<= 4;
        if (c >= '0' && c <= '9') *v |= (uint32_t) (c - '0');
        else if (c >= 'a' && c <= 'f') *v |= (uint32_t) (c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') *v |= (uint32_t) (c - 'A' + 10);
        else return 0;
    }
    return 1;
}
// A JSON string, decoded to UTF-8 into b (cleared first).
static int str(Sc* s, Buf* b) {
    b->n = 0;
    buf_put(b, "", 0);
    ws(s);
    if (s->p >= s->e || *s->p != '"') return 0;
    ++s->p;
    while (s->p < s->e && *s->p != '"') {
        const char* run = s->p;
        while (s->p < s->e && *s->p != '"' && *s->p != '\\') ++s->p;
        if (s->p > run) buf_put(b, run, (size_t) (s->p - run));
        if (s->p < s->e && *s->p == '\\') {
            if (s->p + 1 >= s->e) return 0;
            const char c = s->p[1];
            s->p += 2;
            switch (c) {
            case 'n': buf_put(b, "\n", 1); break;
            case 't': buf_put(b, "\t", 1); break;
            case 'r': buf_put(b, "\r", 1); break;
            case 'b': buf_put(b, "\b", 1); break;
            case 'f': buf_put(b, "\f", 1); break;
            case 'u': {
                uint32_t v;
                if (s->e - s->p < 4 || !hex4(s->p, &v)) return 0;
                s->p += 4;
                if (v >= 0xD800 && v < 0xDC00 && s->e - s->p >= 6 && s->p[0] == '\\' && s->p[1] == 'u') {
                    uint32_t lo;
                    if (hex4(s->p + 2, &lo) && lo >= 0xDC00 && lo < 0xE000) { v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00); s->p += 6; }
                }
                put_utf8(b, v);
                break;
            }
            default: buf_put(b, &c, 1);
            }
        }
    }
    if (s->p >= s->e) return 0;
    ++s->p;
    return 1;
}
static int skip(Sc* s) {   // any value
    ws(s);
    if (s->p >= s->e) return 0;
    if (*s->p == '"') {
        ++s->p;
        while (s->p < s->e && *s->p != '"') s->p += *s->p == '\\' ? 2 : 1;
        if (s->p >= s->e) return 0;
        ++s->p;
        return 1;
    }
    if (*s->p == '{' || *s->p == '[') {
        int depth = 0;
        for (; s->p < s->e; ++s->p) {
            const char c = *s->p;
            if (c == '"') { if (!skip(s)) return 0; --s->p; continue; }
            if (c == '{' || c == '[') ++depth;
            else if ((c == '}' || c == ']') && --depth == 0) { ++s->p; return 1; }
        }
        return 0;
    }
    while (s->p < s->e && *s->p != ',' && *s->p != '}' && *s->p != ']') ++s->p;
    return 1;
}
static long num(Sc* s) {
    ws(s);
    char* q;
    const long v = strtol(s->p, &q, 10);
    s->p = q;
    return v;
}
// Iterate an object: calls f(key, scanner at the value) for each member; f must consume the value.
typedef int (*MemberFn)(void* ctx, const char* key, Sc* s);
static int object(Sc* s, MemberFn f, void* ctx) {
    if (!expect(s, '{')) return 0;
    Buf k = {0};
    if (expect(s, '}')) return 1;
    for (;;) {
        if (!str(s, &k) || !expect(s, ':')) { free(k.p); return 0; }
        if (!f(ctx, k.p, s)) { free(k.p); return 0; }
        if (expect(s, ',')) continue;
        if (expect(s, '}')) break;
        free(k.p);
        return 0;
    }
    free(k.p);
    return 1;
}

// ---- the tokenizer --------------------------------------------------------------------------------------------------

typedef struct {
    char* s;
    size_t n;
    int32_t id;
} Added;

struct Tok {
    Map vocab;          // byte-level string -> id
    PairMap merges;
    char** id_to_str;   // id -> vocab string or added-token content (UTF-8)
    uint8_t* is_added;
    int32_t nid;
    Added* added;       // longest first
    int nadded;
    URegularExpression* re;
    const UNormalizer2* nfc;
    int32_t byte_id[256];   // id of the single byte-level symbol of each byte
    uint16_t byte2u[256];
    int16_t u2byte[512];
    Map cache;          // word (UTF-8) -> index into cache_ids
    int32_t* cache_ids;   // [len, ids...] records
    size_t cache_n, cache_cap;
    int bos;
    int err_bad;
    pthread_mutex_t mu;   // the BPE cache (a server encodes on many threads; each call matches with its own regex clone)
};

typedef struct {
    Tok* t;
    char* pattern;
    int bad_type;
    Buf tmp;
    long rank;
} Ld;

static void set_id_str(Tok* t, int32_t id, const char* s, int added) {
    if (id < 0) return;
    if (id >= t->nid) {
        const int32_t n = id + 1 > 2 * t->nid ? id + 1 : 2 * t->nid;
        t->id_to_str = (char**) realloc(t->id_to_str, sizeof(char*) * (size_t) n);
        t->is_added = (uint8_t*) realloc(t->is_added, (size_t) n);
        memset(t->id_to_str + t->nid, 0, sizeof(char*) * (size_t) (n - t->nid));
        memset(t->is_added + t->nid, 0, (size_t) (n - t->nid));
        t->nid = n;
    }
    free(t->id_to_str[id]);
    t->id_to_str[id] = strdup(s);
    t->is_added[id] = (uint8_t) added;
}

static int vocab_member(void* ctx, const char* key, Sc* s) {
    Ld* l = (Ld*) ctx;
    const long id = num(s);
    *map_slot(&l->t->vocab, key, strlen(key), 1) = (int32_t) id;
    set_id_str(l->t, (int32_t) id, key, 0);
    return 1;
}
// A merge "a b" (or ["a", "b"]): a, b and a+b must be in the vocabulary (merges are read after it).
static int add_merge(Ld* l, const char* a, size_t na, const char* b, size_t nb) {
    Tok* t = l->t;
    int32_t ia, ib, iab;
    char* ab = (char*) malloc(na + nb + 1);
    memcpy(ab, a, na);
    memcpy(ab + na, b, nb);
    const int ok = map_get(&t->vocab, a, na, &ia) && map_get(&t->vocab, b, nb, &ib) && map_get(&t->vocab, ab, na + nb, &iab);
    free(ab);
    if (ok) pm_put(&t->merges, ((uint64_t) (ia + 1) << 32) | (uint64_t) (ib + 1), (int32_t) l->rank, iab);
    l->rank++;
    return 1;
}
static int merges(Ld* l, Sc* s) {
    if (!expect(s, '[')) return 0;
    if (expect(s, ']')) return 1;
    Buf a = {0}, b = {0};
    int ok = 1;
    for (;;) {
        ws(s);
        if (s->p < s->e && *s->p == '[') {
            ++s->p;
            ok = str(s, &a) && expect(s, ',') && str(s, &b) && expect(s, ']');
            if (ok) add_merge(l, a.p, a.n, b.p, b.n);
        } else {
            ok = str(s, &a);
            const char* sp = ok ? strchr(a.p, ' ') : NULL;
            ok = ok && sp;
            if (ok) add_merge(l, a.p, (size_t) (sp - a.p), sp + 1, a.n - (size_t) (sp - a.p) - 1);
        }
        if (!ok) break;
        if (expect(s, ',')) continue;
        ok = expect(s, ']');
        break;
    }
    free(a.p);
    free(b.p);
    return ok;
}
static int model_member(void* ctx, const char* key, Sc* s) {
    Ld* l = (Ld*) ctx;
    if (!strcmp(key, "type")) {
        if (!str(s, &l->tmp)) return 0;
        if (strcmp(l->tmp.p, "BPE")) l->bad_type = 1;
        return 1;
    }
    if (!strcmp(key, "vocab")) return object(s, vocab_member, l);
    if (!strcmp(key, "merges")) {
        // the vocabulary must be known first; tokenizer.json writes vocab before merges
        l->t->merges.cap = 1u << 20;
        while (l->t->merges.cap < 2 * l->t->vocab.n + 1024) l->t->merges.cap *= 2;
        l->t->merges.key = (uint64_t*) calloc(l->t->merges.cap, 8);
        l->t->merges.rank = (int32_t*) calloc(l->t->merges.cap, 4);
        l->t->merges.id = (int32_t*) calloc(l->t->merges.cap, 4);
        return merges(l, s);
    }
    return skip(s);
}
typedef struct {
    Ld* l;
    long id;
    int has_id;
    Buf content;
} AddedTok;
static int added_member(void* ctx, const char* key, Sc* s) {
    AddedTok* a = (AddedTok*) ctx;
    if (!strcmp(key, "id")) { a->id = num(s); a->has_id = 1; return 1; }
    if (!strcmp(key, "content")) return str(s, &a->content);
    return skip(s);
}
static int added_tokens(Ld* l, Sc* s) {
    if (!expect(s, '[')) return 0;
    if (expect(s, ']')) return 1;
    Tok* t = l->t;
    for (;;) {
        AddedTok a = {l, 0, 0, {0}};
        if (!object(s, added_member, &a) || !a.has_id || !a.content.p) { free(a.content.p); return 0; }
        t->added = (Added*) realloc(t->added, sizeof(Added) * (size_t) (t->nadded + 1));
        t->added[t->nadded].s = a.content.p;
        t->added[t->nadded].n = a.content.n;
        t->added[t->nadded].id = (int32_t) a.id;
        t->nadded++;
        if (expect(s, ',')) continue;
        return expect(s, ']');
    }
}
// The Split regex of the pre-tokenizer (a Sequence, or a single pre-tokenizer); ByteLevel with use_regex is rejected.
static int pre_tok(Ld* l, Sc* s) {
    const char* p0 = s->p;
    if (!skip(s)) return 0;
    char err[128];
    Json* j = json_parse(p0, (size_t) (s->p - p0), err, sizeof err);
    if (!j) return 0;
    const Json* seq = json_get(j, "pretokenizers");
    const int n = seq && seq->t == J_ARR ? seq->n : 1;
    for (int i = 0; i < n; ++i) {
        const Json* p = seq && seq->t == J_ARR ? seq->v[i] : j;
        const char* ty = json_gets(p, "type");
        if (ty && !strcmp(ty, "Split")) {
            const char* re = json_gets(json_get(p, "pattern"), "Regex");
            if (re) { free(l->pattern); l->pattern = strdup(re); }
        }
        if (ty && !strcmp(ty, "ByteLevel") && json_truthy(json_get(p, "use_regex"))) l->t->err_bad = 1;
    }
    json_free(j);
    return 1;
}
static int top_member(void* ctx, const char* key, Sc* s) {
    Ld* l = (Ld*) ctx;
    if (!strcmp(key, "model")) return object(s, model_member, l);
    if (!strcmp(key, "added_tokens")) return added_tokens(l, s);
    if (!strcmp(key, "pre_tokenizer")) return pre_tok(l, s);
    return skip(s);
}
static int added_cmp(const void* a, const void* b) {   // longest first (stable by id for equal lengths)
    const Added* x = (const Added*) a, *y = (const Added*) b;
    if (x->n != y->n) return x->n > y->n ? -1 : 1;
    return x->id < y->id ? -1 : x->id > y->id;
}

void tok_close(Tok* t) {
    if (!t) return;
    map_free(&t->vocab);
    map_free(&t->cache);
    free(t->merges.key); free(t->merges.rank); free(t->merges.id);
    for (int32_t i = 0; i < t->nid; ++i) free(t->id_to_str[i]);
    free(t->id_to_str);
    free(t->is_added);
    for (int i = 0; i < t->nadded; ++i) free(t->added[i].s);
    free(t->added);
    free(t->cache_ids);
    if (t->re) uregex_close(t->re);
    pthread_mutex_destroy(&t->mu);
    free(t);
}

Tok* tok_open(const char* path, char* err, int errlen) {
    FILE* f = fopen(path, "rb");
    if (!f) { snprintf(err, (size_t) errlen, "cannot read %s", path); return NULL; }
    fseek(f, 0, SEEK_END);
    const long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* js = (char*) malloc((size_t) len + 1);
    const size_t got = fread(js, 1, (size_t) len, f);
    fclose(f);
    js[got] = 0;
    Tok* t = (Tok*) calloc(1, sizeof(Tok));
    pthread_mutex_init(&t->mu, NULL);
    map_init(&t->vocab, 300000);
    map_init(&t->cache, 1 << 16);
    Ld l = {t, NULL, 0, {0}, 0};
    Sc s = {js, js + got};
    const int ok = object(&s, top_member, &l);
    free(js);
    free(l.tmp.p);
    if (!ok) { snprintf(err, (size_t) errlen, "cannot parse %s", path); free(l.pattern); tok_close(t); return NULL; }
    if (l.bad_type) { snprintf(err, (size_t) errlen, "not a BPE tokenizer"); free(l.pattern); tok_close(t); return NULL; }
    if (t->err_bad) { snprintf(err, (size_t) errlen, "ByteLevel use_regex not supported"); free(l.pattern); tok_close(t); return NULL; }
    for (int i = 0; i < t->nadded; ++i) set_id_str(t, t->added[i].id, t->added[i].s, 1);
    qsort(t->added, (size_t) t->nadded, sizeof(Added), added_cmp);
    t->bos = -1;
    for (int i = 0; i < t->nadded; ++i) if (!strcmp(t->added[i].s, "<|begin_of_text|>")) t->bos = t->added[i].id;
    // the pre-tokenizer regex (ICU)
    UErrorCode ue = U_ZERO_ERROR;
    if (l.pattern) {
        int32_t n16 = 0;
        u_strFromUTF8(NULL, 0, &n16, l.pattern, -1, &ue);
        ue = U_ZERO_ERROR;
        UChar* pat = (UChar*) malloc(sizeof(UChar) * (size_t) (n16 + 1));
        u_strFromUTF8(pat, n16 + 1, &n16, l.pattern, -1, &ue);
        UParseError pe;
        if (!U_FAILURE(ue)) t->re = uregex_open(pat, n16, 0, &pe, &ue);
        free(pat);
    }
    free(l.pattern);
    if (!t->re || U_FAILURE(ue)) { snprintf(err, (size_t) errlen, "pre-tokenizer regex: ICU error %d", (int) ue); tok_close(t); return NULL; }
    ue = U_ZERO_ERROR;
    t->nfc = unorm2_getNFCInstance(&ue);
    if (U_FAILURE(ue)) { snprintf(err, (size_t) errlen, "ICU NFC: error %d", (int) ue); tok_close(t); return NULL; }
    // GPT-2 bytes_to_unicode
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        const int keep = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        t->byte2u[b] = (uint16_t) (keep ? b : 256 + n++);
    }
    for (int i = 0; i < 512; ++i) t->u2byte[i] = -1;
    for (int b = 0; b < 256; ++b) t->u2byte[t->byte2u[b]] = (int16_t) b;
    for (int b = 0; b < 256; ++b) {
        Buf u = {0};
        put_utf8(&u, t->byte2u[b]);
        t->byte_id[b] = -1;
        map_get(&t->vocab, u.p, u.n, &t->byte_id[b]);
        free(u.p);
    }
    return t;
}

int tok_bos(Tok* t) { return t->bos; }
// BPE work space of one call: symbols (a linked list over the word's bytes) and a min-heap of candidate merges.
typedef struct { int32_t id, prev, next, alive; } Sym;
typedef struct { int32_t rank, pos, id; } Cand;
typedef struct {
    Sym* sym;
    Cand* heap;
    int sym_cap, heap_cap, nheap;
} Bpe;

static int cand_less(const Cand* a, const Cand* b) { return a->rank < b->rank || (a->rank == b->rank && a->pos < b->pos); }
static void heap_push(Bpe* w, Cand c) {
    if (w->nheap == w->heap_cap) { w->heap_cap = 2 * w->heap_cap + 64; w->heap = (Cand*) realloc(w->heap, sizeof(Cand) * (size_t) w->heap_cap); }
    int i = w->nheap++;
    while (i > 0 && cand_less(&c, &w->heap[(i - 1) / 2])) { w->heap[i] = w->heap[(i - 1) / 2]; i = (i - 1) / 2; }
    w->heap[i] = c;
}
static Cand heap_pop(Bpe* w) {
    const Cand top = w->heap[0], last = w->heap[--w->nheap];
    int i = 0;
    for (;;) {
        int c = 2 * i + 1;
        if (c >= w->nheap) break;
        if (c + 1 < w->nheap && cand_less(&w->heap[c + 1], &w->heap[c])) ++c;
        if (!cand_less(&w->heap[c], &last)) break;
        w->heap[i] = w->heap[c];
        i = c;
    }
    if (w->nheap) w->heap[i] = last;
    return top;
}
static int merge_of(const Tok* t, int32_t a, int32_t b, int32_t* rank, int32_t* id) {
    return pm_get(&t->merges, ((uint64_t) (a + 1) << 32) | (uint64_t) (b + 1), rank, id);
}
static void push_pair(const Tok* t, Bpe* w, int pos) {
    const int nx = w->sym[pos].next;
    int32_t r, id;
    if (nx >= 0 && merge_of(t, w->sym[pos].id, w->sym[nx].id, &r, &id)) heap_push(w, (Cand) {r, pos, id});
}

// BPE of one pre-token (UTF-8 bytes) into *ids.  The lowest-rank adjacent pair is merged first, the leftmost among equal
// ranks, until none applies: HF's priority-queue merge (O(n log n), so a long run without spaces stays cheap).  Returns
// the count, or -1 if a byte has no symbol.  The word cache is shared: only it is locked.
static int bpe(Tok* t, Bpe* w, const char* s, size_t n, int32_t** ids, int* cap) {
    int32_t v;
    pthread_mutex_lock(&t->mu);
    if (map_get(&t->cache, s, n, &v)) {
        const int k = t->cache_ids[v];
        if (*cap < k) { *cap = 2 * k; *ids = (int32_t*) realloc(*ids, sizeof(int32_t) * (size_t) *cap); }
        memcpy(*ids, t->cache_ids + v + 1, sizeof(int32_t) * (size_t) k);
        pthread_mutex_unlock(&t->mu);
        return k;
    }
    pthread_mutex_unlock(&t->mu);
    if (w->sym_cap < (int) n) { w->sym_cap = 2 * (int) n + 16; w->sym = (Sym*) realloc(w->sym, sizeof(Sym) * (size_t) w->sym_cap); }
    for (size_t i = 0; i < n; ++i) {
        const int32_t id = t->byte_id[(uint8_t) s[i]];
        if (id < 0) return -1;
        w->sym[i] = (Sym) {id, (int32_t) i - 1, i + 1 < n ? (int32_t) i + 1 : -1, 1};
    }
    w->nheap = 0;
    for (size_t i = 0; i + 1 < n; ++i) push_pair(t, w, (int) i);
    while (w->nheap) {
        const Cand c = heap_pop(w);
        Sym* a = &w->sym[c.pos];
        if (!a->alive || a->next < 0) continue;
        int32_t r, id;
        if (!merge_of(t, a->id, w->sym[a->next].id, &r, &id) || r != c.rank || id != c.id) continue;   // stale entry
        Sym* b = &w->sym[a->next];
        a->id = c.id;
        b->alive = 0;
        a->next = b->next;
        if (a->next >= 0) w->sym[a->next].prev = c.pos;
        if (a->prev >= 0) push_pair(t, w, a->prev);
        push_pair(t, w, c.pos);
    }
    if (*cap < (int) n) { *cap = 2 * (int) n + 16; *ids = (int32_t*) realloc(*ids, sizeof(int32_t) * (size_t) *cap); }
    int k = 0;
    for (int i = n ? 0 : -1; i >= 0; i = w->sym[i].next) (*ids)[k++] = w->sym[i].id;
    if (n <= 256) {   // the cache holds ordinary words only (at most 200000 of them)
        pthread_mutex_lock(&t->mu);
        if (t->cache.n < 200000 && !map_get(&t->cache, s, n, &v)) {
            if (t->cache_n + (size_t) k + 1 > t->cache_cap) {
                t->cache_cap = (t->cache_n + (size_t) k + 1) * 2 + 4096;
                t->cache_ids = (int32_t*) realloc(t->cache_ids, sizeof(int32_t) * t->cache_cap);
            }
            *map_slot(&t->cache, s, n, 1) = (int32_t) t->cache_n;
            t->cache_ids[t->cache_n] = k;
            memcpy(t->cache_ids + t->cache_n + 1, *ids, sizeof(int32_t) * (size_t) k);
            t->cache_n += (size_t) k + 1;
        }
        pthread_mutex_unlock(&t->mu);
    }
    return k;
}

// One segment without added tokens: NFC, regex split, BPE.  Writes ids at out[n..] (up to cap), returns the new count.
static int encode_plain(Tok* t, const char* seg, size_t len, int32_t* out, int cap, int n) {
    UErrorCode ue = U_ZERO_ERROR;
    int32_t n16 = 0;
    u_strFromUTF8(NULL, 0, &n16, seg, (int32_t) len, &ue);
    if (ue != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(ue)) return -1;
    ue = U_ZERO_ERROR;
    UChar* u = (UChar*) malloc(sizeof(UChar) * (size_t) (n16 + 1));
    u_strFromUTF8(u, n16 + 1, &n16, seg, (int32_t) len, &ue);
    // NFC (the output can be longer than the input: retry with the size ICU asks for)
    int32_t cap16 = n16 * 3 + 16;
    UChar* nf = (UChar*) malloc(sizeof(UChar) * (size_t) cap16);
    int32_t nn = unorm2_normalize(t->nfc, u, n16, nf, cap16, &ue);
    if (ue == U_BUFFER_OVERFLOW_ERROR) {
        ue = U_ZERO_ERROR;
        cap16 = nn + 1;
        nf = (UChar*) realloc(nf, sizeof(UChar) * (size_t) cap16);
        nn = unorm2_normalize(t->nfc, u, n16, nf, cap16, &ue);
    }
    free(u);
    if (U_FAILURE(ue)) { free(nf); return -1; }
    URegularExpression* re = uregex_clone(t->re, &ue);   // this call's own matcher (the pattern is shared, read-only)
    if (U_FAILURE(ue)) { free(nf); return -1; }
    uregex_setStackLimit(re, 0, &ue);   // no backtracking-stack cap: one long word (no spaces) must not end the text
    uregex_setText(re, nf, nn, &ue);
    Bpe bw = {0};
    int32_t* ids = NULL;
    int icap = 0;
    char* w = NULL;
    int32_t wcap = 0;
    int k = n;
    while (!U_FAILURE(ue) && uregex_findNext(re, &ue)) {
        const int32_t a = uregex_start(re, 0, &ue), b = uregex_end(re, 0, &ue);
        if (U_FAILURE(ue)) break;
        int32_t n8 = 0;
        UErrorCode e2 = U_ZERO_ERROR;
        u_strToUTF8(NULL, 0, &n8, nf + a, b - a, &e2);
        if (n8 + 1 > wcap) { wcap = 2 * n8 + 16; w = (char*) realloc(w, (size_t) wcap); }
        e2 = U_ZERO_ERROR;
        u_strToUTF8(w, wcap, &n8, nf + a, b - a, &e2);
        const int m = bpe(t, &bw, w, (size_t) n8, &ids, &icap);
        if (m < 0) { k = -1; break; }
        for (int i = 0; i < m; ++i) { if (k < cap) out[k] = ids[i]; ++k; }
    }
    if (U_FAILURE(ue)) k = -1;   // a regex error is an error, not the end of the text
    uregex_close(re);
    free(bw.sym);
    free(bw.heap);
    free(ids);
    free(w);
    free(nf);
    return k;
}

int tok_encode(Tok* t, const char* text, int add_bos, int32_t* out, int cap) {
    const size_t len = strlen(text);
    int n = 0;
    size_t seg = 0, i = 0;
    while (i < len) {
        const Added* hit = NULL;
        for (int a = 0; a < t->nadded; ++a) {   // longest first
            const Added* ad = &t->added[a];
            if (!ad->n || ad->s[0] != text[i] || i + ad->n > len) continue;
            if (!memcmp(text + i, ad->s, ad->n)) { hit = ad; break; }
        }
        if (!hit) { ++i; continue; }
        if (i > seg && (n = encode_plain(t, text + seg, i - seg, out, cap, n)) < 0) return -1;
        if (n < cap) out[n] = hit->id;
        ++n;
        i += hit->n;
        seg = i;
    }
    if (seg < len && (n = encode_plain(t, text + seg, len - seg, out, cap, n)) < 0) return -1;
    if (add_bos && t->bos >= 0 && !(n > 0 && cap > 0 && out[0] == t->bos)) {
        if (n + 1 <= cap) {
            memmove(out + 1, out, (size_t) n * sizeof(int32_t));
            out[0] = t->bos;
        }
        ++n;
    }
    return n;
}

static int utf8_valid(const uint8_t* s, size_t n) {
    for (size_t i = 0; i < n;) {
        const uint8_t c = s[i];
        size_t k = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
        if (!k || i + k > n) return 0;
        uint32_t cp = k == 1 ? c : k == 2 ? c & 31u : k == 3 ? c & 15u : c & 7u;
        for (size_t j = 1; j < k; ++j) {
            if ((s[i + j] & 0xC0) != 0x80) return 0;
            cp = (cp << 6) | (s[i + j] & 63u);
        }
        if ((k == 2 && cp < 0x80) || (k == 3 && cp < 0x800) || (k == 4 && (cp < 0x10000 || cp > 0x10FFFF)) ||
            (cp >= 0xD800 && cp < 0xE000))
            return 0;
        i += k;
    }
    return 1;
}

char* tok_decode(Tok* t, const int32_t* ids, int n, int* valid) {
    Buf b = {0};
    buf_put(&b, "", 0);
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 0 || ids[i] >= t->nid || !t->id_to_str[ids[i]]) continue;
        const char* s = t->id_to_str[ids[i]];
        if (t->is_added[ids[i]]) { buf_puts(&b, s); continue; }
        // byte-level symbols: each code point maps back to one byte
        for (const uint8_t* p = (const uint8_t*) s; *p;) {
            uint32_t cp;
            if (*p < 0x80) cp = *p++;
            else if ((*p & 0xE0) == 0xC0) { cp = ((p[0] & 31u) << 6) | (p[1] & 63u); p += 2; }
            else if ((*p & 0xF0) == 0xE0) { cp = ((p[0] & 15u) << 12) | ((p[1] & 63u) << 6) | (p[2] & 63u); p += 3; }
            else { cp = 0xFFFF; p += 4; }
            const int by = cp < 512 ? t->u2byte[cp] : -1;
            if (by >= 0) { const char c = (char) by; buf_put(&b, &c, 1); }
        }
    }
    // a cut at the end of a fixed-length generation may split one character: that is not invalid output
    int ok = utf8_valid((const uint8_t*) b.p, b.n);
    for (size_t trim = 1; !ok && trim <= 3 && trim <= b.n; ++trim) ok = utf8_valid((const uint8_t*) b.p, b.n - trim);
    if (valid) *valid = ok;
    return buf_take(&b);
}
