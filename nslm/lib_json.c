// nslm/lib_json.c - see json.h.
#include "json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void buf_put(Buf* b, const char* s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        b->cap = (b->n + n + 1) * 2;
        b->p = (char*) realloc(b->p, b->cap);
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}
void buf_puts(Buf* b, const char* s) { buf_put(b, s, strlen(s)); }
char* buf_take(Buf* b) {
    if (!b->p) buf_put(b, "", 0);
    char* p = b->p;
    b->p = NULL;
    b->n = b->cap = 0;
    return p;
}

Json* json_new(JType t) {
    Json* j = (Json*) calloc(1, sizeof(Json));
    j->t = t;
    return j;
}
Json* json_str(const char* s) {
    Json* j = json_new(J_STR);
    j->s = strdup(s);
    return j;
}
void json_push(Json* a, Json* v) {
    a->v = (Json**) realloc(a->v, sizeof(Json*) * (size_t) (a->n + 1));
    a->v[a->n++] = v;
}
void json_set(Json* o, const char* key, Json* v) {
    for (int i = 0; i < o->n; ++i)
        if (!strcmp(o->k[i], key)) { json_free(o->v[i]); o->v[i] = v; return; }
    o->k = (char**) realloc(o->k, sizeof(char*) * (size_t) (o->n + 1));
    o->k[o->n] = strdup(key);
    json_push(o, v);
}
void json_free(Json* j) {
    if (!j) return;
    for (int i = 0; i < j->n; ++i) {
        json_free(j->v[i]);
        if (j->k) free(j->k[i]);
    }
    free(j->v);
    free(j->k);
    free(j->s);
    free(j);
}
Json* json_copy(const Json* j) {
    Json* c = json_new(j->t);
    c->i = j->i;
    c->d = j->d;
    if (j->s) c->s = strdup(j->s);
    for (int i = 0; i < j->n; ++i) {
        if (j->t == J_OBJ) json_set(c, j->k[i], json_copy(j->v[i]));
        else json_push(c, json_copy(j->v[i]));
    }
    return c;
}
Json* json_get(const Json* o, const char* key) {
    if (!o || o->t != J_OBJ) return NULL;
    for (int i = 0; i < o->n; ++i)
        if (!strcmp(o->k[i], key)) return o->v[i];
    return NULL;
}
const char* json_gets(const Json* o, const char* key) {
    const Json* v = json_get(o, key);
    return v && v->t == J_STR ? v->s : NULL;
}
int json_truthy(const Json* j) {
    if (!j) return 0;
    switch (j->t) {
    case J_NULL: case J_FALSE: return 0;
    case J_TRUE: return 1;
    case J_INT: return j->i != 0;
    case J_NUM: return j->d != 0;
    case J_STR: return j->s[0] != 0;
    default: return j->n > 0;
    }
}

// ---- parser ----

typedef struct {
    const char* s;
    size_t n, i;
    char* err;
    int errlen;
    int depth;
} P;

static Json* fail(P* p, const char* msg) {
    if (p->err && !p->err[0]) snprintf(p->err, (size_t) p->errlen, "JSON: %s at byte %zu", msg, p->i);
    return NULL;
}
static void ws(P* p) {
    while (p->i < p->n && (p->s[p->i] == ' ' || p->s[p->i] == '\t' || p->s[p->i] == '\n' || p->s[p->i] == '\r')) ++p->i;
}
static int hex4(P* p, unsigned* u) {
    if (p->i + 4 > p->n) return 0;
    *u = 0;
    for (int k = 0; k < 4; ++k) {
        const char c = p->s[p->i++];
        *u = *u * 16 + (unsigned) (c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10
                                   : c >= 'A' && c <= 'F' ? c - 'A' + 10 : 99);
        if (*u > 0xFFFF) return 0;
    }
    return 1;
}
static void utf8(Buf* b, unsigned c) {
    char o[4];
    if (c < 0x80) { o[0] = (char) c; buf_put(b, o, 1); }
    else if (c < 0x800) { o[0] = (char) (0xC0 | c >> 6); o[1] = (char) (0x80 | (c & 63)); buf_put(b, o, 2); }
    else if (c < 0x10000) {
        o[0] = (char) (0xE0 | c >> 12); o[1] = (char) (0x80 | ((c >> 6) & 63)); o[2] = (char) (0x80 | (c & 63));
        buf_put(b, o, 3);
    } else {
        o[0] = (char) (0xF0 | c >> 18); o[1] = (char) (0x80 | ((c >> 12) & 63));
        o[2] = (char) (0x80 | ((c >> 6) & 63)); o[3] = (char) (0x80 | (c & 63));
        buf_put(b, o, 4);
    }
}
static char* pstring(P* p) {
    Buf b = {0};
    ++p->i;   // opening quote
    for (;;) {
        if (p->i >= p->n) { free(b.p); fail(p, "unterminated string"); return NULL; }
        const char c = p->s[p->i++];
        if (c == '"') break;
        if ((unsigned char) c < 0x20) { free(b.p); fail(p, "control character in string"); return NULL; }
        if (c != '\\') { buf_put(&b, &c, 1); continue; }
        if (p->i >= p->n) break;
        const char e = p->s[p->i++];
        unsigned u, lo;
        switch (e) {
        case '"': buf_put(&b, "\"", 1); break;
        case '\\': buf_put(&b, "\\", 1); break;
        case '/': buf_put(&b, "/", 1); break;
        case 'b': buf_put(&b, "\b", 1); break;
        case 'f': buf_put(&b, "\f", 1); break;
        case 'n': buf_put(&b, "\n", 1); break;
        case 'r': buf_put(&b, "\r", 1); break;
        case 't': buf_put(&b, "\t", 1); break;
        case 'u':
            if (!hex4(p, &u)) { free(b.p); fail(p, "bad \\u escape"); return NULL; }
            if (u >= 0xD800 && u < 0xDC00 && p->i + 6 <= p->n && p->s[p->i] == '\\' && p->s[p->i + 1] == 'u') {
                const size_t save = p->i;
                p->i += 2;
                if (hex4(p, &lo) && lo >= 0xDC00 && lo < 0xE000) u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                else p->i = save;
            }
            utf8(&b, u);
            break;
        default: free(b.p); fail(p, "bad escape"); return NULL;
        }
    }
    return buf_take(&b);
}
static Json* pvalue(P* p);
static Json* pvalue(P* p) {
    ws(p);
    if (p->i >= p->n) return fail(p, "unexpected end");
    if (++p->depth > 512) return fail(p, "nesting too deep");
    const char c = p->s[p->i];
    Json* j = NULL;
    if (c == '{') {
        j = json_new(J_OBJ);
        ++p->i;
        ws(p);
        if (p->i < p->n && p->s[p->i] == '}') ++p->i;
        else for (;;) {
            ws(p);
            if (p->i >= p->n || p->s[p->i] != '"') { json_free(j); return fail(p, "expected key"); }
            char* key = pstring(p);
            if (!key) { json_free(j); return NULL; }
            ws(p);
            if (p->i >= p->n || p->s[p->i] != ':') { free(key); json_free(j); return fail(p, "expected ':'"); }
            ++p->i;
            Json* v = pvalue(p);
            if (!v) { free(key); json_free(j); return NULL; }
            json_set(j, key, v);   // a repeated key keeps its first position and the last value, as Python
            free(key);
            ws(p);
            if (p->i < p->n && p->s[p->i] == ',') { ++p->i; continue; }
            if (p->i < p->n && p->s[p->i] == '}') { ++p->i; break; }
            json_free(j);
            return fail(p, "expected ',' or '}'");
        }
    } else if (c == '[') {
        j = json_new(J_ARR);
        ++p->i;
        ws(p);
        if (p->i < p->n && p->s[p->i] == ']') ++p->i;
        else for (;;) {
            Json* v = pvalue(p);
            if (!v) { json_free(j); return NULL; }
            json_push(j, v);
            ws(p);
            if (p->i < p->n && p->s[p->i] == ',') { ++p->i; continue; }
            if (p->i < p->n && p->s[p->i] == ']') { ++p->i; break; }
            json_free(j);
            return fail(p, "expected ',' or ']'");
        }
    } else if (c == '"') {
        char* s = pstring(p);
        if (!s) return NULL;
        j = json_new(J_STR);
        j->s = s;
    } else if (!strncmp(p->s + p->i, "true", 4)) { p->i += 4; j = json_new(J_TRUE); }
    else if (!strncmp(p->s + p->i, "false", 5)) { p->i += 5; j = json_new(J_FALSE); }
    else if (!strncmp(p->s + p->i, "null", 4)) { p->i += 4; j = json_new(J_NULL); }
    else if (c == '-' || (c >= '0' && c <= '9')) {
        const size_t st = p->i;
        int flt = 0;
        if (p->s[p->i] == '-') ++p->i;
        while (p->i < p->n && strchr("0123456789.eE+-", p->s[p->i])) {
            if (strchr(".eE", p->s[p->i])) flt = 1;
            ++p->i;
        }
        char tmp[64];
        const size_t len = p->i - st;
        if (len >= sizeof tmp) return fail(p, "number too long");
        memcpy(tmp, p->s + st, len);
        tmp[len] = 0;
        char* end;
        if (!flt) {
            const long long v = strtoll(tmp, &end, 10);
            if (*end) return fail(p, "bad number");
            j = json_new(J_INT);
            j->i = v;
        } else {
            const double v = strtod(tmp, &end);
            if (*end) return fail(p, "bad number");
            j = json_new(J_NUM);
            j->d = v;
        }
    } else return fail(p, "unexpected character");
    --p->depth;
    return j;
}
Json* json_parse(const char* s, size_t n, char* err, int errlen) {
    P p = {s, n, 0, err, errlen, 0};
    if (err && errlen) err[0] = 0;
    Json* j = pvalue(&p);
    if (!j) return NULL;
    ws(&p);
    if (p.i != p.n) { json_free(j); return fail(&p, "trailing data"); }
    return j;
}

Json* json_parse_prefix(const char* s, size_t n, size_t* used) {
    P p = {s, n, 0, NULL, 0, 0};
    Json* j = pvalue(&p);
    if (j && used) *used = p.i;
    return j;
}

// ---- output ----

void py_num(Buf* b, const Json* j) {
    char t[64];
    if (j->t == J_INT) { snprintf(t, sizeof t, "%lld", (long long) j->i); buf_puts(b, t); return; }
    const double d = j->d;
    if (isnan(d)) { buf_puts(b, "NaN"); return; }
    if (isinf(d)) { buf_puts(b, d < 0 ? "-Infinity" : "Infinity"); return; }
    // the shortest digits that round-trip, then Python's repr layout
    int prec = 1;
    for (; prec < 17; ++prec) {
        snprintf(t, sizeof t, "%.*e", prec - 1, d);
        if (strtod(t, NULL) == d) break;
    }
    snprintf(t, sizeof t, "%.*e", prec - 1, d);
    char digits[32];
    int nd = 0, neg = t[0] == '-';
    const char* q = t + neg;
    for (; *q && *q != 'e'; ++q) if (*q != '.') digits[nd++] = *q;
    while (nd > 1 && digits[nd - 1] == '0') --nd;
    digits[nd] = 0;
    const int exp10 = atoi(q + 1);
    if (neg) buf_puts(b, "-");
    if (exp10 < -4 || exp10 >= 16) {
        buf_put(b, digits, 1);
        if (nd > 1) { buf_puts(b, "."); buf_put(b, digits + 1, (size_t) nd - 1); }
        snprintf(t, sizeof t, "e%c%02d", exp10 < 0 ? '-' : '+', exp10 < 0 ? -exp10 : exp10);
        buf_puts(b, t);
    } else if (exp10 < 0) {
        buf_puts(b, "0.");
        for (int k = 0; k < -exp10 - 1; ++k) buf_puts(b, "0");
        buf_put(b, digits, (size_t) nd);
    } else {
        for (int k = 0; k <= exp10; ++k) buf_put(b, k < nd ? digits + k : "0", 1);
        buf_puts(b, ".");
        if (nd > exp10 + 1) buf_put(b, digits + exp10 + 1, (size_t) (nd - exp10 - 1));
        else buf_puts(b, "0");
    }
}
// Length of the valid UTF-8 sequence at s (1 .. 4), or 0 (overlong forms, surrogates and code points past U+10FFFF are
// invalid, as RFC 3629).
static int utf8_seq(const unsigned char* s) {
    const unsigned c = s[0];
    if (c < 0x80) return 1;
    int n;
    unsigned cp;
    if (c >= 0xC2 && c <= 0xDF) { n = 2; cp = c & 0x1F; }
    else if (c >= 0xE0 && c <= 0xEF) { n = 3; cp = c & 0x0F; }
    else if (c >= 0xF0 && c <= 0xF4) { n = 4; cp = c & 0x07; }
    else return 0;
    for (int i = 1; i < n; ++i) {
        if ((s[i] & 0xC0) != 0x80) return 0;   // also stops at the terminating 0
        cp = cp << 6 | (s[i] & 0x3Fu);
    }
    if ((n == 3 && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))) || (n == 4 && (cp < 0x10000 || cp > 0x10FFFF))) return 0;
    return n;
}
// A JSON string; bytes that are not valid UTF-8 (a model can sample any byte token) become U+FFFD, one per byte, so the
// output is always valid JSON text (RFC 8259).
static void dump_str(Buf* b, const char* s) {
    buf_puts(b, "\"");
    for (; *s; ++s) {
        const unsigned char c = (unsigned char) *s;
        char e[8];
        if (c >= 0x80) {
            const int n = utf8_seq((const unsigned char*) s);
            if (n) { buf_put(b, s, (size_t) n); s += n - 1; }
            else buf_puts(b, "\xEF\xBF\xBD");
            continue;
        }
        if (c == '"') buf_puts(b, "\\\"");
        else if (c == '\\') buf_puts(b, "\\\\");
        else if (c == '\n') buf_puts(b, "\\n");
        else if (c == '\r') buf_puts(b, "\\r");
        else if (c == '\t') buf_puts(b, "\\t");
        else if (c == '\b') buf_puts(b, "\\b");
        else if (c == '\f') buf_puts(b, "\\f");
        else if (c < 0x20) { snprintf(e, sizeof e, "\\u%04x", c); buf_puts(b, e); }
        else buf_put(b, (const char*) &c, 1);
    }
    buf_puts(b, "\"");
}
void json_dump(Buf* b, const Json* j, int compact) {
    switch (j->t) {
    case J_NULL: buf_puts(b, "null"); break;
    case J_FALSE: buf_puts(b, "false"); break;
    case J_TRUE: buf_puts(b, "true"); break;
    case J_INT: case J_NUM: py_num(b, j); break;
    case J_STR: dump_str(b, j->s); break;
    case J_ARR:
        buf_puts(b, "[");
        for (int i = 0; i < j->n; ++i) {
            if (i) buf_puts(b, compact ? "," : ", ");
            json_dump(b, j->v[i], compact);
        }
        buf_puts(b, "]");
        break;
    case J_OBJ:
        buf_puts(b, "{");
        for (int i = 0; i < j->n; ++i) {
            if (i) buf_puts(b, compact ? "," : ", ");
            dump_str(b, j->k[i]);
            buf_puts(b, compact ? ":" : ": ");
            json_dump(b, j->v[i], compact);
        }
        buf_puts(b, "}");
        break;
    }
}
char* json_dumps(const Json* j, int compact) {
    Buf b = {0};
    json_dump(&b, j, compact);
    return buf_take(&b);
}

static void nl(Buf* b, int depth) {
    buf_puts(b, "\n");
    for (int i = 0; i < depth; ++i) buf_puts(b, "  ");
}
void json_dump_indent(Buf* b, const Json* j, int depth) {
    if ((j->t != J_ARR && j->t != J_OBJ) || !j->n) { json_dump(b, j, 0); return; }
    buf_puts(b, j->t == J_ARR ? "[" : "{");
    for (int i = 0; i < j->n; ++i) {
        if (i) buf_puts(b, ",");
        nl(b, depth + 1);
        if (j->t == J_OBJ) { dump_str(b, j->k[i]); buf_puts(b, ": "); }
        json_dump_indent(b, j->v[i], depth + 1);
    }
    nl(b, depth);
    buf_puts(b, j->t == J_ARR ? "]" : "}");
}
