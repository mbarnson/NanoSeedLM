// harness/nslm-serve.c - OpenAI-compatible HTTP server for K2-Horizon-MoVA on the nslm engine.
//
//   nslm-serve --model DIR [--res out/res] [--host 127.0.0.1] [--port 8080]
//              [--ctx 65536] [--kv bf16|q8] [--model-id ID] [--quiet]
//   nslm-serve --render REQUEST.json     print the prompt a chat request renders to, and exit
//
// GET /v1/models, GET /health, POST /v1/chat/completions, POST /v1/completions (stream or not).
// Chat prompts: harness/chat_template.c (the model's chat_template.jinja, tools included).  Tool calls in the output
// come back as OpenAI tool_calls (harness/tool_calls.c).  The thinking span is returned as reasoning_content.
// Sampling defaults: IFM's model card (temperature 1.0, top_p 0.95); temperature 0 is greedy.
// One request at a time on one sequence slot; the KV cache of the previous request's common prefix is reused.
// --kv q8: the 8-bit KV cache (long contexts in less memory; see engine_api.h).
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sock_t;
#define sock_close closesocket
#define SEND_FLAGS 0
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int sock_t;
#define INVALID_SOCKET (-1)
#define sock_close close
#ifdef MSG_NOSIGNAL
#define SEND_FLAGS MSG_NOSIGNAL
#else
#define SEND_FLAGS 0
#endif
#endif
#include <pthread.h>

#include "chat_template.h"
#include "engine_api.h"
#include "json.h"
#include "platform.h"
#include "tokenizer.h"
#include "tool_calls.h"

#define CHUNK 16          // tokens per engine call: the streaming granularity
#define FIRST_CHUNK 4     // a short first call, so the first token arrives early
#define MAX_BODY (64 << 20)
#define TOK_EOS 1         // <|ifm|endoftext|>
#define TOK_IM_END 250019 // <|ifm|im_end|>

static char g_model_id[256];
static Eng* g_eng;
static Tok* g_tok;
static pthread_mutex_t g_eng_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_ctx = 65536;
static int g_verbose = 1;

// ---- JSON building ----

static Json* jstr(const char* s) { return json_str(s ? s : ""); }
static Json* jstrn(const char* s, size_t n) {
    char* t = (char*) malloc(n + 1);
    memcpy(t, s, n);
    t[n] = 0;
    Json* j = json_str(t);
    free(t);
    return j;
}
static Json* jint(int64_t v) { Json* j = json_new(J_INT); j->i = v; return j; }
static Json* jnum(double v) { Json* j = json_new(J_NUM); j->d = v; return j; }
static Json* jobj(void) { return json_new(J_OBJ); }
static Json* jarr(void) { return json_new(J_ARR); }
static const Json* jfield(const Json* o, const char* k, JType t) {   // member of type t (J_NUM accepts J_INT)
    const Json* v = json_get(o, k);
    if (!v) return NULL;
    if (v->t == t || (t == J_NUM && v->t == J_INT)) return v;
    return NULL;
}
static double jnumber(const Json* v) { return v->t == J_INT ? (double) v->i : v->d; }

// ---- HTTP ----

typedef struct {
    sock_t fd;
    bool dead;   // the client went away: stop generating
} Conn;

static bool send_all(Conn* c, const void* p, size_t n) {
    if (!c || c->dead) return false;
    const char* b = (const char*) p;
    while (n) {
        const int w = (int) send(c->fd, b, (int) (n > (1u << 30) ? (1u << 30) : n), SEND_FLAGS);
        if (w <= 0) { c->dead = true; return false; }
        b += w;
        n -= (size_t) w;
    }
    return true;
}
static bool send_cstr(Conn* c, const char* s) { return send_all(c, s, strlen(s)); }
static const char* kCors = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\n"
                           "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
static void send_json(Conn* c, int status, Json* obj) {   // consumes obj
    char* body = json_dumps(obj, 1);
    json_free(obj);
    const char* reason = status == 200 ? "OK" : status == 400 ? "Bad Request" : status == 404 ? "Not Found"
                       : status == 413 ? "Payload Too Large" : "Internal Server Error";
    char head[512];
    snprintf(head, sizeof head, "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n%sConnection: close\r\n\r\n",
             status, reason, strlen(body), kCors);
    send_cstr(c, head);
    send_cstr(c, body);
    free(body);
}
static void send_error(Conn* c, int status, const char* msg) {
    Json* e = jobj();
    json_set(e, "message", jstr(msg));
    json_set(e, "type", jstr(status == 500 ? "server_error" : "invalid_request_error"));
    Json* o = jobj();
    json_set(o, "error", e);
    send_json(c, status, o);
}
static bool sse_begin(Conn* c) {
    char head[512];
    snprintf(head, sizeof head, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n%sConnection: close\r\n\r\n", kCors);
    return send_cstr(c, head);
}
static bool sse_event(Conn* c, Json* obj) {   // consumes obj
    char* d = json_dumps(obj, 1);
    json_free(obj);
    const bool ok = send_all(c, "data: ", 6) && send_cstr(c, d) && send_all(c, "\n\n", 2);
    free(d);
    return ok;
}

typedef struct {
    char method[16], path[512];
    char* body;
    size_t body_len;
} Request;

// One request.  0 with r filled, -1 on a closed connection, or an HTTP status to answer with.
static int read_request(Conn* c, Request* r) {
    Buf buf = {0};
    char tmp[65536];
    char* end = NULL;
    memset(r, 0, sizeof *r);
    while (!end) {
        const int n = (int) recv(c->fd, tmp, (int) sizeof tmp, 0);
        if (n <= 0) { free(buf.p); return -1; }
        buf_put(&buf, tmp, (size_t) n);
        if (buf.n > (1u << 20)) { free(buf.p); return 413; }
        end = strstr(buf.p, "\r\n\r\n");
    }
    const size_t hlen = (size_t) (end - buf.p);
    char* line = buf.p;
    char* eol = strstr(line, "\r\n");
    *eol = 0;
    if (sscanf(line, "%15s %511s", r->method, r->path) != 2) { free(buf.p); return 400; }
    char* q = strchr(r->path, '?');
    if (q) *q = 0;
    long len = 0;
    bool expect = false;
    for (char* l = eol + 2; l < buf.p + hlen;) {
        char* e = strstr(l, "\r\n");
        if (!e || e > buf.p + hlen) e = buf.p + hlen;
        *e = 0;
        char* colon = strchr(l, ':');
        if (colon) {
            *colon = 0;
            char* v = colon + 1;
            while (*v == ' ' || *v == '\t') ++v;
            for (char* k = l; *k; ++k) if (*k >= 'A' && *k <= 'Z') *k = (char) (*k + 32);
            if (!strcmp(l, "content-length")) len = atol(v);
            if (!strcmp(l, "expect")) {
                char lv[32];
                snprintf(lv, sizeof lv, "%s", v);
                for (char* k = lv; *k; ++k) if (*k >= 'A' && *k <= 'Z') *k = (char) (*k + 32);
                expect = !strncmp(lv, "100-continue", 12);
            }
        }
        l = e + 2;
    }
    if (len < 0 || len > MAX_BODY) { free(buf.p); return 413; }
    Buf b = {0};
    buf_put(&b, end + 4, buf.n - hlen - 4);
    free(buf.p);
    if (expect && (long) b.n < len) send_all(c, "HTTP/1.1 100 Continue\r\n\r\n", 25);
    while ((long) b.n < len) {
        const int n = (int) recv(c->fd, tmp, (int) sizeof tmp, 0);
        if (n <= 0) { free(b.p); return -1; }
        buf_put(&b, tmp, (size_t) n);
    }
    if ((long) b.n > len) b.n = (size_t) len;
    if (!b.p) buf_put(&b, "", 0);
    r->body = b.p;
    r->body_len = b.n;
    return 0;
}

// ---- prompt ----

static const char* str_or(const Json* o, const char* k, const char* def) {
    const Json* v = jfield(o, k, J_STR);
    return v ? v->s : def;
}

// The chat prompt for a request (malloc'd), or NULL with err.  *close_tag: the thinking close of the effort;
// *tools_out: the tools in effect (or NULL).
static char* chat_prompt(Json* req, const char** close_tag, const Json** tools_out, char* err, int errlen) {
    const Json* kw = jfield(req, "chat_template_kwargs", J_OBJ);
    const char* effort = str_or(req, "reasoning_effort", NULL);
    const Json* rs = jfield(req, "reasoning", J_OBJ);
    if (rs) effort = str_or(rs, "effort", effort);
    effort = str_or(kw, "reasoning_effort", effort ? effort : "high");
    if (!strcmp(effort, "minimal")) effort = "low";
    Json* messages = json_get(req, "messages");
    if (ct_normalize(messages, err, errlen)) return NULL;
    const Json* tools = json_get(req, "tools");
    const Json* tc = jfield(req, "tool_choice", J_STR);
    if (tc && !strcmp(tc->s, "none")) tools = NULL;
    CtOpts o = {str_or(kw, "tool_presentation_format", NULL), str_or(kw, "tool_call_format", NULL), effort, 1};
    char* p = ct_render(messages, tools, &o, err, errlen);
    if (!p) return NULL;
    *close_tag = !strcmp(effort, "low") ? "</ifm|think_faster>" : !strcmp(effort, "medium") ? "</ifm|think_fast>" : "</ifm|think>";
    *tools_out = json_truthy(tools) ? tools : NULL;
    return p;
}

// ---- generation ----

// Bytes of buf[0..n) that end on a complete UTF-8 character.
static size_t utf8_complete(const char* buf, size_t n) {
    size_t i = n;
    for (int back = 0; back < 4 && i > 0; ++back) {
        const unsigned char c = (unsigned char) buf[i - 1];
        if ((c & 0xC0) == 0x80) { --i; continue; }
        const size_t need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        return (n - (i - 1)) >= need ? n : i - 1;
    }
    return n;
}

typedef struct {
    int max_tokens, top_k;
    double temperature, top_p;
    uint64_t rng;
    bool chat, stream, include_usage;
    char** stops;
    int nstops;
    const char* close_tag;          // the prompt's "</ifm|think...>" for chat (any of the three closes); NULL for raw completions
    const Json* tools;              // tools in effect: content is scanned for tool calls
    char id_[64];
    const char* model;
    long created;
    Buf reasoning, content;         // content: everything after the thinking span, tool markup included
    size_t content_sent;            // bytes of content already streamed
    const char* finish;
    int prompt_tokens, completion_tokens, cached_tokens;
    double prefill_s, decode_s, ttft_s;
} Gen;

typedef struct {
    size_t emitted;
    bool in_content, content_started;
} Splitter;

static Json* chunk(Gen* g, Json* delta, const char* finish) {   // consumes delta
    Json* ch = jobj();
    json_set(ch, "index", jint(0));
    json_set(ch, "delta", delta);
    json_set(ch, "finish_reason", finish ? jstr(finish) : json_new(J_NULL));
    Json* cs = jarr();
    json_push(cs, ch);
    Json* o = jobj();
    json_set(o, "id", jstr(g->id_));
    json_set(o, "object", jstr("chat.completion.chunk"));
    json_set(o, "created", jint(g->created));
    json_set(o, "model", jstr(g->model));
    json_set(o, "choices", cs);
    return o;
}
static Json* text_choice(Gen* g, Json* text, const char* finish) {   // completions (consumes text)
    Json* ch = jobj();
    json_set(ch, "index", jint(0));
    json_set(ch, "text", text);
    json_set(ch, "finish_reason", finish ? jstr(finish) : json_new(J_NULL));
    Json* cs = jarr();
    json_push(cs, ch);
    Json* o = jobj();
    json_set(o, "id", jstr(g->id_));
    json_set(o, "object", jstr("text_completion"));
    json_set(o, "created", jint(g->created));
    json_set(o, "model", jstr(g->model));
    json_set(o, "choices", cs);
    return o;
}
static bool stream_content(Conn* c, Gen* g, const char* t, size_t n) {
    if (!n || !g->stream) return true;
    if (!g->chat) return sse_event(c, text_choice(g, jstrn(t, n), NULL));
    Json* d = jobj();
    json_set(d, "content", jstrn(t, n));
    return sse_event(c, chunk(g, d, NULL));
}
// Streams the content that cannot belong to a tool-call envelope.
static bool flush_content(Conn* c, Gen* g, bool final) {
    const char* b = g->content.p ? g->content.p : "";
    size_t safe = g->content.n;
    if (g->tools && !final) {
        const char* hit = strstr(b, TC_OPEN);
        if (hit) safe = (size_t) (hit - b);
        else {
            const size_t ol = strlen(TC_OPEN);
            for (size_t k = ol - 1; k > 0; --k)
                if (g->content.n >= k && !memcmp(b + g->content.n - k, TC_OPEN, k)) { safe = g->content.n - k; break; }
        }
    }
    if (g->tools && final) return true;   // the end of the request parses and sends the rest
    safe = utf8_complete(b, safe);
    if (safe <= g->content_sent) return true;
    const size_t from = g->content_sent;
    g->content_sent = safe;
    return stream_content(c, g, b + from, safe - from);
}
static bool emit_text(Conn* c, Gen* g, const char* text, size_t n, bool reasoning) {
    if (!n) return true;
    if (reasoning) {
        buf_put(&g->reasoning, text, n);
        if (!g->stream) return true;
        Json* d = jobj();
        json_set(d, "reasoning_content", jstrn(text, n));
        return sse_event(c, chunk(g, d, NULL));
    }
    buf_put(&g->content, text, n);
    return flush_content(c, g, false);
}
// The template already opens thinking, but models can echo an opening tag.
static const char* thinking_tags[] = {
    "<ifm|think>", "<ifm|think_fast>", "<ifm|think_faster>",
    "</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"
};
static const char* find_thinking_tag(const char* b, size_t from, size_t upto, size_t* tl, bool* closing) {
    for (size_t i = from; i < upto; ++i)
        for (size_t t = 0; t < sizeof thinking_tags / sizeof *thinking_tags; ++t) {
            const size_t l = strlen(thinking_tags[t]);
            if (i + l <= upto && !memcmp(b + i, thinking_tags[t], l)) {
                *tl = l;
                *closing = thinking_tags[t][1] == '/';
                return b + i;
            }
        }
    return NULL;
}
// Keep a partial delimiter until the next decoded chunk; never emit its prefix as reasoning.
static size_t thinking_tag_suffix(const char* b, size_t from, size_t upto) {
    size_t keep = 0;
    for (size_t t = 0; t < sizeof thinking_tags / sizeof *thinking_tags; ++t) {
        const size_t tl = strlen(thinking_tags[t]) - 1, max = tl < upto - from ? tl : upto - from;
        for (size_t k = max; k > keep; --k)
            if (!memcmp(b + upto - k, thinking_tags[t], k)) { keep = k; break; }
    }
    return keep;
}
// Hands out raw[sp->emitted .. upto), consuming thinking delimiters before the answer.
static bool emit_upto(Conn* c, Gen* g, Splitter* sp, const char* b, size_t upto, bool final) {
    while (sp->emitted < upto) {
        size_t end = upto;
        if (g->close_tag && !sp->in_content) {
            size_t tl = 0;
            bool closing = false;
            const char* hit = find_thinking_tag(b, sp->emitted, upto, &tl, &closing);
            if (hit) end = (size_t) (hit - b);
            else if (!final) end -= thinking_tag_suffix(b, sp->emitted, upto);
            size_t keep = end;   // newlines before a possible closing tag are held back
            while (keep > sp->emitted && b[keep - 1] == '\n') --keep;
            const size_t give = (final && !hit) ? end : keep;
            const size_t from = sp->emitted;
            if (hit) { sp->in_content = closing; sp->emitted = end + tl; }
            else sp->emitted = give;
            if (!emit_text(c, g, b + from, give - from, true)) return false;
            if (!hit) return true;
            continue;
        }
        const char* t = b + sp->emitted;
        size_t n = end - sp->emitted;
        sp->emitted = end;
        if (!sp->content_started && g->close_tag) {
            while (n && *t == '\n') { ++t; --n; }
            if (!n) continue;
        }
        sp->content_started = true;
        if (!emit_text(c, g, t, n, false)) return false;
    }
    return true;
}

static uint64_t splitmix64(uint64_t* x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
typedef struct { float p; int id; } Cand;
static int cand_cmp(const void* a, const void* b) {
    const Cand *x = (const Cand*) a, *y = (const Cand*) b;
    return x->p > y->p ? -1 : x->p < y->p ? 1 : x->id - y->id;
}
// Temperature, top_p and top_k from one logits row.  Tokens below max - temp * ln(1e8) are left out (< 1e-8 each).
static int sample_row(const float* l, int V, double temp, double top_p, int top_k, uint64_t* rng) {
    static Cand* c = NULL;
    if (!c) c = (Cand*) malloc(sizeof(Cand) * (size_t) V);
    float mx = l[0];
    for (int i = 1; i < V; ++i) mx = l[i] > mx ? l[i] : mx;
    const float th = mx - (float) (temp * 18.420681);
    int n = 0;
    double tot = 0;
    for (int i = 0; i < V; ++i)
        if (l[i] >= th) { c[n].p = (float) exp(((double) l[i] - mx) / temp); c[n].id = i; tot += c[n].p; ++n; }
    qsort(c, (size_t) n, sizeof(Cand), cand_cmp);
    int keep = n;
    double cum = 0;
    for (int i = 0; i < n; ++i) { cum += c[i].p / tot; if (cum >= top_p) { keep = i + 1; break; } }
    if (top_k > 0 && keep > top_k) keep = top_k;
    double ks = 0;
    for (int i = 0; i < keep; ++i) ks += c[i].p;
    const double u = (double) (splitmix64(rng) >> 11) * (1.0 / 9007199254740992.0) * ks;
    double a = 0;
    for (int i = 0; i < keep; ++i) { a += c[i].p; if (a > u) return c[i].id; }
    return c[keep - 1].id;
}
static int gen_tokens(Gen* g, int k, int32_t* out) {
    int seq = 0;
    if (g->temperature <= 0) return eng_generate(g_eng, &seq, 1, k, ENG_MODE_AR, out, NULL);
    static float* logits = NULL;
    if (!logits) logits = (float*) malloc(sizeof(float) * (size_t) eng_vocab(g_eng));
    for (int j = 0; j < k; ++j) {
        if (eng_step(g_eng, 0, logits)) return -1;
        out[j] = sample_row(logits, eng_vocab(g_eng), g->temperature, g->top_p, g->top_k, &g->rng);
        if (out[j] == TOK_EOS || out[j] == TOK_IM_END) { for (int r = j + 1; r < k; ++r) out[r] = out[j]; return 0; }
        if (eng_push(g_eng, 0, out[j])) return -1;
    }
    return 0;
}
static bool run_generation(Conn* c, Gen* g, const int32_t* ids, int n) {
    const double t0 = now_s();
    if (eng_prefill_cached(g_eng, 0, ids, n, &g->cached_tokens)) return false;
    const double t1 = now_s();
    g->prefill_s = t1 - t0;
    g->prompt_tokens = n;
    size_t hold = 0;   // a stop string split across chunks is never streamed
    for (int i = 0; i < g->nstops; ++i) if (strlen(g->stops[i]) - 1 > hold) hold = strlen(g->stops[i]) - 1;
    int32_t* outp = (int32_t*) malloc(sizeof(int32_t) * (size_t) (g->max_tokens + CHUNK));
    int nout = 0;
    Splitter sp = {0};
    char* raw = NULL;
    size_t rawlen = 0;
    g->finish = "length";
    bool done = false, first = true;
    while (!done && nout < g->max_tokens) {
        const int k = (first ? FIRST_CHUNK : CHUNK) < g->max_tokens - nout ? (first ? FIRST_CHUNK : CHUNK) : g->max_tokens - nout;
        if (gen_tokens(g, k, outp + nout)) { free(outp); free(raw); return false; }
        if (first) { g->ttft_s = now_s() - t0; first = false; }
        for (int j = 0; j < k; ++j)
            if (outp[nout + j] == TOK_EOS || outp[nout + j] == TOK_IM_END) { nout += j; g->finish = "stop"; done = true; break; }
        if (!done) nout += k;
        free(raw);
        raw = tok_decode(g_tok, outp, nout, NULL);
        rawlen = strlen(raw);
        for (int i = 0; i < g->nstops; ++i) {
            const char* hit = strstr(raw, g->stops[i]);
            if (hit && (size_t) (hit - raw) < rawlen) { rawlen = (size_t) (hit - raw); g->finish = "stop"; done = true; }
        }
        const size_t upto = done ? rawlen : utf8_complete(raw, rawlen > hold ? rawlen - hold : 0);
        if (!emit_upto(c, g, &sp, raw, upto, done)) { g->finish = "cancelled"; done = true; }
    }
    if (!done && raw) emit_upto(c, g, &sp, raw, rawlen, true);
    flush_content(c, g, true);
    g->decode_s = now_s() - t1;
    g->completion_tokens = nout;
    free(outp);
    free(raw);
    return true;
}

static Json* usage_of(Gen* g) {
    Json* u = jobj();
    json_set(u, "prompt_tokens", jint(g->prompt_tokens));
    json_set(u, "completion_tokens", jint(g->completion_tokens));
    json_set(u, "total_tokens", jint(g->prompt_tokens + g->completion_tokens));
    Json* d = jobj();
    json_set(d, "cached_tokens", jint(g->cached_tokens));
    json_set(u, "prompt_tokens_details", d);
    return u;
}
static Json* timings_of(Gen* g) {
    const int computed = g->prompt_tokens - g->cached_tokens;
    Json* t = jobj();
    json_set(t, "prompt_n", jint(computed));
    json_set(t, "prompt_ms", jnum(g->prefill_s * 1e3));
    json_set(t, "prompt_per_second", jnum(g->prefill_s > 0 ? computed / g->prefill_s : 0));
    json_set(t, "cache_n", jint(g->cached_tokens));
    json_set(t, "predicted_n", jint(g->completion_tokens));
    json_set(t, "predicted_ms", jnum(g->decode_s * 1e3));
    json_set(t, "predicted_per_second", jnum(g->decode_s > 0 ? g->completion_tokens / g->decode_s : 0));
    json_set(t, "ttft_ms", jnum(g->ttft_s * 1e3));
    return t;
}
static void random_id(char* out, size_t cap, const char* prefix) {
    static const char* abc = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    char s[25];
    for (int i = 0; i < 24; ++i) s[i] = abc[plat_random_u64() % 62];
    s[24] = 0;
    snprintf(out, cap, "%s%s", prefix, s);
}
// OpenAI tool_calls from the parsed calls.
static Json* openai_calls(const Json* calls, bool stream) {
    Json* out = jarr();
    for (int i = 0; i < calls->n; ++i) {
        char* args = json_dumps(json_get(calls->v[i], "arguments"), 0);
        char id[64];
        random_id(id, sizeof id, "call_");
        Json* f = jobj();
        json_set(f, "name", jstr(json_gets(calls->v[i], "name")));
        json_set(f, "arguments", jstr(args));
        free(args);
        Json* tc = jobj();
        json_set(tc, "id", jstr(id));
        json_set(tc, "type", jstr("function"));
        json_set(tc, "function", f);
        if (stream) json_set(tc, "index", jint(i));
        json_push(out, tc);
    }
    return out;
}

static void handle_generate(Conn* c, const char* body, size_t body_len, bool chat) {
    char perr[256] = "";
    Json* req = json_parse(body, body_len, perr, sizeof perr);
    if (!req || req->t != J_OBJ) {
        json_free(req);
        send_error(c, 400, "body must be a JSON object");
        return;
    }
    Gen g;
    memset(&g, 0, sizeof g);
    g.chat = chat;
    g.created = (long) time(NULL);
    random_id(g.id_, sizeof g.id_, chat ? "chatcmpl-" : "cmpl-");
    g.model = g_model_id;
    const Json* nn = jfield(req, "n", J_NUM);
    if (nn && jnumber(nn) != 1) { json_free(req); send_error(c, 400, "only n=1 is supported"); return; }
    const Json* mt = jfield(req, "max_completion_tokens", J_NUM);
    if (!mt) mt = jfield(req, "max_tokens", J_NUM);
    g.max_tokens = mt ? (int) jnumber(mt) : g_ctx;
    const Json* v = jfield(req, "temperature", J_NUM);
    g.temperature = v ? jnumber(v) : 1.0;
    v = jfield(req, "top_p", J_NUM);
    g.top_p = v ? jnumber(v) : 0.95;
    if (g.top_p <= 0 || g.top_p > 1) g.top_p = 1;
    v = jfield(req, "top_k", J_NUM);
    g.top_k = v ? (int) jnumber(v) : 0;
    v = jfield(req, "seed", J_NUM);
    g.rng = v ? (uint64_t) (int64_t) jnumber(v) : plat_random_u64();
    if (g.max_tokens < 1) { json_free(req); send_error(c, 400, "max_tokens must be >= 1"); return; }
    const Json* sv = json_get(req, "stream");
    g.stream = sv && sv->t == J_TRUE;
    const Json* so = jfield(req, "stream_options", J_OBJ);
    g.include_usage = so && json_truthy(json_get(so, "include_usage"));
    const Json* stop = json_get(req, "stop");
    g.stops = (char**) calloc(stop && stop->t == J_ARR ? (size_t) stop->n + 1 : 2, sizeof(char*));
    if (stop && stop->t == J_STR && stop->s[0]) g.stops[g.nstops++] = stop->s;
    if (stop && stop->t == J_ARR)
        for (int i = 0; i < stop->n; ++i) if (stop->v[i]->t == J_STR && stop->v[i]->s[0]) g.stops[g.nstops++] = stop->v[i]->s;

    char* prompt = NULL;
    if (chat) {
        if (!json_truthy(json_get(req, "messages"))) { free(g.stops); json_free(req); send_error(c, 400, "messages is required"); return; }
        char err[512] = "";
        const char* tag = NULL;
        const Json* tools = NULL;
        prompt = chat_prompt(req, &tag, &tools, err, sizeof err);
        if (!prompt) { free(g.stops); json_free(req); send_error(c, 400, err); return; }
        g.close_tag = tag;
        g.tools = tools;
    } else {
        const Json* p = json_get(req, "prompt");
        if (p && p->t == J_ARR && p->n == 1) p = p->v[0];
        if (!p || p->t != J_STR) { free(g.stops); json_free(req); send_error(c, 400, "prompt must be a string"); return; }
        prompt = strdup(p->s);
    }
    const int cap_ids = (int) strlen(prompt) + 16;
    int32_t* ids = (int32_t*) malloc(sizeof(int32_t) * (size_t) cap_ids);
    const int n = tok_encode(g_tok, prompt, 0, ids, cap_ids);
    free(prompt);
    if (n < 1 || n >= g_ctx) {
        char m[160];
        snprintf(m, sizeof m, "prompt is %d tokens; the context is %d (--ctx)", n, g_ctx);
        free(ids); free(g.stops); json_free(req);
        send_error(c, 400, n < 1 ? "empty prompt" : m);
        return;
    }
    if (g.max_tokens > g_ctx - n - 1) g.max_tokens = g_ctx - n - 1;
    if (g.stream) {
        if (!sse_begin(c)) { free(ids); free(g.stops); json_free(req); return; }
        if (chat) {
            Json* d = jobj();
            json_set(d, "role", jstr("assistant"));
            json_set(d, "content", jstr(""));
            sse_event(c, chunk(&g, d, NULL));
        }
    }

    pthread_mutex_lock(&g_eng_lock);
    const bool ok = run_generation(c, &g, ids, n);
    pthread_mutex_unlock(&g_eng_lock);
    free(ids);
    buf_put(&g.content, "", 0);
    buf_put(&g.reasoning, "", 0);

    // tool calls in the content
    char* content = strdup(g.content.p);
    Json* calls = NULL;
    if (ok && g.tools) {
        char* vis = NULL;
        Json* parsed = NULL;
        char err[256];
        if (tc_parse(g.content.p, g.tools, &vis, &parsed, err, sizeof err) == 0) {
            free(content);
            content = vis;
            if (parsed->n) { calls = openai_calls(parsed, g.stream); g.finish = "tool_calls"; }
            json_free(parsed);
        } else if (g_verbose) fprintf(stderr, "nslm-serve: tool-call parse failed (%s); returned as text\n", err);
        if (g.stream) {   // the content held back for the envelope
            const size_t cl = strlen(content);
            if (cl > g.content_sent) stream_content(c, &g, content + g.content_sent, cl - g.content_sent);
            if (calls) {
                Json* d = jobj();
                json_set(d, "tool_calls", json_copy(calls));
                sse_event(c, chunk(&g, d, NULL));
            }
        }
    }

    if (g_verbose) {
        char ncalls[32] = "";
        if (calls) snprintf(ncalls, sizeof ncalls, " (%d calls)", calls->n);
        fprintf(stderr, "nslm-serve: prompt %5d tok (%5d cached) %7.1f ms | gen %5d tok %7.2f s = %5.1f tok/s | ttft %6.1f ms | %s%s\n",
                g.prompt_tokens, g.cached_tokens, g.prefill_s * 1e3, g.completion_tokens, g.decode_s,
                g.decode_s > 0 ? g.completion_tokens / g.decode_s : 0.0, g.ttft_s * 1e3, ok ? g.finish : "ENGINE ERROR", ncalls);
    }
    if (!ok) {
        if (g.stream) {
            Json* e = jobj();
            json_set(e, "message", jstr("engine error"));
            json_set(e, "type", jstr("server_error"));
            Json* o = jobj();
            json_set(o, "error", e);
            sse_event(c, o);
        } else send_error(c, 500, "engine error");
    } else if (strcmp(g.finish, "cancelled")) {
        if (g.stream) {
            if (chat) sse_event(c, chunk(&g, jobj(), g.finish));
            else sse_event(c, text_choice(&g, jstr(""), g.finish));
            if (g.include_usage) {
                Json* o = jobj();
                json_set(o, "id", jstr(g.id_));
                json_set(o, "object", jstr(chat ? "chat.completion.chunk" : "text_completion"));
                json_set(o, "created", jint(g.created));
                json_set(o, "model", jstr(g.model));
                json_set(o, "choices", jarr());
                json_set(o, "usage", usage_of(&g));
                json_set(o, "timings", timings_of(&g));
                sse_event(c, o);
            }
            send_all(c, "data: [DONE]\n\n", 14);
        } else if (chat) {
            Json* msg = jobj();
            json_set(msg, "role", jstr("assistant"));
            json_set(msg, "content", calls && !content[0] ? json_new(J_NULL) : jstr(content));
            if (g.reasoning.n) json_set(msg, "reasoning_content", jstr(g.reasoning.p));
            if (calls) json_set(msg, "tool_calls", json_copy(calls));
            Json* ch = jobj();
            json_set(ch, "index", jint(0));
            json_set(ch, "message", msg);
            json_set(ch, "finish_reason", jstr(g.finish));
            Json* cs = jarr();
            json_push(cs, ch);
            Json* o = jobj();
            json_set(o, "id", jstr(g.id_));
            json_set(o, "object", jstr("chat.completion"));
            json_set(o, "created", jint(g.created));
            json_set(o, "model", jstr(g.model));
            json_set(o, "choices", cs);
            json_set(o, "usage", usage_of(&g));
            json_set(o, "timings", timings_of(&g));
            send_json(c, 200, o);
        } else {
            Json* o = text_choice(&g, jstr(content), g.finish);
            json_set(o, "usage", usage_of(&g));
            json_set(o, "timings", timings_of(&g));
            send_json(c, 200, o);
        }
    }
    json_free(calls);
    free(content);
    free(g.content.p);
    free(g.reasoning.p);
    free(g.stops);
    json_free(req);
}

static void* handle_conn(void* arg) {
    Conn c = {(sock_t) (intptr_t) arg, false};
    Request r;
    const int rc = read_request(&c, &r);
    if (rc > 0) send_error(&c, rc, "bad request");
    if (rc) { sock_close(c.fd); return NULL; }
    const char* m = r.method, *p = r.path;
    char msg[600];
    if (!strcmp(m, "OPTIONS")) {
        char head[512];
        snprintf(head, sizeof head, "HTTP/1.1 204 No Content\r\n%sContent-Length: 0\r\nConnection: close\r\n\r\n", kCors);
        send_cstr(&c, head);
    } else if (!strcmp(m, "GET") && (!strcmp(p, "/v1/models") || !strcmp(p, "/models"))) {
        Json* md = jobj();
        json_set(md, "id", jstr(g_model_id));
        json_set(md, "object", jstr("model"));
        json_set(md, "created", jint(0));
        json_set(md, "owned_by", jstr("nslm"));
        Json* data = jarr();
        json_push(data, md);
        Json* o = jobj();
        json_set(o, "object", jstr("list"));
        json_set(o, "data", data);
        send_json(&c, 200, o);
    } else if (!strcmp(m, "GET") && (!strcmp(p, "/health") || !strcmp(p, "/"))) {
        Json* o = jobj();
        json_set(o, "status", jstr("ok"));
        json_set(o, "engine", jstr(eng_describe(g_eng)));
        send_json(&c, 200, o);
    } else if (!strcmp(m, "POST") && (!strcmp(p, "/v1/chat/completions") || !strcmp(p, "/v1/completions") ||
                                      !strcmp(p, "/chat/completions") || !strcmp(p, "/completions"))) {
        const size_t pl = strlen(p);
        handle_generate(&c, r.body, r.body_len, pl >= 16 && !strcmp(p + pl - 16, "chat/completions"));
    } else {
        snprintf(msg, sizeof msg, "no route %s %s", m, p);
        send_error(&c, 404, msg);
    }
    free(r.body);
    sock_close(c.fd);
    return NULL;
}

// --render: the prompt of a chat request file, for template checks.
static int render_file(const char* path) {
    size_t len = 0;
    char* d = plat_slurp(path, &len);
    if (!d) { fprintf(stderr, "cannot read %s\n", path); return 2; }
    Json* req = json_parse(d, len, NULL, 0);
    free(d);
    if (!req || req->t != J_OBJ) { fprintf(stderr, "bad JSON in %s\n", path); return 2; }
    char err[512] = "";
    const char* tag = NULL;
    const Json* tools = NULL;
    char* p = chat_prompt(req, &tag, &tools, err, sizeof err);
    json_free(req);
    if (!p) { fprintf(stderr, "%s\n", err); return 1; }
    fputs(p, stdout);
    free(p);
    return 0;
}

// The last component of a path (either separator), without trailing separators.
static void base_name(const char* path, char* out, size_t cap) {
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s", path);
    size_t n = strlen(tmp);
    while (n > 1 && (tmp[n - 1] == '/' || tmp[n - 1] == '\\')) tmp[--n] = 0;
    const char* b = tmp;
    for (const char* q = tmp; *q; ++q) if (*q == '/' || *q == '\\') b = q + 1;
    snprintf(out, cap, "%s", b);
}

int main(int argc, char** argv) {
    plat_init(&argc, &argv);
    if (opt(argc, argv, "--render", NULL)) return render_file(opt(argc, argv, "--render", NULL));
    const char *model = opt(argc, argv, "--model", NULL), *host = opt(argc, argv, "--host", "127.0.0.1");
    const int port = atoi(opt(argc, argv, "--port", "8080"));
    g_ctx = atoi(opt(argc, argv, "--ctx", "65536"));
    if (opt_flag(argc, argv, "--quiet")) g_verbose = 0;
    if (!model || g_ctx < 64) {
        fprintf(stderr, "usage: nslm-serve --model DIR [--res out/res] [--host 127.0.0.1] [--port 8080] "
                        "[--ctx 65536] [--kv bf16|q8] [--model-id ID] [--quiet]\n       nslm-serve --render REQUEST.json\n");
        return 2;
    }
    char base[256];
    base_name(model, base, sizeof base);
    snprintf(g_model_id, sizeof g_model_id, "%s", opt(argc, argv, "--model-id", base));
    char res[1100];
    {   // default resources: ../res next to the binary
        snprintf(res, sizeof res, "%s", argv[0]);
        char* sl = NULL;
        for (char* q = res; *q; ++q) if (*q == '/' || *q == '\\') sl = q;
        if (sl) snprintf(sl, sizeof res - (size_t) (sl - res), "/../res");
        else snprintf(res, sizeof res, "../res");
    }
    EngOpts o;
    memset(&o, 0, sizeof o);
    o.model_dir = model;
    o.resource_dir = opt(argc, argv, "--res", res);
    o.max_seqs = 1;
    o.kv_tokens = g_ctx;
    o.kv_format = !strcmp(opt(argc, argv, "--kv", "bf16"), "q8") ? ENG_KV_Q8 : ENG_KV_BF16;

    char err[512] = "", tpath[1100];
    snprintf(tpath, sizeof tpath, "%s/tokenizer.json", model);
    g_tok = tok_open(tpath, err, sizeof err);
    if (!g_tok) { fprintf(stderr, "tokenizer: %s\n", err); return 1; }
    const double t0 = now_s();
    g_eng = eng_open(&o, err, sizeof err);
    if (!g_eng) { fprintf(stderr, "engine: %s\n", err); return 1; }
    EngMem m;
    eng_mem(g_eng, &m);
    fprintf(stderr, "nslm-serve: loaded in %.1f s: %s; weights %.2f GB, KV %.2f GB, GPU %.2f GB\n", now_s() - t0,
            eng_describe(g_eng), (m.weights + m.lut) / 1e9, m.kv / 1e9, m.gpu_allocated / 1e9);
    {   // build the pipelines before the first request
        int32_t w[64], out[8];
        const int n = tok_encode(g_tok, "<|ifm|begin_of_text|><|ifm|im_start|>user\nHi<|ifm|im_end|><|ifm|im_start|>assistant\n", 0, w, 64);
        int seq = 0;
        if (eng_prefill(g_eng, 0, w, n) || eng_generate(g_eng, &seq, 1, 8, ENG_MODE_AR, out, NULL)) {
            fprintf(stderr, "engine: warm-up failed\n");
            return 1;
        }
    }
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa)) { fprintf(stderr, "nslm-serve: WSAStartup failed\n"); return 1; }
#else
    signal(SIGPIPE, SIG_IGN);
#endif
    const sock_t ls = socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*) &one, sizeof one);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) { fprintf(stderr, "bad --host %s\n", host); return 2; }
    if (bind(ls, (struct sockaddr*) &addr, sizeof addr) || listen(ls, 64)) { fprintf(stderr, "nslm-serve: cannot listen on %s:%d\n", host, port); return 1; }
    fprintf(stderr, "nslm-serve: http://%s:%d/v1, model \"%s\", context %d tokens\n", host, port, g_model_id, g_ctx);
    for (;;) {
        const sock_t fd = accept(ls, NULL, NULL);
        if (fd == INVALID_SOCKET) continue;
#ifdef SO_NOSIGPIPE
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*) &one, sizeof one);
        pthread_t th;
        if (pthread_create(&th, NULL, handle_conn, (void*) (intptr_t) fd) == 0) pthread_detach(th);
        else sock_close(fd);
    }
}
