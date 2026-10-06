// harness/nslm-serve.m - OpenAI-compatible HTTP server for K2-Horizon-MoVA on the nslm engine.
//
//   nslm-serve --model DIR [--res out/res] [--host 127.0.0.1] [--port 8080]
//              [--ctx 4096] [--model-id ID] [--quiet]
//   nslm-serve --render REQUEST.json     print the prompt a chat request renders to, and exit
//
// GET /v1/models, GET /health, POST /v1/chat/completions, POST /v1/completions (stream or not).
// Chat prompts: harness/chat_template.c (the model's chat_template.jinja, tools included).  Tool calls in the output
// come back as OpenAI tool_calls (harness/tool_calls.c).  The thinking span is returned as reasoning_content.
// Sampling defaults: IFM's model card (temperature 1.0, top_p 0.95); temperature 0 is greedy.
// One request at a time on one sequence slot; the KV cache of the previous request's common prefix is reused.
#import <Foundation/Foundation.h>

#include <arpa/inet.h>
#include <math.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "chat_template.h"
#include "engine_api.h"
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
static int g_ctx = 4096;
static int g_verbose = 1;

static const char* opt(int argc, char** argv, const char* name, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static int opt_flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], name)) return 1;
    return 0;
}
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

// ---- HTTP ----

typedef struct {
    int fd;
    bool dead;   // the client went away: stop generating
} Conn;

static bool send_all(Conn* c, const void* p, size_t n) {
    if (c->dead) return false;
    const char* b = (const char*) p;
    while (n) {
        const ssize_t w = send(c->fd, b, n, 0);
        if (w <= 0) { c->dead = true; return false; }
        b += w;
        n -= (size_t) w;
    }
    return true;
}
static bool send_str(Conn* c, NSString* s) {
    NSData* d = [s dataUsingEncoding:NSUTF8StringEncoding];
    return send_all(c, d.bytes, d.length);
}
static const char* kCors = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\n"
                           "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
static void send_json(Conn* c, int status, id obj) {
    NSData* body = [NSJSONSerialization dataWithJSONObject:obj options:NSJSONWritingWithoutEscapingSlashes error:nil];
    const char* reason = status == 200 ? "OK" : status == 400 ? "Bad Request" : status == 404 ? "Not Found"
                       : status == 413 ? "Payload Too Large" : "Internal Server Error";
    NSString* head = [NSString stringWithFormat:@"HTTP/1.1 %d %s\r\nContent-Type: application/json\r\n"
                                                 "Content-Length: %lu\r\n%sConnection: close\r\n\r\n",
                                                status, reason, (unsigned long) body.length, kCors];
    send_str(c, head);
    send_all(c, body.bytes, body.length);
}
static void send_error(Conn* c, int status, NSString* msg) {
    send_json(c, status, @{@"error": @{@"message": msg, @"type": status == 500 ? @"server_error" : @"invalid_request_error"}});
}
static bool sse_begin(Conn* c) {
    NSString* head = [NSString stringWithFormat:@"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                                                 "Cache-Control: no-cache\r\n%sConnection: close\r\n\r\n", kCors];
    return send_str(c, head);
}
static bool sse_event(Conn* c, id obj) {
    NSData* d = [NSJSONSerialization dataWithJSONObject:obj options:NSJSONWritingWithoutEscapingSlashes error:nil];
    return send_all(c, "data: ", 6) && send_all(c, d.bytes, d.length) && send_all(c, "\n\n", 2);
}
// One request.  0 with method / path / body, -1 on a closed connection, or an HTTP status to answer with.
static int read_request(Conn* c, NSString** method, NSString** path, NSData** body) {
    NSMutableData* buf = [NSMutableData data];
    char tmp[65536];
    NSRange end = {NSNotFound, 0};
    NSData* sep = [NSData dataWithBytes:"\r\n\r\n" length:4];
    while (end.location == NSNotFound) {
        const ssize_t r = recv(c->fd, tmp, sizeof tmp, 0);
        if (r <= 0) return -1;
        [buf appendBytes:tmp length:(NSUInteger) r];
        if (buf.length > (1 << 20)) return 413;
        end = [buf rangeOfData:sep options:0 range:NSMakeRange(0, buf.length)];
    }
    NSString* head = [[NSString alloc] initWithData:[buf subdataWithRange:NSMakeRange(0, end.location)]
                                           encoding:NSUTF8StringEncoding];
    if (!head) return 400;
    NSArray* lines = [head componentsSeparatedByString:@"\r\n"];
    NSArray* rl = [lines[0] componentsSeparatedByString:@" "];
    if (rl.count < 2) return 400;
    *method = rl[0];
    NSString* p = rl[1];
    const NSRange q = [p rangeOfString:@"?"];
    *path = q.location == NSNotFound ? p : [p substringToIndex:q.location];
    long len = 0;
    bool expect = false;
    for (NSUInteger i = 1; i < lines.count; ++i) {
        NSString* l = lines[i];
        const NSRange colon = [l rangeOfString:@":"];
        if (colon.location == NSNotFound) continue;
        NSString* k = [[l substringToIndex:colon.location] lowercaseString];
        NSString* v = [[l substringFromIndex:colon.location + 1] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
        if ([k isEqualToString:@"content-length"]) len = v.integerValue;
        if ([k isEqualToString:@"expect"] && [v.lowercaseString isEqualToString:@"100-continue"]) expect = true;
    }
    if (len < 0 || len > MAX_BODY) return 413;
    NSMutableData* b = [[buf subdataWithRange:NSMakeRange(end.location + 4, buf.length - end.location - 4)] mutableCopy];
    if (expect && (long) b.length < len) send_all(c, "HTTP/1.1 100 Continue\r\n\r\n", 25);
    while ((long) b.length < len) {
        const ssize_t r = recv(c->fd, tmp, sizeof tmp, 0);
        if (r <= 0) return -1;
        [b appendBytes:tmp length:(NSUInteger) r];
    }
    if ((long) b.length > len) [b setLength:(NSUInteger) len];
    *body = b;
    return 0;
}

// ---- prompt ----

static NSString* str_or(id v, NSString* def) { return [v isKindOfClass:NSString.class] ? v : def; }

// The chat prompt for a request (ordered JSON), or nil with *err.  *tools_out: the tools in effect (or NULL).
static NSString* chat_prompt(Json* req, NSDictionary* r, NSString** close_tag, Json** tools_out, NSString** err) {
    NSDictionary* kw = [r[@"chat_template_kwargs"] isKindOfClass:NSDictionary.class] ? r[@"chat_template_kwargs"] : @{};
    NSString* effort = str_or(r[@"reasoning_effort"], nil);
    if ([r[@"reasoning"] isKindOfClass:NSDictionary.class]) effort = str_or(r[@"reasoning"][@"effort"], effort);
    effort = str_or(kw[@"reasoning_effort"], effort ?: @"high");
    if ([effort isEqualToString:@"minimal"]) effort = @"low";
    Json* messages = json_get(req, "messages");
    char e[512] = "";
    if (ct_normalize(messages, e, sizeof e)) { *err = @(e); return nil; }
    Json* tools = json_get(req, "tools");
    if ([r[@"tool_choice"] isKindOfClass:NSString.class] && [r[@"tool_choice"] isEqualToString:@"none"]) tools = NULL;
    CtOpts o = {str_or(kw[@"tool_presentation_format"], nil).UTF8String, str_or(kw[@"tool_call_format"], nil).UTF8String,
                effort.UTF8String, 1};
    char* p = ct_render(messages, tools, &o, e, sizeof e);
    if (!p) { *err = @(e); return nil; }
    NSString* s = @(p);
    free(p);
    *close_tag = [effort isEqualToString:@"low"] ? @"</ifm|think_faster>" : [effort isEqualToString:@"medium"] ? @"</ifm|think_fast>" : @"</ifm|think>";
    *tools_out = json_truthy(tools) ? tools : NULL;
    return s;
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
    NSArray<NSString*>* stops;
    NSString* close_tag;            // the prompt's "</ifm|think...>" for chat (any of the three closes); nil for raw completions
    const Json* tools;              // tools in effect: content is scanned for tool calls
    NSString *id_, *model;
    long created;
    NSMutableString *reasoning, *content;   // content: everything after the thinking span, tool markup included
    size_t content_sent;                     // bytes of content already streamed
    NSString* finish;
    int prompt_tokens, completion_tokens, cached_tokens;
    double prefill_s, decode_s, ttft_s;
} Gen;

typedef struct {
    size_t emitted;
    bool in_content, content_started;
} Splitter;

static NSDictionary* chunk(Gen* g, NSDictionary* delta, id finish) {
    return @{@"id": g->id_, @"object": @"chat.completion.chunk", @"created": @(g->created), @"model": g->model,
             @"choices": @[@{@"index": @0, @"delta": delta, @"finish_reason": finish ?: NSNull.null}]};
}
static bool stream_content(Conn* c, Gen* g, NSString* t) {
    if (!t.length || !g->stream) return true;
    if (!g->chat)
        return sse_event(c, @{@"id": g->id_, @"object": @"text_completion", @"created": @(g->created), @"model": g->model,
                              @"choices": @[@{@"index": @0, @"text": t, @"finish_reason": NSNull.null}]});
    return sse_event(c, chunk(g, @{@"content": t}, nil));
}
// Streams the content that cannot belong to a tool-call envelope.
static bool flush_content(Conn* c, Gen* g, bool final) {
    NSData* all = [g->content dataUsingEncoding:NSUTF8StringEncoding];
    const char* b = (const char*) all.bytes;
    size_t safe = all.length;
    if (g->tools && !final) {
        const char* hit = strstr(b, TC_OPEN);
        if (hit) safe = (size_t) (hit - b);
        else {
            const size_t ol = strlen(TC_OPEN);
            for (size_t k = ol - 1; k > 0; --k)
                if (all.length >= k && !memcmp(b + all.length - k, TC_OPEN, k)) { safe = all.length - k; break; }
        }
    }
    if (g->tools && final) return true;   // the end of the request parses and sends the rest
    safe = utf8_complete(b, safe);
    if (safe <= g->content_sent) return true;
    NSString* t = [[NSString alloc] initWithBytes:b + g->content_sent length:safe - g->content_sent encoding:NSUTF8StringEncoding] ?: @"";
    g->content_sent = safe;
    return stream_content(c, g, t);
}
static bool emit_text(Conn* c, Gen* g, NSString* text, bool reasoning) {
    if (!text.length) return true;
    if (reasoning) {
        [g->reasoning appendString:text];
        return !g->stream || sse_event(c, chunk(g, @{@"reasoning_content": text}, nil));
    }
    [g->content appendString:text];
    return flush_content(c, g, false);
}
// The first closing think tag in b[from, upto): the model may close with any of the three, whatever the prompt opened.
static const char* find_close(const char* b, size_t from, size_t upto, size_t* tl) {
    static const char* tags[] = {"</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"};
    for (size_t i = from; i < upto; ++i)
        for (int t = 0; t < 3; ++t) {
            const size_t l = strlen(tags[t]);
            if (i + l <= upto && !memcmp(b + i, tags[t], l)) { *tl = l; return b + i; }
        }
    return NULL;
}
// Hands out raw[sp->emitted .. upto), split at the closing think tag.
static bool emit_upto(Conn* c, Gen* g, Splitter* sp, NSData* raw, size_t upto, bool final) {
    while (sp->emitted < upto) {
        const char* b = (const char*) raw.bytes;
        size_t end = upto;
        if (g->close_tag && !sp->in_content) {
            size_t tl = 0;
            const char* hit = find_close(b, sp->emitted, upto, &tl);
            if (hit) end = (size_t) (hit - b);
            size_t keep = end;   // newlines before a possible closing tag are held back
            while (keep > sp->emitted && b[keep - 1] == '\n') --keep;
            const size_t give = (final && !hit) ? end : keep;
            NSString* t = [[NSString alloc] initWithBytes:b + sp->emitted length:give - sp->emitted encoding:NSUTF8StringEncoding] ?: @"";
            if (hit) { sp->in_content = true; sp->emitted = end + tl; }
            else sp->emitted = give;
            if (!emit_text(c, g, t, true)) return false;
            if (!hit) return true;
            continue;
        }
        NSString* t = [[NSString alloc] initWithBytes:b + sp->emitted length:end - sp->emitted encoding:NSUTF8StringEncoding] ?: @"";
        sp->emitted = end;
        if (!sp->content_started && g->close_tag) {
            NSUInteger i = 0;
            while (i < t.length && [t characterAtIndex:i] == '\n') ++i;
            t = [t substringFromIndex:i];
            if (!t.length) continue;
        }
        sp->content_started = true;
        if (!emit_text(c, g, t, false)) return false;
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
static BOOL run_generation(Conn* c, Gen* g, const int32_t* ids, int n) {
    const double t0 = now_s();
    if (eng_prefill_cached(g_eng, 0, ids, n, &g->cached_tokens)) return NO;
    const double t1 = now_s();
    g->prefill_s = t1 - t0;
    g->prompt_tokens = n;
    size_t hold = 0;   // a stop string split across chunks is never streamed
    for (NSString* s in g->stops) hold = MAX(hold, strlen(s.UTF8String) - 1);
    int32_t* outp = (int32_t*) malloc(sizeof(int32_t) * (size_t) (g->max_tokens + CHUNK));
    int nout = 0;
    Splitter sp = {0};
    NSData* raw = [NSData data];
    g->finish = @"length";
    bool done = false, first = true;
    while (!done && nout < g->max_tokens) {
        const int k = MIN(first ? FIRST_CHUNK : CHUNK, g->max_tokens - nout);
        if (gen_tokens(g, k, outp + nout)) { free(outp); return NO; }
        if (first) { g->ttft_s = now_s() - t0; first = false; }
        for (int j = 0; j < k; ++j)
            if (outp[nout + j] == TOK_EOS || outp[nout + j] == TOK_IM_END) { nout += j; g->finish = @"stop"; done = true; break; }
        if (!done) nout += k;
        @autoreleasepool {
            char* txt = tok_decode(g_tok, outp, nout, NULL);
            size_t len = strlen(txt);
            for (NSString* s in g->stops) {
                const char* hit = strstr(txt, s.UTF8String);
                if (hit && (size_t) (hit - txt) < len) { len = (size_t) (hit - txt); g->finish = @"stop"; done = true; }
            }
            raw = [NSData dataWithBytes:txt length:len];
            free(txt);
            size_t upto = done ? len : utf8_complete(raw.bytes, len > hold ? len - hold : 0);
            if (!emit_upto(c, g, &sp, raw, upto, done)) { g->finish = @"cancelled"; done = true; }
        }
    }
    if (!done) emit_upto(c, g, &sp, raw, raw.length, true);
    flush_content(c, g, true);
    g->decode_s = now_s() - t1;
    g->completion_tokens = nout;
    free(outp);
    return YES;
}

static NSDictionary* usage_of(Gen* g) {
    return @{@"prompt_tokens": @(g->prompt_tokens), @"completion_tokens": @(g->completion_tokens),
             @"total_tokens": @(g->prompt_tokens + g->completion_tokens),
             @"prompt_tokens_details": @{@"cached_tokens": @(g->cached_tokens)}};
}
static NSDictionary* timings_of(Gen* g) {
    const int computed = g->prompt_tokens - g->cached_tokens;
    return @{@"prompt_n": @(computed), @"prompt_ms": @(g->prefill_s * 1e3),
             @"prompt_per_second": @(g->prefill_s > 0 ? computed / g->prefill_s : 0), @"cache_n": @(g->cached_tokens),
             @"predicted_n": @(g->completion_tokens), @"predicted_ms": @(g->decode_s * 1e3),
             @"predicted_per_second": @(g->decode_s > 0 ? g->completion_tokens / g->decode_s : 0), @"ttft_ms": @(g->ttft_s * 1e3)};
}
static NSString* random_id(NSString* prefix) {
    static const char* abc = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    char s[25];
    for (int i = 0; i < 24; ++i) s[i] = abc[arc4random_uniform(62)];
    s[24] = 0;
    return [NSString stringWithFormat:@"%@%s", prefix, s];
}
// OpenAI tool_calls from the parsed calls.
static NSArray* openai_calls(const Json* calls, bool stream) {
    NSMutableArray* out = [NSMutableArray array];
    for (int i = 0; i < calls->n; ++i) {
        char* args = json_dumps(json_get(calls->v[i], "arguments"), 0);
        NSMutableDictionary* tc = [@{@"id": random_id(@"call_"), @"type": @"function",
                                     @"function": @{@"name": @(json_gets(calls->v[i], "name")), @"arguments": @(args)}} mutableCopy];
        if (stream) tc[@"index"] = @(i);
        free(args);
        [out addObject:tc];
    }
    return out;
}

static void handle_generate(Conn* c, NSData* body, bool chat) {
    NSDictionary* r = [NSJSONSerialization JSONObjectWithData:body options:0 error:nil];
    char perr[256] = "";
    Json* req = json_parse(body.bytes, body.length, perr, sizeof perr);
    if (![r isKindOfClass:NSDictionary.class] || !req || req->t != J_OBJ) {
        json_free(req);
        send_error(c, 400, @"body must be a JSON object");
        return;
    }
    Gen g = {0};
    g.chat = chat;
    g.created = (long) time(NULL);
    g.id_ = random_id(chat ? @"chatcmpl-" : @"cmpl-");
    g.model = @(g_model_id);
    if ([r[@"n"] isKindOfClass:NSNumber.class] && [r[@"n"] intValue] != 1) { json_free(req); send_error(c, 400, @"only n=1 is supported"); return; }
    id mt = r[@"max_completion_tokens"] ?: r[@"max_tokens"];
    g.max_tokens = [mt isKindOfClass:NSNumber.class] ? [mt intValue] : g_ctx;
    g.temperature = [r[@"temperature"] isKindOfClass:NSNumber.class] ? [r[@"temperature"] doubleValue] : 1.0;
    g.top_p = [r[@"top_p"] isKindOfClass:NSNumber.class] ? [r[@"top_p"] doubleValue] : 0.95;
    if (g.top_p <= 0 || g.top_p > 1) g.top_p = 1;
    g.top_k = [r[@"top_k"] isKindOfClass:NSNumber.class] ? [r[@"top_k"] intValue] : 0;
    g.rng = [r[@"seed"] isKindOfClass:NSNumber.class] ? (uint64_t) [r[@"seed"] longLongValue] : ((uint64_t) arc4random() << 32 | arc4random());
    if (g.max_tokens < 1) { json_free(req); send_error(c, 400, @"max_tokens must be >= 1"); return; }
    g.stream = [r[@"stream"] isKindOfClass:NSNumber.class] && [r[@"stream"] boolValue];
    NSDictionary* so = [r[@"stream_options"] isKindOfClass:NSDictionary.class] ? r[@"stream_options"] : nil;
    g.include_usage = so && [so[@"include_usage"] boolValue];
    NSMutableArray* stops = [NSMutableArray array];
    id stop = r[@"stop"];
    if ([stop isKindOfClass:NSString.class] && [stop length]) [stops addObject:stop];
    if ([stop isKindOfClass:NSArray.class])
        for (id s in (NSArray*) stop) if ([s isKindOfClass:NSString.class] && [s length]) [stops addObject:s];
    g.stops = stops;
    g.reasoning = [NSMutableString string];
    g.content = [NSMutableString string];

    NSString* prompt;
    if (chat) {
        if (!json_truthy(json_get(req, "messages"))) { json_free(req); send_error(c, 400, @"messages is required"); return; }
        NSString *err = nil, *tag = nil;
        Json* tools = NULL;
        prompt = chat_prompt(req, r, &tag, &tools, &err);
        if (!prompt) { json_free(req); send_error(c, 400, err); return; }
        g.close_tag = tag;
        g.tools = tools;
    } else {
        id p = r[@"prompt"];
        if ([p isKindOfClass:NSArray.class] && [p count] == 1) p = [p firstObject];
        if (![p isKindOfClass:NSString.class]) { json_free(req); send_error(c, 400, @"prompt must be a string"); return; }
        prompt = p;
    }
    const int cap_ids = (int) strlen(prompt.UTF8String) + 16;
    int32_t* ids = (int32_t*) malloc(sizeof(int32_t) * (size_t) cap_ids);
    const int n = tok_encode(g_tok, prompt.UTF8String, 0, ids, cap_ids);
    if (n < 1 || n >= g_ctx) {
        free(ids);
        json_free(req);
        send_error(c, 400, n < 1 ? @"empty prompt" : [NSString stringWithFormat:@"prompt is %d tokens; the context is %d (--ctx)", n, g_ctx]);
        return;
    }
    if (g.max_tokens > g_ctx - n - 1) g.max_tokens = g_ctx - n - 1;
    if (g.stream) {
        if (!sse_begin(c)) { free(ids); json_free(req); return; }
        if (chat) sse_event(c, chunk(&g, @{@"role": @"assistant", @"content": @""}, nil));
    }

    pthread_mutex_lock(&g_eng_lock);
    const BOOL ok = run_generation(c, &g, ids, n);
    pthread_mutex_unlock(&g_eng_lock);
    free(ids);

    // tool calls in the content
    NSString* content = g.content;
    NSArray* calls = nil;
    if (ok && g.tools) {
        char* vis = NULL;
        Json* parsed = NULL;
        char err[256];
        if (tc_parse(g.content.UTF8String, g.tools, &vis, &parsed, err, sizeof err) == 0) {
            content = @(vis);
            if (parsed->n) { calls = openai_calls(parsed, g.stream); g.finish = @"tool_calls"; }
            free(vis);
            json_free(parsed);
        } else if (g_verbose) fprintf(stderr, "nslm-serve: tool-call parse failed (%s); returned as text\n", err);
        if (g.stream) {   // the content held back for the envelope
            NSData* d = [content dataUsingEncoding:NSUTF8StringEncoding];
            if (d.length > g.content_sent)
                stream_content(c, &g, [[NSString alloc] initWithBytes:(const char*) d.bytes + g.content_sent
                                                               length:d.length - g.content_sent encoding:NSUTF8StringEncoding]);
            if (calls) sse_event(c, chunk(&g, @{@"tool_calls": calls}, nil));
        }
    }
    json_free(req);

    if (g_verbose)
        fprintf(stderr, "nslm-serve: prompt %5d tok (%5d cached) %7.1f ms | gen %5d tok %7.2f s = %5.1f tok/s | ttft %6.1f ms | %s%s\n",
                g.prompt_tokens, g.cached_tokens, g.prefill_s * 1e3, g.completion_tokens, g.decode_s,
                g.decode_s > 0 ? g.completion_tokens / g.decode_s : 0.0, g.ttft_s * 1e3, ok ? g.finish.UTF8String : "ENGINE ERROR",
                calls ? [NSString stringWithFormat:@" (%lu calls)", (unsigned long) calls.count].UTF8String : "");
    if (!ok) {
        if (g.stream) sse_event(c, @{@"error": @{@"message": @"engine error", @"type": @"server_error"}});
        else send_error(c, 500, @"engine error");
        return;
    }
    if ([g.finish isEqualToString:@"cancelled"]) return;
    if (g.stream) {
        if (chat) sse_event(c, chunk(&g, @{}, g.finish));
        else sse_event(c, @{@"id": g.id_, @"object": @"text_completion", @"created": @(g.created), @"model": g.model,
                            @"choices": @[@{@"index": @0, @"text": @"", @"finish_reason": g.finish}]});
        if (g.include_usage)
            sse_event(c, @{@"id": g.id_, @"object": chat ? @"chat.completion.chunk" : @"text_completion", @"created": @(g.created),
                           @"model": g.model, @"choices": @[], @"usage": usage_of(&g), @"timings": timings_of(&g)});
        send_all(c, "data: [DONE]\n\n", 14);
        return;
    }
    if (chat) {
        NSMutableDictionary* msg = [@{@"role": @"assistant", @"content": (calls && !content.length) ? (id) NSNull.null : content} mutableCopy];
        if (g.reasoning.length) msg[@"reasoning_content"] = g.reasoning;
        if (calls) msg[@"tool_calls"] = calls;
        send_json(c, 200, @{@"id": g.id_, @"object": @"chat.completion", @"created": @(g.created), @"model": g.model,
                            @"choices": @[@{@"index": @0, @"message": msg, @"finish_reason": g.finish}],
                            @"usage": usage_of(&g), @"timings": timings_of(&g)});
    } else {
        send_json(c, 200, @{@"id": g.id_, @"object": @"text_completion", @"created": @(g.created), @"model": g.model,
                            @"choices": @[@{@"index": @0, @"text": content, @"finish_reason": g.finish}],
                            @"usage": usage_of(&g), @"timings": timings_of(&g)});
    }
}

static void handle_conn(int fd) {
    @autoreleasepool {
        Conn c = {fd, false};
        NSString *method = nil, *path = nil;
        NSData* body = nil;
        const int rc = read_request(&c, &method, &path, &body);
        if (rc > 0) send_error(&c, rc, @"bad request");
        if (rc) { close(fd); return; }
        if ([method isEqualToString:@"OPTIONS"]) {
            send_str(&c, [NSString stringWithFormat:@"HTTP/1.1 204 No Content\r\n%sContent-Length: 0\r\nConnection: close\r\n\r\n", kCors]);
        } else if ([method isEqualToString:@"GET"] && ([path isEqualToString:@"/v1/models"] || [path isEqualToString:@"/models"])) {
            send_json(&c, 200, @{@"object": @"list", @"data": @[@{@"id": @(g_model_id), @"object": @"model", @"created": @0,
                                                                  @"owned_by": @"nslm"}]});
        } else if ([method isEqualToString:@"GET"] && ([path isEqualToString:@"/health"] || [path isEqualToString:@"/"])) {
            send_json(&c, 200, @{@"status": @"ok", @"engine": @(eng_describe(g_eng))});
        } else if ([method isEqualToString:@"POST"] &&
                   ([path isEqualToString:@"/v1/chat/completions"] || [path isEqualToString:@"/v1/completions"] ||
                    [path isEqualToString:@"/chat/completions"] || [path isEqualToString:@"/completions"])) {
            handle_generate(&c, body, [path hasSuffix:@"chat/completions"]);
        } else {
            send_error(&c, 404, [NSString stringWithFormat:@"no route %@ %@", method, path]);
        }
        close(fd);
    }
}

// --render: the prompt of a chat request file, for template checks.
static int render_file(const char* path) {
    NSData* d = [NSData dataWithContentsOfFile:@(path)];
    if (!d) { fprintf(stderr, "cannot read %s\n", path); return 2; }
    Json* req = json_parse(d.bytes, d.length, NULL, 0);
    NSDictionary* r = [NSJSONSerialization JSONObjectWithData:d options:0 error:nil];
    if (!req || ![r isKindOfClass:NSDictionary.class]) { fprintf(stderr, "bad JSON in %s\n", path); return 2; }
    NSString *err = nil, *tag = nil;
    Json* tools = NULL;
    NSString* p = chat_prompt(req, r, &tag, &tools, &err);
    json_free(req);
    if (!p) { fprintf(stderr, "%s\n", err.UTF8String); return 1; }
    fputs(p.UTF8String, stdout);
    return 0;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        if (opt(argc, argv, "--render", NULL)) return render_file(opt(argc, argv, "--render", NULL));
        const char *model = opt(argc, argv, "--model", NULL), *host = opt(argc, argv, "--host", "127.0.0.1");
        const int port = atoi(opt(argc, argv, "--port", "8080"));
        g_ctx = atoi(opt(argc, argv, "--ctx", "4096"));
        if (opt_flag(argc, argv, "--quiet")) g_verbose = 0;
        if (!model || g_ctx < 64) {
            fprintf(stderr, "usage: nslm-serve --model DIR [--res out/res] [--host 127.0.0.1] [--port 8080] "
                            "[--ctx 4096] [--model-id ID] [--quiet]\n       nslm-serve --render REQUEST.json\n");
            return 2;
        }
        NSString* dir = [@(model) stringByStandardizingPath];
        snprintf(g_model_id, sizeof g_model_id, "%s", opt(argc, argv, "--model-id", dir.lastPathComponent.UTF8String));
        EngOpts o;
        memset(&o, 0, sizeof o);
        o.model_dir = dir.UTF8String;
        o.resource_dir = opt(argc, argv, "--res", NULL) ?: [[@(argv[0]) stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"../res"].UTF8String;
        o.max_seqs = 1;
        o.kv_tokens = g_ctx;

        char err[512] = "";
        g_tok = tok_open([dir stringByAppendingPathComponent:@"tokenizer.json"].UTF8String, err, sizeof err);
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
        signal(SIGPIPE, SIG_IGN);
        const int ls = socket(AF_INET, SOCK_STREAM, 0);
        const int one = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        struct sockaddr_in addr = {0};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t) port);
        if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) { fprintf(stderr, "bad --host %s\n", host); return 2; }
        if (bind(ls, (struct sockaddr*) &addr, sizeof addr) || listen(ls, 64)) { perror("nslm-serve: bind/listen"); return 1; }
        fprintf(stderr, "nslm-serve: http://%s:%d/v1, model \"%s\", context %d tokens\n", host, port, g_model_id, g_ctx);
        for (;;) {
            const int fd = accept(ls, NULL, NULL);
            if (fd < 0) continue;
            setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{ handle_conn(fd); });
        }
    }
}
