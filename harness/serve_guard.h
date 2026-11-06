// harness/serve_guard.h - who may use nslm-serve. With an API key, each request needs "Authorization: Bearer KEY". Without
// one, the server listens on loopback only and takes only a loopback Host (DNS rebinding). A browser Origin needs --allow-origin.
#pragma once
#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    const char* api_key;        // NULL: none (then loopback only)
    const char* allow_origin;   // NULL: no browser origin; "*": any; else one origin, as "http://localhost:3000"
} ServeGuard;

static inline int guard_ieq(const char* a, const char* b, size_t n) {   // the first n bytes, case-insensitive
    for (size_t i = 0; i < n; ++i) {
        const char x = a[i] >= 'A' && a[i] <= 'Z' ? (char) (a[i] + 32) : a[i], y = b[i] >= 'A' && b[i] <= 'Z' ? (char) (b[i] + 32) : b[i];
        if (x != y) return 0;
    }
    return 1;
}
// A Host header that names this machine: localhost, 127.0.0.1 or [::1], with or without ":port". Empty: no header
static inline int guard_loopback_host(const char* h) {
    static const char* names[] = {"localhost", "127.0.0.1", "[::1]"};
    if (!h || !*h) return 1;
    for (int i = 0; i < 3; ++i) {
        const size_t n = strlen(names[i]);
        if (strlen(h) >= n && guard_ieq(h, names[i], n) && (h[n] == 0 || h[n] == ':')) {
            const char* p = h + n + (h[n] == ':');
            while (*p >= '0' && *p <= '9') ++p;
            if (!*p) return 1;
        }
    }
    return 0;
}
static inline int guard_loopback_ip(const char* ip) { return !strncmp(ip, "127.", 4); }   // the --host to bind
static inline int guard_origin_ok(const ServeGuard* g, const char* origin) {
    return !origin || !*origin || (g->allow_origin && (!strcmp(g->allow_origin, "*") || !strcmp(g->allow_origin, origin)));
}
static inline int guard_key_ok(const char* key, const char* auth) {   // "Bearer KEY", compared in constant time
    if (!auth || strncmp(auth, "Bearer ", 7)) return 0;
    const char* a = auth + 7;
    const size_t n = strlen(key), m = strlen(a);
    unsigned char d = (unsigned char) (n != m);
    for (size_t i = 0; i < n; ++i) d |= (unsigned char) (key[i] ^ a[i < m ? i : 0]);
    return d == 0;
}
// 0: the request may go on. Else the HTTP status: 401 without the key, 403 for a foreign Origin or Host
static inline int guard_check(const ServeGuard* g, const char* method, const char* host, const char* origin, const char* auth) {
    if (!guard_origin_ok(g, origin)) return 403;
    if (!strcmp(method, "OPTIONS")) return 0;   // a browser's preflight carries no key
    if (g->api_key) return guard_key_ok(g->api_key, auth) ? 0 : 401;
    return guard_loopback_host(host) ? 0 : 403;
}
// The CORS lines of a reply ("" unless --allow-origin admits this Origin)
static inline void guard_cors(const ServeGuard* g, const char* origin, char* out, size_t n) {
    if (!origin || !*origin || !g->allow_origin || !guard_origin_ok(g, origin)) { if (n) out[0] = 0; return; }
    snprintf(out, n, "Access-Control-Allow-Origin: %s\r\nVary: Origin\r\nAccess-Control-Allow-Headers: Authorization, Content-Type\r\n"
             "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n", strcmp(g->allow_origin, "*") ? origin : "*");
}
