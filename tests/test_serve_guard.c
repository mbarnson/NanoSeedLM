// tests/test_serve_guard.c - harness/serve_guard.h: the API key, the browser origins and the loopback Host of nslm-serve.
#include <stdio.h>
#include <string.h>

#include "../harness/serve_guard.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(void) {
    // Loopback Host headers, with and without a port, and the names that only look like them
    const char* local[] = {"localhost", "localhost:8080", "LocalHost:8080", "127.0.0.1", "127.0.0.1:8080", "[::1]", "[::1]:8080", "", NULL};
    for (int i = 0; i < 9; ++i) CHECK(guard_loopback_host(local[i]), "\"%s\" is loopback", local[i] ? local[i] : "(none)");
    const char* foreign[] = {"evil.com", "evil.com:8080", "localhost.evil.com", "127.0.0.1.evil.com", "localhost:80a", "[::1]x", "192.168.1.5:8080"};
    for (int i = 0; i < 7; ++i) CHECK(!guard_loopback_host(foreign[i]), "\"%s\" is not loopback", foreign[i]);
    CHECK(guard_loopback_ip("127.0.0.1") && !guard_loopback_ip("0.0.0.0") && !guard_loopback_ip("192.168.1.5"), "the --host to bind");

    // No key, no origins (the default): a local client passes; DNS rebinding and a browser page do not
    const ServeGuard d = {NULL, NULL};
    CHECK(guard_check(&d, "POST", "127.0.0.1:8080", NULL, NULL) == 0, "a local client");
    CHECK(guard_check(&d, "POST", "evil.com:8080", NULL, NULL) == 403, "DNS rebinding");
    CHECK(guard_check(&d, "POST", "localhost:8080", "http://evil.com", NULL) == 403, "a page of another origin");
    CHECK(guard_check(&d, "POST", "localhost:8080", "null", NULL) == 403, "a sandboxed page (Origin: null)");
    CHECK(guard_check(&d, "OPTIONS", "localhost:8080", "http://evil.com", NULL) == 403, "a preflight of another origin");

    // An API key: each request needs it, whatever the Host (a LAN address, a reverse proxy)
    const ServeGuard k = {"s3cret", NULL};
    CHECK(guard_check(&k, "POST", "192.168.1.5:8080", NULL, "Bearer s3cret") == 0, "the key");
    CHECK(guard_check(&k, "GET", "192.168.1.5:8080", NULL, NULL) == 401, "no key");
    CHECK(guard_check(&k, "GET", "192.168.1.5:8080", NULL, "Bearer s3cre") == 401, "a shorter key");
    CHECK(guard_check(&k, "GET", "192.168.1.5:8080", NULL, "Bearer s3cretX") == 401, "a longer key");
    CHECK(guard_check(&k, "GET", "192.168.1.5:8080", NULL, "Bearer S3cret") == 401, "another key");
    CHECK(guard_check(&k, "GET", "192.168.1.5:8080", NULL, "Basic s3cret") == 401, "another scheme");
    CHECK(guard_check(&k, "GET", "192.168.1.5:8080", NULL, "Bearer ") == 401, "an empty key");

    // --allow-origin: that origin passes and gets CORS lines, others do not; "*" admits any
    const ServeGuard o = {NULL, "http://localhost:3000"};
    char cors[512];
    CHECK(guard_check(&o, "POST", "localhost:8080", "http://localhost:3000", NULL) == 0, "the allowed origin");
    CHECK(guard_check(&o, "OPTIONS", "localhost:8080", "http://localhost:3000", NULL) == 0, "its preflight");
    CHECK(guard_check(&o, "POST", "localhost:8080", "http://localhost:3001", NULL) == 403, "another port");
    guard_cors(&o, "http://localhost:3000", cors, sizeof cors);
    CHECK(strstr(cors, "Access-Control-Allow-Origin: http://localhost:3000\r\n") && strstr(cors, "Vary: Origin"), "CORS lines: %s", cors);
    guard_cors(&o, "http://localhost:3001", cors, sizeof cors);
    CHECK(!cors[0], "no CORS lines for another origin");
    guard_cors(&d, "http://localhost:3000", cors, sizeof cors);
    CHECK(!cors[0], "no CORS lines by default");
    const ServeGuard any = {"s3cret", "*"};
    guard_cors(&any, "http://example.org", cors, sizeof cors);
    CHECK(strstr(cors, "Access-Control-Allow-Origin: *\r\n") != NULL, "\"*\": %s", cors);
    CHECK(guard_check(&any, "POST", "example.org", "http://example.org", "Bearer s3cret") == 0, "\"*\" with the key");
    CHECK(guard_check(&any, "POST", "example.org", "http://example.org", NULL) == 401, "\"*\" still needs the key");

    printf("test_serve_guard: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
