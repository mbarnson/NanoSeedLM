// tests/test_sha256.c - SHA-256 known-answer vectors (the .nslm trailer hash).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sha256.h"

static void kat(const char* msg, size_t reps, const char* want) {
    Sha256 s;
    uint8_t d[32];
    char hex[65];
    sha256_init(&s);
    for (size_t i = 0; i < reps; ++i) sha256_update(&s, msg, strlen(msg));
    sha256_final(&s, d);
    sha256_hex(d, hex);
    if (strcmp(hex, want)) { printf("sha256 KAT failed: %s\n", hex); exit(1); }
}

int main(void) {
    kat("abc", 1, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    kat("", 1, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    kat("a", 1000000, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    printf("sha256 known-answer vectors: ok\ntest_sha256: PASS\n");
    return 0;
}
