// nslm/sha256.h - SHA-256 (FIPS 180-4) in portable C99, for the .nslm trailer and source hashes.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t h[8];
    uint64_t len;
    uint8_t buf[64];
    size_t nbuf;
} Sha256;

void sha256_init(Sha256* s);
void sha256_update(Sha256* s, const void* data, size_t n);
void sha256_final(Sha256* s, uint8_t out[32]);
void sha256_hex(const uint8_t d[32], char out[65]);
