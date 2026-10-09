// harness/kv_disk.h - the cold KV cache: sequence caches saved to disk in blocks of KVD_BLOCK positions, so a later
// request with the same token prefix restores them instead of computing its prompt (as oMLX's SSD cache does).
//
// Block b of a token sequence is named by the chained hash h_b = sha256(h_{b-1} || its KVD_BLOCK token ids), with h_-1
// the model fingerprint (the engine's description and KV size, the weights' names, formats, shapes and leading bytes):
// another model or quantization never matches.  One file per block, DIR/<fingerprint>/<h_b>.kv, written to a temporary
// file, synced and renamed (a crash leaves no partial block); a file that fails its checks reads as a miss and is
// deleted.  Beyond the byte budget the least recently used blocks go; a block's file time is its last read or write, so
// the order survives restarts.  Blocks unused for longer than a maximum age (kvd_set_max_age) are deleted too.
// Reads are plain reads: the OS page cache keeps recent blocks in memory as it sees fit.  Thread-safe.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KVD_BLOCK 256
#define KVD_HEAD_BYTES (88 + 4 * KVD_BLOCK)   // a block file's bytes before its KV (header, token ids)

typedef struct KvDisk KvDisk;

// The fingerprint of a model folder for an engine (desc: eng_describe; kv_bytes: eng_kv_bytes).  0, or -1.
int kvd_fingerprint(const char* model_dir, const char* desc, int64_t kv_bytes, uint8_t out[32]);
// dir is created if needed; block_bytes: one block's KV (eng_kv_bytes x KVD_BLOCK).  NULL with err, or for budget 0.
KvDisk* kvd_open(const char* dir, uint64_t budget, const uint8_t fingerprint[32], uint64_t block_bytes, char* err, int errlen);
void kvd_close(KvDisk* d);
// The chained hashes of the first nb blocks of ids.
void kvd_hashes(const KvDisk* d, const int32_t* ids, int nb, uint8_t (*h)[32]);
// How many of the blocks h[0 .. nb) are on disk, counting from the first.
int kvd_count(KvDisk* d, const uint8_t (*h)[32], int nb);
// Block h into dst (block_bytes); its tokens must be ids[0 .. KVD_BLOCK).  0, or -1: missing or invalid (then deleted).
int kvd_load(KvDisk* d, const uint8_t h[32], const int32_t* ids, void* dst);
// Saves block h (nothing to do if it is there), then deletes least recently used blocks beyond the budget.  0 or -1.
int kvd_store(KvDisk* d, const uint8_t h[32], const int32_t* ids, const void* src);
void kvd_drop(KvDisk* d, const uint8_t h[32]);   // deletes block h
// Blocks of every model not read or written for longer than seconds are deleted, now and on each store (0: no limit).
void kvd_set_max_age(KvDisk* d, double seconds);
uint64_t kvd_used(KvDisk* d);                     // bytes of every model's blocks under the directory
void kvd_path(const KvDisk* d, const uint8_t h[32], char* out, int cap);

#ifdef __cplusplus
}
#endif
