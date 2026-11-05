// engine/qwen4_nomtp.c - the MTP calls of engine/qwen4_eng.h for an engine without them (the Metal one until plan step
// M4 of docs/plans/2026-11-04-qwen38-mtp.md): each fails. Its build drops this file when it has its own.
#include "qwen4_eng.h"

int q4e_draft(Q4Eng* e, int k, int32_t* drafts) { (void) e; (void) k; (void) drafts; return -1; }
int q4e_forward_keep(Q4Eng* e, const int32_t* ids, int n, float* logits) { (void) e; (void) ids; (void) n; (void) logits; return -1; }
int q4e_accept(Q4Eng* e, int m) { (void) e; (void) m; return -1; }
