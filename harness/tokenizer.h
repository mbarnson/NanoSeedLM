// harness/tokenizer.h - byte-level BPE tokenizer read from an HF tokenizer.json.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Tok Tok;

Tok* tok_open(const char* tokenizer_json, char* err, int errlen);
void tok_close(Tok* t);
int tok_bos(Tok* t);   // id of the BOS token (<|begin_of_text|>)

// Encodes UTF-8 text.  Added tokens (special or not) in the text map to their ids, as in HF.  add_bos != 0 prepends
// BOS unless the encoded text already starts with it: BOS is never doubled.  Returns the count (written up to cap;
// call again with a larger buffer if the count exceeds cap), or -1 on error.
int tok_encode(Tok* t, const char* text, int add_bos, int32_t* out, int cap);

// Decodes ids to UTF-8 (added tokens as their text).  Returns a malloc'd string.  *valid is set to 0 when the bytes
// were not valid UTF-8.
char* tok_decode(Tok* t, const int32_t* ids, int n, int* valid);

#ifdef __cplusplus
}
#endif
