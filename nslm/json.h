// nslm/json.h - JSON values that keep object key order (C99), with Python-compatible output.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef enum { J_NULL, J_FALSE, J_TRUE, J_INT, J_NUM, J_STR, J_ARR, J_OBJ } JType;

typedef struct Json {
    JType t;
    int64_t i;            // J_INT
    double d;             // J_NUM
    char* s;              // J_STR (NUL-terminated UTF-8)
    int n;                // J_ARR / J_OBJ element count
    struct Json** v;      // elements / values
    char** k;             // J_OBJ keys
} Json;

typedef struct {
    char* p;
    size_t n, cap;
} Buf;

void buf_put(Buf* b, const char* s, size_t n);
void buf_puts(Buf* b, const char* s);
char* buf_take(Buf* b);   // NUL-terminated; the caller frees

Json* json_parse(const char* s, size_t n, char* err, int errlen);   // NULL on error
Json* json_parse_prefix(const char* s, size_t n, size_t* used);    // one value at s (Python raw_decode); NULL on error
void json_free(Json* j);
Json* json_new(JType t);
Json* json_str(const char* s);
void json_push(Json* a, Json* v);                  // J_ARR append
void json_set(Json* o, const char* key, Json* v);  // J_OBJ: replace in place, or append
Json* json_copy(const Json* j);

Json* json_get(const Json* o, const char* key);    // NULL if o is not an object or has no key
const char* json_gets(const Json* o, const char* key);   // string member or NULL
int json_truthy(const Json* j);                    // Python truthiness (NULL = undefined = false)

// json.dumps(x, ensure_ascii=False): separators ", " and ": " (compact = ",", ":").
void json_dump(Buf* b, const Json* j, int compact);
char* json_dumps(const Json* j, int compact);
// json.dumps(x, indent=2, ensure_ascii=False)
void json_dump_indent(Buf* b, const Json* j, int depth);
// Python repr(float) / str(int).
void py_num(Buf* b, const Json* j);
