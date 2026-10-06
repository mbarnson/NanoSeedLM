// harness/chat_template.c - see chat_template.h.  Each function mirrors the template macro of the same name; the
// tests compare the output with transformers' apply_chat_template byte for byte (tests/data/template_golden.json).
#include "chat_template.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* BOS = "<|ifm|begin_of_text|>";

typedef struct {
    Buf o;
    char* err;
    int errlen;
    int failed;
    const Json* defs;     // REFS.defs
    Buf seen;             // REFS.seen: "|name|name|"
    int ok;               // RB.ok
    Json** tmp;           // merged schemas, freed at the end
    int ntmp;
} R;

static int is_obj(const Json* j) { return j && j->t == J_OBJ; }
static int is_arr(const Json* j) { return j && j->t == J_ARR; }
static int is_str(const Json* j) { return j && j->t == J_STR; }
static int is_seq(const Json* j) { return j && (j->t == J_ARR || j->t == J_OBJ || j->t == J_STR); }   // Jinja "sequence"
static Json* get(const Json* j, const char* k) { return json_get(j, k); }
static void put(R* r, const char* s) { buf_puts(&r->o, s); }

static void raise(R* r, const char* a, const char* b, const char* c, const char* d, const char* e) {
    if (r->failed) return;
    r->failed = 1;
    snprintf(r->err, (size_t) r->errlen, "%s%s%s%s%s", a, b ? b : "", c ? c : "", d ? d : "", e ? e : "");
}

static Json* keep(R* r, Json* j) {
    r->tmp = (Json**) realloc(r->tmp, sizeof(Json*) * (size_t) (r->ntmp + 1));
    r->tmp[r->ntmp++] = j;
    return j;
}

// Python str.split() + " ".join: whitespace runs (Unicode) to single spaces, trimmed.
static int ws_len(const unsigned char* s) {
    if (*s == ' ' || (*s >= 0x09 && *s <= 0x0D) || (*s >= 0x1C && *s <= 0x1F)) return 1;
    if (s[0] == 0xC2 && (s[1] == 0x85 || s[1] == 0xA0)) return 2;
    if (s[0] == 0xE1 && s[1] == 0x9A && s[2] == 0x80) return 3;
    if (s[0] == 0xE2 && s[1] == 0x80 && ((s[2] >= 0x80 && s[2] <= 0x8A) || s[2] == 0xA8 || s[2] == 0xA9 || s[2] == 0xAF))
        return 3;
    if (s[0] == 0xE2 && s[1] == 0x81 && s[2] == 0x9F) return 3;
    if (s[0] == 0xE3 && s[1] == 0x80 && s[2] == 0x80) return 3;
    return 0;
}
static void put_collapsed(Buf* b, const char* s, int escape) {
    const unsigned char* p = (const unsigned char*) s;
    int pending = 0, any = 0;
    while (*p) {
        const int w = ws_len(p);
        if (w) { pending = any; p += w; continue; }
        if (pending) { buf_puts(b, " "); pending = 0; }
        if (escape && *p == '\\') buf_puts(b, "\\\\");
        else if (escape && *p == '\'') buf_puts(b, "\\'");
        else buf_put(b, (const char*) p, 1);
        any = 1;
        ++p;
    }
}

// s.replace(from, to)
static void put_replace(Buf* b, const char* s, const char* from, const char* to) {
    const size_t fl = strlen(from);
    for (const char* q; (q = strstr(s, from)); s = q + fl) {
        buf_put(b, s, (size_t) (q - s));
        buf_puts(b, to);
    }
    buf_puts(b, s);
}

static void py_repr(Buf* b, const Json* v);
// Python str(value)
static void py_str(Buf* b, const Json* v) {
    if (!v) return;
    if (is_str(v)) buf_puts(b, v->s);
    else if (v->t == J_INT || v->t == J_NUM) py_num(b, v);
    else py_repr(b, v);
}
// Python repr() (for containers inside str())
static void py_repr(Buf* b, const Json* v) {
    switch (v->t) {
    case J_NULL: buf_puts(b, "None"); break;
    case J_TRUE: buf_puts(b, "True"); break;
    case J_FALSE: buf_puts(b, "False"); break;
    case J_INT: case J_NUM: py_num(b, v); break;
    case J_STR: {
        const int dq = strchr(v->s, '\'') && !strchr(v->s, '"');
        buf_puts(b, dq ? "\"" : "'");
        for (const char* s = v->s; *s; ++s) {
            if (*s == '\\') buf_puts(b, "\\\\");
            else if (*s == '\n') buf_puts(b, "\\n");
            else if (*s == '\r') buf_puts(b, "\\r");
            else if (*s == '\t') buf_puts(b, "\\t");
            else if (*s == '\'' && !dq) buf_puts(b, "\\'");
            else if ((unsigned char) *s < 0x20) { char e[8]; snprintf(e, sizeof e, "\\x%02x", *s); buf_puts(b, e); }
            else buf_put(b, s, 1);
        }
        buf_puts(b, dq ? "\"" : "'");
        break;
    }
    case J_ARR:
        buf_puts(b, "[");
        for (int i = 0; i < v->n; ++i) { if (i) buf_puts(b, ", "); py_repr(b, v->v[i]); }
        buf_puts(b, "]");
        break;
    case J_OBJ:
        buf_puts(b, "{");
        for (int i = 0; i < v->n; ++i) {
            if (i) buf_puts(b, ", ");
            Json k = {.t = J_STR, .s = v->k[i]};
            py_repr(b, &k);
            buf_puts(b, ": ");
            py_repr(b, v->v[i]);
        }
        buf_puts(b, "}");
        break;
    }
}
static void tojson(R* r, const Json* v) { json_dump(&r->o, v, 0); }

// ---- value rendering ----

static void render_python_repr(R* r, const Json* v) {
    if (is_str(v)) { put(r, "'"); put_collapsed(&r->o, v->s, 1); put(r, "'"); }
    else if (v->t == J_TRUE) put(r, "True");
    else if (v->t == J_FALSE) put(r, "False");
    else if (v->t == J_NULL) put(r, "None");
    else if (is_obj(v)) {
        put(r, "{");
        for (int i = 0; i < v->n; ++i) {
            Json k = {.t = J_STR, .s = v->k[i]};
            render_python_repr(r, &k);
            put(r, ": ");
            render_python_repr(r, v->v[i]);
            if (i + 1 < v->n) put(r, ", ");
        }
        put(r, "}");
    } else if (is_arr(v)) {
        put(r, "[");
        for (int i = 0; i < v->n; ++i) { render_python_repr(r, v->v[i]); if (i + 1 < v->n) put(r, ", "); }
        put(r, "]");
    } else py_num(&r->o, v);
}
static void render_markdown_value(R* r, const Json* v) {
    if (is_str(v)) put(r, v->s[0] ? v->s : "\"\"");
    else render_python_repr(r, v);
}
static void render_markdown_literal(R* r, const Json* v) {
    if (is_str(v) && !v->s[0]) { put(r, "\"\""); return; }
    put(r, "`");
    if (is_str(v)) put_replace(&r->o, v->s, "\n", "\\n");
    else render_python_repr(r, v);
    put(r, "`");
}
static void render_allowed_values(R* r, const Json* values) {
    if (!is_arr(values)) return;
    for (int i = 0; i < values->n; ++i) { render_markdown_literal(r, values->v[i]); if (i + 1 < values->n) put(r, ", "); }
}
// str(value).replace("\n", nl) as a string Json
static Json* str_replaced(R* r, const Json* v, const char* nl) {
    Buf t = {0}, u = {0};
    py_str(&t, v);
    put_replace(&u, t.p ? t.p : "", "\n", nl);
    free(t.p);
    Json* j = json_new(J_STR);
    j->s = buf_take(&u);
    return keep(r, j);
}
static void render_markdown_detail(R* r, const char* indent, const char* label, const Json* v) {
    put(r, "\n"); put(r, indent); put(r, "  - "); put(r, label); put(r, ": ");
    render_markdown_value(r, v);
}
static void render_markdown_metadata_detail(R* r, const char* label, const Json* v) {
    put(r, "\n- "); put(r, label); put(r, ": ");
    render_markdown_value(r, v);
}

// ---- types ----

static void render_markdown_type(R* r, const Json* spec);
static void render_markdown_type_name(R* r, const Json* name, const Json* spec) {
    if (is_str(name) && !strcmp(name->s, "array")) {
        put(r, "array of ");
        if (get(spec, "items")) render_markdown_type(r, get(spec, "items"));
        else put(r, "any");
    } else if (json_truthy(name)) py_str(&r->o, name);
    else put(r, "any");
}
static const char* ref_tail(const char* ref) { const char* s = strrchr(ref, '/'); return s ? s + 1 : ref; }
static void render_markdown_type(R* r, const Json* spec) {
    const Json* t = get(spec, "type");
    if (spec && spec->t == J_TRUE) put(r, "True");
    else if (spec && spec->t == J_FALSE) put(r, "False");
    else if (!is_obj(spec)) put(r, "any");
    else if (is_arr(t) && t->n > 0) {
        for (int i = 0; i < t->n; ++i) { render_markdown_type_name(r, t->v[i], spec); if (i + 1 < t->n) put(r, " or "); }
    } else if (is_arr(t)) put(r, "any");
    else if (json_truthy(t)) render_markdown_type_name(r, t, spec);
    else if (is_str(get(spec, "$ref"))) put(r, ref_tail(get(spec, "$ref")->s));
    else if (json_truthy(get(spec, "oneOf")) || json_truthy(get(spec, "anyOf"))) {
        const int one = json_truthy(get(spec, "oneOf"));
        const Json* vs = get(spec, one ? "oneOf" : "anyOf");
        put(r, one ? "oneOf[" : "anyOf[");
        for (int i = 0; is_arr(vs) && i < vs->n; ++i) { render_markdown_type(r, vs->v[i]); if (i + 1 < vs->n) put(r, " or "); }
        put(r, "]");
    } else if (json_truthy(get(spec, "properties"))) put(r, "object");
    else if (get(spec, "items")) { put(r, "array of "); render_markdown_type(r, get(spec, "items")); }
    else put(r, "any");
}

static void render_compact_type(R* r, const Json* spec);
static void render_compact_type_name(R* r, const Json* name, const Json* spec) {
    if (is_str(name) && !strcmp(name->s, "array")) {
        put(r, "array[");
        if (get(spec, "items")) render_compact_type(r, get(spec, "items"));
        else put(r, "any");
        put(r, "]");
    } else if (json_truthy(name)) py_str(&r->o, name);
    else put(r, "any");
}
static void render_compact_type(R* r, const Json* spec) {
    const Json* t = get(spec, "type");
    if (!is_obj(spec)) put(r, "any");
    else if (is_arr(t) && t->n > 0) {
        for (int i = 0; i < t->n; ++i) { render_compact_type_name(r, t->v[i], spec); if (i + 1 < t->n) put(r, "|"); }
    } else if (is_arr(t)) put(r, "any");
    else if (json_truthy(t)) render_compact_type_name(r, t, spec);
    else if (is_str(get(spec, "$ref"))) put(r, ref_tail(get(spec, "$ref")->s));
    else if (json_truthy(get(spec, "oneOf")) || json_truthy(get(spec, "anyOf"))) {
        const int one = json_truthy(get(spec, "oneOf"));
        const Json* vs = get(spec, one ? "oneOf" : "anyOf");
        put(r, one ? "oneOf[" : "anyOf[");
        for (int i = 0; is_arr(vs) && i < vs->n; ++i) { render_compact_type(r, vs->v[i]); if (i + 1 < vs->n) put(r, "|"); }
        put(r, "]");
    } else if (json_truthy(get(spec, "properties"))) put(r, "object");
    else if (get(spec, "items")) { put(r, "array["); render_compact_type(r, get(spec, "items")); put(r, "]"); }
    else put(r, "any");
}
static const char* render_value_type(const Json* v) {
    switch (v->t) {
    case J_NULL: return "null";
    case J_TRUE: case J_FALSE: return "boolean";
    case J_INT: return "integer";
    case J_NUM: return "number";
    case J_STR: return "string";
    case J_OBJ: return "object";
    default: return "array";
    }
}
static int schema_has_combinator(const Json* spec) {
    const Json* t = get(spec, "type");
    if (json_truthy(get(spec, "oneOf")) || json_truthy(get(spec, "anyOf"))) return 1;
    if (is_arr(t) && t->n > 1) return 1;
    if (is_str(t) && !strcmp(t->s, "array") && get(spec, "items")) return schema_has_combinator(get(spec, "items"));
    const Json* p = get(spec, "properties");
    if (json_truthy(p)) {
        int found = 0;
        for (int i = 0; is_obj(p) && i < p->n; ++i) found |= schema_has_combinator(p->v[i]);
        return found;
    }
    return 0;
}

// ---- $ref inlining ----

static const char* ref_key(const char* ref) {
    if (!strncmp(ref, "#/$defs/", 8)) return ref + 8;
    if (!strncmp(ref, "#/definitions/", 14)) return ref + 14;
    return NULL;
}
static int seen(R* r, const char* k) {
    char t[512];
    snprintf(t, sizeof t, "|%s|", k);
    return r->seen.p && strstr(r->seen.p, t) != NULL;
}
static void mark_seen(R* r, const char* k) { buf_puts(&r->seen, k); buf_puts(&r->seen, "|"); }
// dict(def items + spec items without $ref)
static const Json* merge(R* r, const Json* def, const Json* spec) {
    Json* m = json_copy(def);
    for (int i = 0; i < spec->n; ++i)
        if (strcmp(spec->k[i], "$ref")) json_set(m, spec->k[i], json_copy(spec->v[i]));
    return keep(r, m);
}
static const Json* inline_ref(R* r, const Json* spec) {
    if (!is_obj(spec) || !is_str(get(spec, "$ref"))) return spec;
    const char* k = ref_key(get(spec, "$ref")->s);
    if (!k || seen(r, k) || !is_obj(get(r->defs, k))) return spec;
    spec = merge(r, get(r->defs, k), spec);
    mark_seen(r, k);
    if (is_str(get(spec, "$ref"))) {
        const char* k2 = ref_key(get(spec, "$ref")->s);
        if (k2 && is_obj(get(r->defs, k2))) {
            spec = merge(r, get(r->defs, k2), spec);
            mark_seen(r, k2);
        }
    }
    return spec;
}

// ---- markdown schema ----

static const char* RENDERED[] = {"type", "description", "enum", "default", "properties", "required", "items", "oneOf",
                                 "anyOf", "additionalProperties", "patternProperties", "returns", NULL};
static int in_list(const char* k, const char** list) {
    for (; *list; ++list) if (!strcmp(k, *list)) return 1;
    return 0;
}
static int in_required(const Json* req, const char* name) {
    if (is_arr(req)) {
        for (int i = 0; i < req->n; ++i) if (is_str(req->v[i]) && !strcmp(req->v[i]->s, name)) return 1;
        return 0;
    }
    if (is_str(req)) return strstr(req->s, name) != NULL;
    if (is_obj(req)) return get(req, name) != NULL;
    return 0;
}
static const char* ind(R* r, const char* indent, const char* more) {
    Buf b = {0};
    buf_puts(&b, indent);
    buf_puts(&b, more);
    Json* j = json_new(J_STR);
    j->s = buf_take(&b);
    return keep(r, j)->s;
}

static void render_markdown_param(R* r, const char* name, const Json* spec, const Json* req, const char* indent);
static void render_markdown_schema_details(R* r, const Json* spec, const char* indent, int values);

static void render_markdown_schema_annotations(R* r, const Json* spec, const char* indent, int values) {
    if (values && get(spec, "description"))
        render_markdown_detail(r, indent, "Description", str_replaced(r, get(spec, "description"), ind(r, "\n", ind(r, indent, "    "))));
    if (values && get(spec, "enum")) { put(r, "\n"); put(r, indent); put(r, "  - Allowed values: "); render_allowed_values(r, get(spec, "enum")); }
    if (values && get(spec, "default")) { put(r, "\n"); put(r, indent); put(r, "  - Default: "); render_markdown_literal(r, get(spec, "default")); }
    const Json* ap = get(spec, "additionalProperties");
    if (ap) {
        if (is_obj(ap)) {
            put(r, "\n"); put(r, indent); put(r, "  - Additional properties *("); render_markdown_type(r, ap); put(r, ")*");
            render_markdown_schema_details(r, ap, ind(r, indent, "  "), 1);
        } else render_markdown_detail(r, indent, "Additional properties", ap);
    }
}
static void render_markdown_metadata_annotations(R* r, const Json* spec) {
    if (get(spec, "description")) render_markdown_metadata_detail(r, "Description", str_replaced(r, get(spec, "description"), "\n    "));
    if (get(spec, "enum")) { put(r, "\n- Allowed values: "); render_allowed_values(r, get(spec, "enum")); }
    if (get(spec, "default")) { put(r, "\n- Default: "); render_markdown_literal(r, get(spec, "default")); }
    const Json* ap = get(spec, "additionalProperties");
    if (ap) {
        if (is_obj(ap)) {
            put(r, "\n- Additional properties *("); render_markdown_type(r, ap); put(r, ")*");
            render_markdown_schema_details(r, ap, "", 1);
        } else render_markdown_metadata_detail(r, "Additional properties", ap);
    }
}
static void render_markdown_schema_extras(R* r, const Json* spec, const char* indent) {
    for (int i = 0; i < spec->n; ++i)
        if (!in_list(spec->k[i], RENDERED)) {
            put(r, "\n"); put(r, indent); put(r, "  - "); put(r, spec->k[i]); put(r, ": ");
            render_markdown_value(r, spec->v[i]);
        }
}
static void render_markdown_metadata_extras(R* r, const Json* spec) {
    for (int i = 0; i < spec->n; ++i)
        if (!in_list(spec->k[i], RENDERED)) {
            put(r, "\n- "); put(r, spec->k[i]); put(r, ": ");
            render_markdown_value(r, spec->v[i]);
        }
}
static void variants(R* r, const Json* vs, const char* head, const char* indent) {
    put(r, head);
    for (int i = 0; is_arr(vs) && i < vs->n; ++i) {
        char n[16];
        snprintf(n, sizeof n, "%d", i + 1);
        put(r, "\n"); put(r, indent); put(r, "  - Variant "); put(r, n); put(r, " *("); render_markdown_type(r, vs->v[i]); put(r, ")*");
        render_markdown_schema_details(r, vs->v[i], ind(r, indent, "  "), 1);
    }
}
static void render_markdown_schema_structure(R* r, const Json* spec, const char* indent, int props) {
    const Json* p = get(spec, "properties");
    if (props && json_truthy(p))
        for (int i = 0; is_obj(p) && i < p->n; ++i)
            render_markdown_param(r, p->k[i], p->v[i], json_truthy(get(spec, "required")) ? get(spec, "required") : NULL, ind(r, indent, "  "));
    const Json* items = get(spec, "items");
    if (items && is_obj(items)) {
        put(r, "\n"); put(r, indent); put(r, "  - Items *("); render_markdown_type(r, items); put(r, ")*");
        render_markdown_schema_details(r, items, ind(r, indent, "  "), 1);
    } else if (items) render_markdown_detail(r, indent, "Items", items);
    if (json_truthy(get(spec, "oneOf"))) variants(r, get(spec, "oneOf"), ind(r, ind(r, "\n", indent), "  - oneOf:"), ind(r, indent, "  "));
    if (json_truthy(get(spec, "anyOf"))) variants(r, get(spec, "anyOf"), ind(r, ind(r, "\n", indent), "  - anyOf:"), ind(r, indent, "  "));
    const Json* pp = get(spec, "patternProperties");
    if (is_obj(pp)) {
        put(r, "\n"); put(r, indent); put(r, "  - Pattern properties:");
        for (int i = 0; i < pp->n; ++i) {
            if (is_obj(pp->v[i])) {
                put(r, "\n"); put(r, indent); put(r, "    - `"); put(r, pp->k[i]); put(r, "` *("); render_markdown_type(r, pp->v[i]); put(r, ")*");
                render_markdown_schema_details(r, pp->v[i], ind(r, indent, "    "), 1);
            } else {
                put(r, "\n"); put(r, indent); put(r, "    - `"); put(r, pp->k[i]); put(r, "`: ");
                render_markdown_value(r, pp->v[i]);
            }
        }
    } else if (pp) render_markdown_detail(r, indent, "Pattern properties", pp);
    const Json* ret = get(spec, "returns");
    if (is_obj(ret)) {
        put(r, "\n"); put(r, indent); put(r, "  - Returns *("); render_markdown_type(r, ret); put(r, ")*");
        render_markdown_schema_details(r, ret, ind(r, indent, "  "), 1);
    } else if (ret) render_markdown_detail(r, indent, "Returns", ret);
}
static void render_markdown_schema_details(R* r, const Json* spec, const char* indent, int values) {
    spec = inline_ref(r, spec);
    if (is_obj(spec)) {
        render_markdown_schema_annotations(r, spec, indent, values);
        render_markdown_schema_structure(r, spec, indent, 1);
        render_markdown_schema_extras(r, spec, indent);
    } else if (spec->t != J_TRUE && spec->t != J_FALSE) {
        put(r, "\n"); put(r, indent); put(r, "  - Value: "); render_markdown_literal(r, spec);
    }
}
static void render_markdown_parameter_schema(R* r, const Json* spec) {
    spec = inline_ref(r, spec);
    if (!is_obj(spec)) return;
    render_markdown_metadata_annotations(r, spec);
    const Json* items = get(spec, "items");
    if (items && is_obj(items)) {
        put(r, "\n- Items *("); render_markdown_type(r, items); put(r, ")*");
        render_markdown_schema_details(r, items, "", 1);
    } else if (items) render_markdown_metadata_detail(r, "Items", items);
    if (json_truthy(get(spec, "oneOf"))) variants(r, get(spec, "oneOf"), "\n- oneOf:", "");
    if (json_truthy(get(spec, "anyOf"))) variants(r, get(spec, "anyOf"), "\n- anyOf:", "");
    const Json* pp = get(spec, "patternProperties");
    if (is_obj(pp)) {
        put(r, "\n- Pattern properties:");
        for (int i = 0; i < pp->n; ++i) {
            if (is_obj(pp->v[i])) {
                put(r, "\n  - `"); put(r, pp->k[i]); put(r, "` *("); render_markdown_type(r, pp->v[i]); put(r, ")*");
                render_markdown_schema_details(r, pp->v[i], "  ", 1);
            } else {
                put(r, "\n  - `"); put(r, pp->k[i]); put(r, "`: ");
                render_markdown_value(r, pp->v[i]);
            }
        }
    } else if (pp) render_markdown_metadata_detail(r, "Pattern properties", pp);
    const Json* ret = get(spec, "returns");
    if (is_obj(ret)) {
        put(r, "\n- Returns *("); render_markdown_type(r, ret); put(r, ")*");
        render_markdown_schema_details(r, ret, "", 1);
    } else if (ret) render_markdown_metadata_detail(r, "Returns", ret);
    render_markdown_metadata_extras(r, spec);
}
static void render_markdown_param(R* r, const char* name, const Json* spec, const Json* req, const char* indent) {
    spec = inline_ref(r, spec);
    put(r, "\n"); put(r, indent); put(r, "- `"); put(r, name); put(r, "` *(");
    render_markdown_type(r, spec);
    if (in_required(req, name)) put(r, ", required");
    put(r, ")*");
    if (json_truthy(get(spec, "description"))) {
        put(r, " - ");
        put(r, str_replaced(r, get(spec, "description"), ind(r, "\n", ind(r, indent, "  ")))->s);
    }
    if (json_truthy(get(spec, "enum"))) { put(r, "\n"); put(r, indent); put(r, "  - Allowed values: "); render_allowed_values(r, get(spec, "enum")); }
    if (get(spec, "default")) { put(r, "\n"); put(r, indent); put(r, "  - Default: "); render_markdown_literal(r, get(spec, "default")); }
    render_markdown_schema_details(r, spec, indent, 0);
}

static const Json* fn_of(const Json* tool) { return get(tool, "function") ? get(tool, "function") : tool; }
static const char* name_of(const Json* fn) { return json_gets(fn, "name") ? json_gets(fn, "name") : ""; }

static void render_tools_markdown(R* r, const Json* tools) {
    put(r, "<ifm|tools>");
    for (int t = 0; t < tools->n; ++t) {
        const Json* fn = fn_of(tools->v[t]);
        const Json* params = get(fn, "parameters");
        r->defs = is_obj(get(params, "$defs")) ? get(params, "$defs") : is_obj(get(params, "definitions")) ? get(params, "definitions") : NULL;
        free(r->seen.p);
        r->seen = (Buf) {0};
        buf_puts(&r->seen, "|");
        const Json* p = params;
        if (is_obj(p) && is_str(get(p, "$ref"))) {
            const char* k = ref_key(get(p, "$ref")->s);
            if (k && is_obj(get(r->defs, k))) { p = merge(r, get(r->defs, k), p); mark_seen(r, k); }
        }
        put(r, "\n## "); put(r, name_of(fn));
        if (json_truthy(get(fn, "description"))) { put(r, "\n"); py_str(&r->o, get(fn, "description")); }
        put(r, "\n\n**Parameters**");
        if (json_truthy(p) && json_truthy(get(p, "properties"))) {
            const Json* props = get(p, "properties");
            const Json* req = json_truthy(get(p, "required")) ? get(p, "required") : NULL;
            for (int i = 0; is_obj(props) && i < props->n; ++i) render_markdown_param(r, props->k[i], props->v[i], req, "");
        } else if (is_obj(p) && (json_truthy(get(p, "oneOf")) || json_truthy(get(p, "anyOf")) || get(p, "items"))) {
            render_markdown_parameter_schema(r, p);
        } else put(r, "\n- None");
        const Json* ret = get(fn, "returns") ? get(fn, "returns") : get(fn, "response");
        if (is_obj(ret)) {
            put(r, "\n\n**Returns**");
            put(r, "\n- Return *("); render_markdown_type(r, ret); put(r, ")*");
            render_markdown_schema_details(r, ret, "", 1);
        } else if (ret) { put(r, "\n\n**Returns**\n- "); render_markdown_value(r, ret); }
        if (t + 1 < tools->n) put(r, "\n");
    }
    put(r, "\n</ifm|tools>");
}
static void render_tools_json(R* r, const Json* tools) {
    put(r, "<ifm|tools>");
    for (int t = 0; t < tools->n; ++t) { put(r, "\n"); tojson(r, tools->v[t]); }
    put(r, "\n</ifm|tools>");
}

// ---- validation and renderability ----

static int value_contains_mapping(const Json* v) {
    if (is_obj(v)) return 1;
    if (is_arr(v)) for (int i = 0; i < v->n; ++i) if (value_contains_mapping(v->v[i])) return 1;
    return 0;
}
static const char* path2(R* r, const char* a, const char* b, const char* c, const char* d) {
    return ind(r, ind(r, a, b), ind(r, c ? c : "", d ? d : ""));
}
static void validate_schema(R* r, const Json* spec, const char* path, int lenient, int classify, int in_variant) {
    if (!is_obj(spec) || r->failed) return;
    const Json* req = get(spec, "required");
    const Json* props = get(spec, "properties");
    if (!lenient && req) {
        if (is_str(req) || !is_seq(req)) { raise(r, "Schema '", path, "' has 'required' but it is not a list.", NULL, NULL); return; }
        if (req->n > 0 && !json_truthy(props) && !in_variant) {
            raise(r, "Schema '", path, "' has required fields but no properties object to define them.", NULL, NULL);
            return;
        }
        if (json_truthy(props) && is_arr(req))
            for (int i = 0; i < req->n; ++i)
                if (is_str(req->v[i]) && !get(props, req->v[i]->s)) {
                    raise(r, "Schema '", path, "' marks '", req->v[i]->s, "' as required, but that property is not defined in properties.");
                    return;
                }
    }
    if (classify) {
        static const char* fine[] = {"description", "default", "title", "examples", "properties", "patternProperties",
                                     "additionalProperties", "returns", NULL};
        for (int i = 0; i < spec->n; ++i) {
            const char* k = spec->k[i];
            const Json* v = spec->v[i];
            if (!strcmp(k, "$ref")) {
                if (!is_str(v) || !ref_key(v->s)) r->ok = 0;
            } else if (!strcmp(k, "$defs") || !strcmp(k, "definitions")) {
                if (is_obj(v)) for (int d = 0; d < v->n; ++d) validate_schema(r, v->v[d], path2(r, path, ".$defs.", v->k[d], NULL), 1, 1, 0);
                else r->ok = 0;
            } else if (!strcmp(k, "type")) {
                if (is_obj(v)) r->ok = 0;
            } else if (!strcmp(k, "enum")) {
                if (is_str(v) || is_obj(v) || !is_seq(v)) r->ok = 0;
            } else if (!strcmp(k, "items")) {
            } else if (!strcmp(k, "oneOf") || !strcmp(k, "anyOf")) {
                if (is_obj(v) || is_str(v) || !is_seq(v)) r->ok = 0;
            } else if (!strcmp(k, "required")) {
                if (json_truthy(v) && !json_truthy(props)) r->ok = 0;
            } else if (in_list(k, fine)) {
            } else if (is_obj(v)) {
                for (int u = 0; u < v->n; ++u) if (value_contains_mapping(v->v[u])) r->ok = 0;
            } else if (is_arr(v)) {
                if (value_contains_mapping(v)) r->ok = 0;
            }
        }
    }
    if (json_truthy(props) && is_obj(props))
        for (int i = 0; i < props->n; ++i) validate_schema(r, props->v[i], path2(r, path, ".", props->k[i], NULL), lenient, classify, 0);
    if (get(spec, "items")) validate_schema(r, get(spec, "items"), ind(r, path, "[]"), lenient, classify, 0);
    for (int w = 0; w < 2; ++w) {
        const Json* vs = get(spec, w ? "anyOf" : "oneOf");
        if (!json_truthy(vs) || !is_arr(vs)) continue;
        for (int i = 0; i < vs->n; ++i) {
            char n[16];
            snprintf(n, sizeof n, "%d]", i);
            validate_schema(r, vs->v[i], path2(r, path, w ? ".anyOf[" : ".oneOf[", n, NULL), lenient, classify, 1);
        }
    }
    if (is_obj(get(spec, "additionalProperties")))
        validate_schema(r, get(spec, "additionalProperties"), ind(r, path, ".additionalProperties"), lenient, classify, 0);
    const Json* pp = get(spec, "patternProperties");
    if (is_obj(pp))
        for (int i = 0; i < pp->n; ++i)
            validate_schema(r, pp->v[i], path2(r, path, ".patternProperties[", pp->k[i], "]"), lenient, classify, 0);
    if (is_obj(get(spec, "returns"))) validate_schema(r, get(spec, "returns"), ind(r, path, ".returns"), lenient, classify, 0);
}
// 1 when some tool must render as JSON (RB.bad), 0 otherwise; r->failed on a template error.
static int validate_tools(R* r, const Json* tools, int classify) {
    static const char* root_ok[] = {"type", "description", "enum", "default", "properties", "required", "optional",
                                    "title", "items", "oneOf", "anyOf", "additionalProperties", "patternProperties",
                                    "returns", "examples", "$defs", "definitions", "$ref", NULL};
    static const char* fn_ok[] = {"name", "description", "parameters", "returns", "response", "type", "function", NULL};
    int bad = 0;
    for (int t = 0; t < tools->n && !r->failed; ++t) {
        const Json* fn = fn_of(tools->v[t]);
        const Json* params = get(fn, "parameters");
        const char* name = name_of(fn);
        r->ok = 1;
        if (is_str(params)) {
            raise(r, "tool.function.parameters must be a dict, not a JSON string. Parse it before passing to the template.", NULL, NULL, NULL, NULL);
            break;
        }
        if (!params || params->t == J_NULL) {
            if (get(fn, "arguments")) raise(r, "Tool '", name, "' has 'arguments' instead of 'parameters'. Rename 'arguments' to 'parameters'.", NULL, NULL);
            else raise(r, "Tool '", name, "' is missing required 'parameters' field. Each tool must have a 'parameters' dict with 'type', 'properties', and 'required' keys.", NULL, NULL);
            break;
        }
        validate_schema(r, params, path2(r, "tool.", name, ".parameters", NULL), 0, classify, 0);
        if (classify) {
            if (is_obj(params)) {
                for (int i = 0; i < params->n; ++i)
                    if (!in_list(params->k[i], root_ok) && (is_obj(params->v[i]) || is_arr(params->v[i]))) r->ok = 0;
            } else r->ok = 0;
        }
        if (is_obj(get(fn, "returns"))) validate_schema(r, get(fn, "returns"), path2(r, "tool.", name, ".returns", NULL), 0, classify, 0);
        if (classify && !get(fn, "returns") && is_obj(get(fn, "response")))
            validate_schema(r, get(fn, "response"), path2(r, "tool.", name, ".response", NULL), 1, 1, 0);
        if (classify && is_obj(fn))
            for (int i = 0; i < fn->n; ++i)
                if (!in_list(fn->k[i], fn_ok) && (is_obj(fn->v[i]) || is_arr(fn->v[i]))) r->ok = 0;
        if (!r->ok) bad = 1;
    }
    return bad;
}

// ---- messages ----

static const char* CALL_JSON =
    "Wrap all tool calls in a single <ifm|tool_calls></ifm|tool_calls> block. For each call, emit one JSON object with "
    "the function name and arguments on the same line inside <ifm|tool_call></ifm|tool_call> tags:\n\n<ifm|tool_calls>\n"
    "<ifm|tool_call>{\"name\": <function-name>, \"arguments\": <args-json-object>}</ifm|tool_call>\n</ifm|tool_calls>";
static const char* CALL_XML =
    "Wrap all tool calls in a single <ifm|tool_calls></ifm|tool_calls> block. For each call, write the function name at "
    "the start of <ifm|tool_call>, followed by paired <ifm|arg_key> and <ifm|arg_value> tags for each argument:\n\n"
    "<ifm|tool_calls>\n<ifm|tool_call>$FUNCTION_NAME\n<ifm|arg_key>$PARAMETER_NAME</ifm|arg_key>\n"
    "<ifm|arg_value>$PARAMETER_VALUE</ifm|arg_value>\n...\n</ifm|tool_call>\n</ifm|tool_calls>\n\nString and scalar "
    "parameters should be written as plain text. Array and object parameters should be written as JSON literals.";
static const char* CALL_XML_TYPED =
    "Wrap all tool calls in a single <ifm|tool_calls></ifm|tool_calls> block. For each call, write the function name at "
    "the start of <ifm|tool_call>, followed by <ifm|arg_key>, <ifm|arg_type>, and <ifm|arg_value> tags for each "
    "argument:\n\n<ifm|tool_calls>\n<ifm|tool_call>$FUNCTION_NAME\n<ifm|arg_key>$PARAMETER_NAME</ifm|arg_key>\n"
    "<ifm|arg_type>$ARGUMENT_TYPE</ifm|arg_type>\n<ifm|arg_value>$PARAMETER_VALUE</ifm|arg_value>\n...\n"
    "</ifm|tool_call>\n</ifm|tool_calls>\n\nUse the parameter type shown in the tool definition. If that type contains "
    "anyOf or oneOf, use the actual argument value type instead. String and scalar parameters should be written as "
    "plain text. Array and object parameters should be written as JSON literals.";

static void render_arg_type(R* r, const Json* tools, const char* tool, const char* arg, const Json* value) {
    Buf found = {0};
    buf_puts(&found, "any");
    for (int t = 0; is_arr(tools) && t < tools->n; ++t) {
        const Json* fn = fn_of(tools->v[t]);
        const Json* params = get(fn, "parameters");
        const Json* props = get(params, "properties");
        if (strcmp(name_of(fn), tool) || !json_truthy(params) || !json_truthy(props) || !get(props, arg)) continue;
        const Json* spec = get(props, arg);
        if (is_obj(spec) && is_str(get(spec, "$ref"))) {
            const char* k = ref_key(get(spec, "$ref")->s);
            const Json* d = is_obj(get(params, "$defs")) ? get(params, "$defs") : get(params, "definitions");
            if (k && is_obj(d) && is_obj(get(d, k))) spec = merge(r, get(d, k), spec);
        }
        Buf save = r->o;
        r->o = (Buf) {0};
        if (schema_has_combinator(spec)) put(r, render_value_type(value));
        else render_compact_type(r, spec);
        free(found.p);
        found = r->o;
        r->o = save;
    }
    put(r, found.p);
    free(found.p);
}
static void render_tool_calls_block(R* r, const Json* calls, const char* fmt, const Json* tools) {
    put(r, "<ifm|tool_calls>");
    for (int i = 0; is_arr(calls) && i < calls->n && !r->failed; ++i) {
        const Json* tc = json_truthy(get(calls->v[i], "function")) ? get(calls->v[i], "function") : calls->v[i];
        const Json* args = get(tc, "arguments");
        if (is_str(args)) {
            raise(r, "tool_call.arguments must be a dict, not a JSON string. Parse it before passing to the template.", NULL, NULL, NULL, NULL);
            return;
        }
        const char* name = name_of(tc);
        if (!strcmp(fmt, "json")) {
            put(r, "\n<ifm|tool_call>{\"name\": \""); put(r, name); put(r, "\", \"arguments\": ");
            if (args) tojson(r, args); else put(r, "{}");
            put(r, "}</ifm|tool_call>");
        } else {
            put(r, "\n<ifm|tool_call>"); put(r, name); put(r, "\n");
            for (int a = 0; is_obj(args) && a < args->n; ++a) {
                put(r, "<ifm|arg_key>"); put(r, args->k[a]); put(r, "</ifm|arg_key>\n");
                if (!strcmp(fmt, "xml_typed")) {
                    put(r, "<ifm|arg_type>"); render_arg_type(r, tools, name, args->k[a], args->v[a]); put(r, "</ifm|arg_type>\n");
                }
                put(r, "<ifm|arg_value>");
                if (is_str(args->v[a])) put(r, args->v[a]->s); else tojson(r, args->v[a]);
                put(r, "</ifm|arg_value>\n");
            }
            put(r, "</ifm|tool_call>");
        }
    }
    put(r, "\n</ifm|tool_calls>");
}
static void render_tool_response(R* r, const Json* c) {
    if (is_str(c)) { put(r, "<|ifm|im_start|>tool\n"); put(r, c->s); put(r, "<|ifm|im_end|>"); return; }
    if (is_arr(c)) {
        if (!c->n) { raise(r, "tool message content list must not be empty.", NULL, NULL, NULL, NULL); return; }
        put(r, "<|ifm|im_start|>tool\n");
        for (int i = 0; i < c->n; ++i) {
            if (i) put(r, "\n");
            const Json* it = c->v[i];
            if (is_str(it)) put(r, it->s);
            else if (is_obj(it) && is_str(get(it, "text"))) put(r, get(it, "text")->s);
            else tojson(r, it);
        }
        put(r, "<|ifm|im_end|>");
        return;
    }
    put(r, "<|ifm|im_start|>tool\n");
    if (c) tojson(r, c); else put(r, "null");
    put(r, "<|ifm|im_end|>");
}
static const char* role_of(const Json* m) { return json_gets(m, "role") ? json_gets(m, "role") : ""; }

char* ct_render(const Json* messages, const Json* tools, const CtOpts* o, char* err, int errlen) {
    R r = {0};
    r.err = err;
    r.errlen = errlen;
    if (err && errlen) err[0] = 0;
    const char* pres = o && o->presentation ? o->presentation : "markdown";
    const char* call = o && o->call_format ? o->call_format : "xml";
    if (strcmp(pres, "json") && strcmp(pres, "xml") && strcmp(pres, "markdown"))
        raise(&r, "Unsupported tool_presentation_format: '", pres, "'. Supported formats: json, xml, markdown.", NULL, NULL);
    else if (strcmp(call, "json") && strcmp(call, "xml") && strcmp(call, "xml_typed"))
        raise(&r, "Unsupported tool_call_format: '", call, "'. Supported formats: json, xml, xml_typed.", NULL, NULL);
    else if (!is_arr(messages) || !messages->n) raise(&r, "messages must be a non-empty list", NULL, NULL, NULL, NULL);
    const Json* first = r.failed ? NULL : messages->v[0];
    put(&r, BOS);
    const Json* avail = json_truthy(tools) ? tools : NULL;
    if (!r.failed && !avail && !strcmp(role_of(first), "system") && json_truthy(get(first, "tools"))) avail = get(first, "tools");
    if (!r.failed && avail && is_arr(avail)) {
        const int bad = validate_tools(&r, avail, strcmp(pres, "json") != 0);
        if (!r.failed) {
            put(&r, "<|ifm|im_start|>system\n# Tools\nYou may call one or more tools to assist with the user query.\n\n"
                    "Available tools are:\n\n");
            if (!strcmp(pres, "json") || bad) render_tools_json(&r, avail);
            else if (!strcmp(pres, "markdown")) render_tools_markdown(&r, avail);
            else raise(&r, "tool_presentation_format 'xml' is not supported here; use markdown or json", NULL, NULL, NULL, NULL);
            put(&r, "\n\nWhen calling tools, you MUST follow the tool-call format below:\n\n");
            put(&r, !strcmp(call, "json") ? CALL_JSON : !strcmp(call, "xml") ? CALL_XML : CALL_XML_TYPED);
            if (!strcmp(role_of(first), "system") && json_truthy(get(first, "content"))) {
                put(&r, "\n\n");
                py_str(&r.o, get(first, "content"));
            }
            put(&r, "<|ifm|im_end|>");
        }
    } else if (!r.failed && !strcmp(role_of(first), "system")) {
        put(&r, "<|ifm|im_start|>system\n");
        if (is_str(get(first, "content"))) put(&r, get(first, "content")->s);
        put(&r, "<|ifm|im_end|>");
    }
    for (int i = 0; !r.failed && i < messages->n; ++i) {
        const Json* m = messages->v[i];
        const char* role = role_of(m);
        const char* content = json_gets(m, "content") ? json_gets(m, "content") : "";
        if (!strcmp(role, "user") || (!strcmp(role, "system") && i > 0)) {
            put(&r, "<|ifm|im_start|>"); put(&r, role); put(&r, "\n"); put(&r, content); put(&r, "<|ifm|im_end|>");
        } else if (!strcmp(role, "assistant")) {
            static const char* fields[] = {"think", "think_fast", "think_faster", "reasoning_content", "reasoning"};
            static const char* tags[] = {"ifm|think", "ifm|think_fast", "ifm|think_faster", "ifm|think", "ifm|think"};
            const char *think = NULL, *tag = NULL;
            int any = 0;
            for (int f = 0; f < 5; ++f) {
                any |= get(m, fields[f]) != NULL;
                if (!think && is_str(get(m, fields[f]))) { think = get(m, fields[f])->s; tag = tags[f]; }
            }
            if (!think) {
                if (!any) raise(&r, "Assistant message is missing a thinking field. Provide one of: think, reasoning, reasoning_content, think_fast, think_faster.", NULL, NULL, NULL, NULL);
                else raise(&r, "Assistant thinking fields must be strings. Provide one of: think, reasoning, reasoning_content, think_fast, think_faster as a string.", NULL, NULL, NULL, NULL);
                break;
            }
            put(&r, "<|ifm|im_start|>assistant\n<"); put(&r, tag); put(&r, ">\n"); put(&r, think);
            put(&r, "</"); put(&r, tag); put(&r, ">"); put(&r, content);
            if (json_truthy(get(m, "tool_calls"))) render_tool_calls_block(&r, get(m, "tool_calls"), call, avail);
            put(&r, "<|ifm|im_end|>");
        } else if (!strcmp(role, "tool")) render_tool_response(&r, get(m, "content"));
    }
    if (!r.failed && (!o || o->add_generation_prompt)) {
        const char* effort = o && o->effort ? o->effort : "high";
        if (!strcmp(effort, "high")) put(&r, "<|ifm|im_start|>assistant\n<ifm|think>\n");
        else if (!strcmp(effort, "medium")) put(&r, "<|ifm|im_start|>assistant\n<ifm|think_fast>\n");
        else if (!strcmp(effort, "low")) put(&r, "<|ifm|im_start|>assistant\n<ifm|think_faster>\n");
        else raise(&r, "Unsupported reasoning_effort: '", effort, "'. Supported values: high, medium, low.", NULL, NULL);
    }
    for (int i = 0; i < r.ntmp; ++i) json_free(r.tmp[i]);
    free(r.tmp);
    free(r.seen.p);
    if (r.failed) { free(r.o.p); return NULL; }
    return buf_take(&r.o);
}

// ---- OpenAI messages -> template input ----

static char* parts_text(const Json* c) {
    Buf b = {0};
    for (int i = 0; i < c->n; ++i) {
        const Json* p = c->v[i];
        const char* t = is_str(p) ? p->s : json_gets(p, "text");
        if (!t) continue;
        if (b.n) buf_puts(&b, "\n");
        buf_puts(&b, t);
    }
    return buf_take(&b);
}
int ct_normalize(Json* messages, char* err, int errlen) {
    if (!is_arr(messages)) { snprintf(err, (size_t) errlen, "messages must be a list"); return -1; }
    for (int i = 0; i < messages->n; ++i) {
        Json* m = messages->v[i];
        if (!is_obj(m)) { snprintf(err, (size_t) errlen, "messages[%d] must be an object", i); return -1; }
        const char* role = role_of(m);
        if (!strcmp(role, "developer")) json_set(m, "role", json_str("system"));
        role = role_of(m);
        Json* c = get(m, "content");
        if (is_arr(c) && strcmp(role, "tool")) {
            char* t = parts_text(c);
            json_set(m, "content", json_str(t));
            free(t);
        }
        if (strcmp(role, "assistant")) continue;
        Json* calls = get(m, "tool_calls");
        for (int k = 0; is_arr(calls) && k < calls->n; ++k) {
            Json* fn = get(calls->v[k], "function");
            Json* args = get(fn, "arguments");
            if (!is_str(args)) continue;
            Json* parsed = args->s[0] ? json_parse(args->s, strlen(args->s), NULL, 0) : json_new(J_OBJ);
            if (!is_obj(parsed)) {
                json_free(parsed);
                snprintf(err, (size_t) errlen, "messages[%d].tool_calls[%d].function.arguments is not a JSON object", i, k);
                return -1;
            }
            json_set(fn, "arguments", parsed);
        }
        static const char* fields[] = {"think", "think_fast", "think_faster", "reasoning_content", "reasoning"};
        int any = 0;
        for (int f = 0; f < 5; ++f) any |= is_str(get(m, fields[f]));
        if (any) continue;
        const char* text = json_gets(m, "content") ? json_gets(m, "content") : "";
        static const char* tags[] = {"ifm|think", "ifm|think_fast", "ifm|think_faster"};
        int split = 0;
        for (int t = 0; t < 3 && !split; ++t) {
            char open[32], close[32];
            snprintf(open, sizeof open, "<%s>", tags[t]);
            snprintf(close, sizeof close, "</%s>", tags[t]);
            const char* a = strstr(text, open);
            const char* z = strstr(text, close);
            if (!z) continue;
            const char* s = a && a < z ? a + strlen(open) : text;
            while (*s == '\n') ++s;
            Buf th = {0}, rest = {0};
            buf_put(&th, s, (size_t) (z - s));
            buf_puts(&rest, z + strlen(close));
            json_set(m, t == 0 ? "think" : t == 1 ? "think_fast" : "think_faster", json_str(th.p ? th.p : ""));
            json_set(m, "content", json_str(rest.p ? rest.p : ""));
            free(th.p);
            free(rest.p);
            split = 1;
        }
        if (!split) json_set(m, "reasoning_content", json_str(""));
    }
    return 0;
}
