// harness/tokenizer.m - byte-level BPE from an HF tokenizer.json (see tokenizer.h).
// Pipeline: added tokens are split out first (leftmost, longest match, never
// normalized); each remaining segment is NFC-normalized, split by the pre-tokenizer regex (ICU via
// NSRegularExpression, the same pattern string), mapped byte-to-unicode (GPT-2 byte level), and merged by BPE rank.
// The post-processor is plain ByteLevel: no BOS is added by the tokenizer itself.
#import <Foundation/Foundation.h>
#include "tokenizer.h"

struct Tok {
    NSDictionary<NSString*, NSNumber*>* vocab;
    NSDictionary<NSString*, NSNumber*>* ranks;   // "a b" -> merge rank
    NSArray<NSString*>* id_to_str;               // id -> vocab string or added-token content
    NSArray<NSString*>* added;                   // added-token contents, longest first
    NSDictionary<NSString*, NSNumber*>* added_ids;
    NSSet<NSNumber*>* added_set;
    NSRegularExpression* re;
    NSMutableDictionary<NSString*, NSArray<NSNumber*>*>* cache;
    unichar byte2u[256];
    int u2byte[512];
    int bos;
};

Tok* tok_open(const char* path, char* err, int errlen) {
    @autoreleasepool {
        NSData* d = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
        NSDictionary* j = d ? [NSJSONSerialization JSONObjectWithData:d options:0 error:nil] : nil;
        if (!j) { snprintf(err, errlen, "cannot read %s", path); return NULL; }
        Tok* t = (Tok*) calloc(1, sizeof(Tok));
        NSDictionary* model = j[@"model"];
        if (![model[@"type"] isEqualToString:@"BPE"]) { snprintf(err, errlen, "not a BPE tokenizer"); free(t); return NULL; }
        t->vocab = model[@"vocab"];
        NSMutableDictionary* ranks = [NSMutableDictionary dictionaryWithCapacity:[model[@"merges"] count]];
        NSInteger r = 0;
        for (id m in model[@"merges"]) {
            NSString* k = [m isKindOfClass:[NSArray class]] ? [NSString stringWithFormat:@"%@ %@", m[0], m[1]] : m;
            ranks[k] = @(r++);
        }
        t->ranks = ranks;
        NSInteger maxid = 0;
        for (NSNumber* v in t->vocab.allValues) maxid = MAX(maxid, v.integerValue);
        NSMutableDictionary* aids = [NSMutableDictionary new];
        for (NSDictionary* a in j[@"added_tokens"]) {
            aids[a[@"content"]] = a[@"id"];
            maxid = MAX(maxid, [a[@"id"] integerValue]);
        }
        t->added_ids = aids;
        t->added_set = [NSSet setWithArray:aids.allValues];
        t->added = [aids.allKeys sortedArrayUsingComparator:^NSComparisonResult(NSString* a, NSString* b) {
            return a.length > b.length ? NSOrderedAscending : a.length < b.length ? NSOrderedDescending : NSOrderedSame;
        }];
        NSMutableArray* i2s = [NSMutableArray arrayWithCapacity:(NSUInteger) maxid + 1];
        for (NSInteger i = 0; i <= maxid; ++i) [i2s addObject:@""];
        for (NSString* k in t->vocab) i2s[t->vocab[k].unsignedIntegerValue] = k;
        for (NSString* k in aids) i2s[((NSNumber*) aids[k]).unsignedIntegerValue] = k;
        t->id_to_str = i2s;
        NSNumber* bos = aids[@"<|begin_of_text|>"];
        t->bos = bos ? bos.intValue : -1;

        // the pre-tokenizer: a Sequence of one Split regex (Isolated) and a ByteLevel without its own regex
        NSString* pat = nil;
        NSDictionary* pre = j[@"pre_tokenizer"];
        NSArray* seq = [pre[@"type"] isEqualToString:@"Sequence"] ? pre[@"pretokenizers"] : @[ pre ];
        for (NSDictionary* p in seq) {
            if ([p[@"type"] isEqualToString:@"Split"]) pat = p[@"pattern"][@"Regex"];
            if ([p[@"type"] isEqualToString:@"ByteLevel"] && [p[@"use_regex"] boolValue]) {
                snprintf(err, errlen, "ByteLevel use_regex not supported"); free(t); return NULL;
            }
        }
        NSError* re_err = nil;
        t->re = pat ? [NSRegularExpression regularExpressionWithPattern:pat options:0 error:&re_err] : nil;
        if (!t->re) { snprintf(err, errlen, "pre-tokenizer regex: %s", re_err.localizedDescription.UTF8String); free(t); return NULL; }
        t->cache = [NSMutableDictionary new];

        // GPT-2 bytes_to_unicode
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            const bool keep = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            t->byte2u[b] = keep ? (unichar) b : (unichar) (256 + n++);
        }
        for (int i = 0; i < 512; ++i) t->u2byte[i] = -1;
        for (int b = 0; b < 256; ++b) t->u2byte[t->byte2u[b]] = b;
        return t;
    }
}

void tok_close(Tok* t) {
    if (!t) return;
    t->vocab = nil; t->ranks = nil; t->id_to_str = nil; t->added = nil; t->added_ids = nil; t->added_set = nil;
    t->re = nil; t->cache = nil;
    free(t);
}

int tok_bos(Tok* t) { return t->bos; }

static NSArray<NSNumber*>* bpe(Tok* t, NSString* word) {
    NSArray* hit = t->cache[word];
    if (hit) return hit;
    NSData* u8 = [word dataUsingEncoding:NSUTF8StringEncoding];
    const uint8_t* b = u8.bytes;
    NSMutableArray<NSString*>* sym = [NSMutableArray arrayWithCapacity:u8.length];
    for (NSUInteger i = 0; i < u8.length; ++i) [sym addObject:[NSString stringWithCharacters:&t->byte2u[b[i]] length:1]];
    while (sym.count > 1) {
        NSInteger best = NSIntegerMax, at = -1;
        for (NSUInteger i = 0; i + 1 < sym.count; ++i) {
            NSNumber* r = t->ranks[[NSString stringWithFormat:@"%@ %@", sym[i], sym[i + 1]]];
            if (r && r.integerValue < best) { best = r.integerValue; at = (NSInteger) i; }
        }
        if (at < 0) break;
        // merge every occurrence of the best pair, left to right (HF merges by rank; equal pairs merge in one pass)
        NSString* a = sym[at], *c = sym[at + 1];
        NSMutableArray* next = [NSMutableArray arrayWithCapacity:sym.count];
        for (NSUInteger i = 0; i < sym.count;) {
            if (i + 1 < sym.count && [sym[i] isEqualToString:a] && [sym[i + 1] isEqualToString:c]) {
                [next addObject:[a stringByAppendingString:c]];
                i += 2;
            } else {
                [next addObject:sym[i]];
                i += 1;
            }
        }
        sym = next;
    }
    NSMutableArray* ids = [NSMutableArray arrayWithCapacity:sym.count];
    for (NSString* s in sym) {
        NSNumber* id_ = t->vocab[s];
        if (!id_) return nil;
        [ids addObject:id_];
    }
    if (t->cache.count < 200000) t->cache[word] = ids;
    return ids;
}

static int encode_plain(Tok* t, NSString* seg, int32_t* out, int cap, int n) {
    NSString* s = seg.precomposedStringWithCanonicalMapping;   // NFC
    __block int k = n;
    __block bool bad = false;
    [t->re enumerateMatchesInString:s options:0 range:NSMakeRange(0, s.length)
                         usingBlock:^(NSTextCheckingResult* m, NSMatchingFlags f, BOOL* stop) {
        (void) f;
        NSArray* ids = bpe(t, [s substringWithRange:m.range]);
        if (!ids) { bad = true; *stop = YES; return; }
        for (NSNumber* v in ids) { if (k < cap) out[k] = v.intValue; ++k; }
    }];
    return bad ? -1 : k;
}

int tok_encode(Tok* t, const char* text, int add_bos, int32_t* out, int cap) {
    @autoreleasepool {
        NSString* s = [NSString stringWithUTF8String:text];
        if (!s) return -1;
        int n = 0;
        NSUInteger seg = 0, i = 0;
        const NSUInteger len = s.length;
        while (i < len) {
            NSString* hit = nil;
            const unichar c0 = [s characterAtIndex:i];
            for (NSString* a in t->added) {   // longest first
                if (a.length == 0 || [a characterAtIndex:0] != c0 || i + a.length > len) continue;
                if ([s compare:a options:NSLiteralSearch range:NSMakeRange(i, a.length)] == NSOrderedSame) { hit = a; break; }
            }
            if (!hit) { ++i; continue; }
            if (i > seg) {
                n = encode_plain(t, [s substringWithRange:NSMakeRange(seg, i - seg)], out, cap, n);
                if (n < 0) return -1;
            }
            if (n < cap) out[n] = t->added_ids[hit].intValue;
            ++n;
            i += hit.length;
            seg = i;
        }
        if (seg < len) {
            n = encode_plain(t, [s substringFromIndex:seg], out, cap, n);
            if (n < 0) return -1;
        }
        if (add_bos && t->bos >= 0 && !(n > 0 && cap > 0 && out[0] == t->bos)) {
            if (n + 1 <= cap) {
                memmove(out + 1, out, (size_t) n * sizeof(int32_t));
                out[0] = t->bos;
            }
            ++n;
        }
        return n;
    }
}

char* tok_decode(Tok* t, const int32_t* ids, int n, int* valid) {
    @autoreleasepool {
        NSMutableData* bytes = [NSMutableData data];
        for (int i = 0; i < n; ++i) {
            if (ids[i] < 0 || (NSUInteger) ids[i] >= t->id_to_str.count) continue;
            NSString* s = t->id_to_str[(NSUInteger) ids[i]];
            if ([t->added_set containsObject:@(ids[i])]) {
                [bytes appendData:[s dataUsingEncoding:NSUTF8StringEncoding]];
                continue;
            }
            for (NSUInteger k = 0; k < s.length; ++k) {
                const unichar c = [s characterAtIndex:k];
                const int b = c < 512 ? t->u2byte[c] : -1;
                if (b >= 0) { uint8_t v = (uint8_t) b; [bytes appendBytes:&v length:1]; }
            }
        }
        NSString* str = [[NSString alloc] initWithData:bytes encoding:NSUTF8StringEncoding];
        // a cut at the end of a fixed-length generation may split one character: that is not invalid output
        for (NSUInteger trim = 1; !str && trim <= 3 && trim <= bytes.length; ++trim)
            str = [[NSString alloc] initWithBytes:bytes.bytes length:bytes.length - trim encoding:NSUTF8StringEncoding];
        if (valid) *valid = str != nil;
        char* out = (char*) malloc(bytes.length + 1);
        memcpy(out, bytes.bytes, bytes.length);
        out[bytes.length] = 0;
        return out;
    }
}
