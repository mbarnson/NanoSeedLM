// Exercise the production splitter without loading a model or opening a server.
#define main nslm_server_main
#include "../harness/nslm-serve.m"
#undef main

static int check_split(NSString* rawText, NSString* wantReasoning, NSString* wantContent, bool chat, size_t cut) {
    NSData* raw = [rawText dataUsingEncoding:NSUTF8StringEncoding];
    Gen g = {0};
    g.close_tag = chat ? @"</ifm|think>" : nil;
    g.reasoning = [NSMutableString string];
    g.content = [NSMutableString string];
    Splitter sp = {0};
    if (cut <= raw.length) {
        emit_upto(NULL, &g, &sp, raw, utf8_complete(raw.bytes, cut), false);
    } else {
        for (size_t n = 1; n < raw.length; ++n)
            emit_upto(NULL, &g, &sp, raw, utf8_complete(raw.bytes, n), false);
    }
    emit_upto(NULL, &g, &sp, raw, raw.length, true);
    if ([g.reasoning isEqualToString:wantReasoning] && [g.content isEqualToString:wantContent]) return 0;
    fprintf(stderr, "split at %zu: %s\n  reasoning: %s\n  content: %s\n", cut, rawText.UTF8String,
            g.reasoning.UTF8String, g.content.UTF8String);
    return 1;
}
static int check(NSString* raw, NSString* reasoning, NSString* content, bool chat) {
    const size_t n = [raw lengthOfBytesUsingEncoding:NSUTF8StringEncoding];
    for (size_t cut = 0; cut <= n + 1; ++cut)
        if (check_split(raw, reasoning, content, chat, cut)) return 1;
    return 0;
}
int main(void) {
    @autoreleasepool {
        int failed = 0;
        NSString* answer = @"{\"say\":\"Hello\",\"commands\":[],\"stop\":true}";
        for (NSString* tag in @[@"ifm|think", @"ifm|think_fast", @"ifm|think_faster"]) {
            failed += check([NSString stringWithFormat:@"plan</%@>\n%@", tag, answer], @"plan", answer, true);
            failed += check([NSString stringWithFormat:@"<%@>plan</%@>%@", tag, tag, answer], @"plan", answer, true);
            failed += check([NSString stringWithFormat:@"<%@>\nplan\n</%@>\n%@", tag, tag, answer], @"\nplan", answer, true);
            failed += check([NSString stringWithFormat:@"<%@>plan", tag], @"plan", @"", true);
        }
        failed += check(@"<ifm|think><ifm|think_fast>plan</ifm|think_faster>answer", @"plan", @"answer", true);
        failed += check(@"café ☕</ifm|think>réponse", @"café ☕", @"réponse", true);
        failed += check(@"plan</ifm|thi", @"plan</ifm|thi", @"", true);
        failed += check(@"plan<unknown>tag</ifm|think>answer", @"plan<unknown>tag", @"answer", true);
        // Missing closes never promote reasoning (including JSON) into executable answer content.
        failed += check(answer, answer, @"", true);
        failed += check(@"</ifm|think><ifm|think>quoted", @"", @"<ifm|think>quoted", true);
        failed += check(@"<ifm|think>raw</ifm|think>", @"", @"<ifm|think>raw</ifm|think>", false);
        printf("thinking splitter: %s\n", failed ? "FAILED" : "passed (all byte splits and one-byte feeds)");
        return failed ? 1 : 0;
    }
}
