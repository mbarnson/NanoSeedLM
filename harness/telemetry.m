// harness/telemetry.m - GPU and ANE activity over a window, sudoless, from IOReport.
//   GPU busy %: "GPU Stats" / "GPU Performance States" / GPUPH - share of the window not in state OFF.
//   ANE busy %: "PMP" / "AF BW" / "ANE0 L1 RD+WR" - a histogram of sampled ANE fabric bandwidth; busy = the share of
//               samples above the lowest (idle) bin.  An activity proxy, not compute utilization; GPU-only work shows
//               some crosstalk, so small values mean no ANE work.
// A missing channel reports -1, never 0.
#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <unistd.h>
#include <IOKit/IOKitLib.h>
#include "telemetry.h"

typedef CFMutableDictionaryRef (*CopyGroupFn)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
typedef void* (*CreateSubFn)(void*, CFMutableDictionaryRef, CFMutableDictionaryRef*, uint64_t, CFTypeRef);
typedef CFDictionaryRef (*SamplesFn)(void*, CFMutableDictionaryRef, CFTypeRef);
typedef CFDictionaryRef (*DeltaFn)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
typedef CFStringRef (*StrFn)(CFDictionaryRef);
typedef int32_t (*CountFn)(CFDictionaryRef);
typedef CFStringRef (*StateNameFn)(CFDictionaryRef, int32_t);
typedef int64_t (*StateResFn)(CFDictionaryRef, int32_t);
typedef void (*MergeFn)(CFMutableDictionaryRef, CFMutableDictionaryRef, CFTypeRef);

static struct {
    bool ok;
    void* sub;
    CFMutableDictionaryRef subbed;
    SamplesFn samples;
    DeltaFn delta;
    StrFn group, subgroup, name;
    CountFn count;
    StateNameFn state_name;
    StateResFn state_res;
} g;

static bool telem_init(void) {
    static bool tried = false;
    if (tried) return g.ok;
    tried = true;
    void* h = dlopen("/usr/lib/libIOReport.dylib", RTLD_LAZY);
    if (!h) return false;
    CopyGroupFn cg = (CopyGroupFn) dlsym(h, "IOReportCopyChannelsInGroup");
    CreateSubFn cs = (CreateSubFn) dlsym(h, "IOReportCreateSubscription");
    MergeFn mg = (MergeFn) dlsym(h, "IOReportMergeChannels");
    g.samples = (SamplesFn) dlsym(h, "IOReportCreateSamples");
    g.delta = (DeltaFn) dlsym(h, "IOReportCreateSamplesDelta");
    g.group = (StrFn) dlsym(h, "IOReportChannelGetGroup");
    g.subgroup = (StrFn) dlsym(h, "IOReportChannelGetSubGroup");
    g.name = (StrFn) dlsym(h, "IOReportChannelGetChannelName");
    g.count = (CountFn) dlsym(h, "IOReportStateGetCount");
    g.state_name = (StateNameFn) dlsym(h, "IOReportStateGetNameForIndex");
    g.state_res = (StateResFn) dlsym(h, "IOReportStateGetResidency");
    // every symbol that is dereferenced below; a missing one degrades to "telemetry unavailable" (-1), never a crash
    if (!cg || !cs || !mg || !g.samples || !g.delta || !g.group || !g.subgroup || !g.name || !g.count ||
        !g.state_name || !g.state_res) return false;
    CFMutableDictionaryRef gpu = cg(CFSTR("GPU Stats"), CFSTR("GPU Performance States"), 0, 0, 0);
    CFMutableDictionaryRef ane = cg(CFSTR("PMP"), CFSTR("AF BW"), 0, 0, 0);
    if (!gpu) return false;
    if (ane) mg(gpu, ane, NULL);
    g.sub = cs(NULL, gpu, &g.subbed, 0, NULL);
    g.ok = g.sub != NULL;
    return g.ok;
}

TelemMark telem_mark(void) {
    TelemMark m = {NULL};
    if (telem_init()) m.sample = (void*) g.samples(g.sub, g.subbed, NULL);
    return m;
}

TelemResult telem_since(TelemMark m) {
    TelemResult r = {-1, -1, 0};
    if (!m.sample || !g.ok) return r;
    CFDictionaryRef now = g.samples(g.sub, g.subbed, NULL);
    CFDictionaryRef d = now ? g.delta((CFDictionaryRef) m.sample, now, NULL) : NULL;
    if (!d) {
        if (now) CFRelease(now);
        CFRelease((CFDictionaryRef) m.sample);
        return r;
    }
    NSArray* chs = ((__bridge NSDictionary*) d)[@"IOReportChannels"];
    double gpu_tot = 0, gpu_act = 0, ane_tot = 0, ane_act = 0;
    bool have_ane = false;
    for (id c in chs) {
        CFDictionaryRef ch = (__bridge CFDictionaryRef) c;
        NSString* n = (__bridge NSString*) g.name(ch);
        NSString* s = (__bridge NSString*) g.subgroup(ch);
        if (!n) continue;
        const int32_t k = g.count(ch);
        if ([n isEqualToString:@"GPUPH"]) {
            for (int32_t i = 0; i < k; ++i) {
                const double v = (double) g.state_res(ch, i);
                gpu_tot += v;
                if (![(__bridge NSString*) g.state_name(ch, i) isEqualToString:@"OFF"]) gpu_act += v;
            }
        } else if ([s isEqualToString:@"AF BW"] && [n isEqualToString:@"ANE0 L1 RD+WR"]) {
            have_ane = true;
            for (int32_t i = 0; i < k; ++i) {
                const double v = (double) g.state_res(ch, i);
                ane_tot += v;
                if (i > 0) ane_act += v;   // state 0 is the lowest bandwidth bin (idle)
            }
        }
    }
    if (gpu_tot > 0) {
        r.gpu_busy_pct = 100.0 * gpu_act / gpu_tot;
        // the histogram takes no samples while the ANE is powered down: present but empty means 0% busy
        if (have_ane) r.ane_busy_pct = ane_tot > 0 ? 100.0 * ane_act / ane_tot : 0.0;
    }
    CFRelease(d);
    CFRelease(now);
    CFRelease((CFDictionaryRef) m.sample);
    r.ok = 1;
    return r;
}

long long telem_self_gpu_ns(void) {
    NSString* want = [NSString stringWithFormat:@"pid %d,", getpid()];
    io_iterator_t it;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &it) != KERN_SUCCESS) return -1;
    long long tot = 0;
    io_object_t acc;
    while ((acc = IOIteratorNext(it))) {
        io_iterator_t ch;
        if (IORegistryEntryGetChildIterator(acc, kIOServicePlane, &ch) == KERN_SUCCESS) {
            io_object_t c;
            while ((c = IOIteratorNext(ch))) {
                CFTypeRef cr = IORegistryEntryCreateCFProperty(c, CFSTR("IOUserClientCreator"), kCFAllocatorDefault, 0);
                if (cr && CFGetTypeID(cr) == CFStringGetTypeID() && [(__bridge NSString*) cr hasPrefix:want]) {
                    CFTypeRef au = IORegistryEntryCreateCFProperty(c, CFSTR("AppUsage"), kCFAllocatorDefault, 0);
                    if (au && CFGetTypeID(au) == CFArrayGetTypeID())
                        for (NSDictionary* d in (__bridge NSArray*) au)
                            if ([d isKindOfClass:[NSDictionary class]]) tot += [d[@"accumulatedGPUTime"] longLongValue];
                    if (au) CFRelease(au);
                }
                if (cr) CFRelease(cr);
                IOObjectRelease(c);
            }
            IOObjectRelease(ch);
        }
        IOObjectRelease(acc);
    }
    IOObjectRelease(it);
    return tot;
}
