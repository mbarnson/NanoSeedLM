#!/usr/bin/env python3
"""tools/qwen4_cache_sim.py - replay a routing trace of the qwen4 CUDA engine (Q4E_ROUTES=FILE) through expert cache
policies of SLOTS slots: the engine's LRU (its hits must match the engine's), LFU, a static hot set with LRU for the
rest, and Belady's optimum (the most hits any policy can get).  Counts are per (forward, layer, distinct expert).
--window K admits the union of K consecutive decode tokens at once, as a verify step of K rows (MTP drafts) would.

  python3 tools/qwen4_cache_sim.py --trace routes.bin --layers 48 --experts 512 --top-k 10 --slots 3255 4069
"""
import argparse
import heapq
import struct
from collections import Counter, OrderedDict, defaultdict


def read_trace(path, layers, k):
    """[(T, [set of units for each layer])] in the engine's admit order; unit = layer * experts + expert."""
    out, data, at = [], open(path, "rb").read(), 0
    while at < len(data):
        (t,) = struct.unpack_from("<i", data, at)
        n = layers * t * k
        ids = struct.unpack_from("<%di" % n, data, at + 4)
        at += 4 + 4 * n
        out.append((t, [sorted(set(ids[l * t * k:(l + 1) * t * k])) for l in range(layers)]))
    return out


def window(trace, k):
    """Runs of k consecutive one-row forwards as one forward of k rows (each layer: the union of their experts)."""
    out, run = [], []
    def flush():
        if run: out.append((len(run), [sorted(set().union(*(r[l] for r in run))) for l in range(len(run[0]))]))
        run.clear()
    for t, layer_sets in trace:
        if t != 1:
            flush()
            out.append((t, layer_sets))
            continue
        run.append(layer_sets)
        if len(run) == k: flush()
    flush()
    return out


def admits(trace, experts):
    """The admits in order: (decode, [units])."""
    return [(t <= 8, [l * experts + x for x in xs]) for t, layer_sets in trace for l, xs in enumerate(layer_sets)]


def initial(slots, layers, experts):   # the engine's fill: expert-major, every layer the same share
    return [l * experts + x for x in range(experts) for l in range(layers)][:slots]


def lfu_victim(heap, res, cnt, now, tick):
    keep = []
    while heap:
        c, t, v = heapq.heappop(heap)
        if v not in res or c != cnt[v]: continue
        if v in now: keep.append((c, t, v)); continue
        for e in keep: heapq.heappush(heap, e)
        return v
    for e in keep: heapq.heappush(heap, e)
    return None


def run(adm, slots, layers, experts, policy, hot=()):
    """Hits and misses (all admits, decode admits) for policy 'lru' or 'lfu', with the units of hot pinned."""
    hot = set(hot)
    free = slots - len(hot)
    res = OrderedDict((u, None) for u in initial(slots, layers, experts) if u not in hot)
    while len(res) > free: res.popitem(last=False)   # the pinned units take slots of the fill
    cnt, st = Counter(), [0, 0, 0, 0]
    heap = [(0, 0, u) for u in res]   # lfu: (uses, tick, unit), stale entries skipped
    for tick, (dec, units) in enumerate(adm, 1):
        now = set(units)
        for u in units:
            cnt[u] += 1
            hit = u in hot or u in res
            st[0 if hit else 1] += 1
            if dec: st[2 if hit else 3] += 1
            if u in hot: continue
            if u in res:
                res.move_to_end(u)
            elif len(res) < free:
                res[u] = None
            else:   # the victim: not needed now, the oldest (lru) or the least used (lfu); none: not admitted (the engine)
                victim = next((v for v in res if v not in now), None) if policy == "lru" else lfu_victim(heap, res, cnt, now, tick)
                if victim is None: continue
                del res[victim]
                res[u] = None
            if policy == "lfu": heapq.heappush(heap, (cnt[u], tick, u))
    return st


def belady(adm, slots, layers, experts):
    """The optimum: on a miss, replace the unit used again latest (or never)."""
    uses = defaultdict(list)
    for i, (_, units) in enumerate(adm):
        for u in units: uses[u].append(i)
    nxt = {u: 0 for u in uses}
    INF = 1 << 60
    def next_use(u, i):
        p = uses.get(u, [])
        j = nxt.get(u, 0)
        while j < len(p) and p[j] <= i: j += 1
        nxt[u] = j
        return p[j] if j < len(p) else INF
    res, heap, st = set(initial(slots, layers, experts)), [], [0, 0, 0, 0]
    for u in res: heapq.heappush(heap, (-next_use(u, -1), u))
    for i, (dec, units) in enumerate(adm):
        now = set(units)
        for u in units:
            hit = u in res
            st[0 if hit else 1] += 1
            if dec: st[2 if hit else 3] += 1
            if not hit:
                keep, v = [], None
                while heap:   # the farthest next use among the resident units not needed now (stale entries skipped)
                    nu, w = heapq.heappop(heap)
                    if w not in res: continue
                    if w in now: keep.append((nu, w)); continue
                    if -nu != next_use(w, i): heapq.heappush(heap, (-next_use(w, i), w)); continue
                    v = w
                    break
                for e in keep: heapq.heappush(heap, e)
                if v is None: continue
                res.discard(v)
                res.add(u)
        for u in now: heapq.heappush(heap, (-next_use(u, i), u))
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--slots", type=int, nargs="+", required=True)
    ap.add_argument("--window", type=int, default=1)
    a = ap.parse_args()
    trace = read_trace(a.trace, a.layers, a.top_k)
    ntok = sum(t for t, _ in trace if t == 1)
    if a.window > 1: trace = window(trace, a.window)
    adm = admits(trace, a.experts)
    half = len(adm) // 2
    freq = Counter(u for _, units in adm[:half] for u in units)   # the hot set from the first half of the admits only
    print("%d admits (%d decode), %d decode tokens, window %d" % (len(adm), sum(d for d, _ in adm), ntok, a.window))
    for s in a.slots:
        rows = [("lru (the engine)", run(adm, s, a.layers, a.experts, "lru")), ("lfu", run(adm, s, a.layers, a.experts, "lfu"))]
        for share in (0.25, 0.5, 0.75):
            hot = [u for u, _ in freq.most_common(int(s * share))]
            rows.append(("hot %d%% + lru" % (100 * share), run(adm, s, a.layers, a.experts, "hot", hot)))
        rows.append(("belady (optimum)", belady(adm, s, a.layers, a.experts)))
        print("slots %d:" % s)
        for name, (h, m, dh, dm) in rows:
            print("  %-18s hits %7d  misses %7d  (%.1f%%)   decode: hits %7d  misses %7d  (%.1f%%), %.1f misses a token"
                  % (name, h, m, 100 * h / max(1, h + m), dh, dm, 100 * dh / max(1, dh + dm), dm / max(1, ntok)))


if __name__ == "__main__":
    main()
