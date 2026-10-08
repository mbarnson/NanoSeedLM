#!/usr/bin/env python3
"""Concurrent clients against a running nslm-serve (continuous batching).

    tools/serve_concurrency_test.py [--url http://127.0.0.1:8080] [--n 6]

Checks, with streaming chat requests of mixed sampling parameters (greedy, temperature / top_p / top_k / min_p, seeds):
  - each request alone, then all of them at once: every request's text is the same both times (a request's tokens do
    not depend on what else runs), and the concurrent streams overlap in time;
  - max_tokens ends a request with finish_reason "length" after exactly that many tokens;
  - a client that disconnects mid-stream frees its slot: afterwards --n requests at once still all complete.
Exit status 0 when every check passes.
"""
import argparse
import http.client
import json
import sys
import threading
import time
import urllib.parse

PROMPTS = [
    "Write a haiku about the sea.",
    "List five uses of a paperclip.",
    "Explain what a hash table is in two sentences.",
    "What is 17 * 23? Show your work briefly.",
    "Give three tips for writing clear code.",
    "Describe the color blue to someone who cannot see.",
    "Name four planets and one fact about each.",
    "Write a limerick about a cat.",
]


def request(i, max_tokens=96):
    r = {"messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}], "max_tokens": max_tokens,
         "reasoning_effort": "low", "stream": True, "stream_options": {"include_usage": True}, "seed": 1000 + i}
    kind = i % 4
    if kind == 0:
        r["temperature"] = 0
    elif kind == 1:
        r.update(temperature=1.0, top_p=0.95)
    elif kind == 2:
        r.update(temperature=0.8, top_k=40)
    else:
        r.update(temperature=1.2, min_p=0.05)
    return r


def stream(url, body, out, stop_after=None):
    """POST a streaming request; out gets text (reasoning + content), finish, usage, chunk times."""
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = c.getresponse()
    out.update(text="", reasoning="", finish=None, usage=None, times=[], status=resp.status, done=False)
    n = 0
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        d = line[6:]
        if d == b"[DONE]":
            out["done"] = True
            break
        ev = json.loads(d)
        if ev.get("usage"):
            out["usage"] = ev["usage"]
        for ch in ev.get("choices", []):
            delta = ch.get("delta", {})
            out["reasoning"] += delta.get("reasoning_content") or ""
            out["text"] += delta.get("content") or ""
            if ch.get("finish_reason"):
                out["finish"] = ch["finish_reason"]
        out["times"].append(time.time())
        n += 1
        if stop_after and n >= stop_after:
            resp.close()   # disconnect mid-stream
            c.close()
            return
    c.close()


def run_all(url, bodies):
    outs = [dict() for _ in bodies]
    ths = [threading.Thread(target=stream, args=(url, b, o)) for b, o in zip(bodies, outs)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=6, help="concurrent requests (<= the server's --max-seqs to overlap fully)")
    a = ap.parse_args()
    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("ok    " if ok else "FAIL  ") + what)
        fails += not ok

    bodies = [request(i) for i in range(a.n)]
    alone = []
    for b in bodies:
        o = {}
        stream(a.url, b, o)
        alone.append(o)
    t0 = time.time()
    together = run_all(a.url, bodies)
    wall = time.time() - t0
    for i, (x, y) in enumerate(zip(alone, together)):
        check(x["done"] and y["done"] and x["finish"] and y["finish"], f"request {i}: complete alone and together "
              f"({x['finish']}, {y['finish']})")
        check((x["reasoning"], x["text"]) == (y["reasoning"], y["text"]),
              f"request {i}: same text alone and with {a.n - 1} others ({len(x['reasoning']) + len(x['text'])} chars)")
    first = max(o["times"][0] for o in together)
    last = min(o["times"][-1] for o in together)
    check(first < last, f"{a.n} streams overlap in time (all started before any ended; {wall:.1f} s)")

    o = {}
    stream(a.url, dict(request(1, max_tokens=7), stream_options={"include_usage": True}), o)
    check(o["finish"] == "length" and o["usage"] and o["usage"]["completion_tokens"] == 7,
          f"max_tokens 7: finish {o['finish']}, {o['usage'] and o['usage']['completion_tokens']} tokens")

    o = {}
    stream(a.url, request(2, max_tokens=4000), o, stop_after=5)   # disconnect after a few chunks
    time.sleep(1.0)
    outs = run_all(a.url, [request(i, max_tokens=24) for i in range(a.n)])
    check(all(x["done"] for x in outs) and max(x["times"][0] for x in outs) < min(x["times"][-1] for x in outs),
          f"after a client disconnect: {a.n} requests at once all complete, overlapping (its slot was freed)")
    print("serve_concurrency_test:", "FAIL" if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
