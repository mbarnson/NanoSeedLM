#!/usr/bin/env python3
"""Continuous-batching benchmark against a running nslm-serve: N concurrent streaming requests.

    tools/serve_bench.py [--url http://127.0.0.1:8080] [--streams 1,2,4,8,16,32] [--tokens 128] [--runs 2] [--out FILE]

Each level starts N streaming chat requests at once (distinct short prompts, temperature 1 with fixed seeds,
reasoning_effort high so the model keeps going; max_tokens --tokens).  Reported per level (median of --runs, a third run
when two differ by more than 5%): aggregate decode tok/s (tokens after each stream's first, over the span from the first
stream's first token to the last stream's end), per-request decode tok/s and time to first token, and the machine state
(power, load, the busiest other processes, GPU utilization) before and after; each level first waits (up to
--idle-wait s) for the GPU to be idle (macOS: under 5% busy), so another GPU job does not share the run.  The server's
--max-seqs must be >= the largest N.
"""
import argparse
import http.client
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse

TOPICS = ["the history of lighthouses", "how bees make honey", "the physics of bicycles", "why the sky is blue",
          "the life of a river", "how compilers work", "the first moon landing", "how bread rises", "ocean tides",
          "the invention of paper", "how vaccines train immunity", "the migration of whales", "how GPS finds you",
          "the making of glass", "how volcanoes form", "the story of chess", "how wind turbines work", "coral reefs",
          "the printing press", "how batteries store energy", "desert ecosystems", "the Silk Road", "how clocks keep time",
          "the water cycle", "how airplanes fly", "the origin of zero", "how mushrooms grow", "the Great Wall",
          "how radio works", "the life of stars", "how maps are made", "the discovery of penicillin"]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:   # noqa: BLE001
        return f"({e})"


def machine_state():
    s = {"load": sh("sysctl -n vm.loadavg || cat /proc/loadavg"),
         "power": sh("pmset -g batt | head -1 || echo unknown"),
         "lowpower": sh("pmset -g | grep -i lowpowermode | tr -s ' '"),
         "top_cpu": sh("ps -A -o %cpu=,comm= | sort -rn | head -5 | tr -s ' ' | tr '\\n' ';'"),
         "gpu_util": gpu_util()}
    return s


def gpu_util():
    out = sh("ioreg -r -c AGXAccelerator -d 1 | grep -o '\"Device Utilization %\"=[0-9]*' | head -1")
    return int(out.split("=")[1]) if "=" in out else -1


def wait_idle(limit):
    t0 = time.time()
    while time.time() - t0 < limit:
        u = gpu_util()
        if u < 5:
            return u
        time.sleep(5)
    return gpu_util()


def one(url, i, n_tok, res):
    u = urllib.parse.urlparse(url)
    body = {"messages": [{"role": "user", "content": f"Write a detailed essay about {TOPICS[i % len(TOPICS)]}."}],
            "max_tokens": n_tok, "temperature": 1.0, "seed": 7 + i, "reasoning_effort": "high", "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.time()
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=1200)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    first = None
    usage = None
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
            continue
        ev = json.loads(line[6:])
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            if first is None and (d.get("content") or d.get("reasoning_content")):
                first = time.time()
    t1 = time.time()
    c.close()
    res[i] = {"t0": t0, "first": first, "end": t1, "tokens": usage["completion_tokens"] if usage else 0}


def level(url, n, n_tok):
    res = [None] * n
    ths = [threading.Thread(target=one, args=(url, i, n_tok, res)) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    span = max(r["end"] for r in res) - min(r["first"] for r in res)
    agg = sum(r["tokens"] - 1 for r in res) / span
    per = [(r["tokens"] - 1) / (r["end"] - r["first"]) for r in res]
    ttft = [r["first"] - r["t0"] for r in res]
    return {"agg": agg, "per_med": statistics.median(per), "per_min": min(per), "ttft_med": statistics.median(ttft),
            "ttft_max": max(ttft), "tokens": sum(r["tokens"] for r in res)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--streams", default="1,2,4,8,16,32")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--idle-wait", type=float, default=3600)
    a = ap.parse_args()
    level(a.url, 2, 16)   # warm-up
    rows = []
    for n in [int(x) for x in a.streams.split(",")]:
        idle = wait_idle(a.idle_wait)
        before = machine_state()
        runs = [level(a.url, n, a.tokens) for _ in range(a.runs)]
        if len(runs) >= 2 and abs(runs[0]["agg"] - runs[1]["agg"]) > 0.05 * max(runs[0]["agg"], runs[1]["agg"]):
            runs.append(level(a.url, n, a.tokens))   # noisy: a third run
        after = machine_state()
        med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
        row = {"label": a.label, "streams": n, "runs": len(runs), "agg_runs": [round(r["agg"], 1) for r in runs], **med,
               "state_before": before, "state_after": after}
        rows.append(row)
        print(f"{a.label} {n:3d} streams: aggregate {med['agg']:7.1f} tok/s (runs {row['agg_runs']}), per request "
              f"{med['per_med']:6.1f} tok/s (min {med['per_min']:.1f}), ttft {med['ttft_med']:.2f} s (max {med['ttft_max']:.2f})",
              flush=True)
        print(f"     state: GPU busy before {idle}% | load {after['load']} | {after['power']} | top {after['top_cpu']}",
              flush=True)
    if a.out:
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
