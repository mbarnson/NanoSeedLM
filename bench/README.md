# Decode and prompt speed of the release, long runs (cloud, 2026-11-06)

Three requests to `nslm-serve` (the stock path: the release binaries, `run-PACK` with Unsloth's Q8_0 PLE table,
`--ctx 32768`), thinking on, temperature 0:

| Request | Prompt | Output |
|---|---|---|
| `long-0k.json` | a short task (a B-tree module) | at most 4,096 tokens |
| `depth-15k.json` | `code-15k.txt`: 15,000 tokens of `nslm/lib_*.c` at `ff92415`, then a question | at most 2,048 tokens |
| `depth-30k.json` | `code-30k.txt`: 30,000 tokens of the same, then the question | at most 2,048 tokens |

    curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d @bench/long-0k.json >/dev/null

Report from the server log: the prompt tokens and prompt tok/s, the generated tokens and decode tok/s, and TTFT. Run
each request twice; the second is warm. Record the machine state (other GPU and CPU use, power).
