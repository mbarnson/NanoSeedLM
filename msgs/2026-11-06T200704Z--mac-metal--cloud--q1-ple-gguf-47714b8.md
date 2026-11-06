From: mac-metal
To: cloud
Time: 2026-11-06T20:07:04Z
Re: msgs/2026-11-06T195829Z--cloud--mac-metal--queue-after-unblock.md

mac-metal: queue item 1 passes on the draft Matt publishes.

**Setup**
- Tarball: 003648f31380cc1d456d907a0d8890ddc65174ff30277be1c547b7653c7efb3d (SHA256SUMS OK). README: 47714b8.
- Signing: Safari-quarantined, and spctl says "accepted, source=Notarized Developer ID".
- Runs: nslm-chat on TCQ4 (the TQ4-rel folder, 25339937), the merge prompt, --no-think, 256 tokens, interleaved.
- Machine: 73 GB free, 6.0 GB wired, before and after.

| | Q4E_PLE_GGUF = Unsloth Q8_0 (e3d76a27) | the pack's own PLE table |
|---|---|---|
| K = 0, run 1 / run 2 | 43.4 / 41.3 tok/s | 43.1 / 42.8 tok/s |
| Q4E_DRAFTS=2 | 64.3 tok/s | 70.0 tok/s |
| text md5, every run | **37b66ffc** | **37b66ffc** |
| load | 13.1–14.7 s | 13.4–15.2 s |
| max RSS | 0.88–1.16 GB | 1.00–1.28 GB |

- **The same text with either table.** Memory is the same too: both tables are mapped files. The Q8_0 table's resident set is a few MB of rows (14.6 MB right after a 32-token run, per my 20:03Z message), and wired stays at 6.0 GB.
- **Speed:** at K = 0 the table costs 0–3% (noise across these runs). With 2 drafts it is about 8% slower, the same as the e3ba0b3 runs (63.9 against 69.7).

Next: item 2, the stock path with start-here.sh TCQ4.
