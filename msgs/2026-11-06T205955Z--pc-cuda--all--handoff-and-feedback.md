From: pc-cuda
To: all
Time: 2026-11-06T20:59:55Z

pc-cuda: Matt is closing this effort and clearing my context. A handoff, then feedback on how we worked.

HANDOFF: gates GA/GB on da4e1ce are still running (windows PASS posted)
- The script is G:\devel\qwen4-runs\gate-ga.sh. Its log is gate-ga.log (one line per run), per-run server logs ga-*.log, responses ga-*.json. TCQ3-D4, Q8_0 PLE, --ctx 32768, long-0k, 8 runs, started 20:49Z, done ~21:10Z.
- So far at temperature 1.0: unset 38.2 tok/s (3383 slots). 2 drafts sampled from q (A2): 34.7, 2174 of 3470 kept (62.7%; before A2, 38-46%). 2 drafts at min_p 0.3: 37.6, 1795 of 2326 kept (77.2%). A2 lifts acceptance a lot, and A1 trims the verify. 2 drafts is near break-even at 0.3 but not past it yet.
- Not yet run: the second runs of the best arm and of unset at each temperature, and Strata at temperature 1.0 on long-0k (Matt wants it: "time is irrelevant"). Strata runs with G:\Strata\.venv\Scripts\python.exe G:\Strata\serve\server.py --engine strata --config G:\Strata\strata-iq3_s.json --port 8090 (no --open). Request temperature 1.0 explicitly, and read draft_n / draft_n_accepted in timings.
- If the next pc-cuda session does not pick it up, cloud please read gate-ga.log yourself on your next PC request.

FEEDBACK (what cost us time today, and a fix each)
1. Holds race. I started a GPU run 24 s after a hold message (01:00Z). Messages are polled; a hold should be state. Proposal: claims/hold.md (Matt or cloud writes "PC held until ..."), and run scripts check it in their first line and refuse.
2. Never kill by command-line pattern. My stop of "start-here" also killed a download that matched (17:33Z). Scripts should write PID files, and stops should use those PIDs.
3. Hand-typed SHAs and counts. Twice a result carried a typo (a wrong full SHA at 21:5xZ on 11-05, "44/44... 46/46"). agents-post.sh could resolve --sha with git rev-parse and refuse a SHA that does not exist. Ideally the result body is the ctest summary pasted by the script, not retyped.
4. Windows shell escaping. bash heredocs, printf and sed turned "\r" or "\\" in Windows paths into control characters, three times. That broke start-script answers and one build script. Write such files with an editor tool, or use forward slashes.
5. Incremental CI misses new GLOB'd sources. A new nslm/lib_*.c failed to link until I re-ran CMake. Every CI run should configure (cmake -S . -B build) before it builds. My win-ci.bat now does.
6. Noise bars on a deterministic engine. Repeating a KL run gives the same numbers bit for bit, so a spread needs other text slices. And any BF16-level change costs ~0.044 KL on 1,024 wikitext rows with nslm-qwen4-kld (T5). Worth a line in docs/concepts so no one reads 0.04 as a format's cost again.
7. Check the request, not the server, before quoting a rate. My Strata note assumed temperature 1.0, and the probe had sent 0. Benchmark scripts should log their request bodies beside the responses.
8. One slow HTTPS connection stalled a 95 GB download for 35 min, while fresh connections ran 50x faster. Proposal for the start scripts: curl --speed-limit 1000000 --speed-time 60 plus a retry loop.
9. Benchmark conventions for the cards: note warm against cold page cache (TCQ4 at 95 GB is always cold on 128 GB), the desktop's own GPU load (here 1.2 GB, 13-43% utilization idle), and that thinking runs hit max_tokens before any answer.
10. Approvals: this session was not started with remote-control and has no allow rules. Matt has the choice; I asked him in session.
Thanks, both. It was a good run.
