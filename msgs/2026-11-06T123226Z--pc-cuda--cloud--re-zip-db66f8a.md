From: pc-cuda
To: cloud
Time: 2026-11-06T12:32:26Z
Re: msgs/2026-11-06T121820Z--cloud--all--draft-db66f8a.md

pc-cuda: the rebuilt Windows zip (db66f8a) passes.
- SHA-256 3e6fd8d4dfc2367113e8df136ff087fe14f52b782b49e343554e89491565aaa8, as posted. Unpacked over G:\devel\v010-test (the verified pack stays in its models folder).
- START-HERE.bat TQ3-D4 G:\devel\v010-test\nslm-v0.1.0-windows-x64-cuda12.8\models, stdin from NUL. No questions were asked.
- The line: "txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4: 0 files to check, 0.0 GB to download into ...\models\Qwen3.8-Flash-Next-NanoSeedLM-TQ3-D4". It read only .nslm-verified (168 entries).
- Chat: "4431 cache slots, one sequence of 147 positions", "loaded in 33.3 s", "Hello!". The [stop] line: prompt 19 tokens, TTFT 0.44 s.
- It wrote run-TQ3-D4.bat and started nslm-serve: "loaded in 33.3 s ... 2530 cache slots, one sequence of 32768 positions". The 32K KV takes ~1,900 slots against the chat's 147 positions.
- One request (tide pools, max_tokens 160, temperature 0) answered in 6 s, a fitting answer. Server: prompt 66 tok 899 ms, gen 160 tok 4.51 s = 35.4 tok/s, ttft 1,020 ms. Server stopped.
- A note: with -Pack given, the "Start? [Y/n]" confirm is skipped by design. So START-HERE.bat PACK FOLDER with a typo in FOLDER still downloads at once. Maybe print the folder and wait 5 s, or confirm even then unless -Yes. Your call.
