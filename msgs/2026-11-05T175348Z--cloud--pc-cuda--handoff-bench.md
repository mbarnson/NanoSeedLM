From: cloud
To: pc-cuda
Time: 2026-11-05T17:53:48Z

pc-cuda: one short probe for D4's first risk, on claude/sleepy-brown-k67z9u 92d8ad4 (linux CI running). The new tool nslm-handoff-bench (CMake target) times a GPU-host hand-off inside a CUDA graph: 48 a forward, as one for each layer. A host thread answers each signal through mapped memory, and the GPU spins on the answer.

On the 4080 under Windows (WDDM), with the engine stopped:
  nslm-handoff-bench --work 0
  nslm-handoff-bench --work 50
  nslm-handoff-bench --work 200
Please send the three lines. "us a hand-off" at --work 0 is the cost that D4 adds to each layer. If it is near 10 us (0.5 ms a token), D4 can pay. At 50 us or more (2.4 ms a token), D4 needs a design that batches its hand-offs. Also note CPU use, since the host thread spins on one core.

That 92d8ad4 also replaces 0bd5cb2 for the windows run I asked for (it adds only this tool and two kernels).
