# The library with A = I against lfsr_width (same 2048 blocks): P = 3 and 4, K = 16; P = 3 K = 18; P = 3 stride 4.
import sys, time, numpy as np, seedlib
d = sys.argv[1]
gu = np.fromfile(f"{d}/gu.f32", np.float32); dn = np.fromfile(f"{d}/dn.f32", np.float32)
bg, bd = 1536, 512
X = np.concatenate([np.stack([gu[8 * (i * (len(gu) // 8) // bg):][:8] for i in range(bg)]),
                    np.stack([dn[8 * (i * (len(dn) // 8) // bd):][:8] for i in range(bd)])])
for P, K, st in [(3, 16, 1), (4, 16, 1), (3, 18, 1), (3, 16, 4)]:
    seedlib.setup(P, K, st); t = time.time()
    wd, s, e, er = seedlib.search(np.eye(8), X)
    print(f"P {P} K {K} stride {st}: mse {((X - wd) ** 2).sum() / (X ** 2).sum():.5f}  search {time.time() - t:.1f} s")
