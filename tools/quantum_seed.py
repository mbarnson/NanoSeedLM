#!/usr/bin/env python3
# quantum_seed.py - the SeedLM seed search on qubits, at toy sizes, with Qiskit: an exact simulator, a noisy local copy
# of an IBM device, or an IBM Quantum device.  docs/plans/2026-11-06-quantum-seed-search.md.
#
# A mini seed format with the structure of SeedLM's: a K-bit Fibonacci LFSR (K qubits, seeds 1 .. 2^K - 1), blocks of
# C weights, P coefficients, w_r = scale * sum_p q_p S[P r + p + 1], with S_k = V_k - (2^K - 1) / 2 for state k.  Stream
# bit j of seed s is a GF(2) linear form <a_j, s>, so with c_j = (-1)^x_j each state is S_k = -sum_i 2^(i-1) c_(k+i).
# At a fixed (q, scale) the block error is then a sum of Walsh characters of the seed:
#     err(s) = V_0 + sum_b V_b (-1)^<b, s>   =   <s| V_0 I + sum_b V_b Z^b |s>,   Z^b = product of Z on the bits of b.
# So the search for the seed is the ground state of a diagonal Hamiltonian of Pauli Z products on K qubits, with no
# arithmetic in the circuit (docs/results/2026-11-06-lfsr-width.md checks this form for real P = 3 blocks).
#
# Methods, each scored by P(best): the probability that one measurement gives the best seed (1 / (2^K - 1) at random).
#   fourier  prepare sum_b A_b |b>, A = Walsh coefficients of (E_max - err), then H on every qubit: P(s) ~ (E_max - err(s))^2
#   qaoa     p layers of exp(-i gamma H_C) exp(-i beta sum X), angles fitted on the exact simulator
#   grover   an ideal oracle that marks the best seed (a diagonal gate), floor(pi/4 sqrt(N)) iterations: the reference
#
#   python3 tools/quantum_seed.py --k 4 --problem planted                       exact simulation
#   python3 tools/quantum_seed.py --k 4 --problem gauss --mode fake --backend FakeFez   noisy local copy of ibm_fez
#   python3 tools/quantum_seed.py --k 4 --problem planted --mode ibm --shots 2000      an IBM device (saved account)
#   --problem FILE[:OFFSET] takes C float32 weights from FILE (e.g. tools/seed_kl_probe's data/gu.f32).
# Needs qiskit and scipy; --mode fake needs qiskit-aer and qiskit-ibm-runtime; --mode ibm needs qiskit-ibm-runtime and an
# account saved with QiskitRuntimeService.save_account (IBM Quantum Platform: an API key and an instance).
import argparse, json, math, sys
import numpy as np


# ---- the mini format --------------------------------------------------------------------------------------------------
def primitive_taps(K):
    """A tap mask (bit 0 set) whose Fibonacci LFSR has period 2^K - 1; the first found, as tools/seed_kl_probe/seedlib.c."""
    full = (1 << K) - 1
    def period(t):
        s, n = 1, 0
        while True:
            s = (s >> 1) | ((bin(s & t).count("1") & 1) << (K - 1)); n += 1
            if s == 1 or n > full: return n
    cands = [1 | (1 << j) for j in range(1, K)]
    cands += [1 | (1 << a) | (1 << b) | (1 << c) for a in range(1, K) for b in range(a + 1, K) for c in range(b + 1, K)]
    for t in cands:
        if period(t) == full: return t
    raise ValueError(f"no primitive taps for K = {K}")

def stream_forms(K, taps, n):
    """a_j: stream bit j as a mask over the seed bits (x_j = parity(a_j & s)). x_(t+K) = XOR of x_(t+i), i in taps."""
    a = [1 << j for j in range(K)]
    while len(a) < n:
        t = len(a) - K
        v = 0
        for i in range(K):
            if taps >> i & 1: v ^= a[t + i]
        a.append(v)
    return a

def states(K, taps, s, n):
    """The n states after seed s (state_k = stream bits k .. k + K - 1), centred: S = V - (2^K - 1) / 2."""
    out = []
    for _ in range(n):
        s = (s >> 1) | ((bin(s & taps).count("1") & 1) << (K - 1))
        out.append(s - ((1 << K) - 1) / 2)
    return np.array(out)

def decode(K, taps, C, P, s, q, scale):
    S = states(K, taps, s, C * P).reshape(C, P)
    return scale * (S @ np.asarray(q, float))

def walsh(K, taps, C, P, x, q, scale):
    """V[b], b = 0 .. 2^K - 1: err(s) = sum_b V[b] (-1)^popcount(b & s) at the fixed (q, scale)."""
    a = stream_forms(K, taps, C * P + K)
    V = np.zeros(1 << K)
    for r in range(C):
        al = {}
        for p in range(P):
            k = P * r + p + 1                       # state k = stream bits k .. k + K - 1
            for i in range(K): al[k + i] = al.get(k + i, 0.0) + scale * q[p] * 2.0 ** (i - 1)
        js = sorted(j for j in al if al[j] != 0.0)
        V[0] += x[r] ** 2 + sum(al[j] ** 2 for j in js)
        for u, j in enumerate(js):
            V[a[j]] += 2 * x[r] * al[j]
            for k2 in js[u + 1:]: V[a[j] ^ a[k2]] += 2 * al[j] * al[k2]
    return V

def walsh_eval(V):
    """All err(s) from V by a fast Walsh-Hadamard transform."""
    f = V.copy(); h = 1
    while h < len(f):
        for i in range(0, len(f), 2 * h):
            u, v = f[i:i + h].copy(), f[i + h:i + 2 * h].copy(); f[i:i + h], f[i + h:i + 2 * h] = u + v, u - v
        h *= 2
    return f

def fit(K, taps, C, P, x, qmax=7):
    """The classical search: every seed, least-squares coefficients, a shared scale max|t| / qmax, rounded q."""
    best = (math.inf, None, None, None)
    for s in range(1, 1 << K):
        S = states(K, taps, s, C * P).reshape(C, P)
        t = np.linalg.lstsq(S, x, rcond=None)[0]
        m = np.abs(t).max()
        if m == 0: continue
        scale = m / qmax
        q = np.clip(np.round(t / scale), -qmax - 1, qmax).astype(int)
        e = float(((x - scale * (S @ q)) ** 2).sum())
        if e < best[0]: best = (e, s, q, scale)
    return best


# ---- the problems -----------------------------------------------------------------------------------------------------
def make_problem(args, K, taps, rng):
    C, P = args.c, args.p
    if args.problem == "planted":
        s_star = int(rng.integers(1, 1 << K))
        q = rng.choice([v for v in range(-3, 4) if v], size=P)
        x = decode(K, taps, C, P, s_star, q, 1.0)
        x = x + args.noise * np.sqrt((x ** 2).mean()) * rng.standard_normal(C)
        return x, q, 1.0, f"planted seed {s_star}, q {list(map(int, q))}, noise {args.noise}"
    if args.problem == "gauss": x = rng.standard_normal(C)
    else:
        path, _, off = args.problem.partition(":")
        x = np.fromfile(path, dtype=np.float32)[int(off or 0): int(off or 0) + C].astype(float)
        if len(x) < C: sys.exit(f"{path}: fewer than {C} weights at offset {off or 0}")
    e, s, q, scale = fit(K, taps, C, P, x)
    return x, q, scale, f"{args.problem}: classical search seed {s}, q {list(map(int, q))}, error {e:.4g}"


# ---- the circuits -----------------------------------------------------------------------------------------------------
def cost_op(V, K):
    from qiskit.quantum_info import SparsePauliOp
    terms = [("Z" * bin(b).count("1"), [i for i in range(K) if b >> i & 1], V[b]) for b in range(1, 1 << K) if abs(V[b]) > 1e-12]
    norm = max(abs(t[2]) for t in terms)
    return SparsePauliOp.from_sparse_list([(z, idx, c / norm) for z, idx, c in terms], num_qubits=K), len(terms)

def qaoa_circuit(Hc, K, layers):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import PauliEvolutionGate
    g, b = ParameterVector("g", layers), ParameterVector("b", layers)
    qc = QuantumCircuit(K)
    qc.h(range(K))
    for l in range(layers):
        qc.append(PauliEvolutionGate(Hc, time=g[l]), range(K))
        qc.rx(2 * b[l], range(K))
    return qc, list(g) + list(b)

def fourier_circuit(V, K):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation
    E = walsh_eval(V)
    A = -V.copy()
    A[0] = E[1:].max() - V[0]          # Walsh coefficients of E_max - err (0 at the worst seed, largest at the best)
    qc = QuantumCircuit(K)
    qc.append(StatePreparation(A / np.linalg.norm(A)), range(K))
    qc.h(range(K))
    return qc

def grover_circuit(K, best, iters):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import DiagonalGate
    N = 1 << K
    oracle = [(-1.0 if s == best else 1.0) for s in range(N)]
    diffuse = [1.0] + [-1.0] * (N - 1)   # 2|0><0| - I, up to a global phase
    qc = QuantumCircuit(K)
    qc.h(range(K))
    for _ in range(iters):
        qc.append(DiagonalGate(oracle), range(K))
        qc.h(range(K)); qc.append(DiagonalGate(diffuse), range(K)); qc.h(range(K))
    return qc

def probs(qc):
    from qiskit.quantum_info import Statevector
    return Statevector(qc).probabilities()

def qaoa_probs(diag, K, th):
    """Exact QAOA in NumPy: |+>^K, then for each layer exp(-i g diag) and RX(2 b) on every qubit (qaoa_circuit)."""
    layers = len(th) // 2
    psi = np.full(1 << K, (1 << K) ** -0.5, dtype=complex)
    for l in range(layers):
        g, b = th[l], th[layers + l]
        psi = psi * np.exp(-1j * g * diag)
        c, sn = math.cos(b), -1j * math.sin(b)
        for i in range(K):   # qubit i = bit i of the index
            v = psi.reshape(-1, 2, 1 << i)
            psi = np.stack([c * v[:, 0] + sn * v[:, 1], sn * v[:, 0] + c * v[:, 1]], 1).reshape(-1)
    return np.abs(psi) ** 2

def fit_qaoa(Hc, K, layers, starts, rng):
    """Angles that minimize <H_C> on the exact simulator; the bound circuit, checked against Qiskit's Statevector."""
    from scipy.optimize import minimize
    diag = np.real(np.asarray(Hc.to_matrix(sparse=True).diagonal())).ravel()
    def energy(th): return float(qaoa_probs(diag, K, th) @ diag)
    top = None
    for _ in range(starts):
        r = minimize(energy, rng.uniform(0, math.pi, 2 * layers), method="COBYLA", options={"maxiter": 600})
        if top is None or r.fun < top.fun: top = r
    qc, params = qaoa_circuit(Hc, K, layers)
    bound = qc.assign_parameters(dict(zip(params, top.x)))
    p = qaoa_probs(diag, K, top.x)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.allclose(p, probs(bound), atol=1e-9), "NumPy QAOA != the circuit's statevector"
    return bound, p


# ---- scoring and runs -------------------------------------------------------------------------------------------------
def score(p, E, best):
    """P(best), and the mean error of a measured seed (seed 0, not a seed, counts as the worst error)."""
    p = np.asarray(p, float); p = p / p.sum()
    Ez = E.copy(); Ez[0] = E[1:].max()
    return float(p[best]), float(p @ Ez)

def run_sampler(circuits, args, K):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    try:   # qiskit-ibm-runtime 0.50 and later: the client-side Sampler; SamplerV2 before
        from qiskit_ibm_runtime.executor_sampler import Sampler
    except ImportError:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    if args.mode == "fake":
        from qiskit_ibm_runtime import fake_provider
        backend = getattr(fake_provider, args.backend)()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService()
        backend = svc.backend(args.backend) if args.backend else svc.least_busy(operational=True, simulator=False, min_num_qubits=K)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=args.rng)
    isa = []
    for qc in circuits:
        m = qc.copy(); m.measure_all(); isa.append(pm.run(m))
    shape = [(c.depth(), sum(1 for i in c.data if i.operation.num_qubits == 2)) for c in isa]
    res = Sampler(mode=backend).run(isa, shots=args.shots).result()
    out = []
    for r in res:
        p = np.zeros(1 << K)
        for key, n in r.data.meas.get_counts().items(): p[int(key, 2)] += n
        out.append(p)
    return backend.name, shape, out

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=4, help="LFSR bits = qubits (3 to 10)")
    ap.add_argument("--c", type=int, default=4, help="weights in a block")
    ap.add_argument("--p", type=int, default=1, help="coefficients in a block")
    ap.add_argument("--problem", default="planted", help="planted | gauss | FILE[:OFFSET]")
    ap.add_argument("--noise", type=float, default=0.05, help="planted: noise, relative to the block's RMS")
    ap.add_argument("--layers", type=int, default=1, help="QAOA layers p")
    ap.add_argument("--starts", type=int, default=20, help="QAOA: random starts of the angle fit")
    ap.add_argument("--method", default="fourier,qaoa,grover")
    ap.add_argument("--mode", default="exact", choices=["exact", "fake", "ibm"])
    ap.add_argument("--backend", default=None, help="fake: a qiskit_ibm_runtime.fake_provider class (FakeFez); ibm: a device name")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--rng", type=int, default=1)
    ap.add_argument("--json", default=None, help="write the results here")
    args = ap.parse_args()
    if args.mode == "fake" and not args.backend: args.backend = "FakeFez"
    K, C, P = args.k, args.c, args.p
    rng = np.random.default_rng(args.rng)
    taps = primitive_taps(K)
    x, q, scale, desc = make_problem(args, K, taps, rng)
    V = walsh(K, taps, C, P, x, q, scale)
    E = walsh_eval(V)
    direct = np.array([((x - decode(K, taps, C, P, s, q, scale)) ** 2).sum() for s in range(1 << K)])
    assert np.allclose(E[1:], direct[1:], rtol=1e-9, atol=1e-9 * max(1.0, abs(direct).max())), "Walsh expansion != direct error"
    best = 1 + int(np.argmin(E[1:]))
    Hc, nterms = cost_op(V, K)
    N = (1 << K) - 1
    out = {"k": K, "c": C, "p": P, "taps": taps, "problem": desc, "terms": nterms, "best_seed": best,
           "best_error": float(E[best]), "mean_error": float(E[1:].mean()), "methods": {}}
    print(f"K = {K} qubits, {N} seeds, C = {C}, P = {P}, taps 0x{taps:x}; {desc}")
    print(f"cost Hamiltonian: {nterms} Pauli Z products (of {N}); best seed {best} (error {E[best]:.4g}, mean {E[1:].mean():.4g}); "
          f"Walsh form checked against direct evaluation")
    print(f"{'method':24s} {'P(best)':>9s} {'x random':>9s} {'mean error':>11s}")
    print(f"{'random seed':24s} {1 / N:9.4f} {1.0:9.2f} {E[1:].mean():11.4g}")
    circuits, names = [], []
    for m in args.method.split(","):
        if m == "fourier": qc = fourier_circuit(V, K); p = probs(qc); name = "fourier"
        elif m == "qaoa": qc, p = fit_qaoa(Hc, K, args.layers, args.starts, rng); name = f"qaoa p={args.layers}"
        elif m == "grover":
            it = max(1, int(math.pi / 4 * math.sqrt(1 << K))); qc = grover_circuit(K, best, it); p = probs(qc); name = f"grover x{it} (ideal oracle)"
        else: sys.exit(f"unknown method {m}")
        pb, me = score(p, E, best)
        out["methods"][name] = {"exact_p_best": pb, "exact_mean_error": me}
        print(f"{name + ' (exact)':24s} {pb:9.4f} {pb * N:9.2f} {me:11.4g}")
        circuits.append(qc); names.append(name)
    if args.mode != "exact":
        dev, shape, meas = run_sampler(circuits, args, K)
        for name, (depth, n2), p in zip(names, shape, meas):
            pb, me = score(p, E, best)
            out["methods"][name].update({"device": dev, "shots": args.shots, "depth": depth, "two_qubit_gates": n2,
                                         "measured_p_best": pb, "measured_mean_error": me})
            print(f"{name + ' (' + dev + ')':24s} {pb:9.4f} {pb * N:9.2f} {me:11.4g}   depth {depth}, two-qubit gates {n2}, {args.shots} shots")
    if args.json:
        with open(args.json, "w") as f: json.dump(out, f, indent=1)

if __name__ == "__main__":
    main()
