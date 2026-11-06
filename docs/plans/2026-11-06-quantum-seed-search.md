# Plan: the seed search on qubits, at toy sizes (2026-11-06)

**Goal.** Explore the seed search as a quantum problem with Matt, on a simulator and on IBM's free tier. Find where the
structure of the LFSR helps a quantum method, and where device noise ends the gain. No speedup is expected at these
sizes. The plan records failures as clearly as gains.

**Status.** Draft. Q1 is done: `tools/quantum_seed.py` runs on an exact simulator and on a noisy local copy of
`ibm_fez`. Q2 needs Matt's IBM account.

## Why

- At a fixed coefficient vector q and exponent, the error of a seed block is a sum of Walsh characters of the seed. A
  real P = 3 block has 452 to 529 such terms over its 16 seed bits. Read [Wider LFSR seeds](../results/2026-11-06-lfsr-width.md).
- That sum is a diagonal Hamiltonian of Pauli Z products. A circuit applies exp(-i g H) with CNOT ladders and Z
  rotations, and it needs no arithmetic and no extra qubits.
- QAOA uses such a Hamiltonian directly. Decoded quantum interferometry (Jordan et al., 2024, arXiv:2408.08292)
  maximizes objectives of the same form by decoding.
- A GPU searches a 16-bit block in about 1.3 microseconds. Error-corrected devices are far slower for this problem. By a
  rough estimate, Grover breaks even with one GPU only near 74 seed bits. So this plan is a study of structure, not a
  path to faster packs.

## The tool

`tools/quantum_seed.py` builds a mini seed format with SeedLM's structure:

- A K-bit Fibonacci LFSR on K qubits, with seeds 1 to 2^K - 1. Seed 0 is not a seed, and a measurement of 0 counts as
  a miss.
- Blocks of C weights with P coefficients: w_r = scale * sum_p q_p S[P r + p + 1], with S = V - (2^K - 1) / 2.
- Problems:
  - `planted`: a known seed plus noise.
  - `gauss`: random weights, and a classical search for q and the scale.
  - `FILE[:OFFSET]`: real weights, for example the rows of `tools/seed_kl_probe/fetch_qwen_rows.py`.
- Methods, each scored by P(best), the probability that one measurement gives the best seed:
  - `fourier`: amplitudes from the Walsh coefficients of (E_max - err), then a Hadamard on each qubit. So P(s) is
    proportional to (E_max - err(s))^2.
  - `qaoa`: p layers, with angles fitted on an exact NumPy simulator. The tool checks the NumPy state against Qiskit's
    statevector of the device circuit.
  - `grover`: an ideal oracle that marks the best seed. It knows the answer, so it is only the reference for what
    amplitude amplification can reach.
- Modes:
  - `exact`: the exact simulator.
  - `fake`: Qiskit Aer with the noise of an IBM device, for example `FakeFez`.
  - `ibm`: a real device.
  - The modes `fake` and `ibm` transpile with Qiskit's preset pass manager at level 3. They run through the
    runtime's Sampler. The tool prints the depth and the two-qubit gates of each circuit.

```bash
pip install qiskit qiskit-ibm-runtime qiskit-aer scipy
python3 tools/quantum_seed.py --k 4 --problem planted
python3 tools/quantum_seed.py --k 4 --problem planted --mode fake --backend FakeFez
python3 tools/quantum_seed.py --k 4 --problem planted --mode ibm --shots 2000 --json run-k4.json
```

For `--mode ibm`, save an account once, with an API key and an instance from IBM Quantum Platform:
`QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=..., instance=..., set_as_default=True)`.
Send the JSON files to a session, so that the next step can read them.

## Measured so far (Q1)

Exact simulation and the noisy copy of `ibm_fez`, 4,000 shots. "Gain" is P(best) over the random rate 1 / (2^K - 1).

| K | Problem | Method | Exact gain | `FakeFez` gain | Two-qubit gates on `FakeFez` |
|---|---|---|---|---|---|
| 4 | planted | fourier | 2.30 | 2.21 | 13 |
| 4 | planted | qaoa p = 1 | 2.25 | 2.03 | 31 |
| 4 | planted | grover, 3 iterations (ideal oracle) | 14.4 | 9.7 | 129 |
| 5 | gauss, C = 6 | qaoa p = 2 | 8.8 | 5.3 | 160 |
| 5 | gauss, C = 6 | fourier | 2.9 | 2.8 | 41 |

Exact simulation, `gauss` with C = 8 and 10 random starts of the angle fit:

| K | qaoa p = 1 | qaoa p = 2 | qaoa p = 3 | fourier |
|---|---|---|---|---|
| 4 | 3.1 | 4.8 | 7.5 | 2.9 |
| 6 | 4.6 | 3.7 | 4.2 | 2.7 |
| 8 | 3.5 | 8.4 | 5.0 | 2.3 |

At K = 6 and 8 the gain does not rise with p. The fit minimizes the mean energy, not P(best), and 10 starts find local
optima.

## Steps and gates

| Step | Work | Who | Gate |
|---|---|---|---|
| Q1 | The tool: the mini format, the check of the Walsh form, the exact and the noisy local simulators | cloud | QA |
| Q2 | An IBM Quantum Platform account (Open Plan). `--mode fake`, then `--mode ibm` at K = 3 and 4, with `fourier` and `qaoa` p = 1 | Matt | QB |
| Q3 | A sweep on the exact simulator: K = 4 to 10, p = 1 to 4, more starts, and an angle fit that maximizes P(best). Planted, random and real blocks | cloud or Matt | QC |
| Q4 | The device at the largest K and p that keep their gain on `FakeFez` | Matt | QD |
| Q5 | The coefficients too: a quantum seed step that alternates with a classical step for q, or the bits of q on qubits | cloud, then Matt | none |
| Q6 | Optional: an Ising form for an annealer. The Z products must be reduced to pairs with extra qubits | Matt | none |

- **QA.** The Walsh form equals the direct error for every seed. The NumPy QAOA equals the circuit's statevector. The
  `fake` mode runs. Done.
- **QB.** On the device, `fourier` and `qaoa` p = 1 at K = 4 give a gain of at least 1.5 with 2,000 shots. `FakeFez`
  gives 2.2 and 2.0.
- **QC.** QAOA on the exact simulator reaches a gain of 5 at some p of 4 or less for K = 8. Record the circuit size.
- **QD.** The device keeps at least half of the exact gain.

## Hypotheses

| ID | Hypothesis | Gate |
|---|---|---|
| HQ1 | QAOA p = 1 gives a gain of about 2 to 5 at K = 4 to 8 | Q1: holds (2.0 to 4.6) |
| HQ2 | More layers give more gain when the angle fit finds the optimum | QC (K = 4, 5: holds; K = 6, 8: not yet) |
| HQ3 | Device noise removes most of the gain above about 150 two-qubit gates | QD (`FakeFez` kept 60% at 160 gates) |
| HQ4 | Fourier sampling stays at a gain of 2 to 3 at any K, because it weights each seed by its squared error gap | QC (K = 4 to 8: 2.3 to 2.9) |

## Access and costs

Free and low-cost access, checked on 2026-11-06. Terms change, so check each provider's page before a run.

| Provider | Free part | Notes |
|---|---|---|
| IBM Quantum Platform, Open Plan | 10 minutes of device time in each 28-day window. Since March 2026, an account that used 20 minutes in 12 months can opt in to 180 minutes for the next 12 months | Heron devices, Qiskit. Paid: about $96 a minute |
| Qiskit Aer with fake backends | Local and free | Noise models of IBM devices (`FakeFez`, `FakeMarrakesh`, `FakeTorino`) |
| Amazon Braket | Simulator time in the first 12 months | Devices (IonQ, Rigetti, QuEra) are paid for each task and shot |
| qBraid | A free base plan; devices with credits | Third-party sources only. Check the current offer |
| D-Wave Leap | A trial of the annealer | The current free time is not confirmed. Ising problems only (Q6) |

- Q1 and Q3: CPU only.
- Q2 and Q4: a few seconds of device time for each circuit at 2,000 to 4,000 shots. The platform shows the use of each
  job. The 10 free minutes may cover several dozen runs.

Sources: [IBM plans overview](https://quantum.cloud.ibm.com/docs/guides/plans-overview),
[IBM Quantum pricing](https://www.ibm.com/quantum/pricing),
[Open Plan news, 2026-03-16](https://quantum.cloud.ibm.com/announcements/en/product-updates/2026-03-16-open-plan-news),
[Cloud-based quantum computing](https://en.wikipedia.org/wiki/Cloud-based_quantum_computing).
