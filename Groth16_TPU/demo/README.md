# TPU Groth16 — GUI Demo

A two-process **multi-circuit** demo of the BLS12-377 Groth16 prover on TPU:

```
┌──────────────────────┐  HTTP/JSON  ┌─────────────────────────────┐
│  Streamlit UI        │ ──────────▶ │  Prover Daemon              │
│  • env progress bar  │             │  • owns the TPU             │
│  • per-circuit cards │             │  • TPU env warmed first     │
│  • build/rebuild/use │ ◀────────── │  • per-circuit registry     │
│  • witness editor    │             │  • in-process builds        │
│  • compare panel     │             │  • only active in HBM       │
└──────────────────────┘             └─────────────────────────────┘
       port 8501                              port 8000
```

The UI process never imports JAX, so Streamlit reruns and hot-reloads
don't fight the daemon for the TPU.  Any frontend (curl, CLI, a future
React app) can drive the same daemon.

Eight circuits ship out of the box:

| Key                 | Statement                              | Wires | Constraints | Category  |
|---                  |---                                     |---    |---          |---        |
| `cubic`             | `x³ + x + 5 = y`                       | 5     | 3 + pad     | practical |
| `power_11`          | `x¹¹ = y`                              | 12    | 10 + pad    | archive   |
| `fibonacci_20`      | `f₂₀ = y` from seeds `(a, b)`          | 22    | 19 + pad    | practical |
| `range_32`          | `0 ≤ x < 2³²`, hidden `x`, sentinel ok | 35    | 34 + pad    | archive   |
| `mimc_preimage`     | `MiMC(x) = h`, hidden `x` (20 rounds)  | 62    | 60 + pad    | practical |
| `poseidon_preimage` | `Poseidon(x) = h`, hidden `x`          | 51    | 49 + pad    | practical |
| `merkle_membership` | `leaf ∈ tree(root R)`, depth-2 (MiMC)  | 126   | 122 + pad   | practical |
| `sudoku_4x4`        | valid 4×4 Sudoku solution, hidden grid | 106   | 129 + pad   | practical |

All eight pad to the same TPU shape (`max_size=1024`), so the prover
JIT-compiles once and reuses for every circuit.

The sidebar splits circuits by **category**: `practical` ones render
directly; `archive` ones (small teaching toys like `power_11` /
`range_32`) live behind a collapsible *▸ Archived demos* expander.  Each
card has a *Move to practical / archive* button that re-files it live
(persisted to `circuit_categories.json`); no restart needed.

---

## Quick start

```bash
# 0) prerequisites — TPU VM, MORPH repo cloned.  Paths are from repo root.
cd <repo root>

# Install everything (core + demo + recommended acceleration deps):
pip install -r requirements.txt

# 1) start the demo
./Groth16_TPU/demo/demo.sh start

# 2) tunnel to your laptop and open the UI
ssh -L 8501:localhost:8501 user@tpu-host
# open http://localhost:8501
```

**On first boot the daemon spends ~12 min compiling TPU kernels.**  No
specific circuit is loaded yet — the user picks what to build from the
sidebar.  After kernels are warm, building any one circuit takes ~135 s
(CPU, in-process); activating an already-built circuit takes ~5-10 s.

JAX's compile cache survives in `~/.cache/maple-jax-compile/` between
daemon restarts; subsequent boots warm in ~30 s.

### Lifecycle commands

```bash
./Groth16_TPU/demo/demo.sh start     # start daemon + UI
./Groth16_TPU/demo/demo.sh stop      # stop both, plus any forkserver children
./Groth16_TPU/demo/demo.sh status    # show what's running + daemon's /status
./Groth16_TPU/demo/demo.sh logs      # tail both logs
./Groth16_TPU/demo/demo.sh restart   # stop then start
```

PIDs and logs live in `Groth16_TPU/demo/.demo_run/` — no manual
clean-up needed.  Default ports `8000` (daemon) and `8501` (UI);
override with `MAPLE_DAEMON_PORT` / `MAPLE_UI_PORT`.

---

## User flow

```
┌─ Sidebar ──────────────────────────────────────┐
│                                                │
│ TPU env: ████████████████████ 100% ✓ warm     │
│ No circuit active — Use one below.            │
│ ───────────────────────────────────────────── │
│ Circuits                                       │
│                                                │
│ ┌─ cubic        ⭐ active        ┐            │
│ │  [Rebuild] [Stop] [Active]    │            │
│ │  📌 showing in main panel     │            │
│ └────────────────────────────────┘            │
│ ┌─ mimc_preimage 🔵 building     ┐            │
│ │  ████████░░ 78% encode_qap…    │            │
│ │  [Build] [Stop] [Use]          │            │
│ │  [Show in main panel ⤴]        │            │
│ └────────────────────────────────┘            │
│ ┌─ fibonacci_20  🟢 ready        ┐            │
│ │  [Rebuild] [Stop] [Use]        │            │
│ │  [Show in main panel ⤴]        │            │
│ └────────────────────────────────┘            │
│ ┌─ poseidon…    ⚪ available     ┐            │
│ │  [Build] [Stop] [Use]          │            │
│ │  [Show in main panel ⤴]        │            │
│ └────────────────────────────────┘            │
└────────────────────────────────────────────────┘
```

**Status semantics**

| Badge | Status      | Meaning                                                 |
|---    |---          |---                                                      |
| ⚪    | `available` | No pkl on disk; click **Build** to create it.           |
| 🔵    | `building`  | A `compile_circuit` worker is running; live progress.   |
| 🟢    | `ready`     | Pkl on disk, not bound to prover.  Click **Use**; **Rebuild** to refresh the pkl. |
| ⭐    | `active`    | Pkl on disk AND bound to prover.  `▶ Generate proof` enabled; **Rebuild** to refresh the pkl. |
| 🔴    | `error`     | Last build failed; click **Rebuild**.                   |

The first button in each card is **Build** when no pkl exists, and
**Rebuild** once one does (`ready` / `active` / `error`).  See
*Rebuilding a circuit* below.

Only **one** circuit can be active at a time (TPU HBM budget).  Clicking
**Use** on a different ready circuit drops the previous one's tensors and
binds the new circuit's setup — typically ~5-10 s thanks to the warm
JIT cache.

### Rebuilding a circuit

A built circuit's cached pkl can be regenerated from scratch via the
**Rebuild** button (or `POST /circuits/{key}/build?force=true`).  This
runs the full trusted setup again, ignoring and overwriting the existing
file — useful when:

* the cached pkl is **stale or version-incompatible** (e.g. copied from
  another machine / built under a different NumPy — the pickle load can
  fail with a `_frombuffer` arity error), or
* you simply want a fresh trusted setup.

Because a rebuild reruns the trusted setup, it produces a **new
proving/verification key**.  If you rebuild the *currently active*
circuit, the in-memory binding keeps the old key until you click
**Use** again — re-activate to pick up the freshly written pkl.

### What works while a build is in flight

* Editing the witness for any circuit.
* Clicking **Check witness** (pure CPython R1CS verifier — never blocks).
* Building OTHER circuits' pkls — well, almost: see *TPU lock* below.

### TPU lock — what serialises

The TPU is single-tenant.  These operations all take the same lock:

* TPU env warmup (one time, at daemon start)
* A build's TPU phases — `qap_build`, `encode_pk`, `encode_qap`
* `activate_circuit` (uploads the setup's tensors to HBM)
* `/prove`

A build and a prove can both be **in progress**, but the TPU executes
them serially.  In practice the prove pauses for ~20 s (the `qap_build`
slice) and a few seconds each for the encode phases; the rest of the
build is CPU and overlaps freely.

---

## Files

| File | Role |
|---|---|
| `daemon.py`        | FastAPI app.  Owns the TPU.  Endpoints below. |
| `init_worker.py`   | Per-circuit registry; in-process builds under `_tpu_lock`. |
| `client.py`        | Thin `requests`-based wrapper.  Used by the Streamlit UI. |
| `protocol.py`      | Shared dataclasses — `StatusReply`, `CircuitInfo`, `BuildInfo`, `ProveReply`. |
| `prep_circuit.py`  | CLI shim — kept for the standalone `python -m dev_loop prep` workflow.  Daemon builds run in-process, not via this. |
| `demo_circuits.py` | Registry of predefined circuits.  Add new ones via `register_circuit(...)`. |
| `demo_app.py`      | The Streamlit UI.  Pure HTTP client; no JAX. |
| `demo.sh`          | Lifecycle wrapper (start/stop/status/logs/restart). |

---

## HTTP API

### `GET /healthz`

```json
{"status": "ok"}
```

### `GET /status`

```json
{
  "phase":      "warmup",
  "progress":   1.0,
  "detail":     "Warmup complete",
  "elapsed":    127.4,
  "done":       true,
  "error":      null,
  "start_time": 1716711234.5,
  "active":     "cubic"
}
```

The daemon's TPU-env state.  `done == true` means the prover is warm
(synthetic Setup discarded; awaiting an `/activate` call).  `active` is
the currently-bound circuit key, or `null` when no circuit is bound.

Env phases:

| Phase          | Cost      | What |
|---             |---        |---   |
| `tpu_contexts` | ~70 s     | Stand up MSM / NTT / Fr DRNS contexts. |
| `aot_compile`  | ~58 s     | `jax.jit(F).lower().compile()` for 6 composites. |
| `warmup`       | ~600 s cold / ~10 s warm | Throwaway prove on a synthetic Setup. |
| `ready`        | —         | `done=true`. |

### `GET /circuits`

```json
{
  "env_ready": true,
  "active":    "cubic",
  "circuits": [
    {
      "key": "cubic",
      "name": "Cubic: x³ + x + 5 = y",
      "description": "...",
      "public_label": "y",
      "num_wires_orig": 5,
      "num_public_orig": 1,
      "wire_labels": ["wire 0 — constant 1", ...],
      "input_schema": [
        {"name": "x", "label": "x  (private input)", "kind": "int",
         "default": 3, "min": null, "max": null, "help": "..."}
      ],
      "status":     "active",
      "pkl_exists": true,
      "is_active":  true,
      "build":      null
    },
    {
      "key": "power_11",
      "status":     "building",
      "pkl_exists": false,
      "is_active":  false,
      "build": {
        "phase":      "running",
        "progress":   0.78,
        "detail":     "encode_qap: DRNS-encoding qap_tpu (U/V/W polynomials)",
        "elapsed":    87.4,
        "start_time": 1716711350.0,
        "error":      null
      },
      ...
    }
  ]
}
```

### `POST /circuits/{key}/build`

Spawn an in-process `compile_circuit` worker for `key`.  Idempotent —
returns one of `started`, `already_running`, `already_built`.

```json
{"key": "power_11", "outcome": "started"}
```

Pass `?force=true` to **rebuild** even when a pkl already exists: the
cached file is ignored and overwritten with a fresh trusted setup
(otherwise the call short-circuits with `already_built`).

```bash
curl -X POST "http://localhost:8000/circuits/mimc_preimage/build?force=true"
```

The build serialises against any other TPU work via `_tpu_lock`; see
`GET /circuits` for live progress.

### `POST /circuits/{key}/cancel`

Set a cancel flag picked up at the next `compile_circuit` phase
boundary.  Active TPU kernels can't be preempted, so cancel takes
effect within a few seconds.  Returns `cancelling` or `not_running`.

### `POST /circuits/{key}/activate`

Load `key`'s setup pkl into the warm prover, dropping the previous
active circuit's HBM tensors.  Returns `409` if the pkl isn't built;
`503` if TPU env isn't warm yet.

```json
{"key": "fibonacci_20", "outcome": "active"}
```

### `POST /circuits/deactivate`

Free the active circuit's HBM tensors; prover stays warm.

### `POST /prove`

Either provide high-level inputs (the daemon's `witness_builder` fills
in intermediates) **or** a complete witness vector — exactly one of
`inputs` / `witness`.

```json
{
  "circuit": "cubic",
  "inputs":  {"x": 3},
  "seed":    42,
  "verify":  true
}
```

Reply mirrors the previous schema (proof bytes, witness, per-phase
timings, verification result).  See `protocol.py:ProveReply` for the
authoritative shape.

Errors:

* `503` — TPU env still warming up, or requested circuit isn't active.
* `400` — bad request (witness fails R1CS).
* `404` — unknown circuit key.
* `500` — unexpected exception; full traceback in `detail`.

EC coordinates and witness values are encoded as **decimal-int strings**
— Groth16 coordinates are 377-bit, JSON ints are too small.

---

## Architectural notes

### Why a daemon

`init_worker.py` could run inside the Streamlit process directly — and an
earlier draft did.  Streamlit reruns the script on every interaction,
and JAX multiprocessing pools re-exec the page module in their workers.
Those workers try to import JAX and crash because the Streamlit parent
already owns the TPU.  Hoisting the prover into its own process makes
the UI side strictly HTTP, and immune to its own reruns.

It also means **editing `demo_app.py` doesn't lose the 12-min compile**.
The daemon keeps running with its warm kernels; only the UI hot-reloads.

### Why builds run *in-process*

An earlier T25 design spawned `prep_circuit` as a subprocess (so
forkserver children for parallel DRNS encoding don't re-import the
daemon's `__main__`).  That worked when the daemon owned no specific
circuit and the subprocess could grab the TPU for `qap_build`.

It does NOT work when the daemon already owns the TPU.  On a TPU VM
each device is one-process-only — the subprocess gets
`The TPU is already in use by process N`.

Solution: builds run in the daemon process under `_tpu_lock`.  The
forkserver children that do the per-wire DRNS encoding only touch CPU
NumPy, not JAX, so they don't fight for the TPU.  `prep_circuit.py`
stays around for the standalone `python -m dev_loop prep` CLI workflow.

### Why TPU env warms first

User experience.  The full cold-boot is ~14 min — letting the user pick
a circuit before that finishes makes the UI feel responsive.  When the
user clicks **Build**, it queues behind whatever the TPU is doing
(usually the warmup); when warmup finishes, the build phase that needs
the TPU (`qap_build`, ~20 s) runs.  Everything else is CPU and runs
freely.

The trade-off: the warmup uses a **synthetic Setup** with all-zero
tensors at the correct shapes (see `prover_class.make_warmup_setup`).
We can't decode the EC output of that prove (zero limbs decode to
divide-by-zero), so the warmup goes through `_prove_pure_tpu` directly
and skips the host-side decode.  The JIT trace cache is what we care
about, and that's populated regardless.

### Why HTTP, not unix sockets / shared memory

JSON over HTTP is the cheapest interop layer that lets a non-TPU host
(your laptop, a CI runner, a frontend in another language) drive the
same daemon over the network.  At ~700 ms per prove, the ~5 ms
serialisation cost is noise.  Swap in gRPC behind the same
`protocol.py` if you ever need sub-ms RPC.

### CPU acceleration in `trusted_setup`

For any circuit at `max_size=1024`, `trusted_setup` is **~6 s** with all
optional deps installed (`python-flint` + `gmpy2`), **~12 s** without:

1. **`python-flint` (`fmpz_mod_poly`)** for `U(τ), V(τ), W(τ)` polynomial
   evaluation — C-backed Horner, ~5-10× the CPython loop.
2. **4-worker `ProcessPoolExecutor`** over the EC scalar-mul lists.
   ~3× wall-clock.
3. **`gmpy2.mpz`** for the per-wire `(β·U + α·V + W) · δ⁻¹` reductions
   and the `tau_powers` running product.  ~2× on the scalar math.

All three are guarded by `try: import …`; pure-CPython fallback if a
dep is missing.

### JAX persistent compilation cache

`path_setup.py` configures
`jax.experimental.compilation_cache → ~/.cache/maple-jax-compile/`.
That means the ~10 min G2 fused MSM cold compile is paid **once per
user**, not once per daemon restart.  Override with `MAPLE_JAX_CACHE_DIR`.

---

## Extending

### Add a predefined circuit

Edit `demo_circuits.py`:

```python
register_circuit(CircuitSpec(
    key             = "my_circuit",
    name            = "My circuit: f(x) = y",
    description     = "...",
    public_label    = "y",
    input_schema    = [InputField(name="x", label="x", default=0)],
    r1cs_builder    = my_r1cs_function,
    witness_builder = my_witness_function,
    public_output   = lambda w: w[-1],
    num_wires_orig  = N,
    wire_labels     = ["wire 0 — constant", ...],
))
```

Restart the daemon.  The sidebar shows the new circuit with status
`available`; click **Build** to compile the pkl, **Use** to bind it.

The cache pkl lives at the path returned by
`spec.cache_path(cache_dir, max_size=1024)` — by convention
`{cache_dir}/{key}_{max_size}_setup.pkl`.

### Arbitrary R1CS upload

Reserved hook in `demo_circuits.register_arbitrary_r1cs` — currently a
`NotImplementedError` placeholder.  Out of scope for v1.

---

## Troubleshooting

### "Can't reach prover daemon at ..."

The daemon isn't running, isn't reachable, or `--host`/`--port`
mismatch.  Check / restart:

```bash
./Groth16_TPU/demo/demo.sh status
./Groth16_TPU/demo/demo.sh restart
```

### TPU env stuck on `warmup` for > 30 min

Cold first-ever process on a fresh user account can hit ~27 min in
`warmup` because the JAX persistent cache is empty.  Restart the
daemon and the cache populates in `~/.cache/maple-jax-compile/`,
bringing the next warmup down to ~5 min.

If it never finishes: tail the daemon stderr — XLA errors surface there.

### Build stuck on `qap_build` / `encode_pk` / `encode_qap`

These are the TPU-touching build phases.  They serialise behind any
active prove or the env warmup via `_tpu_lock`.  If you see them
genuinely hung (>5× expected time), restart the daemon — TPU state
sometimes wedges in ways the JAX runtime can't recover from.

### `activate` returns 500 / `_frombuffer() takes 4 ... but 5 were given`

The circuit's cached pkl was serialized by a **different NumPy version**
(commonly: the `.dev_loop_cache/*.pkl` files were copied from another
checkout/machine).  `pickle.load` can't reconstruct the buffer-backed
arrays, so `Setup.load` throws and `activate` 500s.  Fix: regenerate the
pkl in *this* environment — click **Rebuild** in the sidebar, or:

```bash
curl -X POST "http://localhost:8000/circuits/<key>/build?force=true"
```

Don't copy `.dev_loop_cache` between environments with mismatched NumPy;
rebuild in place instead.

### "circuit X is not active"

`POST /prove` returned `503` because the requested circuit isn't the
one currently bound to the prover.  Either the UI auto-activates as
part of the submit handler, or you can hit
`POST /circuits/{key}/activate` directly.  Only one circuit may be
active at a time.

### "Witness does not satisfy the R1CS"

The local **Check witness** button must pass before **Generate proof**
enables.  Click **Autofill** to populate the right intermediate values
from your high-level inputs.

---
