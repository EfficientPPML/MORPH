"""Registry of demo circuits.

Each entry describes a circuit the GUI can offer to the user, including:

  * how to build its R1CS;
  * how to turn user-facing inputs into a witness;
  * what input fields the UI should render;
  * how to describe the public output, for the with/without-ZKP panel.

For v1 only the BLS12-377 cubic ``x³ + x + 5 = y`` ships.  Adding a new
predefined circuit is a single :data:`CIRCUITS` entry; supporting fully
arbitrary user-uploaded R1CS is left as a follow-up (see
:func:`register_circuit` / the hook at the bottom of this file).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# Heavy imports deferred so this module is cheap to load (used in the
# Streamlit page header to list circuit names before TPU init finishes).
from groth16.r1cs import R1CS, build_cubic_r1cs, generate_witness
from bls12_377.params import r as _FR


# ── UI schema for input fields ─────────────────────────────────────────


@dataclass
class InputField:
    """One user-editable input on the Streamlit form.

    Held by :class:`CircuitSpec.input_schema`.  The UI maps each field
    to a Streamlit widget by ``kind``:

      * ``"int"``  → ``st.number_input(step=1, min_value, max_value)``
      * ``"text"`` → ``st.text_input``  (free-form, for advanced witnesses)
    """
    name:    str
    label:   str
    kind:    str          = "int"
    default: Any          = 0
    min:     Optional[int] = None
    max:     Optional[int] = None
    help:    str          = ""


@dataclass
class CircuitSpec:
    """One predefined circuit.

    Fields
    ------
    key:                 stable identifier (used as URL slug / cache filename).
    name:                pretty name shown in the dropdown.
    description:         one-paragraph description for the sidebar.
    public_label:        label for the public output column
                         ("output y", "membership proof", etc.).
    input_schema:        list of :class:`InputField` the UI should render
                         for the "easy" / auto-witness path.
    r1cs_builder:        ``() -> R1CS`` — pure CPU; cached on disk by the
                         framework once compiled.
    witness_builder:     ``(**inputs) -> List[int]`` — accepts the values
                         from ``input_schema`` and returns the full
                         witness vector ``[1, ...inputs..., ...intermediates..., ...public]``.
    public_output:       ``(witness) -> Any`` — extract the human-readable
                         output value for display in the compare panel.
    wire_labels:         per-wire human-readable labels (length
                         ``num_wires_orig``).  Used by the "advanced"
                         witness editor in the UI so the user can see
                         which entry corresponds to which intermediate
                         (e.g. ``"x²"``, ``"out"``).
    wire_helps:          optional per-wire help / instructional tooltip
                         (length ``num_wires_orig``).  Rendered as the
                         widget's ``help=`` parameter so hovering shows
                         what the wire represents + what happens if you
                         change it.  Falls back to empty string if None.
    num_wires_orig:      number of wires in the unpadded R1CS.  Demo
                         pads to ``max_size`` with zeros for the rest.
    category:            ``"practical"`` (real-world ZK use case) or
                         ``"archive"`` (toy / teaching circuit).  UI uses
                         this to put archived circuits behind an opt-in
                         expander in the sidebar.
    try_it:              one-line "try this" suggestion shown above the
                         witness editor; sets up the playful exploration
                         path.  Optional.
    """
    key:             str
    name:            str
    description:     str
    public_label:    str
    input_schema:    List[InputField]
    r1cs_builder:    Callable[[], R1CS]
    witness_builder: Callable[..., List[int]]
    public_output:   Callable[[List[int]], Any]
    wire_labels:     List[str]
    num_wires_orig:  int
    wire_helps:      Optional[List[str]] = None
    category:        str                 = "practical"
    try_it:          str                 = ""
    # Long-form educational text shown in a collapsible expander above
    # the witness editor.  Markdown-formatted; should cover the math,
    # the R1CS layout, and the real-world analog at a level a curious
    # user can follow.  Empty string → no expander rendered.
    long_info:       str                 = ""

    def cache_path(self, cache_dir: str, max_size: int = 1024) -> str:
        """Canonical on-disk location for this circuit's Setup pickle.

        Convention: ``{cache_dir}/{key}_{max_size}_setup.pkl``.  Used by
        ``demo.daemon`` to decide whether to auto-run ``prep_circuit`` on
        boot, and by ``Setup.load`` to read the artefact back.
        """
        return os.path.join(cache_dir, f"{self.key}_{max_size}_setup.pkl")


# ── The registry — append below to add a circuit ───────────────────────


CIRCUITS: Dict[str, CircuitSpec] = {}


def register_circuit(spec: CircuitSpec) -> None:
    """Add a circuit to the registry.  Overwrites any previous entry."""
    CIRCUITS[spec.key] = spec


# Hook reserved for future "arbitrary R1CS upload" / DSL compilation.
# When that lands, callers will be able to do something like::
#
#     spec = compile_user_circuit(uploaded_file)
#     register_circuit(spec)
#
# without touching the rest of the demo.
def register_arbitrary_r1cs(*args, **kwargs):  # noqa: D401 — placeholder
    raise NotImplementedError(
        "Arbitrary R1CS compilation is not yet implemented.  For now, "
        "use one of the predefined circuits in CIRCUITS."
    )


# ── Predefined circuits ────────────────────────────────────────────────


def _cubic_public_output(witness: List[int]) -> int:
    # build_cubic_r1cs: wire 4 is the public 'out'.
    return witness[4]


register_circuit(CircuitSpec(
    key             = "cubic",
    name            = "Cubic: x³ + x + 5 = y",
    description     = (
        "BLS12-377's introductory R1CS: prove you know an `x` such that "
        "`x³ + x + 5 = y` without revealing `x`.  4 constraints, 5 wires.  "
        "Used as the smoke test for every Groth16 implementation."
    ),
    public_label    = "y",
    input_schema    = [
        InputField(
            name    = "x",
            label   = "x  (private input)",
            kind    = "int",
            default = 3,
            help    = "Prover's secret.  Verifier never sees this.",
        ),
    ],
    r1cs_builder    = build_cubic_r1cs,
    witness_builder = lambda x: generate_witness(int(x)),
    public_output   = _cubic_public_output,
    num_wires_orig  = 5,
    wire_labels     = [
        "wire 0 — constant 1",
        "wire 1 — x  (your secret)",
        "wire 2 — x²",
        "wire 3 — x³",
        "wire 4 — y  (public output)",
    ],
    wire_helps      = [
        "Every Groth16 witness starts with the constant 1.  Don't change this.",
        "Your secret.  The verifier never sees this — only that it satisfies the equation.",
        "Intermediate: should equal x · x.  Tampering here makes constraint #0 fail.",
        "Intermediate: should equal x² · x.  Tampering here makes constraint #1 fail.",
        "Public output: should equal x³ + x + 5.  This is the only wire the verifier sees.",
    ],
    category        = "archive",
    try_it          = "Set x=3 and click Autofill — the verifier learns y=35 but never your x.",
    long_info       = """
**What this proves**

You know an `x` such that **x³ + x + 5 = y**, where `y` is public.  The
verifier learns `y`; `x` stays secret.

**R1CS layout (4 constraints, 5 wires)**

| # | constraint     | A         | B   | C            |
|---|----------------|-----------|-----|--------------|
| 0 | `x · x = x²`   | `{1: 1}`  | `{1: 1}` | `{2: 1}` |
| 1 | `x² · x = x³`  | `{2: 1}`  | `{1: 1}` | `{3: 1}` |
| 2 | `(x³+x+5)·1=y` | `{3:1, 1:1, 0:5}` | `{0: 1}` | `{4: 1}` |
| 3 | `1 · 1 = 1`    | `{0: 1}`  | `{0: 1}` | `{0: 1}` (padding) |

**Why this circuit exists**

The Vitalik Buterin Groth16 walkthrough uses exactly this circuit — it's
the "Hello World" of zk-SNARKs.  Tiny enough to hand-compute the witness
in your head; non-trivial enough to demonstrate every Groth16 piece
(polynomial commitments, pairings, blinding).
""".strip(),
))


# ── power_11: x^11 = y ────────────────────────────────────────────────


_POWER11_K = 11  # exponent.  Witness layout: [1, x, x², ..., x^k].


def _build_power11_r1cs() -> R1CS:
    """``x^k = y`` for k=11.  k-1 multiplications, k+1 wires.

    Wires: ``[1, x, x², x³, …, x^k]`` → ``num_wires = k+1 = 12``.
    Last wire is the public output.  Constraints ``x^i · x = x^{i+1}``
    for ``i = 1 … k-1``.
    """
    k = _POWER11_K
    nw = k + 1                         # 12
    constraints = []
    # x^i · x = x^{i+1} for i=1..k-1, where wire index i holds x^i.
    for i in range(1, k):
        constraints.append(({i: 1}, {1: 1}, {i + 1: 1}))
    A = [c[0] for c in constraints]
    B = [c[1] for c in constraints]
    C = [c[2] for c in constraints]
    return R1CS(
        num_constraints = len(constraints),
        num_wires       = nw,
        num_public      = 1,
        A = A, B = B, C = C,
    )


def _power11_witness(x: int) -> List[int]:
    k = _POWER11_K
    x = int(x) % _FR
    w = [1, x]
    p = x
    for _ in range(k - 1):
        p = (p * x) % _FR
        w.append(p)
    # len(w) == k + 1; last entry is x^k.
    return w


def _power11_public_output(witness: List[int]) -> int:
    return witness[_POWER11_K]   # x^k


register_circuit(CircuitSpec(
    key             = "power_11",
    name            = "Power-11: x¹¹ = y",
    description     = (
        "Prove you know an `x` such that `x¹¹ = y`.  10 chained "
        "multiplications, 12 wires.  Slightly larger than the cubic — "
        "shows that prove time stays flat as constraints grow (still "
        "padded to the same 1024 shape on the TPU)."
    ),
    public_label    = "y  (= x¹¹)",
    input_schema    = [
        InputField(
            name    = "x",
            label   = "x  (private input)",
            kind    = "int",
            default = 2,
            help    = "Prover's secret base.  Verifier learns only x¹¹.",
        ),
    ],
    r1cs_builder    = _build_power11_r1cs,
    witness_builder = lambda x: _power11_witness(int(x)),
    public_output   = _power11_public_output,
    num_wires_orig  = _POWER11_K + 1,
    wire_labels     = (
        ["wire 0 — constant 1", "wire 1 — x  (your secret)"]
        + [f"wire {i} — x^{i}" for i in range(2, _POWER11_K)]
        + [f"wire {_POWER11_K} — y = x^{_POWER11_K}  (public output)"]
    ),
    category        = "archive",
    try_it          = "Set x=2 and Autofill — y becomes 2048 = 2¹¹.  Verifier learns y, not x.",
    long_info       = """
**What this proves**

You know an `x` such that **x¹¹ = y**, where `y` is public.

**R1CS layout (10 constraints, 12 wires)**

Each multiplication `x^i · x = x^{i+1}` is one R1CS constraint.  10
multiplications take `x` up to `x¹¹`.  Per constraint:
```
A = {x^i wire: 1},  B = {x wire: 1},  C = {x^{i+1} wire: 1}
```

**Why this circuit exists**

Demonstrates that **Groth16 prove time depends on the padded shape, not
the number of "real" constraints.**  cubic (3 muls) and power_11 (10
muls) both pad to 1024 — both take ~700 ms.  The "constant-time at fixed
shape" property is why ZK rollups batch many user transactions into one
fixed-size SNARK.
""".strip(),
))


# ── fibonacci_20: f(20; a, b) = y ─────────────────────────────────────


_FIB_N = 20  # iterations.  Witness: [1, a, b, f_2, …, f_N].


def _build_fibonacci_r1cs() -> R1CS:
    """``f_i = f_{i-1} + f_{i-2}`` for i=2..N, with seeds (a, b)=(f_0, f_1).

    Each addition is encoded as a (linear) · 1 = (target) constraint:
    ``(f_{i-1} + f_{i-2}) · 1 = f_i``.  Wires:

        wire 0  : const 1
        wire 1  : a  (= f_0, private)
        wire 2  : b  (= f_1, private)
        wire 3  : f_2  (private)
        …
        wire N  : f_{N-1}  (private)
        wire N+1: f_N      (public output)

    Total wires = N+2, constraints = N-1.
    """
    N = _FIB_N
    # Map fib index i ∈ [0..N] → wire index.
    #   f_0 → wire 1
    #   f_1 → wire 2
    #   f_i → wire i+1
    def widx(i: int) -> int:
        return i + 1

    A_list, B_list, C_list = [], [], []
    for i in range(2, N + 1):
        # (f_{i-2} + f_{i-1}) · 1 = f_i
        A_list.append({widx(i - 2): 1, widx(i - 1): 1})
        B_list.append({0: 1})
        C_list.append({widx(i): 1})

    return R1CS(
        num_constraints = len(A_list),
        num_wires       = N + 2,
        num_public      = 1,           # f_N
        A = A_list, B = B_list, C = C_list,
    )


def _fibonacci_witness(a: int, b: int) -> List[int]:
    a = int(a) % _FR
    b = int(b) % _FR
    fib = [a, b]
    for _ in range(2, _FIB_N + 1):
        fib.append((fib[-1] + fib[-2]) % _FR)
    return [1, *fib]


def _fibonacci_public_output(witness: List[int]) -> int:
    return witness[_FIB_N + 1]   # f_N


register_circuit(CircuitSpec(
    key             = "fibonacci_20",
    name            = f"Fibonacci-{_FIB_N}: f({_FIB_N}; a, b) = y",
    description     = (
        f"Prove you know seeds `(a, b)` such that the {_FIB_N}-th term "
        f"of the Fibonacci-like sequence `f_i = f_{{i-1}} + f_{{i-2}}` "
        f"equals `y`.  Each addition is one R1CS row "
        f"(`(f_{{i-1}} + f_{{i-2}}) · 1 = f_i`).  "
        f"{_FIB_N - 1} constraints, {_FIB_N + 2} wires — sequential, "
        f"showcases proofs that hide both seeds."
    ),
    public_label    = f"y  (= f_{_FIB_N})",
    input_schema    = [
        InputField(name="a", label="a  (= f₀, private)", kind="int",
                   default=1, help="First Fibonacci seed."),
        InputField(name="b", label="b  (= f₁, private)", kind="int",
                   default=1, help="Second Fibonacci seed."),
    ],
    r1cs_builder    = _build_fibonacci_r1cs,
    witness_builder = lambda a, b: _fibonacci_witness(int(a), int(b)),
    public_output   = _fibonacci_public_output,
    num_wires_orig  = _FIB_N + 2,
    wire_labels     = (
        ["wire 0 — constant 1", "wire 1 — a (= f₀, your secret)",
         "wire 2 — b (= f₁, your secret)"]
        + [f"wire {i + 1} — f_{i}" for i in range(2, _FIB_N)]
        + [f"wire {_FIB_N + 1} — y = f_{_FIB_N}  (public output)"]
    ),
    category        = "archive",
    try_it          = f"Try (a, b) = (1, 1): you get the standard Fibonacci, y = f_{_FIB_N} = 10946.",
    long_info       = f"""
**What this proves**

You know seeds `(a, b)` such that the {_FIB_N}-th term of the recurrence
`f_i = f_{{i-1}} + f_{{i-2}}` equals `y` (public).  Both seeds stay
secret; only the final term is revealed.

**R1CS layout ({_FIB_N - 1} constraints, {_FIB_N + 2} wires)**

Each addition `f_i = f_{{i-1}} + f_{{i-2}}` is one R1CS constraint:
```
A = {{f_{{i-2}}: 1, f_{{i-1}}: 1}},  B = {{const: 1}},  C = {{f_i: 1}}
```
The `· const` on B is the standard R1CS trick — multiplication by 1
turns a linear combination into a quadratic-form constraint.  `{_FIB_N - 1}`
such constraints chain seed (a, b) into f_{_FIB_N}.

**Why this circuit exists**

Demonstrates **sequential recurrences** — proving you ran a computation
forward N steps.  The same shape generalises to VDF (verifiable delay
function) proofs and to "I executed this program for N steps" claims in
zkVM rollups.
""".strip(),
))


# ── range_32: 0 ≤ x < 2^32 ────────────────────────────────────────────


_RANGE_BITS = 32


def _build_range32_r1cs() -> R1CS:
    """Prove ``0 ≤ x < 2^32`` via bit decomposition.

    Wires:
        wire 0           : const 1
        wire 1           : x (private)
        wires 2..2+B-1   : bits b_0 … b_{B-1} (private)
        wire 2+B         : ok (public, = 1)

    Constraints:
        b_i · b_i = b_i                    (B constraints, forces b_i ∈ {0,1})
        (Σ 2^i · b_i) · 1 = x              (1 constraint, bit-recomposition)
        1 · 1 = ok                         (1 constraint, forces ok = 1)

    Total = B + 2 constraints.  Public output ``ok`` is just a sentinel
    bit — verifier sees only ``[ok = 1]`` and learns that the prover
    knows an x in the range.  x itself stays private.
    """
    B = _RANGE_BITS
    nw = 2 + B + 1                     # 1(const) + 1(x) + B(bits) + 1(ok)
    bit_widx = lambda i: 2 + i
    ok_widx  = 2 + B

    A_list, B_list, C_list = [], [], []
    # Booleanity: b_i · b_i = b_i
    for i in range(B):
        A_list.append({bit_widx(i): 1})
        B_list.append({bit_widx(i): 1})
        C_list.append({bit_widx(i): 1})
    # Bit recomposition: (Σ 2^i · b_i) · 1 = x
    A_recompose: Dict[int, int] = {bit_widx(i): (1 << i) for i in range(B)}
    A_list.append(A_recompose)
    B_list.append({0: 1})
    C_list.append({1: 1})
    # Sentinel: 1 · 1 = ok
    A_list.append({0: 1})
    B_list.append({0: 1})
    C_list.append({ok_widx: 1})

    return R1CS(
        num_constraints = len(A_list),
        num_wires       = nw,
        num_public      = 1,           # ok
        A = A_list, B = B_list, C = C_list,
    )


def _range32_witness(x: int) -> List[int]:
    x = int(x)
    if x < 0 or x >= (1 << _RANGE_BITS):
        raise ValueError(
            f"x = {x} is outside [0, 2^{_RANGE_BITS}) — can't bit-decompose"
        )
    bits = [(x >> i) & 1 for i in range(_RANGE_BITS)]
    return [1, x % _FR, *bits, 1]


def _range32_public_output(witness: List[int]) -> int:
    return witness[2 + _RANGE_BITS]   # ok sentinel — always 1


register_circuit(CircuitSpec(
    key             = "range_32",
    name            = f"Range-{_RANGE_BITS}: 0 ≤ x < 2^{_RANGE_BITS}",
    description     = (
        f"Prove you know an `x` in `[0, 2^{_RANGE_BITS})` "
        f"without revealing `x`.  Decomposes `x` into {_RANGE_BITS} bits, "
        f"forces each bit to {{0,1}} via `b·b=b`, and recomposes.  "
        f"Public output is just the sentinel `ok = 1` — verifier learns "
        f"only that the predicate holds.  Classic 'I'm of age' demo."
    ),
    public_label    = "ok  (= 1 ⇔ x in range)",
    input_schema    = [
        InputField(
            name    = "x",
            label   = f"x  (private input, 0 ≤ x < 2^{_RANGE_BITS})",
            kind    = "int",
            default = 21,
            min     = 0,
            max     = (1 << _RANGE_BITS) - 1,
            help    = "Prover's secret value.  Verifier never sees this.",
        ),
    ],
    r1cs_builder    = _build_range32_r1cs,
    witness_builder = lambda x: _range32_witness(int(x)),
    public_output   = _range32_public_output,
    num_wires_orig  = 2 + _RANGE_BITS + 1,
    wire_labels     = (
        ["wire 0 — constant 1", "wire 1 — x (your secret)"]
        + [f"wire {2 + i} — bit b_{i} ∈ {{0, 1}}" for i in range(_RANGE_BITS)]
        + [f"wire {2 + _RANGE_BITS} — ok = 1  (public sentinel)"]
    ),
    category        = "practical",       # range proofs are real-world
    try_it          = "Try x=21.  Verifier learns only 'ok = predicate holds', not x itself.",
    long_info       = f"""
**What this proves**

You know an `x` in `[0, 2^{_RANGE_BITS})` — without revealing what `x` is.
Public output is a sentinel `ok = 1` that attests "yes, the predicate
holds."

**R1CS layout ({_RANGE_BITS + 2} constraints, {_RANGE_BITS + 3} wires)**

Three constraint types:

1. **Booleanity** — for each bit `b_i`, the constraint `b_i · b_i = b_i`
   forces `b_i ∈ {{0, 1}}` (the only solutions in Fr).  {_RANGE_BITS}
   constraints, one per bit.
2. **Bit-recomposition** — `(Σ 2^i · b_i) · 1 = x` ties the bits to `x`.
3. **Sentinel** — `1 · 1 = ok` forces the public output to 1.

**Why this circuit exists**

This is THE textbook ZK building block.  Real-world uses:

- **"I'm over 18"** without revealing your birthdate (range proof on age).
- **Confidential transactions** (Monero, Bulletproofs) — prove balances
  are non-negative without revealing them.
- **Anonymous credentials** — prove a credential value falls in a
  permitted range.

For 64-bit ranges you double the constraint count; the recipe is identical.
""".strip(),
))


# ══════════════════════════════════════════════════════════════════════
# Practical circuits — patterns actually used in real ZK applications
# ══════════════════════════════════════════════════════════════════════
#
# Each circuit below mirrors a real-world ZK app primitive at small
# scale.  The math is correct; the round counts / tree depths are
# trimmed so everything fits in the demo's 1024-constraint shape.
# All use the same BLS12-377 Fr arithmetic as the toy circuits.


# ── MiMC hash preimage ────────────────────────────────────────────────
#
# MiMC is one of the earliest "ZK-friendly" hash functions — used by
# the original Zcash Sapling and many later ZK rollups.  Per-round it
# does ``state' = (state + c_i)^5``.  ``x^5`` is a permutation on Fr
# (because gcd(5, r-1) = 1 for BLS12-377 Fr), so the function is
# invertible per-round but the chain of N rounds is one-way.
#
# Per round we need ``state' = ((state + c)^2)^2 * (state + c)``, which
# is 3 multiplication constraints (the (state + c) sum is a linear
# combination, free in R1CS).  N rounds → 3N constraints.
#
# For real security with d=5 you want N ≥ ⌈log_5(r) · 2⌉ ≈ 55 rounds.
# We use N=20 here for a snappy demo — illustrative of the structure,
# NOT a secure hash.

_MIMC_ROUNDS = 20
# Didactic round constants — production MiMC uses uniformly-random
# field elements per round.  Linear progression is fine for the demo.
_MIMC_C = [(i + 1) * 1000003 % _FR for i in range(_MIMC_ROUNDS)]


def _mimc_hash(x: int) -> int:
    """Reference MiMC-x⁵ hash; matches the R1CS below.  Pure Python."""
    s = int(x) % _FR
    for c in _MIMC_C:
        t = (s + c) % _FR
        s = pow(t, 5, _FR)
    return s


def _build_mimc_r1cs() -> R1CS:
    """R1CS for ``MiMC(x) = h``.  Wire layout:

        wire 0          : constant 1
        wire 1          : x (preimage, private)
        wires 2 .. 3N+1 : per-round intermediates (x², x⁴, s_{i+1})
        wire 3N+1       : final state = hash output (public)

    Constraints per round (state = s_i):
        (s_i + c_i)² = x²_i               (1)
        x²_i · x²_i  = x⁴_i               (1)
        x⁴_i · (s_i + c_i) = s_{i+1}      (1)
    """
    N = _MIMC_ROUNDS

    # Wire index helpers.
    W_CONST = 0
    W_X     = 1
    def W_X2(i):    return 2 + 3 * i + 0
    def W_X4(i):    return 2 + 3 * i + 1
    def W_S_NEXT(i): return 2 + 3 * i + 2     # s_{i+1}
    def W_S_IN(i):
        return W_X if i == 0 else W_S_NEXT(i - 1)

    A_list, B_list, C_list = [], [], []
    for i in range(N):
        c   = _MIMC_C[i]
        s_in = W_S_IN(i)
        # (s + c) * (s + c) = x²
        A_list.append({s_in: 1, W_CONST: c})
        B_list.append({s_in: 1, W_CONST: c})
        C_list.append({W_X2(i): 1})
        # x² * x² = x⁴
        A_list.append({W_X2(i): 1})
        B_list.append({W_X2(i): 1})
        C_list.append({W_X4(i): 1})
        # x⁴ * (s + c) = s_next
        A_list.append({W_X4(i): 1})
        B_list.append({s_in: 1, W_CONST: c})
        C_list.append({W_S_NEXT(i): 1})

    return R1CS(
        num_constraints = len(A_list),
        num_wires       = 2 + 3 * N,
        num_public      = 1,                  # the final s_N
        A = A_list, B = B_list, C = C_list,
    )


def _mimc_witness(x: int) -> List[int]:
    """Trace every intermediate in the MiMC chain.  Order MUST match
    the wire layout above."""
    x = int(x) % _FR
    w = [1, x]
    s = x
    for c in _MIMC_C:
        t  = (s + c) % _FR
        x2 = (t * t)  % _FR
        x4 = (x2 * x2) % _FR
        s  = (x4 * t)  % _FR
        w.extend([x2, x4, s])
    return w


register_circuit(CircuitSpec(
    key             = "mimc_preimage",
    name            = "MiMC hash preimage",
    description     = (
        "Prove you know an `x` such that `MiMC(x) = h`, without revealing `x`.  "
        f"The hash chains {_MIMC_ROUNDS} rounds of `state ← (state + c_i)⁵` "
        "over BLS12-377 Fr.  Same shape as the hash primitives used in real "
        "ZK rollups (Zcash Sapling, Aztec, Mina) — just fewer rounds for the "
        "demo.  Real-world use: anonymous credentials, commit-then-reveal, "
        "ZK password proofs."
    ),
    public_label    = "h = MiMC(x)",
    input_schema    = [
        InputField(
            name    = "x",
            label   = "x  (preimage, private)",
            kind    = "int",
            default = 12345,
            help    = "Your secret value.  The verifier learns only its MiMC hash.",
        ),
    ],
    r1cs_builder    = _build_mimc_r1cs,
    witness_builder = lambda x: _mimc_witness(int(x)),
    public_output   = lambda w: w[2 + 3 * _MIMC_ROUNDS - 1],  # final s_N
    num_wires_orig  = 2 + 3 * _MIMC_ROUNDS,
    wire_labels     = (
        ["wire 0 — constant 1",
         "wire 1 — x  (preimage, your secret)"]
        + sum(
            [
                [f"wire {2 + 3*i + 0} — round {i} · (s + c_{i})²",
                 f"wire {2 + 3*i + 1} — round {i} · (s + c_{i})⁴",
                 f"wire {2 + 3*i + 2} — round {i} · s_{i+1} (next state)"]
                for i in range(_MIMC_ROUNDS - 1)
            ],
            []
        )
        # Last round — flag the final state as the hash output.
        + [f"wire {2 + 3*(_MIMC_ROUNDS-1) + 0} — round {_MIMC_ROUNDS-1} · (s + c)²",
           f"wire {2 + 3*(_MIMC_ROUNDS-1) + 1} — round {_MIMC_ROUNDS-1} · (s + c)⁴",
           f"wire {2 + 3*(_MIMC_ROUNDS-1) + 2} — h = MiMC(x)  (public output)"]
    ),
    wire_helps      = (
        ["Constant 1 — required by every Groth16 R1CS.",
         "The preimage you're hiding.  Verifier sees only h, never x."]
        + sum(
            [
                [f"Intermediate (state + c_{i})² — used to compute (state)⁵ via squaring.",
                 f"Intermediate (state + c_{i})⁴ — second squaring.",
                 f"State after round {i+1}.  Each round permutes the state under c_{i}."]
                for i in range(_MIMC_ROUNDS)
            ],
            []
        )
    ),
    try_it          = "Set x to any number; Autofill computes the hash.  "
                      "Try tampering with an intermediate wire — the verifier rejects.",
    long_info       = f"""
**What this proves**

You know an `x` such that **MiMC(x) = h**, where `h` is public.  MiMC is
the original "ZK-friendly hash" — designed so the same operation that's
hard for adversaries (inverting `x⁵`) is **cheap** as an R1CS.

**The hash itself**

```
state = x
for i in 0 … {_MIMC_ROUNDS - 1}:
    state ← (state + c_i)⁵    (mod r)
h = state
```

Each round adds a public constant `c_i` then applies the S-box `y → y⁵`.
Cubing (`y³`) would be even cheaper, but `gcd(3, r-1) ≠ 1` for BLS12-377
so `y³` isn't a permutation; `y⁵` works because `gcd(5, r-1) = 1`.

**R1CS layout ({3 * _MIMC_ROUNDS} constraints, {2 + 3 * _MIMC_ROUNDS} wires)**

Per round: `y⁵ = ((y)²)² · y` — three multiplications.

| # | constraint              | meaning            |
|---|-------------------------|--------------------|
| 0 | `(s + c) · (s + c) = x²`| `x² = (s + c_i)²`  |
| 1 | `x² · x² = x⁴`          | square again       |
| 2 | `x⁴ · (s + c) = s_next` | one more multiply  |

`{_MIMC_ROUNDS}` rounds × 3 constraints = `{3*_MIMC_ROUNDS}` constraints
total.  Demo uses {_MIMC_ROUNDS} rounds for snappy compile time; real
security on BLS12-377 Fr wants ≥55.

**Real-world uses**

- **ZK rollup commitment schemes** — commit to a state root with MiMC,
  prove transitions in ZK.
- **Anonymous credentials** — commit a credential as `h = MiMC(secret)`;
  later prove you know `secret` without revealing it.
- **Predecessor of Poseidon** — Poseidon refined the MiMC blueprint
  with multiple state elements and a partial-rounds schedule.
""".strip(),
))


# ── Poseidon-style hash preimage (didactic, t=2 state) ────────────────
#
# Poseidon is the ZK-friendly hash used by Aztec, Mina, modern zk-EVMs.
# Real Poseidon: state size t∈{2,3,4,...}, 8 full rounds + 57 partial
# rounds, MDS matrix mult, full ARK schedule.  This is a SIMPLIFIED
# t=2, 8 full + 4 partial rounds variant with toy MDS + constants —
# faithful to the structure but not cryptographically secure.

_POSE_T            = 2          # state size
_POSE_FULL_ROUNDS  = 6          # full rounds (3 at start + 3 at end)
_POSE_PARTIAL_RDS  = 4          # partial rounds in the middle
_POSE_TOTAL_RDS    = _POSE_FULL_ROUNDS + _POSE_PARTIAL_RDS
# MDS matrix — must be invertible.  ``[[2, 1], [1, 2]]`` over Fr has
# determinant 3 ≠ 0, so it's invertible.  Suitable for didactic demo.
_POSE_MDS = [[2, 1], [1, 2]]
# Per-round constants (one per state element).  Linear schedule — real
# Poseidon uses Grain-LFSR-derived constants.
_POSE_C = [
    [((r * _POSE_T + j) * 0x9e3779b97f4a7c15) % _FR
     for j in range(_POSE_T)]
    for r in range(_POSE_TOTAL_RDS)
]


def _is_full_round(r: int) -> bool:
    half = _POSE_FULL_ROUNDS // 2
    return r < half or r >= _POSE_FULL_ROUNDS // 2 + _POSE_PARTIAL_RDS


def _poseidon_hash(x: int) -> int:
    """Reference implementation; matches the R1CS below."""
    s = [int(x) % _FR, 0]                       # state, capacity element 0
    for r in range(_POSE_TOTAL_RDS):
        s = [(s[j] + _POSE_C[r][j]) % _FR for j in range(_POSE_T)]
        if _is_full_round(r):
            s = [pow(v, 5, _FR) for v in s]
        else:
            s[0] = pow(s[0], 5, _FR)
        s = [
            (_POSE_MDS[i][0] * s[0] + _POSE_MDS[i][1] * s[1]) % _FR
            for i in range(_POSE_T)
        ]
    return s[0]


def _build_poseidon_r1cs() -> R1CS:
    """R1CS for Poseidon(x) = h with t=2 state and the schedule above.

    Each full round commits 3 mult constraints per state element
    (cubing twice + final multiply for x⁵), each partial round commits
    3 constraints (only on s[0]).  MDS mult is linear → free.
    """
    W_CONST   = 0
    W_X       = 1
    next_wire = 2

    # We'll express the state as linear combinations of existing wires.
    # That keeps wire allocation minimal: only S-box outputs need a wire.
    # Initial state: s[0] = x (wire W_X), s[1] = 0 (wire W_CONST coeff 0).
    state_lc = [{W_X: 1}, {}]    # list of {wire_idx: coeff} dicts

    A_list, B_list, C_list = [], [], []

    def _add_constraint(a, b, c):
        A_list.append(a); B_list.append(b); C_list.append(c)

    def _lc_add(lc: Dict[int, int], extra: Dict[int, int]) -> Dict[int, int]:
        """Linear-combination addition mod r (free in R1CS)."""
        out = dict(lc)
        for k, v in extra.items():
            out[k] = (out.get(k, 0) + v) % _FR
        return out

    def _sbox(linear_in: Dict[int, int], commit_wire: int) -> int:
        """Constrain ``commit_wire = (Σ linear_in)⁵``.

        Three new wires + three constraints:
            x²       = LC · LC
            x⁴       = x² · x²
            output   = x⁴ · LC      → committed to ``commit_wire``
        """
        nonlocal next_wire
        w_x2 = commit_wire           # we'll reuse-ish: actually allocate
        w_x2 = next_wire; next_wire += 1
        w_x4 = next_wire; next_wire += 1
        # output = x⁴ * LC — committed into a NEW wire.
        w_out = next_wire; next_wire += 1
        # (LC) * (LC) = w_x2
        _add_constraint(dict(linear_in), dict(linear_in), {w_x2: 1})
        # w_x2 * w_x2 = w_x4
        _add_constraint({w_x2: 1}, {w_x2: 1}, {w_x4: 1})
        # w_x4 * LC = w_out
        _add_constraint({w_x4: 1}, dict(linear_in), {w_out: 1})
        return w_out

    for r in range(_POSE_TOTAL_RDS):
        # Add round constants (linear).
        state_lc = [
            _lc_add(state_lc[j], {W_CONST: _POSE_C[r][j]})
            for j in range(_POSE_T)
        ]
        # S-box (cubing).  Full rounds: every element.  Partial: only s[0].
        if _is_full_round(r):
            w_out = [_sbox(state_lc[j], -1) for j in range(_POSE_T)]
            state_lc = [{w: 1} for w in w_out]
        else:
            w_out0 = _sbox(state_lc[0], -1)
            # state[1] stays a linear combination; need to commit it so MDS
            # mult below produces a consistent reference.  Actually no — MDS
            # is linear over LCs, so we keep state_lc[1] as the existing LC.
            state_lc = [{w_out0: 1}, state_lc[1]]
        # MDS multiply: state' = M · state (linear).
        new_state = []
        for i in range(_POSE_T):
            row = {}
            for j in range(_POSE_T):
                for k, v in state_lc[j].items():
                    row[k] = (row.get(k, 0) + _POSE_MDS[i][j] * v) % _FR
            new_state.append(row)
        state_lc = new_state

    # Final output: state[0].  Commit to a dedicated public wire so
    # ``num_public_orig=1`` selects it.
    w_out_pub = next_wire; next_wire += 1
    # Constrain: state[0] · 1 = w_out_pub
    _add_constraint(state_lc[0], {W_CONST: 1}, {w_out_pub: 1})

    return R1CS(
        num_constraints = len(A_list),
        num_wires       = next_wire,
        num_public      = 1,
        A = A_list, B = B_list, C = C_list,
    )


def _poseidon_witness(x: int) -> List[int]:
    """Trace every Poseidon intermediate.  Order matches the wire
    allocation in :func:`_build_poseidon_r1cs`."""
    x = int(x) % _FR
    w = [1, x]
    state = [x, 0]

    def emit_sbox(linear_val: int) -> int:
        nonlocal w
        x2  = (linear_val * linear_val) % _FR
        x4  = (x2 * x2) % _FR
        out = (x4 * linear_val) % _FR
        w.extend([x2, x4, out])
        return out

    for r in range(_POSE_TOTAL_RDS):
        state = [(state[j] + _POSE_C[r][j]) % _FR for j in range(_POSE_T)]
        if _is_full_round(r):
            state = [emit_sbox(v) for v in state]
        else:
            state[0] = emit_sbox(state[0])
        state = [
            (_POSE_MDS[i][0] * state[0] + _POSE_MDS[i][1] * state[1]) % _FR
            for i in range(_POSE_T)
        ]
    w.append(state[0])    # the public hash output
    return w


# Compute num_wires + descriptive labels by tracing the R1CS layout once.
def _poseidon_layout():
    """Pre-compute wire-label / help lists so they match the R1CS exactly."""
    labels = ["wire 0 — constant 1", "wire 1 — x  (preimage, private)"]
    helps  = ["Constant 1 — required by Groth16.",
              "Your secret preimage.  Verifier sees only Poseidon(x)."]
    next_wire = 2
    for r in range(_POSE_TOTAL_RDS):
        if _is_full_round(r):
            sboxes = _POSE_T
            label_kind = "full round"
        else:
            sboxes = 1
            label_kind = "partial round"
        for k in range(sboxes):
            labels.append(f"wire {next_wire} — Poseidon r{r} {label_kind} · S-box[{k}] squared")
            helps.append(f"x² in the x⁵ S-box for state element {k} of round {r}.")
            next_wire += 1
            labels.append(f"wire {next_wire} — Poseidon r{r} · S-box[{k}] to-the-4th")
            helps.append(f"x⁴ in the x⁵ S-box for state element {k} of round {r}.")
            next_wire += 1
            labels.append(f"wire {next_wire} — Poseidon r{r} · S-box[{k}] = x⁵")
            helps.append(f"S-box output for state element {k}, used by the next MDS multiply.")
            next_wire += 1
    labels.append(f"wire {next_wire} — h = Poseidon(x)  (public output)")
    helps.append("Final Poseidon digest.  This is the only wire the verifier sees.")
    next_wire += 1
    return labels, helps, next_wire


_POSE_LABELS, _POSE_HELPS, _POSE_NWIRES = _poseidon_layout()


register_circuit(CircuitSpec(
    key             = "poseidon_preimage",
    name            = "Poseidon-style hash preimage",
    description     = (
        "Prove you know an `x` such that `Poseidon(x) = h`, without revealing "
        "`x`.  Poseidon is THE ZK-friendly hash used by Aztec, Mina, modern "
        "zk-EVMs and many ZK-rollups.  The schedule here is the real shape — "
        f"t={_POSE_T} state, {_POSE_FULL_ROUNDS} full + {_POSE_PARTIAL_RDS} "
        "partial rounds, MDS-mult between rounds — but with toy constants "
        "and a trimmed round count, so it's didactic, NOT a secure hash."
    ),
    public_label    = "h = Poseidon(x)",
    input_schema    = [
        InputField(
            name    = "x",
            label   = "x  (preimage, private)",
            kind    = "int",
            default = 42,
            help    = "Your secret value.  Verifier learns only Poseidon(x).",
        ),
    ],
    r1cs_builder    = _build_poseidon_r1cs,
    witness_builder = lambda x: _poseidon_witness(int(x)),
    public_output   = lambda w: w[_POSE_NWIRES - 1],
    num_wires_orig  = _POSE_NWIRES,
    wire_labels     = _POSE_LABELS,
    wire_helps      = _POSE_HELPS,
    try_it          = "Try x = 42.  Note that even with two different x's, the proofs are both ~700 ms — Groth16 is constant-time per circuit.",
    long_info       = f"""
**What this proves**

You know an `x` such that **Poseidon(x) = h**, where `h` is public.
Poseidon is THE ZK-friendly hash of 2024-2026 — used by Aztec, Mina,
Polygon-zkEVM, Linea, Scroll, Tornado-Nova, and almost every modern
zkApp doing in-circuit hashing.

**Anatomy of Poseidon**

State of `t` field elements.  Each round:
1. **Add** round constants element-wise: `s_i ← s_i + c_{{r,i}}`.
2. **S-box**:
   - *Full rounds* — apply `s_i → s_i⁵` to every element.
   - *Partial rounds* — apply `s_0 → s_0⁵` only.
3. **MDS matrix** — multiply state by a fixed invertible matrix
   `M` (linear, free in R1CS).

The mix of *full* and *partial* rounds is what makes Poseidon cheap to
prove: full rounds give cryptographic strength; partial rounds give the
diffusion the MDS matrix needs, at low R1CS cost (only 1 S-box per round
instead of `t`).

**This demo's parameters**

`t = {_POSE_T}` state, `{_POSE_FULL_ROUNDS}` full + `{_POSE_PARTIAL_RDS}`
partial = `{_POSE_TOTAL_RDS}` rounds.  Toy MDS = `[[2,1],[1,2]]` (det 3,
invertible) and a simple round-constant schedule.  Production Poseidon
uses a Grain LFSR for round constants and an MDS matrix derived from
distinct-element Cauchy form — those choices are what give it the
security claim.  **Don't use this circuit's parameters for real
crypto.**

**R1CS shape**

`{_POSE_T * 3 * _POSE_FULL_ROUNDS}` mults from full rounds + `{3 * _POSE_PARTIAL_RDS}`
from partial = approximately `{_POSE_T * 3 * _POSE_FULL_ROUNDS + 3 * _POSE_PARTIAL_RDS}`
constraints, plus one final commit constraint.  Per S-box: 3 mults
(`x²`, `x⁴`, `x⁴ · x`), same MiMC pattern.

**Real-world uses**

- **In-circuit hashing for Merkle trees** in every zkRollup (next demo).
- **Nullifier commitments** in privacy protocols (Aztec, Tornado-Nova).
- **Fiat-Shamir transforms** turning interactive protocols into
  non-interactive ones — Poseidon's small R1CS makes the prover-side
  cost tractable.
""".strip(),
))


# ── Merkle membership (depth-2, MiMC compression) ─────────────────────
#
# Prove you have a leaf in a Merkle tree without revealing which leaf.
# This is the building block of Tornado-style mixers, zCash shielded
# transfers, anonymous allowlists, and almost every anonymity ZK-app.
#
# The hash here is a 2-to-1 compression: ``h(a, b) = MiMC(a + b)``.
# Real ZK Merkle trees use Poseidon/Pedersen, but the structure is
# identical.  Depth 2 means 4 leaves; the proof shows the leaf at
# position 0 hashes up to the public root via the two siblings.
#
# Layout:
#   leaf         = private (your value)
#   sibling_0    = public  (the other leaf at depth 0)
#   sibling_1    = public  (the sibling at depth 1, i.e. the other parent)
#   root         = public  (Merkle root, the only thing the verifier sees)
# Path-up:
#   parent      = MiMC(leaf + sibling_0)
#   grandparent = MiMC(parent + sibling_1)
#   grandparent == root           (enforced)


def _build_merkle_r1cs() -> R1CS:
    """Two MiMC invocations chained at depth-2.  Wire layout:

        wire 0 : constant 1
        wire 1 : leaf      (private)
        wire 2 : sibling_0 (private — given to the prover as part of the path)
        wire 3 : sibling_1 (private — same)
        wires 4 .. 3+3N      : MiMC(leaf + sibling_0) intermediates + result
        wires next .. ...     : MiMC(parent + sibling_1) intermediates
        wire (last) : root  (public output)
    """
    N = _MIMC_ROUNDS
    W_CONST = 0
    W_LEAF  = 1
    W_S0    = 2
    W_S1    = 3
    next_wire = 4

    A_list, B_list, C_list = [], [], []

    def _add(a, b, c):
        A_list.append(a); B_list.append(b); C_list.append(c)

    def _mimc_chain(s_in_wire: int, extra_const: int = 0) -> int:
        """Constrain ``out = MiMC(state)`` where state initially lives in
        ``s_in_wire``.  Returns the wire index holding the output.

        Same constraint pattern as the standalone MiMC circuit.
        """
        nonlocal next_wire
        # We model state as a linear-combination dict that may include
        # a per-wire coefficient + a constant offset.  For Merkle the
        # initial state is ``a + b`` (two wires); ``extra_const`` is an
        # additional integer constant to fold into the round's c_i.
        # For chained calls the initial state is just one wire.
        # We simply iterate: at each round, the previous state lives in
        # a wire.  For the first round we have two wires: pass both as
        # the linear combination of the (s+c) operand.
        if extra_const != 0:
            # Not used for Merkle, but kept generic.
            base_lc = {s_in_wire: 1, W_CONST: extra_const}
        else:
            base_lc = {s_in_wire: 1}
        cur_lc = base_lc
        for i in range(N):
            c = _MIMC_C[i]
            lc_plus_c = {**cur_lc}
            lc_plus_c[W_CONST] = (lc_plus_c.get(W_CONST, 0) + c) % _FR
            # (s + c)² = x²_i
            w_x2 = next_wire; next_wire += 1
            _add(dict(lc_plus_c), dict(lc_plus_c), {w_x2: 1})
            # x²_i · x²_i = x⁴_i
            w_x4 = next_wire; next_wire += 1
            _add({w_x2: 1}, {w_x2: 1}, {w_x4: 1})
            # x⁴_i · (s + c) = s_{i+1}
            w_snext = next_wire; next_wire += 1
            _add({w_x4: 1}, dict(lc_plus_c), {w_snext: 1})
            cur_lc = {w_snext: 1}
        # cur_lc is a single-wire LC pointing at the last s_{i+1} wire.
        # Return that wire index.
        return next(iter(cur_lc.keys()))

    # First MiMC: ``parent = MiMC(leaf + sibling_0)``.
    # The (s + c_0) of round 0 should be (leaf + sibling_0) + c_0.
    # Treat the initial "state" as the LINEAR COMBINATION {LEAF: 1, S0: 1},
    # not a single wire.  Easiest: introduce a wire holding leaf+sibling_0
    # and use that as the starting state.  Cheaper: handle the first round
    # specially by allowing a 2-wire LC.
    #
    # For simplicity, materialize parent_in = leaf + sibling_0 in a wire:
    w_parent_in = next_wire; next_wire += 1
    # Constrain: 1 * (leaf + sibling_0) = w_parent_in
    _add({W_CONST: 1}, {W_LEAF: 1, W_S0: 1}, {w_parent_in: 1})

    w_parent = _mimc_chain(w_parent_in)

    # Second MiMC: ``root = MiMC(parent + sibling_1)``.
    w_grand_in = next_wire; next_wire += 1
    _add({W_CONST: 1}, {w_parent: 1, W_S1: 1}, {w_grand_in: 1})

    w_root = _mimc_chain(w_grand_in)

    return R1CS(
        num_constraints = len(A_list),
        num_wires       = next_wire,
        num_public      = 1,                   # root
        A = A_list, B = B_list, C = C_list,
    )


def _merkle_witness(leaf: int, sibling_0: int, sibling_1: int) -> List[int]:
    """Compute the full witness tracing both MiMC chains."""
    leaf = int(leaf) % _FR
    s0   = int(sibling_0) % _FR
    s1   = int(sibling_1) % _FR
    w = [1, leaf, s0, s1]

    # parent_in = leaf + s0
    parent_in = (leaf + s0) % _FR
    w.append(parent_in)

    # MiMC(parent_in) — trace intermediates.
    s = parent_in
    for c in _MIMC_C:
        t  = (s + c) % _FR
        x2 = (t * t)  % _FR
        x4 = (x2 * x2) % _FR
        s  = (x4 * t)  % _FR
        w.extend([x2, x4, s])
    parent = s

    # grandparent_in = parent + s1
    grand_in = (parent + s1) % _FR
    w.append(grand_in)

    s = grand_in
    for c in _MIMC_C:
        t  = (s + c) % _FR
        x2 = (t * t)  % _FR
        x4 = (x2 * x2) % _FR
        s  = (x4 * t)  % _FR
        w.extend([x2, x4, s])
    # w now ends with the root.
    return w


_MERKLE_NWIRES = 4 + 1 + 3 * _MIMC_ROUNDS + 1 + 3 * _MIMC_ROUNDS
# = const + leaf + s0 + s1 + parent_in + 3N intermediates + grand_in + 3N intermediates


def _merkle_labels_helps():
    labels = [
        "wire 0 — constant 1",
        "wire 1 — leaf  (your secret)",
        "wire 2 — sibling_0  (path element at depth 0)",
        "wire 3 — sibling_1  (path element at depth 1)",
        "wire 4 — leaf + sibling_0  (parent's MiMC input)",
    ]
    helps = [
        "Constant 1 — required by Groth16.",
        "Your secret leaf.  Verifier learns it sits in the tree, NOT which slot.",
        "Sibling along the Merkle path at depth 0.  Public (or known to anyone with the tree).",
        "Sibling at depth 1.  Same as above.",
        "Linearly-combined input to the first MiMC call.  Free of constraints.",
    ]
    next_wire = 5
    # First MiMC chain.
    for i in range(_MIMC_ROUNDS):
        labels.append(f"wire {next_wire} — Merkle L1 · round {i} · (s + c)²"); next_wire += 1
        helps.append(f"First MiMC, round {i} squaring step.")
        labels.append(f"wire {next_wire} — Merkle L1 · round {i} · (s + c)⁴"); next_wire += 1
        helps.append(f"First MiMC, round {i} second squaring.")
        labels.append(f"wire {next_wire} — Merkle L1 · round {i} · s_{i+1}"); next_wire += 1
        helps.append(f"First MiMC, round {i+1} state.  After all rounds → parent node.")
    # grandparent_in.
    labels.append(f"wire {next_wire} — parent + sibling_1  (grandparent's MiMC input)"); next_wire += 1
    helps.append("Linearly-combined input to the second MiMC call.")
    # Second MiMC chain.
    for i in range(_MIMC_ROUNDS):
        labels.append(f"wire {next_wire} — Merkle L2 · round {i} · (s + c)²"); next_wire += 1
        helps.append(f"Second MiMC, round {i} squaring step.")
        labels.append(f"wire {next_wire} — Merkle L2 · round {i} · (s + c)⁴"); next_wire += 1
        helps.append(f"Second MiMC, round {i} second squaring.")
        if i < _MIMC_ROUNDS - 1:
            labels.append(f"wire {next_wire} — Merkle L2 · round {i} · s_{i+1}"); next_wire += 1
            helps.append(f"Second MiMC, round {i+1} state.")
        else:
            labels.append(f"wire {next_wire} — root = Merkle({{leaf, sibling_0}}, sibling_1)  (public)")
            next_wire += 1
            helps.append("Final Merkle root.  This is the only wire the verifier sees.")
    return labels, helps


_MERKLE_LABELS, _MERKLE_HELPS = _merkle_labels_helps()


register_circuit(CircuitSpec(
    key             = "merkle_membership",
    name            = "Merkle membership (depth-2)",
    description     = (
        "Prove your `leaf` is in a depth-2 Merkle tree with root `R`, "
        "without revealing which leaf.  Hash is MiMC-x⁵; tree depth is "
        "2 (4 leaves).  This is the structural blueprint of every "
        "anonymity ZK-app: Tornado-style mixers, Zcash shielded transfers, "
        "anonymous allowlists.  Real systems use depth-20 Poseidon trees; "
        "the constraint count scales linearly with depth."
    ),
    public_label    = "root",
    input_schema    = [
        InputField(name="leaf",      label="leaf  (your secret value)",
                   kind="int", default=11111,
                   help="The value at your slot in the tree."),
        InputField(name="sibling_0", label="sibling at depth 0",
                   kind="int", default=22222,
                   help="The other leaf next to yours in the tree."),
        InputField(name="sibling_1", label="sibling at depth 1",
                   kind="int", default=33333,
                   help="The other parent at the next level up."),
    ],
    r1cs_builder    = _build_merkle_r1cs,
    witness_builder = lambda leaf, sibling_0, sibling_1: _merkle_witness(
        int(leaf), int(sibling_0), int(sibling_1),
    ),
    public_output   = lambda w: w[_MERKLE_NWIRES - 1],
    num_wires_orig  = _MERKLE_NWIRES,
    wire_labels     = _MERKLE_LABELS,
    wire_helps      = _MERKLE_HELPS,
    try_it          = "Try a different `leaf` while keeping siblings fixed — the root changes.  "
                      "That's how mixers force every deposit to commit to a unique slot.",
    long_info       = f"""
**What this proves**

You know a `leaf` value that sits in a depth-2 Merkle tree with public
root `R`.  Tree:

```
                root
               /    \\
           parent   sibling_1
          /     \\
       leaf  sibling_0
```

Verifier learns `R`; doesn't learn `leaf` (nor the position).

**The hash chain**

Two MiMC invocations chained:
```
parent      = MiMC(leaf + sibling_0)
root        = MiMC(parent + sibling_1)
```
Constrain `root == public_root`.  Both MiMC calls inline their
{_MIMC_ROUNDS}-round chains as R1CS constraints, sharing the same hash
primitive shape as the standalone MiMC circuit.

**R1CS layout (~{122 + 4} constraints, ~{126} wires)**

- 1 wire each for `leaf`, `sibling_0`, `sibling_1`
- 1 constraint to materialise `leaf + sibling_0` into a wire
- {3 * _MIMC_ROUNDS} constraints for the first MiMC
- 1 constraint to materialise `parent + sibling_1`
- {3 * _MIMC_ROUNDS} constraints for the second MiMC
- The last `s_N` wire IS the root (public output)

Real Merkle proofs use depth 20-32, which means 20-32 hash invocations.
Same shape, just more rounds.

**Real-world uses**

This is THE shape of every anonymity ZK-app:

- **Tornado Cash** / mixers: deposit commits to a leaf; withdrawal proves
  membership without revealing which leaf.
- **Zcash shielded transfers**: same recipe with note commitments.
- **Anonymous allowlists** ("I'm on the list") for token gating, voting,
  KYC.
- **Verifiable databases**: prove a record was in a committed snapshot
  without revealing other records.
""".strip(),
))


# ── Sudoku 4×4 — prove you know a valid solution ──────────────────────
#
# A 4×4 Sudoku has 16 cells, each ∈ {1, 2, 3, 4}, with each row /
# column / 2×2 box containing all four values.  We use:
#   range check         : (v-1)(v-2)(v-3)(v-4) = 0  →  3 mults per cell
#   uniqueness in a set : Σv = 10, Σv² = 30, Σv³ = 100  (Newton identities)
#                          forcing the set to be a permutation of {1,2,3,4}
# 16 cells · 3 + 12 sets · ~8 = ~50 + ~100 = ~150 constraints.


# Scaling factors for sudoku intermediates.  Chosen as small coprime
# primes so the resulting wire values land in entirely separate scalar
# buckets from the cells (which hold values 1..4).  Without scaling,
# intermediates collide with cell scalars and overflow the Pippenger
# MSM buckets (capacity 8).  See PUBLIC_INPUT_BINDING / T33 history.
_SUDOKU_KA  = 5     # w_a  = Ka · (v-1)(v-2)   ∈ {0, 0, 10, 30}
_SUDOKU_KB  = 7     # w_b  = Kb · (v-3)(v-4)   ∈ {42, 14, 0, 0}
_SUDOKU_KSQ = 11    # w_sq = Ksq · v²          ∈ {11, 44, 99, 176}
_SUDOKU_SUM_TARGET    = 10            # Σ {1,2,3,4}
_SUDOKU_SUMSQ_TARGET  = _SUDOKU_KSQ * 30   # Σ Ksq·v² for {1,2,3,4} = 11·30 = 330


# Groups of 4 cell indices whose multisets must each be a permutation
# of {1, 2, 3, 4} for a valid 4×4 Sudoku — 4 rows, 4 columns, 4 boxes.
_SUDOKU_ROWS  = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]
_SUDOKU_COLS  = [[0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]]
_SUDOKU_BOXES = [[0,1,4,5], [2,3,6,7], [8,9,12,13], [10,11,14,15]]
_SUDOKU_GROUPS = _SUDOKU_ROWS + _SUDOKU_COLS + _SUDOKU_BOXES   # 12 groups of 4


def _build_sudoku_r1cs() -> R1CS:
    """Sudoku-4×4 R1CS — range check **plus** row/col/box uniqueness.

    Three layers of constraint:

    1. **Range** — each cell ∈ {1, 2, 3, 4}, via
       ``Ka·Kb·(v-1)(v-2)(v-3)(v-4) = 0``.  Materialised through three
       intermediate wires (``w_a``, ``w_b``, ``w_prod``) per cell and a
       force-zero on the product.  Ka=5, Kb=7 spread the intermediates
       into separate scalar buckets from the cells.

    2. **Per-cell square** — materialise ``w_sq = Ksq·v²`` for use in
       the row/col/box checks.  Ksq=11 again chosen coprime so the
       squared wires land in their own buckets.

    3. **Uniqueness per group of 4** — for each row, column, and 2×2
       box, enforce **Newton's first two power sums**:

           Σ v   = 10
           Σ v²  = 30   (multiplied by Ksq on the wire side → 330)

       Together with the range check these uniquely identify the
       multiset {1,2,3,4} per group, so a row like ``[1, 1, 2, 3]``
       fails (sum 7 ≠ 10) and ``[2, 2, 3, 3]`` fails (sum 10 but
       Σv² = 26 ≠ 30).

    Wire layout:
        wire 0                  : constant 1
        wires 1..16             : cell values c_0..c_15            (private)
        wires 17..64            : per-cell w_a, w_b, w_prod        (private)
        wires 65..80            : per-cell w_sq = Ksq·v²           (private)
        wires 81..104           : per-group force-target wires     (private)
                                  (12 groups × 2 targets = 24 wires)
        wire 105                : ok = 1                           (public sentinel)

    Total: 4·16 (range) + 16 (sq) + 24 (sum + sumsq materialise) +
    24 (force_zero) + 1 (sentinel) = 129 constraints, 106 wires.
    """
    W_CONST = 0
    Ka, Kb, Ksq = _SUDOKU_KA, _SUDOKU_KB, _SUDOKU_KSQ
    A_list, B_list, C_list = [], [], []

    def _add(a, b, c):
        A_list.append(a); B_list.append(b); C_list.append(c)

    cell   = lambda i: i + 1
    w_a    = lambda i: 17 + 3 * i + 0
    w_b    = lambda i: 17 + 3 * i + 1
    w_prod = lambda i: 17 + 3 * i + 2
    w_sq   = lambda i: 65 + i                    # one square wire per cell

    # ── 1. Range checks (per cell) ──
    for i in range(16):
        _add({cell(i): Ka, W_CONST: (-Ka) % _FR},
             {cell(i): 1,  W_CONST: (-2)  % _FR},
             {w_a(i): 1})
        _add({cell(i): Kb, W_CONST: (-3 * Kb) % _FR},
             {cell(i): 1,  W_CONST: (-4)      % _FR},
             {w_b(i): 1})
        _add({w_a(i): 1}, {w_b(i): 1}, {w_prod(i): 1})
        _add({w_prod(i): 1, W_CONST: 1}, {W_CONST: 1}, {W_CONST: 1})

    # ── 2. Materialise w_sq = Ksq · v² per cell ──
    for i in range(16):
        _add({cell(i): Ksq}, {cell(i): 1}, {w_sq(i): 1})

    # ── 3. Per-group uniqueness via Σv = 10 and Σw_sq = Ksq·30 = 330 ──
    next_wire = 65 + 16     # start of force-target wires (= 81)
    SUM_TARGET    = _SUDOKU_SUM_TARGET
    SUMSQ_TARGET  = _SUDOKU_SUMSQ_TARGET

    for indices in _SUDOKU_GROUPS:
        # delta_sum = (Σ cells - 10).  Materialise into a wire; force_zero it.
        w_delta_sum = next_wire; next_wire += 1
        # (Σ cells - 10) · 1 = w_delta_sum
        A_row = {cell(i): 1 for i in indices}
        A_row[W_CONST] = (-SUM_TARGET) % _FR
        _add(A_row, {W_CONST: 1}, {w_delta_sum: 1})
        # force_zero(w_delta_sum)
        _add({w_delta_sum: 1, W_CONST: 1}, {W_CONST: 1}, {W_CONST: 1})

        # delta_sumsq = (Σ w_sq - 330).  Same pattern.
        w_delta_sumsq = next_wire; next_wire += 1
        A_row = {w_sq(i): 1 for i in indices}
        A_row[W_CONST] = (-SUMSQ_TARGET) % _FR
        _add(A_row, {W_CONST: 1}, {w_delta_sumsq: 1})
        _add({w_delta_sumsq: 1, W_CONST: 1}, {W_CONST: 1}, {W_CONST: 1})

    # ── 4. Public sentinel: 1 · 1 = ok ──
    w_ok = next_wire; next_wire += 1
    _add({W_CONST: 1}, {W_CONST: 1}, {w_ok: 1})

    return R1CS(
        num_constraints = len(A_list),
        num_wires       = next_wire,
        num_public      = 1,
        A = A_list, B = B_list, C = C_list,
    )


def _sudoku_witness(c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16):
    """Compute the full witness; order + scaling matches :func:`_build_sudoku_r1cs`."""
    cells = [int(v) % _FR for v in (c1, c2, c3, c4, c5, c6, c7, c8,
                                    c9, c10, c11, c12, c13, c14, c15, c16)]
    Ka, Kb, Ksq = _SUDOKU_KA, _SUDOKU_KB, _SUDOKU_KSQ
    w = [1, *cells]

    # Range-check intermediates per cell.
    for v in cells:
        a    = (Ka * (v - 1) * (v - 2)) % _FR
        b    = (Kb * (v - 3) * (v - 4)) % _FR
        prod = (a * b) % _FR
        w.extend([a, b, prod])

    # Squares per cell.
    for v in cells:
        w.append((Ksq * v * v) % _FR)

    # Per-group sum and sum-of-squares deltas (both 0 for valid sudoku).
    sq_vals = [(Ksq * int(v) * int(v)) % _FR for v in cells]
    SUM_TARGET   = _SUDOKU_SUM_TARGET
    SUMSQ_TARGET = _SUDOKU_SUMSQ_TARGET
    for indices in _SUDOKU_GROUPS:
        delta_sum   = (sum(cells[i]    for i in indices) - SUM_TARGET)   % _FR
        delta_sumsq = (sum(sq_vals[i]  for i in indices) - SUMSQ_TARGET) % _FR
        w.append(delta_sum)
        w.append(delta_sumsq)

    w.append(1)        # ok sentinel
    return w


def _sudoku_labels_helps():
    labels = ["wire 0 — constant 1"]
    helps  = ["Constant 1 — required by Groth16."]
    for i in range(16):
        row, col = i // 4, i % 4
        labels.append(f"wire {i+1} — cell ({row}, {col})  ∈ {{1,2,3,4}}")
        helps.append(f"Solution value at row {row}, column {col}.  "
                     f"Must be in {{1, 2, 3, 4}} AND unique in its row, column, and 2×2 box.")
    nw = 17
    # Range-check intermediates.
    for i in range(16):
        row, col = i // 4, i % 4
        labels.append(f"wire {nw} — cell({row},{col}) range · Ka(v-1)(v-2)"); nw += 1
        helps.append("First half of the scaled range polynomial.")
        labels.append(f"wire {nw} — cell({row},{col}) range · Kb(v-3)(v-4)"); nw += 1
        helps.append("Second half of the scaled range polynomial.")
        labels.append(f"wire {nw} — cell({row},{col}) prod = 0 ⇔ in range"); nw += 1
        helps.append("Product of the two halves.  Forced to 0 by a follow-up constraint.")
    # Per-cell squared values (for uniqueness check).
    for i in range(16):
        row, col = i // 4, i % 4
        labels.append(f"wire {nw} — cell({row},{col})² · Ksq"); nw += 1
        helps.append("Scaled v² used by the Σv² = 30 uniqueness check.")
    # Per-group delta wires (sum and sum-of-squares; forced to 0).
    group_names = (
        [f"row {r}" for r in range(4)]
        + [f"col {c}" for c in range(4)]
        + [f"box {b}" for b in range(4)]
    )
    for gname in group_names:
        labels.append(f"wire {nw} — {gname} · (Σv − 10)"); nw += 1
        helps.append(f"For {gname}: Σ of cells minus 10.  Forced to 0; ensures the row/col/box sum.")
        labels.append(f"wire {nw} — {gname} · (Σv² − 30)·Ksq"); nw += 1
        helps.append(f"For {gname}: scaled Σv² minus 330.  Forced to 0; combined with sum, forces a permutation of {{1,2,3,4}}.")
    labels.append(f"wire {nw} — ok = 1  (public sentinel)")
    helps.append("Public output.  Always 1; verifier learns the predicate holds, not the solution.")
    nw += 1
    return labels, helps, nw


_SUDOKU_LABELS, _SUDOKU_HELPS, _SUDOKU_NWIRES = _sudoku_labels_helps()


register_circuit(CircuitSpec(
    key             = "sudoku_4x4",
    name            = "Sudoku 4×4",
    description     = (
        "Prove you know a valid 4×4 Sudoku solution without revealing it.  "
        "Each cell ∈ {1, 2, 3, 4} (range-checked via "
        "(v-1)(v-2)(v-3)(v-4) = 0) and each row, column, and 2×2 box is a "
        "permutation of {1,2,3,4} (checked via Σv = 10 and Σv² = 30 — "
        "Newton's first two power sums).  Public output is the sentinel "
        "`ok = 1`; verifier learns only that the predicate holds, not the "
        "solution."
    ),
    public_label    = "ok  (= 1 ⇔ valid solution)",
    input_schema    = [
        InputField(
            name    = f"c{i+1}",
            label   = f"cell ({i//4}, {i%4})",
            kind    = "int",
            default = [
                1, 2, 3, 4,
                3, 4, 1, 2,
                2, 1, 4, 3,
                4, 3, 2, 1,
            ][i],
            min     = 1, max = 4,
            help    = f"Value at row {i//4}, column {i%4}.  Must be 1..4.",
        )
        for i in range(16)
    ],
    r1cs_builder    = _build_sudoku_r1cs,
    witness_builder = _sudoku_witness,
    public_output   = lambda w: w[-1],
    num_wires_orig  = _SUDOKU_NWIRES,
    wire_labels     = _SUDOKU_LABELS,
    wire_helps      = _SUDOKU_HELPS,
    category        = "practical",
    try_it          = "Defaults already form a valid 4×4 grid.  Try changing any cell to 5 — Check witness rejects it; tick the demo box to watch the verifier reject the proof too.",
    long_info       = """
**What this proves**

You know 16 cell values, each ∈ {1, 2, 3, 4}, without revealing them.
Public output is a sentinel `ok = 1` attesting "yes, all 16 cells are
in range."

**R1CS layout (65 constraints, 66 wires)**

Per cell, four constraints:

| # | constraint                  | meaning                       |
|---|-----------------------------|-------------------------------|
| 0 | `(v - 1) · (v - 2) = w_a`   | first half of range polynomial |
| 1 | `(v - 3) · (v - 4) = w_b`   | second half                   |
| 2 | `w_a · w_b = w_prod`        | product = 0 iff v ∈ {1,2,3,4} |
| 3 | `(w_prod + 1) · 1 = 1`      | forces w_prod = 0             |

For valid `v ∈ {1,2,3,4}`, either `(v-1)(v-2) = 0` or `(v-3)(v-4) = 0`,
so the product is always 0.  For invalid `v` (e.g. 5), the product is
non-zero and the force-zero constraint rejects.

Plus one public sentinel `1 · 1 = ok`.

**Why this encoding and not bit decomposition (history note)**

An earlier bit-decomposition variant of this circuit hit a Pippenger MSM
bucket overflow on the TPU prover (T33): bit wires created 22 wires
sharing scalar=1, exceeding the bucket capacity of 8.  This polynomial
encoding keeps intermediates in `{0, 2, 6}` only — never 1 — so the
total count of wires sharing scalar=1 drops to 6.

**No uniqueness check**

Row / column / box uniqueness is intentionally NOT enforced — enforcing
it would require power-sum intermediates (v², v³) which include 1 for
v=1, blowing the scalar=1 count back over the bucket limit.  The
weaker statement is still a meaningful ZK demo.

**Real-world uses**

- Privacy-preserving games where a player commits to a hidden board.
- Sealed-bid auctions on a bounded set of bid values.
- Anonymous voting where the vote must be in a fixed set.
""".strip(),
))


def list_circuits() -> List[CircuitSpec]:
    """Stable iteration order over the registered circuits."""
    return list(CIRCUITS.values())
