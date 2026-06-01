"""Pad R1CS / witness up to a fixed target shape (default 1024) so the
TPU prover sees the same constraint count and wire count on every call.
The prover then JIT-compiles its kernels once and reuses them across
arbitrary circuits that fit under the target.

Convention:

* `target_num_constraints` and `target_num_wires` must be powers of two
  (matching the existing R1CS / QAP conventions).
* Padding constraints are the trivial ``1 · 1 = 1`` form (sourced from
  wire 0, the constant-one wire).  They're always satisfied so the
  padded R1CS accepts the same witnesses as the original.
* Witness padding is zero-fill — extra wires don't appear in any
  original constraint, and they appear in the padding constraints only
  via wire 0, so their value doesn't affect satisfaction.

Padding the R1CS *here* lets the prover assume a fixed shape without
hardcoding constants in its body.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import path_setup  # noqa: F401

from typing import List

from groth16.r1cs import R1CS


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def pad_r1cs(
    r1cs: R1CS,
    target_num_constraints: int = 1024,
    target_num_wires: int = 1024,
) -> R1CS:
    """Return a new R1CS padded to the target shape.

    Padding strategy:

    * **Extra constraints** (``target_num_constraints - r1cs.num_constraints``):
      filled with the trivially-satisfied ``1 · 1 = 1`` constraint
      (``A = B = C = {0: 1}``).  Wire 0 is the constant-one wire in
      every Groth16 R1CS, so these constraints hold for any witness.
    * **Extra wires** (``target_num_wires - r1cs.num_wires``): no
      constraint references them.  They appear in the witness as zeros
      via :func:`pad_witness`.

    Both target sizes must be powers of two, since the QAP's NTT step
    requires it.  Raises ``ValueError`` if the input already exceeds
    a target.
    """
    if (target_num_constraints & (target_num_constraints - 1)) != 0:
        raise ValueError(
            f"target_num_constraints must be a power of 2; got {target_num_constraints}"
        )
    if (target_num_wires & (target_num_wires - 1)) != 0:
        raise ValueError(
            f"target_num_wires must be a power of 2; got {target_num_wires}"
        )
    if r1cs.num_constraints > target_num_constraints:
        raise ValueError(
            f"r1cs has {r1cs.num_constraints} constraints; target is "
            f"{target_num_constraints}"
        )
    if r1cs.num_wires > target_num_wires:
        raise ValueError(
            f"r1cs has {r1cs.num_wires} wires; target is {target_num_wires}"
        )

    pad_constraints = target_num_constraints - r1cs.num_constraints
    A = list(r1cs.A) + [{0: 1}] * pad_constraints
    B = list(r1cs.B) + [{0: 1}] * pad_constraints
    C = list(r1cs.C) + [{0: 1}] * pad_constraints

    return R1CS(
        num_constraints = target_num_constraints,
        num_wires       = target_num_wires,
        num_public      = r1cs.num_public,
        A               = A,
        B               = B,
        C               = C,
    )


def pad_witness(
    witness: List[int],
    target_num_wires: int = 1024,
) -> List[int]:
    """Zero-pad a witness vector up to ``target_num_wires``.

    The added entries are 0 — they don't appear in any original
    constraint (those reference only the original wires) and they only
    appear in padding constraints via wire 0 (constant one), so they
    don't affect satisfaction.
    """
    if len(witness) > target_num_wires:
        raise ValueError(
            f"witness has {len(witness)} entries; target is {target_num_wires}"
        )
    return list(witness) + [0] * (target_num_wires - len(witness))


# ══════════════════════════════════════════════════════════════════════
# v2 — public-wire-permuting padding (T32 soundness fix)
# ══════════════════════════════════════════════════════════════════════
#
# The default :func:`pad_r1cs` / :func:`pad_witness` leave the original
# public wire at its original position (e.g. wire 4 for cubic).  After
# padding to 1024 wires, the QAP / trusted_setup treats wire 1023 (the
# LAST padded wire) as "the public wire" — but that wire is purely
# padding and doesn't appear in any constraint, so its U/V/W
# polynomials are all zero, and the verifier's IC[1] becomes the curve
# identity.  Result: the public-input value passed to the verifier
# isn't actually bound to the proof.  See PUBLIC_INPUT_BINDING.md.
#
# v2 fixes this by **permuting** the original public wires to the END
# of the padded wire space (positions ``[target_num_wires - num_public
# .. target_num_wires - 1]``) before applying padding constraints.
# Constraints are rewritten through the permutation; the witness is
# permuted to match.
#
# Both versions produce identical *witness satisfaction* semantics —
# the same constraints, same witness values, just placed at different
# wire indices.  The Groth16 binding fix is purely about which wire's
# polynomial gets folded into IC[1].


def _build_v2_permutation(num_wires_orig: int, num_public_orig: int,
                            target_num_wires: int) -> List[int]:
    """Return σ such that ``σ[old_wire_idx] = new_wire_idx``.

    Layout produced by σ:

        old wire 0                              → new wire 0   (constant)
        old wires [1 .. num_priv_orig]          → new wires [1 .. num_priv_orig]
        old public wires [num_priv_orig+1 ..    → new wires [target - num_public
                          num_wires_orig - 1]                  .. target - 1]

    where ``num_priv_orig = num_wires_orig - num_public_orig - 1``.
    Original wire 0 is treated as part of the "constant" slot, not as
    a "private input."
    """
    num_priv_orig = num_wires_orig - num_public_orig - 1
    sigma = [0] * num_wires_orig
    sigma[0] = 0                                            # const stays at 0
    for i in range(1, num_priv_orig + 1):
        sigma[i] = i                                         # privates stay in place
    for k in range(num_public_orig):
        old_idx = num_priv_orig + 1 + k
        new_idx = target_num_wires - num_public_orig + k
        sigma[old_idx] = new_idx
    return sigma


def pad_r1cs_v2(
    r1cs: R1CS,
    target_num_constraints: int = 1024,
    target_num_wires: int = 1024,
) -> R1CS:
    """Permuting padding — public wires move to the END of the padded
    space (positions ``target_num_wires - num_public .. target_num_wires - 1``).

    Fixes the Groth16 public-input binding issue documented in
    ``PUBLIC_INPUT_BINDING.md``.  Witnesses for this layout must be
    produced by :func:`pad_witness_v2`.
    """
    if (target_num_constraints & (target_num_constraints - 1)) != 0:
        raise ValueError(
            f"target_num_constraints must be a power of 2; got {target_num_constraints}"
        )
    if (target_num_wires & (target_num_wires - 1)) != 0:
        raise ValueError(
            f"target_num_wires must be a power of 2; got {target_num_wires}"
        )
    if r1cs.num_constraints > target_num_constraints:
        raise ValueError(
            f"r1cs has {r1cs.num_constraints} constraints; target is "
            f"{target_num_constraints}"
        )
    if r1cs.num_wires > target_num_wires:
        raise ValueError(
            f"r1cs has {r1cs.num_wires} wires; target is {target_num_wires}"
        )

    sigma = _build_v2_permutation(
        r1cs.num_wires, r1cs.num_public, target_num_wires,
    )

    def _remap(row):
        return {sigma[wire]: coeff for wire, coeff in row.items()}

    A_new = [_remap(row) for row in r1cs.A]
    B_new = [_remap(row) for row in r1cs.B]
    C_new = [_remap(row) for row in r1cs.C]

    # Append padding constraints (1·1=1) — they only reference wire 0
    # (the constant), so the permutation doesn't affect them.
    pad_constraints = target_num_constraints - r1cs.num_constraints
    A_new += [{0: 1}] * pad_constraints
    B_new += [{0: 1}] * pad_constraints
    C_new += [{0: 1}] * pad_constraints

    return R1CS(
        num_constraints = target_num_constraints,
        num_wires       = target_num_wires,
        num_public      = r1cs.num_public,
        A               = A_new,
        B               = B_new,
        C               = C_new,
    )


def pad_witness_v2(
    witness: List[int],
    num_wires_orig: int,
    num_public_orig: int,
    target_num_wires: int = 1024,
) -> List[int]:
    """Permuting witness pad — matches :func:`pad_r1cs_v2`.

    The original public-wire values are placed at the END of the
    padded witness; original private wires stay at their original
    indices; the gap between them is zero-filled.
    """
    if len(witness) != num_wires_orig:
        raise ValueError(
            f"witness has {len(witness)} entries; expected {num_wires_orig}"
        )
    if num_wires_orig > target_num_wires:
        raise ValueError(
            f"num_wires_orig {num_wires_orig} > target {target_num_wires}"
        )

    sigma = _build_v2_permutation(num_wires_orig, num_public_orig, target_num_wires)
    padded = [0] * target_num_wires
    for old_idx, value in enumerate(witness):
        padded[sigma[old_idx]] = int(value)
    return padded
