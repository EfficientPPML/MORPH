"""
Import path plumbing for the TPU-backed Groth16.

Layout assumptions
------------------
    <repo root>/                  ← MAPLE root (TPU contexts live here)
    <repo root>/Groth16/          ← upstream pure-Python Groth16
    <repo root>/Groth16_TPU/      ← this package (TPU adapters)

What this module does
---------------------
1. Adds MAPLE root to ``sys.path`` so we can ``import number_theory_transform_context``,
   ``import multiscalar_multiplication_context``, ``import utils``, etc.
2. Adds ``Groth16_TPU/`` to ``sys.path`` AHEAD of ``Groth16/`` so that
   ``from kernels.ntt import ...``  resolves to *our* TPU adapter, not
   the CPU one in ``Groth16/kernels``.
3. Adds ``Groth16/`` to ``sys.path`` after that, so that ``from bls12_377...``,
   ``from groth16...`` resolve to the upstream pure-Python protocol code.
4. Loads the upstream CPU kernels into the ``cpu_kernels`` namespace so the
   TPU adapter can fall back / delegate without colliding with our own
   ``kernels`` package.

Usage
-----
    from Groth16_TPU import path_setup  # noqa: F401  (side-effecting)
    # — or, when running a script directly —
    import path_setup  # noqa: F401

Importing this module is idempotent and side-effect-only.
"""

import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
MAPLE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
GROTH16_DIR = os.path.join(MAPLE_ROOT, "Groth16")
GROTH16_TPU_DIR = _HERE


def _prepend_unique(path: str) -> None:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


# Order matters: the entry inserted *last* ends up at sys.path[0], so we
# insert MAPLE root first, then Groth16, then Groth16_TPU.  The final
# resolution order is: Groth16_TPU/  →  Groth16/  →  MAPLE_ROOT/.
_prepend_unique(MAPLE_ROOT)
_prepend_unique(GROTH16_DIR)
_prepend_unique(GROTH16_TPU_DIR)


def _load_module_from_file(name: str, path: str) -> types.ModuleType:
    """Import a Python file under a custom module name without colliding
    with anything else on ``sys.path``."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Expose the upstream CPU kernels under a non-conflicting top-level name.
# Our own ``kernels`` package lives in Groth16_TPU/kernels and is what the
# upstream protocol code (groth16/{qap,setup,prover}.py) will import via
# ``from kernels.ntt import ...``.  We use ``cpu_kernels`` for the original.
def _load_cpu_kernels() -> types.ModuleType:
    if "cpu_kernels" in sys.modules:
        return sys.modules["cpu_kernels"]
    pkg = types.ModuleType("cpu_kernels")
    pkg.__path__ = [os.path.join(GROTH16_DIR, "kernels")]
    sys.modules["cpu_kernels"] = pkg

    cpu_ntt = _load_module_from_file(
        "cpu_kernels.ntt", os.path.join(GROTH16_DIR, "kernels", "ntt.py")
    )
    cpu_msm = _load_module_from_file(
        "cpu_kernels.msm", os.path.join(GROTH16_DIR, "kernels", "msm.py")
    )
    pkg.ntt = cpu_ntt
    pkg.msm = cpu_msm
    return pkg


cpu_kernels = _load_cpu_kernels()


# ── JAX persistent compilation cache ──────────────────────────────────
#
# The G2 fused MSM kernel takes ~10 min to JIT-compile on first call.
# Without a persistent cache, every fresh daemon process pays that cost
# again.  JAX has an on-disk compilation cache we can point at a
# per-user directory — restarts become ~30 s instead of ~10 min.
#
# We do this *here* (in path_setup) so it's the first thing every
# entry-point — daemon, prep_circuit, dev_loop, tests — picks up.
def _enable_jax_compilation_cache() -> None:
    cache_dir = os.environ.get(
        "MAPLE_JAX_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "maple-jax-compile"),
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)
        from jax.experimental.compilation_cache import compilation_cache  # noqa
        compilation_cache.set_cache_dir(cache_dir)
    except Exception:
        # JAX too old, no TPU, or any other failure — silently skip;
        # the prover still works, just pays the full cold-compile each boot.
        pass


_enable_jax_compilation_cache()
