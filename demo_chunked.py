"""Memory-bounded generation via ``chunk_size``.

The per-point Vecchia factorizations in ``refine`` depend only on point
geometry, not on the generated values, so they can be computed in chunks
without changing the result. Passing ``chunk_size`` to ``generate`` caps peak
memory at ``O(chunk_size * (k+1)**2 * d)`` instead of ``O(N * (k+1)**2 * d)``,
which lets a single GPU generate point sets far larger than would otherwise
fit. Output is identical (byte-for-byte) to ``chunk_size=None``.

Run::

    python demo_chunked.py
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial

import graphgp as gp


def main():
    print("device:", jax.devices()[0])

    # 1. Equivalence: chunked == full, exactly.
    pts = jr.normal(jr.key(137), (2000, 3))
    graph = gp.build_graph(pts, n0=100, k=10)
    cov = gp.extras.matern_kernel(
        p=0, variance=1.0, cutoff=1.0, r_min=1e-4, r_max=10, n_bins=1000, jitter=1e-5
    )
    xi = jr.normal(jr.key(1), (graph.points.shape[0],))
    v_full = jax.jit(gp.generate)(graph, cov, xi)
    for cs in (1, 64, 1000):
        v = jax.jit(Partial(gp.generate, chunk_size=cs))(graph, cov, xi)
        d = float(jnp.max(jnp.abs(v - v_full)))
        print(f"  chunk_size={cs:5d}: max|Δ vs full| = {d:.2e}")

    # 2. Scale: a point set whose all-at-once factorization would not fit.
    N, k = 2_000_000, 30
    pts = jr.normal(jr.key(0), (N, 3))
    t0 = time.time()
    graph = gp.build_graph(pts, n0=256, k=k)
    print(f"\n  graph({N:,}, k={k}) built in {time.time()-t0:.1f}s")
    xi = jr.normal(jr.key(1), (N,))
    try:
        v = jax.jit(gp.generate)(graph, cov, xi)
        v.block_until_ready()
        print("  full path: OK")
    except Exception as e:
        print("  full path: OUT OF MEMORY ->", str(e).split("\n")[0][:80])
    t0 = time.time()
    v = jax.jit(Partial(gp.generate, chunk_size=50_000))(graph, cov, xi)
    v.block_until_ready()
    print(f"  chunked(50k): OK in {time.time()-t0:.1f}s  (N={N:,}, no NaN={not bool(jnp.any(jnp.isnan(v)))})")


if __name__ == "__main__":
    main()
