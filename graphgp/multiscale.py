"""Multi-resolution (scale-ladder) graph construction for GraphGP.

The default :func:`graphgp.build_graph` conditions each point on its ``k``
nearest *preceding* neighbours. On a dense point set those nearest neighbours
all sit at the *finest* scale (within a few times the local spacing), so the
Vecchia approximation screens out correlation on scales much larger than the
neighbour spacing. For a single Matern (exponential tail) this is harmless --
there is no long-range power to lose -- but for a **broad / power-law kernel**
(e.g. galaxy angular clustering, which has real power at tens of times the
point spacing) the realized field collapses at large separation: its
correlation function falls far below the input kernel and can even go negative.

The fix, following the "process randomly down-sampled versions of the data
serially, oversampling across scales" idea, is to give every point a
**multi-scale conditioning set**:

1. Randomly permute the points and split them into a geometric ladder of
   nested levels (level 0 coarsest, generated densely). A random subset of a
   uniform cloud is itself uniform, so level ``j`` has spacing ~ ``r ** j``.
2. Each finer point conditions on a *fixed quota* of nearest neighbours drawn
   **from every coarser level** -- a few from the coarsest (global / large
   scale), a few from each intermediate level, and the rest from the
   immediately-coarser level (local / small scale). The conditioning set
   therefore spans the full ladder of scales for *every* point, not just the
   ones inserted early.

Because neighbours are always drawn from strictly coarser levels, the graph has
exactly the batched, topological structure that :func:`graphgp.generate` (and
its ``chunk_size`` path) expect, so generation is unchanged. "Oversampling" --
more levels and/or larger ``k`` -- adds rungs to the ladder and tightens the
large-scale conditioning.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import jax.numpy as jnp

from .graph import Graph


def _level_boundaries(n: int, n0: int, n_levels: int) -> np.ndarray:
    """Cumulative level sizes: a geometric ladder ``n0 = c_0 < ... < c_{L-1} = n``."""
    n0 = int(min(max(n0, 1), n))
    n_levels = int(max(n_levels, 1))
    if n_levels == 1 or n0 >= n:
        return np.array([0, n], dtype=np.int64)
    frac = np.arange(n_levels) / (n_levels - 1)
    c = np.round(n0 * (n / n0) ** frac).astype(np.int64)
    c[0], c[-1] = n0, n
    for i in range(1, len(c)):
        if c[i] <= c[i - 1]:
            c[i] = c[i - 1] + 1
    c = np.minimum(c, n)
    c = np.concatenate([c[:1], c[1:][np.diff(c) > 0]])
    if c[-1] != n:
        c[-1] = n
    return np.concatenate([[0], c])   # prepend 0 so level 0 is [0, c_0)


def _quotas(k: int, n_coarser: int) -> np.ndarray:
    """Split ``k`` neighbours across ``n_coarser`` coarser levels, giving the
    spare neighbours to the *nearer* (finer-coarse, higher-index) levels."""
    base = k // n_coarser
    rem = k % n_coarser
    q = np.full(n_coarser, base, dtype=np.int64)
    if rem:
        q[n_coarser - rem:] += 1
    return q


def build_multiscale_graph(
    points: jnp.ndarray,
    *,
    k: int,
    n0: int = 256,
    n_levels: int = 8,
    seed: int = 0,
    indices: Optional[np.ndarray] = None,
    corr_length: Optional[float] = None,
) -> Graph:
    """Build a multi-scale (scale-ladder) GraphGP dependency graph.

    Each point is conditioned on ``k`` neighbours distributed across *all*
    coarser random sub-samples, so the conditioning sets span many scales and
    the realized field reproduces broad / long-range kernels that the stock
    nearest-neighbour graph screens out. Drop-in for :func:`graphgp.generate`.

    Conditioning (a rule worth banking): the latent covariance becomes
    ill-conditioned when the point spacing falls far *below* the kernel
    correlation length -- neighbouring nodes are then almost perfectly
    correlated, the per-node conditional covariance blocks go singular, and the
    sampler diverges. (Wechsler's v0 field pipeline hit exactly this: an
    N_side=512 voxel ~1 h^-1Mpc against a 5 h^-1Mpc Matern length diverged,
    while N_side=256 -- cells comparable to the length -- converged; it is the
    same pathology as hard-core-flattening a kernel collapsing the Vecchia
    blocks.) Keep node spacing comparable to (not far below) the kernel scale.
    Pass ``corr_length`` to get a warning when the finest-level median
    nearest-neighbour spacing is < 1/3 of it (purely diagnostic; no effect on
    the graph).

    Args:
        points: ``(N, d)`` point locations (already embedded if anisotropic).
        k: Total neighbours per point (levels >= 1).
        n0: Size of the coarsest level (dense GP base). Must be ``>= k``.
        n_levels: Number of geometric resolution levels.
        seed: Seed for the random nested sub-sampling.
        indices: Optional original-index array; defaults to the permutation.
        corr_length: Optional kernel correlation length (same units as
            ``points``) for the conditioning diagnostic above.

    Returns:
        A :class:`~graphgp.graph.Graph` with level-boundary ``offsets`` and
        ``neighbors`` referencing only coarser levels.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points)
    n = len(pts)
    if n0 < k:
        raise ValueError(f"n0 must be at least k. Got n0={n0}, k={k}.")

    if corr_length is not None and n > 1:
        # finest-level median nearest-neighbour spacing vs the kernel length
        sample = pts if n <= 5000 else pts[np.random.default_rng(seed).choice(n, 5000, replace=False)]
        d_nn, _ = cKDTree(sample).query(sample, k=2, workers=-1)
        spacing = float(np.median(d_nn[:, 1]))
        if spacing < corr_length / 3.0:
            import warnings
            warnings.warn(
                f"build_multiscale_graph: median node spacing {spacing:.3g} is < 1/3 of the "
                f"kernel correlation length {corr_length:.3g}; the latent covariance is likely "
                f"ill-conditioned (near-perfectly-correlated neighbours). Consider coarser points "
                f"(spacing ~ correlation length) to keep the conditional blocks well-conditioned.",
                stacklevel=2)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ordered = pts[perm]                       # coarse -> fine in random order

    bnd = _level_boundaries(n, n0, n_levels)  # [0, c0, c1, ..., N]
    n0 = int(bnd[1])
    n_lev = len(bnd) - 1
    trees = [cKDTree(ordered[bnd[j]:bnd[j + 1]]) for j in range(n_lev)]

    neighbors = np.empty((n - n0, k), dtype=np.int64)
    for ell in range(1, n_lev):               # finer levels condition on coarser
        lo, hi = bnd[ell], bnd[ell + 1]
        q = _quotas(k, ell)                    # quota from each coarser level 0..ell-1
        cols = []
        for j in range(ell):
            qj = int(q[j])
            if qj == 0:
                continue
            _, nn = trees[j].query(ordered[lo:hi], k=qj, workers=-1)
            nn = np.atleast_2d(nn.T).T if qj == 1 else nn
            nn = nn.reshape(hi - lo, qj) + bnd[j]   # global indices into `ordered`
            cols.append(nn)
        neighbors[lo - n0:hi - n0] = np.hstack(cols)

    offsets = tuple(int(x) for x in bnd[1:])   # (c0, c1, ..., N); batch ends
    out_indices = perm if indices is None else np.asarray(indices)[perm]
    return Graph(
        points=jnp.asarray(ordered, dtype=points.dtype),
        neighbors=jnp.asarray(neighbors),
        offsets=offsets,
        indices=jnp.asarray(out_indices),
    )
