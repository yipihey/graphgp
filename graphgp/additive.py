"""Conditional generation and additive multi-resolution synthesis for GraphGP.

A single nearest-neighbour graph on a dense point set realizes a kernel only up
to a few times the local point spacing: the conditioning sets are all at the
finest scale, so a **broad / long-range** kernel's large-separation correlation
is screened away. The cure here is *additive multi-resolution synthesis*:

    K = K_1 + K_2 + ... + K_B    (e.g. a sum of tensor-product Matérn bands)

so an independent GP field can be drawn for each band and the fields summed --
their covariances add back to ``K`` -- with **each band realized on a point
density matched to its own scale**. A broad band is generated on a sparse
sub-sample (where the ``k`` nearest neighbours actually span the broad scale,
so it is not screened) and then *kriged up* to the full point set; a fine band
is generated densely. This is the "process randomly down-sampled versions
serially, adding to K" idea made exact.

Two primitives:

- :func:`generate_conditional` -- draw a GP field at ``new_points`` conditioned
  on given values at ``coarse_points`` (k-nearest-coarse Vecchia kriging). This
  is the up-sampling step.
- :func:`generate_additive` -- given embedded points and a list of per-band
  covariances (each carrying its own embedding ``alpha``) with optional
  ``n_coarse`` sub-sample sizes, draw and sum the band fields.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np
import jax.numpy as jnp

from .graph import build_graph
from .refine import generate, refine


def generate_conditional(
    coarse_points: jnp.ndarray,
    coarse_values: jnp.ndarray,
    new_points: jnp.ndarray,
    covariance,
    *,
    k: int,
    seed: int = 0,
    chunk_size: Optional[int] = None,
) -> jnp.ndarray:
    """Draw a GP field at ``new_points`` conditioned on ``coarse_values``.

    Each new point is conditioned on its ``k`` nearest ``coarse_points`` (a
    Vecchia kriging step). For a kernel that is smooth on the coarse spacing
    (i.e. the coarse set resolves it) this reproduces the kernel faithfully;
    used to up-sample a broad band from a sparse sub-sample to the full set.

    Args:
        coarse_points: ``(M, d)`` embedded coarse locations (already generated).
        coarse_values: ``(M,)`` field values at ``coarse_points``.
        new_points: ``(P, d)`` embedded locations to fill in.
        covariance: a ``(cov_bins, cov_vals)`` tuple or ``AnisotropicCovariance``.
        k: neighbours per new point (drawn from the coarse set).
        seed: seed for the conditional innovations.
        chunk_size: passed through to ``refine`` to bound memory.

    Returns:
        ``(P,)`` field values at ``new_points``.
    """
    from scipy.spatial import cKDTree

    cp = np.asarray(coarse_points)
    npn = np.asarray(new_points)
    m = len(cp)
    if m < k:
        raise ValueError(f"need at least k={k} coarse points, got {m}.")
    tree = cKDTree(cp)
    _, nn = tree.query(npn, k=k, workers=-1)
    nn = nn.reshape(len(npn), k).astype(np.int64)   # indices into coarse set

    combined = jnp.concatenate(
        [jnp.asarray(coarse_points), jnp.asarray(new_points)], axis=0)
    offsets = (m, m + len(npn))
    xi = jnp.asarray(np.random.default_rng(seed).standard_normal(len(npn)),
                     dtype=combined.dtype)
    values = refine(combined, jnp.asarray(nn), offsets, covariance,
                    jnp.asarray(coarse_values), xi, chunk_size=chunk_size)
    return values[m:]


def generate_additive(
    points: jnp.ndarray,
    bands: Sequence[dict],
    *,
    k: int = 30,
    n0: int = 256,
    seed: int = 0,
    chunk_size: Optional[int] = None,
    embed: Optional[Callable[[float], jnp.ndarray]] = None,
    verbose: bool = False,
) -> jnp.ndarray:
    """Sum independent GP band fields, each realized at its natural resolution.

    Args:
        points: ``(N, d)`` embedded points for bands that use the default
            embedding. If a band needs a different embedding ``alpha`` (e.g. an
            anisotropic kernel), pass ``embed`` and the band's ``alpha`` so the
            points are re-embedded per band.
        bands: list of dicts, each with
            - ``cov``: the band covariance (tuple or ``AnisotropicCovariance``);
            - ``n_coarse`` (optional): sub-sample size for this band; ``None`` or
              ``>= N`` means generate at full density;
            - ``alpha`` (optional): embedding scale, used with ``embed``.
        k, n0: graph parameters.
        seed: base seed (each band/stage offsets it).
        chunk_size: bound generation memory.
        embed: ``alpha -> (N, d)`` callable to re-embed points per band; if
            ``None`` the same ``points`` are used for every band.
        verbose: print per-band progress.

    Returns:
        ``(N,)`` summed field with covariance ``sum_b cov_b``.
    """
    n = len(points)
    f = np.zeros(n)
    for b, band in enumerate(bands):
        cov = band["cov"]
        nc = band.get("n_coarse")
        alpha = band.get("alpha")
        pts = embed(alpha) if (embed is not None and alpha is not None) else points
        if nc is None or nc >= n:
            graph = build_graph(pts, n0=min(n0, n // 2), k=min(k, n - 1))
            eps = np.random.default_rng(seed + 1 + b).standard_normal(n)
            fb = np.asarray(generate(graph, cov, jnp.asarray(eps, dtype=pts.dtype),
                                     chunk_size=chunk_size))
            if verbose:
                print(f"[additive] band {b}: full density (N={n:,})")
        else:
            rng = np.random.default_rng(seed + 1 + b)
            idx = rng.choice(n, size=int(nc), replace=False)
            mask = np.zeros(n, bool); mask[idx] = True
            coarse = pts[idx]
            gc = build_graph(coarse, n0=min(n0, nc // 2), k=min(k, nc - 1))
            eps_c = rng.standard_normal(int(nc))
            fc = np.asarray(generate(gc, cov, jnp.asarray(eps_c, dtype=pts.dtype),
                                     chunk_size=chunk_size))
            rest = pts[~mask]
            fr = np.asarray(generate_conditional(
                coarse, jnp.asarray(fc, dtype=pts.dtype), rest, cov,
                k=min(k, int(nc)), seed=seed + 1000 + b, chunk_size=chunk_size))
            fb = np.empty(n)
            fb[idx] = fc
            fb[~mask] = fr
            if verbose:
                print(f"[additive] band {b}: coarse N_c={int(nc):,} -> kriged to {n:,}")
        fb = np.where(np.isfinite(fb), fb, 0.0)
        f += fb
    return jnp.asarray(f)
