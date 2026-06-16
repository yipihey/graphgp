import numpy as np
import jax
import jax.numpy as jnp
import pytest

import graphgp as gp
from graphgp.multiscale import _level_boundaries, _quotas

jax.config.update("jax_enable_x64", True)


def test_level_boundaries_monotone_and_endpoints():
    b = _level_boundaries(10000, 256, 8)
    assert b[0] == 0 and b[-1] == 10000
    assert np.all(np.diff(b) > 0)
    assert b[1] == 256  # coarsest level size


def test_quotas_sum_to_k():
    for n_coarser in range(1, 8):
        q = _quotas(30, n_coarser)
        assert q.sum() == 30
        assert np.all(q >= 0)
        # spare neighbours go to the nearer (higher-index) coarse levels
        assert np.all(np.diff(q) >= 0)


def test_multiscale_graph_is_valid():
    rng = np.random.default_rng(0)
    pts = jnp.asarray(rng.uniform(0, 1, size=(3000, 2)))
    g = gp.build_multiscale_graph(pts, k=20, n0=128, n_levels=6, seed=1)
    gp.check_graph(g)  # topological order, batches reference only coarser points
    assert g.offsets[0] == len(g.points) - len(g.neighbors)
    assert g.neighbors.shape[1] == 20


def test_multiscale_generation_runs_and_has_right_variance():
    rng = np.random.default_rng(1)
    n = 5000
    pts = jnp.asarray(rng.uniform(0, 1, size=(n, 2)))
    cb, cv = gp.extras.matern_kernel(p=1, variance=1.0, cutoff=0.05,
                                     r_min=1e-5, r_max=4.0, n_bins=400, jitter=1e-4)
    cov = (cb, cv)
    g = gp.build_multiscale_graph(pts, k=20, n0=128, n_levels=7, seed=2)
    var = 0.0
    n_real = 12
    for s in range(n_real):
        f = np.asarray(gp.generate(g, cov, jnp.asarray(rng.standard_normal(n))))
        assert np.isfinite(f).all()
        var += f.var()
    var /= n_real
    # realized field variance should be in the right ballpark of the kernel
    # variance (1.0); the scale-ladder approximation disperses mildly.
    assert 0.7 < var < 1.35


def test_multiscale_chunked_matches_unchunked():
    rng = np.random.default_rng(3)
    n = 4000
    pts = jnp.asarray(rng.uniform(0, 1, size=(n, 2)))
    cb, cv = gp.extras.matern_kernel(p=1, variance=1.0, cutoff=0.05,
                                     r_min=1e-5, r_max=4.0, n_bins=400, jitter=1e-4)
    cov = (cb, cv)
    g = gp.build_multiscale_graph(pts, k=16, n0=128, n_levels=6, seed=4)
    eps = jnp.asarray(rng.standard_normal(n))
    f_full = np.asarray(gp.generate(g, cov, eps))
    f_chunk = np.asarray(gp.generate(g, cov, eps, chunk_size=500))
    assert np.allclose(f_full, f_chunk, atol=1e-8)
