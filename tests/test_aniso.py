import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import Partial

import numpy as np

import graphgp as gp

from test_tree import check_equal

rng = jr.key(91)


def _setup(n=600):
    k1, k2, k3 = jr.split(rng, 3)
    ra = jr.uniform(k1, (n,), minval=0.0, maxval=np.radians(8.0))
    dec = jr.uniform(k2, (n,), minval=0.0, maxval=np.radians(8.0))
    z = jr.uniform(k3, (n,), minval=0.4, maxval=0.6)
    n_hat = jnp.stack([jnp.cos(dec) * jnp.cos(ra),
                       jnp.cos(dec) * jnp.sin(ra),
                       jnp.sin(dec)], axis=1)
    l_th, l_z, var = np.radians(0.6), 0.04, 1.0
    s_bins = np.concatenate([[0.0], np.linspace(1e-4, np.radians(4.0), 60)])
    z_bins = np.concatenate([[0.0], np.linspace(1e-4, 0.15, 60)])
    SS, ZZ = np.meshgrid(s_bins, z_bins, indexing="ij")
    grid = var * np.exp(-0.5 * (SS / l_th) ** 2) * np.exp(-0.5 * (ZZ / l_z) ** 2)
    alpha = float(l_th / l_z)
    cov = gp.build_anisotropic_covariance(s_bins, z_bins, grid, alpha, jitter=1e-4)
    points = gp.embed_points(n_hat, z, alpha)
    return points, cov


def test_anisotropic_chunked_matches():
    points, cov = _setup()
    graph = gp.build_graph(points, n0=80, k=20)
    xi = jr.normal(rng, (points.shape[0],))
    v_full = jax.jit(gp.generate)(graph, cov, xi)
    for cs in (1, 32, 256):
        v = jax.jit(Partial(gp.generate, chunk_size=cs))(graph, cov, xi)
        check_equal(v_full, v, rtol=1e-12,
                    text=f"Anisotropic chunked (chunk_size={cs}) != full.")


def test_anisotropic_dense_covariance():
    # generate_dense is L @ xi with K = compute_cov_matrix(cov, pts, pts), so the
    # Jacobian J = L and J Jᵀ must equal the anisotropic kernel matrix exactly.
    points, cov = _setup(n=250)
    J = jax.jacfwd(Partial(gp.generate_dense, points, cov))(jnp.zeros(points.shape[0]))
    K_implied = J @ J.T
    K_true = gp.compute_cov_matrix(cov, points, points)
    assert jnp.allclose(K_implied, K_true, atol=1e-6), \
        "Anisotropic implied covariance does not match the kernel."
