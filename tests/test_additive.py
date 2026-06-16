import numpy as np
import jax
import jax.numpy as jnp

import graphgp as gp

jax.config.update("jax_enable_x64", True)


def _matern_cov(cutoff, variance=1.0, jitter=1e-3):
    return gp.extras.matern_kernel(p=1, variance=variance, cutoff=cutoff,
                                   r_min=1e-5, r_max=4.0, n_bins=400, jitter=jitter)


def test_generate_conditional_reproduces_coarse_field():
    # A band conditioned from a coarse sub-sample onto new points should
    # reproduce a field with ~the kernel variance and finite values. As in
    # practice, the coarse spacing resolves the band (cutoff ~ a few times the
    # ~0.035 coarse spacing) so the neighbour blocks are well-conditioned.
    rng = np.random.default_rng(0)
    cov = _matern_cov(0.05)
    coarse = jnp.asarray(rng.uniform(0, 1, size=(1200, 2)))
    new = jnp.asarray(rng.uniform(0, 1, size=(5000, 2)))
    g = gp.build_graph(coarse, n0=128, k=20)
    fc = np.asarray(gp.generate(g, cov, jnp.asarray(rng.standard_normal(1200))))
    fn = np.asarray(gp.generate_conditional(coarse, jnp.asarray(fc), new, cov, k=20, seed=3))
    assert fn.shape == (5000,)
    assert np.isfinite(fn).all()
    # the kriged field should faithfully match the coarse field's variance
    # (the primitive up-samples; it does not change the realised amplitude)
    assert abs(np.var(fn) - np.var(fc)) < 0.2 * np.var(fc)
    assert np.abs(fn).max() < 8.0


def test_generate_additive_sums_band_covariances():
    # Two-band kernel: the additive field's total variance should match the sum
    # of the band variances (independent bands => covariances add).
    rng = np.random.default_rng(1)
    n = 6000
    pts = jnp.asarray(rng.uniform(0, 1, size=(n, 2)))
    cbN, vN = _matern_cov(0.02, variance=0.6)
    _,   vB = _matern_cov(0.30, variance=0.4)
    bands = [
        {"cov": (cbN, vN), "n_coarse": None},        # fine: full density
        {"cov": (cbN, vB), "n_coarse": 700},         # broad: coarse -> kriged
    ]
    var = 0.0
    n_real = 16
    for s in range(n_real):
        f = np.asarray(gp.generate_additive(pts, bands, k=20, n0=128, seed=100 * s))
        assert np.isfinite(f).all()
        var += f.var()
    var /= n_real
    assert 0.75 < var < 1.25   # total variance ~ 0.6 + 0.4 = 1.0


def test_generate_additive_per_band_embedding():
    # bands with their own alpha + an embed callback should run and stay finite.
    rng = np.random.default_rng(2)
    n = 4000
    n_hat = rng.normal(size=(n, 3)); n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True)
    z = rng.uniform(0.4, 0.7, size=n)

    def embed(alpha):
        return jnp.asarray(np.hstack([n_hat, (alpha * z)[:, None]]))

    sb = np.concatenate([[0.0], np.geomspace(1e-4, 0.1, 63)])
    zb = np.concatenate([[0.0], np.geomspace(1e-4, 0.06, 63)])
    grid = np.exp(-(sb[:, None] / 0.05) - (zb[None, :] / 0.01))
    cov = gp.build_anisotropic_covariance(sb, zb, grid, 2.0, jitter=1e-3)
    bands = [{"cov": cov, "alpha": 2.0, "n_coarse": None},
             {"cov": cov, "alpha": 5.0, "n_coarse": 800}]
    f = np.asarray(gp.generate_additive(embed(2.0), bands, k=20, n0=128,
                                        seed=7, embed=embed))
    assert f.shape == (n,)
    assert np.isfinite(f).all()
