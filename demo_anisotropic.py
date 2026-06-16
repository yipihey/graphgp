"""Anisotropic covariance recovery: K(Δθ, Δz) with different angular/radial scales.

Generates a GraphGP field on observed points (n̂, z) using an anisotropic
kernel that is *narrow in angle, wide in redshift* (or vice versa), then
measures the empirical pair covariance ⟨f_i f_j⟩ binned in (Δθ, Δz) and checks
it recovers the input K. This validates the AnisotropicCovariance path
(2-D bilinear lookup dispatched inside compute_cov_matrix), including the
chunked generation, without any reference to cosmology.

Run::  python demo_anisotropic.py
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import graphgp as gp

jax.config.update("jax_enable_x64", True)


def main():
    rng = jr.key(0)
    N = 6000
    # points on a small sky patch (chord ≈ angular separation in rad) + redshift
    k1, k2, k3 = jr.split(rng, 3)
    ra = jr.uniform(k1, (N,), minval=0.0, maxval=np.radians(8.0))   # rad
    dec = jr.uniform(k2, (N,), minval=0.0, maxval=np.radians(8.0))
    z = jr.uniform(k3, (N,), minval=0.40, maxval=0.60)
    n_hat = jnp.stack([jnp.cos(dec) * jnp.cos(ra),
                       jnp.cos(dec) * jnp.sin(ra),
                       jnp.sin(dec)], axis=1)

    # anisotropic kernel: angular scale l_th, radial scale l_z (well separated)
    l_th = np.radians(0.6)    # ~0.01 rad
    l_z = 0.04
    variance = 1.0
    s_bins = np.concatenate([[0.0], np.linspace(1e-4, np.radians(4.0), 80)])
    z_bins = np.concatenate([[0.0], np.linspace(1e-4, 0.15, 80)])
    SS, ZZ = np.meshgrid(s_bins, z_bins, indexing="ij")
    grid = variance * np.exp(-0.5 * (SS / l_th) ** 2) * np.exp(-0.5 * (ZZ / l_z) ** 2)

    alpha = float(l_th / l_z)   # embed radial axis so tree locality is balanced
    cov = gp.build_anisotropic_covariance(s_bins, z_bins, grid, alpha, jitter=1e-3)
    points = gp.embed_points(n_hat, z, alpha)   # (N, 4)

    graph = gp.build_graph(points, n0=128, k=24)
    xi = jr.normal(jr.key(7), (N,))
    f = np.asarray(jax.jit(gp.generate)(graph, cov, xi))
    f_chunk = np.asarray(jax.jit(lambda c, x: gp.generate(graph, c, x, chunk_size=2000))(cov, xi))
    print(f"chunked vs full max|Δ| = {np.max(np.abs(f - f_chunk)):.2e}  (var f = {f.var():.3f})")

    # empirical ⟨f_i f_j⟩ in (Δθ, Δz) bins from a random subset of points
    idx = np.asarray(jr.choice(jr.key(3), N, (2500,), replace=False))
    nh = np.asarray(n_hat)[idx]; zz = np.asarray(z)[idx]; ff = f[idx]
    chord = np.linalg.norm(nh[:, None, :] - nh[None, :, :], axis=-1)
    dz = np.abs(zz[:, None] - zz[None, :])
    prod = ff[:, None] * ff[None, :]
    iu = np.triu_indices(len(idx), k=1)
    chord, dz, prod = chord[iu], dz[iu], prod[iu]

    print("\nEmpirical ⟨f_i f_j⟩ vs input K(Δθ, Δz):")
    print(f"{'Δθ[deg]':>8} {'Δz':>6} {'K_in':>8} {'K_emp':>8} {'ratio':>7}  Npairs")
    th_edges = [0.0, 0.3, 0.6, 1.2]      # degrees
    z_edges = [0.0, 0.02, 0.05, 0.10]
    for ti in range(len(th_edges) - 1):
        for zi in range(len(z_edges) - 1):
            th_lo, th_hi = np.radians(th_edges[ti]), np.radians(th_edges[ti + 1])
            z_lo, z_hi = z_edges[zi], z_edges[zi + 1]
            m = (chord >= th_lo) & (chord < th_hi) & (dz >= z_lo) & (dz < z_hi)
            if m.sum() < 50:
                continue
            th_c = np.degrees(np.sqrt(th_lo * th_hi)) if th_lo > 0 else np.degrees(0.5 * th_hi)
            z_c = np.sqrt(max(z_lo, 1e-3) * z_hi) if z_lo > 0 else 0.5 * z_hi
            chord_c = 2 * np.sin(0.5 * np.radians(th_c))  # back to chord for K lookup
            K_in = variance * np.exp(-0.5 * (np.radians(th_c) / l_th) ** 2) * np.exp(-0.5 * (z_c / l_z) ** 2)
            K_emp = prod[m].mean()
            print(f"{th_c:8.2f} {z_c:6.3f} {K_in:8.3f} {K_emp:8.3f} "
                  f"{K_emp / K_in if K_in > 1e-3 else np.nan:7.2f}  {int(m.sum())}")


if __name__ == "__main__":
    main()
