"""Anisotropic (non-Euclidean) covariance for GraphGP.

The default covariance is isotropic: ``compute_cov_matrix`` reduces a pair of
points to a scalar Euclidean distance and looks the value up in a 1-D table.
Many problems are anisotropic — most importantly survey clustering in
*observed* coordinates, where the correlation depends separately on angular
separation Δθ (on the sky) and redshift separation Δz, and the two are not
interchangeable (redshift-space distortions, photo-z smearing, ...).

``AnisotropicCovariance`` represents such a kernel as a 2-D table
``grid[i, j] = K(spatial_bins[i], z_bins[j])`` and evaluates it with bilinear
interpolation. Points are embedded so that the **last** coordinate is the
"radial" axis (e.g. ``alpha * z``) and the **remaining** coordinates are the
"spatial"/angular axis (e.g. a unit sky vector n̂). For a pair,

    Δspatial = ||a[:-1] - b[:-1]||        (chord distance ≈ Δθ for unit n̂)
    Δz       = |a[-1] - b[-1]| / alpha     (recover raw redshift separation)

``alpha`` is the scale applied to the radial coordinate in the embedding; it
sets the relative weighting of the radial vs spatial axis for the k-d tree
ordering / neighbor search in ``build_graph`` (which uses ordinary Euclidean
distance on the embedded points) while ``compute_cov_matrix`` divides it back
out to index the table in raw Δz units.

Because dispatch happens inside ``compute_cov_matrix``, every higher-level
routine — ``generate``, ``refine`` (incl. ``chunk_size``), ``generate_inv``,
``generate_logdet`` — works with an anisotropic kernel with no other changes.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jax.scipy.ndimage import map_coordinates
from jax.tree_util import register_dataclass


@register_dataclass
@dataclass
class AnisotropicCovariance:
    """2-D tabulated covariance K(Δspatial, Δz), bilinearly interpolated.

    Fields:
        spatial_bins: (n_s,) increasing grid of spatial separations (first 0.0).
        z_bins: (n_z,) increasing grid of redshift separations (first 0.0).
        grid: (n_s, n_z) covariance values.
        alpha: scalar radial embedding scale; points carry ``alpha*z`` in their
            last coordinate, divided back out here to index ``z_bins``.
    """

    spatial_bins: Array
    z_bins: Array
    grid: Array
    alpha: Array


def aniso_evaluate(cov: AnisotropicCovariance, points_a: Array, points_b: Array) -> Array:
    """Covariance matrix between ``points_a`` and ``points_b``.

    Mirrors the broadcasting of the isotropic ``compute_cov_matrix``: for inputs
    of shape ``(..., Ma, D)`` and ``(..., Mb, D)`` returns ``(..., Ma, Mb)``.
    The last coordinate is radial (``alpha*z``); the rest are spatial.
    """
    a = jnp.expand_dims(points_a, -2)  # (..., Ma, 1, D)
    b = jnp.expand_dims(points_b, -3)  # (..., 1, Mb, D)
    diff = a - b                       # (..., Ma, Mb, D)
    d_spatial = jnp.linalg.norm(diff[..., :-1], axis=-1)
    d_z = jnp.abs(diff[..., -1]) / cov.alpha

    n_s = cov.spatial_bins.shape[0]
    n_z = cov.z_bins.shape[0]
    # fractional grid indices via the (possibly non-uniform) bin coordinates
    ix = jnp.interp(d_spatial, cov.spatial_bins, jnp.arange(n_s, dtype=d_spatial.dtype))
    iz = jnp.interp(d_z, cov.z_bins, jnp.arange(n_z, dtype=d_z.dtype))
    return map_coordinates(cov.grid, [ix, iz], order=1, mode="nearest")


def embed_points(n_hat: Array, z: Array, alpha: float) -> Array:
    """Embed observed points for an anisotropic kernel.

    Args:
        n_hat: (N, 3) unit sky vectors (from RA/Dec).
        z: (N,) redshifts.
        alpha: radial scale; the embedded radial coordinate is ``alpha*z``.

    Returns:
        (N, 4) points ``[n̂_x, n̂_y, n̂_z, alpha*z]`` for ``build_graph`` and
        ``AnisotropicCovariance``.
    """
    return jnp.concatenate([n_hat, (alpha * z)[:, None]], axis=1)


def build_anisotropic_covariance(
    spatial_bins: Array,
    z_bins: Array,
    grid: Array,
    alpha: float,
    *,
    jitter: float = 0.0,
) -> AnisotropicCovariance:
    """Construct an ``AnisotropicCovariance`` from a 2-D table.

    ``grid[0, 0]`` (zero separation) is inflated by ``1 + jitter`` for
    positive-definiteness, mirroring the isotropic kernels' nugget.
    """
    grid = jnp.asarray(grid)
    grid = grid.at[0, 0].set(grid[0, 0] * (1.0 + jitter))
    return AnisotropicCovariance(
        spatial_bins=jnp.asarray(spatial_bins),
        z_bins=jnp.asarray(z_bins),
        grid=grid,
        alpha=jnp.asarray(alpha, dtype=grid.dtype),
    )
