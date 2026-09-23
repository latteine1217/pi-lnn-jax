"""Cylinder geometry, fluid mask, and wall BC sampling/loss."""
import jax
import jax.numpy as jnp
import numpy as np
from pi_lnn_jax.boundary import CylinderGeometry, body_sdf, fluid_mask, sample_wall_bc


GEOM = CylinderGeometry(body_center=(0.25, 0.5), body_radius=0.075,
                        Lx=0.32, Ly=0.17, u_inf=0.33)


def test_sdf_sign():
    # center → negative (inside body), far corner → positive (fluid)
    assert float(body_sdf(0.25, 0.5, GEOM)) < 0.0
    assert float(body_sdf(0.9, 0.1, GEOM)) > 0.0


def test_fluid_mask_excludes_body():
    cx = jnp.array([0.25, 0.9], jnp.float32)   # inside body, fluid
    cy = jnp.array([0.5, 0.1], jnp.float32)
    m = fluid_mask(cx, cy, GEOM)
    assert float(m[0]) == 0.0 and float(m[1]) == 1.0


def test_sample_wall_bc_locations():
    inflow, body, slip = sample_wall_bc(jax.random.PRNGKey(0), GEOM, 64)
    assert inflow.shape == (64, 3) and body.shape == (64, 3) and slip.shape == (64, 3)
    assert np.allclose(np.asarray(inflow[:, 0]), 0.0)                   # inflow at x=0
    d = np.sqrt((np.asarray(body[:, 0]) - 0.25) ** 2 + (np.asarray(body[:, 1]) - 0.5) ** 2)
    assert np.all(d <= 0.075 + 1e-5)                                    # body within disk
    ys = np.asarray(slip[:, 1])
    assert np.all((np.isclose(ys, 0.0)) | (np.isclose(ys, 1.0)))        # slip on y=0/1
