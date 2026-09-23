"""Cylinder wall-case geometry, fluid mask, and soft wall BC (inflow/body/slip).

Kolmogorov (periodic) does not use this module — periodicity is structural in the model.
所有 BC loss 在 normalized 預測空間比對（對齊 train_cylinder_v1）。
"""
from __future__ import annotations
from typing import NamedTuple, Callable
import jax
import jax.numpy as jnp


class CylinderGeometry(NamedTuple):
    body_center: tuple    # (cx, cy) normalized
    body_radius: float
    Lx: float
    Ly: float
    u_inf: float


def body_sdf(x, y, geom: CylinderGeometry):
    """Signed distance to body disk (>0 流體域, <0 body 內部)。

    NOTE(C2 圓近似): 用單一 normalized radius = normalized 空間的「圓」。但 x/Lx、y/Ly
    各自正規化後物理圓在 normalized 空間其實是橢圓（Lx≠Ly）。圓近似與真橢圓 IoU≈0.32，
    但影響 negligible：QR sensor 不落 body 內（physics 不受影響），eval body_mask 錯排的
    近體低速格點僅佔 <0.02% KE。正確修需從 body_xy 點雲取 per-axis 半軸做橢圓 mask。
    """
    cx, cy = geom.body_center
    return jnp.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1e-8) - geom.body_radius


def fluid_mask(cx, cy, geom: CylinderGeometry):
    """Float mask: 1.0 在流體域 (sdf>0)，0.0 在 body 內。"""
    return (body_sdf(cx, cy, geom) > 0.0).astype(jnp.float32)


def sample_wall_bc(key, geom: CylinderGeometry, n: int):
    """Return (inflow_pts, body_pts, slip_pts), each [n,3] = (x, y, t in [0,1] time-normalized)。

    時間軸以 [0,1] 取樣；caller 再依實際 sensor_time 線性映射（見 train_cylinder 的 decode_fn）。
    """
    cx, cy = geom.body_center
    r = geom.body_radius
    kk = jax.random.split(key, 7)
    # inflow x=0
    y_in = jax.random.uniform(kk[0], (n,), jnp.float32, 0.0, 1.0)
    t_in = jax.random.uniform(kk[1], (n,), jnp.float32, 0.0, 1.0)
    inflow = jnp.stack([jnp.zeros(n), y_in, t_in], -1)
    # body disk: center + r'·(cosθ,sinθ)
    rr = r * jnp.sqrt(jax.random.uniform(kk[2], (n,), jnp.float32))
    th = jax.random.uniform(kk[3], (n,), jnp.float32, 0.0, 2 * jnp.pi)
    bx = cx + rr * jnp.cos(th)
    by = cy + rr * jnp.sin(th)
    t_bd = jax.random.uniform(kk[4], (n,), jnp.float32, 0.0, 1.0)
    body = jnp.stack([bx, by, t_bd], -1)
    # slip walls: half y=0 half y=1
    half = n // 2
    xs = jax.random.uniform(kk[5], (n,), jnp.float32, 0.0, 1.0)
    ys = jnp.concatenate([jnp.zeros(half), jnp.ones(n - half)])
    ts = jax.random.uniform(kk[6], (n,), jnp.float32, 0.0, 1.0)  # 獨立 key（勿重用 inflow 的 kk[0]）
    slip = jnp.stack([xs, ys, ts], -1)
    return (inflow.astype(jnp.float32), body.astype(jnp.float32), slip.astype(jnp.float32))


def wall_bc_loss(decode_fn: Callable, inflow_pts, body_pts, slip_pts,
                 u_inf_n: float, u_zero_n: float, v_zero_n: float, bc_body_w: float):
    """Soft wall BC loss（normalized 空間）。

    decode_fn: [N,3] (x, y, t_phys) → [N,>=2] normalized 預測。
    inflow: u→u_inf_n, v→v_zero_n; body: u,v→zero (×bc_body_w); slip: v→v_zero_n。
    """
    o_in = decode_fn(inflow_pts)
    loss = jnp.mean((o_in[:, 0] - u_inf_n) ** 2) + jnp.mean((o_in[:, 1] - v_zero_n) ** 2)
    o_bd = decode_fn(body_pts)
    loss = loss + bc_body_w * (jnp.mean((o_bd[:, 0] - u_zero_n) ** 2)
                               + jnp.mean((o_bd[:, 1] - v_zero_n) ** 2))
    o_sl = decode_fn(slip_pts)
    loss = loss + jnp.mean((o_sl[:, 1] - v_zero_n) ** 2)
    return loss


# =============================================================================
# Forced-oscillation (controlled cylinder) — time-varying body geometry.
#
# The body is a rigid disk whose center oscillates along a fixed axis:
#     center(t) = base_center + amp * sin(2*pi*freq*t + phase) * axis
# All spatial quantities are in normalized [0,1]^2; time t is physical (same
# clock as sensor_time), so freq is a physical frequency. The static cylinder
# path above is untouched; amp=0 makes every function here reduce to it.
# =============================================================================


class OscillatingGeometry(NamedTuple):
    """Rigid disk translating along `axis`, in normalized [0,1]^2 query space.

    The body is a CIRCLE IN PHYSICAL SPACE, so its radius is stored in physical
    units and the normalized<->physical mapping goes through Lx/Ly. With Lx != Ly
    a single normalized radius cannot describe it (it would be an ellipse of
    semi-axes (R/Lx, R/Ly)); collapsing those to one number distorts the body per
    axis and, for the controlled rig, pushes the mask outside the domain. So the
    sdf/mask/sampling below all measure distance in physical units.

    (`CylinderGeometry` keeps its normalized-radius convention untouched — this
    anisotropy handling is scoped to the controlled/moving path.)
    """
    base_center: tuple    # (cx, cy) rest position, normalized
    body_radius_phys: float   # body radius in PHYSICAL units (metres)
    Lx: float             # physical domain length along x (metres)
    Ly: float             # physical domain length along y (metres)
    u_inf: float
    amp: float            # oscillation amplitude, normalized space
    freq: float           # physical frequency (cycles per physical time unit)
    phase: float          # phase offset (radians)
    axis: tuple           # (ax, ay) unit vector, oscillation direction


def _osc_disp(geom: OscillatingGeometry, t):
    """Scalar displacement along axis at physical time t."""
    return geom.amp * jnp.sin(2.0 * jnp.pi * geom.freq * t + geom.phase)


def body_center_at(geom: OscillatingGeometry, t):
    """Body center (cx, cy) at physical time t."""
    s = _osc_disp(geom, t)
    ax, ay = geom.axis
    return (geom.base_center[0] + s * ax, geom.base_center[1] + s * ay)


def body_velocity_at(geom: OscillatingGeometry, t):
    """Body wall velocity (vx, vy) = d center/dt at physical time t.

    Rigid-body: every point on the disk moves with this velocity (no rotation).
    """
    v = geom.amp * 2.0 * jnp.pi * geom.freq * jnp.cos(
        2.0 * jnp.pi * geom.freq * t + geom.phase)
    ax, ay = geom.axis
    return (v * ax, v * ay)


def body_sdf_moving(x, y, t, geom: OscillatingGeometry):
    """Signed distance (PHYSICAL metres) to the moving disk at time t.

    >0 fluid, <0 body. x/y are normalized query coords; offsets are mapped to
    physical via Lx/Ly so the body stays a true circle when Lx != Ly.
    """
    cx, cy = body_center_at(geom, t)
    dx = (x - cx) * geom.Lx
    dy = (y - cy) * geom.Ly
    return jnp.sqrt(dx ** 2 + dy ** 2 + 1e-12) - geom.body_radius_phys


def fluid_mask_moving(x, y, t, geom: OscillatingGeometry):
    """Float mask: 1.0 in fluid (sdf>0), 0.0 inside the moving body."""
    return (body_sdf_moving(x, y, t, geom) > 0.0).astype(jnp.float32)


def sample_wall_bc_moving(key, geom: OscillatingGeometry, n: int, t_lo: float, t_hi: float):
    """Sample (inflow, body, slip, body_vel) with the body oscillating.

    Unlike the static `sample_wall_bc`, body points here use PHYSICAL time
    sampled in [t_lo, t_hi] (the body position depends on it), are placed on the
    disk at its center-at-that-time, and carry the rigid-body wall velocity as
    their moving no-slip target.

    Returns:
      inflow[n,3], body[n,3], slip[n,3]  each = (x, y, t_phys)
      body_vel[n,2] = (vx, vy) normalized-space velocity at each body point's time
    """
    r = geom.body_radius_phys
    kk = jax.random.split(key, 7)
    # inflow x=0 (physical time so downstream decode uses the same clock)
    y_in = jax.random.uniform(kk[0], (n,), jnp.float32, 0.0, 1.0)
    t_in = jax.random.uniform(kk[1], (n,), jnp.float32, t_lo, t_hi)
    inflow = jnp.stack([jnp.zeros(n), y_in, t_in], -1)
    # body disk at its moving center for each sampled physical time. Points are
    # drawn area-uniformly on the PHYSICAL circle of radius r, then mapped back to
    # normalized coords via Lx/Ly (an offset r*cos/Lx is not r*cos when Lx != Ly).
    t_bd = jax.random.uniform(kk[2], (n,), jnp.float32, t_lo, t_hi)
    cx_t, cy_t = body_center_at(geom, t_bd)          # each [n]
    rr = r * jnp.sqrt(jax.random.uniform(kk[3], (n,), jnp.float32))
    th = jax.random.uniform(kk[4], (n,), jnp.float32, 0.0, 2 * jnp.pi)
    bx = cx_t + (rr * jnp.cos(th)) / geom.Lx
    by = cy_t + (rr * jnp.sin(th)) / geom.Ly
    body = jnp.stack([bx, by, t_bd], -1)
    vx, vy = body_velocity_at(geom, t_bd)            # each [n]
    body_vel = jnp.stack([vx, vy], -1)
    # slip walls: half y=0 half y=1
    half = n // 2
    xs = jax.random.uniform(kk[5], (n,), jnp.float32, 0.0, 1.0)
    ys = jnp.concatenate([jnp.zeros(half), jnp.ones(n - half)])
    ts = jax.random.uniform(kk[6], (n,), jnp.float32, t_lo, t_hi)
    slip = jnp.stack([xs, ys, ts], -1)
    return (inflow.astype(jnp.float32), body.astype(jnp.float32),
            slip.astype(jnp.float32), body_vel.astype(jnp.float32))


def wall_bc_loss_moving(decode_fn: Callable, inflow_pts, body_pts, slip_pts,
                        body_vel_n, u_inf_n: float, v_zero_n: float, bc_body_w: float):
    """Soft wall BC loss with a MOVING no-slip target (normalized space).

    Same as `wall_bc_loss` except the body target is the (normalized) rigid-body
    wall velocity `body_vel_n[:, :2]` instead of zero. Caller is responsible for
    converting physical body velocity to the normalized prediction space.
    """
    o_in = decode_fn(inflow_pts)
    loss = jnp.mean((o_in[:, 0] - u_inf_n) ** 2) + jnp.mean((o_in[:, 1] - v_zero_n) ** 2)
    o_bd = decode_fn(body_pts)
    loss = loss + bc_body_w * (jnp.mean((o_bd[:, 0] - body_vel_n[:, 0]) ** 2)
                               + jnp.mean((o_bd[:, 1] - body_vel_n[:, 1]) ** 2))
    o_sl = decode_fn(slip_pts)
    loss = loss + jnp.mean((o_sl[:, 1] - v_zero_n) ** 2)
    return loss
