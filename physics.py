"""
Biot-Savart field computation for gradient coil verification.

All inputs/outputs in SI units (meters, Tesla, A).
"""

import numpy as np


MU0 = 4 * np.pi * 1e-7


def biot_savart_B(wires_3d, current=1.0, obs_points=None):
    """Compute B field at observation points from a list of wire paths.

    Parameters
    ----------
    wires_3d : list of dict
        Wire paths as returned by tfm.wire_paths_to_3d.  Each entry has
        'xyz' (Nx3, meters) and 'sign' (+1 / -1).
    current : float
        Current amplitude in amperes.
    obs_points : ndarray (M, 3) in meters, or None
        If None, returns 0.

    Returns
    -------
    B : ndarray (M, 3)  — Tesla
    """
    if obs_points is None or len(obs_points) == 0:
        return np.zeros((0, 3))

    obs = np.asarray(obs_points, dtype=float)  # (M, 3)
    B = np.zeros_like(obs)

    # Flatten every wire into one segment table so the sum is a single
    # broadcast over (points x segments) instead of a Python loop per point.
    dls, mids, currents = [], [], []
    for w in wires_3d:
        xyz = w['xyz']                     # (N, 3) meters
        if len(xyz) < 2:
            continue
        dls.append(np.diff(xyz, axis=0))                  # (N-1, 3)
        mids.append((xyz[:-1] + xyz[1:]) / 2)             # (N-1, 3)
        currents.append(np.full(len(xyz) - 1, current * w.get('sign', 1)))
    if not dls:
        return B

    dl = np.concatenate(dls)               # (S, 3)
    r_mid = np.concatenate(mids)           # (S, 3)
    Idl = dl * np.concatenate(currents)[:, None]          # (S, 3)

    # Chunk over observation points to bound peak memory at roughly
    # chunk * S * 3 floats; 2**22 segment-point pairs is ~100 MB of temporaries.
    chunk = max(1, int(2**22 // max(len(dl), 1)))
    for lo in range(0, len(obs), chunk):
        p = obs[lo:lo + chunk, None, :]                    # (m, 1, 3)
        r_vec = p - r_mid[None, :, :]                      # (m, S, 3)
        r_mag = np.linalg.norm(r_vec, axis=2)              # (m, S)
        np.maximum(r_mag, 1e-12, out=r_mag)
        cross = np.cross(Idl[None, :, :], r_vec)           # (m, S, 3)
        B[lo:lo + chunk] = (MU0 / (4 * np.pi)) * np.einsum(
            'msc,ms->mc', cross, 1.0 / r_mag**3)

    return B


def compute_gradient_profile(wires_3d, axis='x', current=1.0,
                             half_range_mm=50.0, n_pts=25, offset=None):
    """Compute By along a chosen axis to verify gradient linearity.

    Parameters
    ----------
    axis : 'x', 'y', or 'z'
    half_range_mm : float
    n_pts : int
    offset : array-like (3,) or None
        Fixed offset added to every sample point (useful when By=0 on the
        default axis by symmetry, e.g. Gz coil needs y-offset).

    Returns
    -------
    coords_mm, By_T, gradient_mT_m_A, linearity_error_pct
    """
    r = half_range_mm * 1e-3
    t = np.linspace(-r, r, n_pts)
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]

    pts = np.zeros((n_pts, 3))
    pts[:, axis_idx] = t
    if offset is not None:
        pts += np.asarray(offset)

    B = biot_savart_B(wires_3d, current=current, obs_points=pts)
    By = B[:, 1]  # B0 is along Y in Halbach convention

    # Linear fit
    coeffs = np.polyfit(t, By, 1)
    slope = coeffs[0]                    # T/m/A
    By_fit = np.polyval(coeffs, t)
    residuals = By - By_fit
    max_range = max(abs(By.max()), abs(By.min()))
    linearity_err = (np.abs(residuals).max() / max_range * 100) if max_range > 0 else 0.0

    return t * 1000, By, slope * 1000, linearity_err   # (mm, T, mT/m/A, %)


def dsv_sample_points(radius_m, n_shells=6, n_per_shell=120):
    """Deterministic quasi-uniform sample of a solid sphere (the DSV).

    Fibonacci-distributed directions on each of ``n_shells`` shells whose radii
    are spaced as r ~ (i/n)^(1/3) so the points are uniform by volume.  Fixed
    rather than random so a design's reported metrics are reproducible.
    """
    i = np.arange(n_per_shell)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    cos_t = 1.0 - 2.0 * (i + 0.5) / n_per_shell
    sin_t = np.sqrt(np.clip(1.0 - cos_t**2, 0.0, None))
    theta = golden * i
    unit = np.column_stack([sin_t * np.cos(theta), sin_t * np.sin(theta), cos_t])

    pts = [np.zeros((1, 3))]
    for s in range(1, n_shells + 1):
        pts.append(unit * (radius_m * (s / n_shells) ** (1.0 / 3.0)))
    return np.vstack(pts)


def dsv_metrics(wires_3d, gradient_axis, dsv_radius_mm, current=1.0,
                n_shells=6, n_per_shell=120):
    """Gradient efficiency and linearity over a 3-D DSV, not a single line.

    A line profile through isocenter only probes one direction and flatters the
    design: the field can be linear along the gradient axis while bowing badly
    off-axis.  This fits By = G*u + c by least squares over a solid sphere and
    reports the worst-case departure from that ideal, scaled by the ideal field
    at the DSV edge -- the usual definition of gradient nonlinearity.

    Parameters
    ----------
    gradient_axis : 'x', 'y', or 'z'
        The spatial coordinate this coil encodes (Gx -> 'x', and so on).
    dsv_radius_mm : float
        DSV *radius*, not diameter.

    Returns
    -------
    dict with 'efficiency_mT_m_A', 'max_deviation_pct', 'rms_deviation_pct',
    'isocenter_offset_uT_A', 'dsv_radius_mm', 'n_points'.
    """
    R = dsv_radius_mm * 1e-3
    pts = dsv_sample_points(R, n_shells, n_per_shell)
    u = pts[:, {'x': 0, 'y': 1, 'z': 2}[gradient_axis.lower()]]

    By = biot_savart_B(wires_3d, current=current, obs_points=pts)[:, 1]

    G, c = np.polyfit(u, By, 1)            # T/m/A, T/A
    residual = By - (G * u + c)

    ideal_edge = abs(G) * R                # |B| the ideal gradient makes at the edge
    scale = 100.0 / ideal_edge if ideal_edge > 0 else 0.0

    return {
        'efficiency_mT_m_A': G * 1e3,
        'max_deviation_pct': float(np.abs(residual).max() * scale),
        'rms_deviation_pct': float(np.sqrt(np.mean(residual**2)) * scale),
        'isocenter_offset_uT_A': float(c * 1e6),
        'dsv_radius_mm': dsv_radius_mm,
        'n_points': len(pts),
    }


def compute_gradient_map(wires_3d, plane='xy', current=1.0,
                         half_range_mm=40.0, n_pts=15):
    """Compute By on a 2D plane for field map visualization.

    Parameters
    ----------
    plane : 'xy', 'xz', or 'yz'
    half_range_mm : float
        Half-extent of the sampling grid.
    n_pts : int
        Grid resolution per axis.

    Returns
    -------
    A, B_grid, By_grid : each (n_pts, n_pts) arrays
        A, B are the two axis coordinate grids (mm).
        By_grid is the field (Tesla).
    axis_labels : tuple of str
    """
    r = half_range_mm * 1e-3
    u = np.linspace(-r, r, n_pts)
    A2, B2 = np.meshgrid(u, u)
    pts = np.zeros((n_pts * n_pts, 3))

    plane_map = {
        'xy': (0, 1, 'X (mm)', 'Y (mm)'),
        'xz': (0, 2, 'X (mm)', 'Z (mm)'),
        'yz': (1, 2, 'Y (mm)', 'Z (mm)'),
    }
    i0, i1, xl, yl = plane_map[plane.lower()]
    pts[:, i0] = A2.ravel()
    pts[:, i1] = B2.ravel()

    B = biot_savart_B(wires_3d, current=current, obs_points=pts)
    By_grid = B[:, 1].reshape(n_pts, n_pts)

    return A2 * 1000, B2 * 1000, By_grid, (xl, yl)
