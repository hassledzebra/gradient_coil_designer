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

    for w in wires_3d:
        xyz = w['xyz']                    # (N, 3) meters
        sign = w.get('sign', 1)
        I = current * sign

        # Segment midpoints and direction vectors
        dl = np.diff(xyz, axis=0)         # (N-1, 3) segment vectors
        r_mid = (xyz[:-1] + xyz[1:]) / 2 # (N-1, 3) midpoints

        # Vectorised Biot-Savart: dB = (mu0 * I / 4pi) * (dl × r_hat) / |r|²
        for p_idx, p in enumerate(obs):
            r_vec = p - r_mid              # (N-1, 3)  displacement to obs
            r_mag = np.linalg.norm(r_vec, axis=1, keepdims=True)  # (N-1,1)
            r_mag = np.maximum(r_mag, 1e-12)
            r_hat = r_vec / r_mag
            cross = np.cross(dl, r_hat)   # (N-1, 3)
            dB = (MU0 * I / (4 * np.pi)) * cross / r_mag**2
            B[p_idx] += dB.sum(axis=0)

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
