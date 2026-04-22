"""
Target Field Method (Turner, 1986) for gradient coil wire path design.

Coordinate convention (Halbach MRI):
  X = bore axis (axial)
  Y = B0 direction (transverse)
  Z = perpendicular transverse

Gradient definitions:
  Gx = dBy/dx  — cos(phi) symmetry, saddle coil
  Gy = dBy/dy  — cos(2*phi) symmetry, 45° rotated quadrupole
  Gz = dBy/dz  — sin(2*phi) symmetry, quadrupole
"""

import warnings
import numpy as np
from scipy import special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Bessel function coefficients
# ─────────────────────────────────────────────────────────────────────────────

def _calcP(mm, aa, bb, kk):
    mu0 = 4 * np.pi * 1e-7
    dBesselk_m = special.kvp(mm, abs(kk) * aa, 1)
    dBesseli_m = special.ivp(mm, abs(kk) * bb, 1)
    return aa * mu0 * kk * dBesseli_m * dBesselk_m


def _calcQ(mm, aa, bb, kk):
    mu0 = 4 * np.pi * 1e-7
    dBesselk_m = special.kvp(mm, abs(kk) * aa, 1)
    Besseli_m = special.iv(mm, abs(kk) * bb)
    return mm * aa * mu0 / bb * abs(kk) / kk * Besseli_m * dBesselk_m


# ─────────────────────────────────────────────────────────────────────────────
# Contour extraction
# ─────────────────────────────────────────────────────────────────────────────

def compute_contour_paths(streamF, num_wires, phi, z):
    """Extract wire paths as (phi, z) contours from a stream function.

    Returns
    -------
    wire_paths : list of (level, Nx2 ndarray)
        Each entry is (stream_level, array_of_(phi,z) points).
    cs : matplotlib QuadContourSet
        Raw contour object (useful for plotting unwrapped patterns).
    """
    phi2D, z2D = np.meshgrid(phi, z)
    levels = np.linspace(np.min(streamF), np.max(streamF), num_wires * 2 + 4)
    levels = levels[1:-1]
    midpoint_levels = [(levels[i] + levels[i + 1]) / 2 for i in range(len(levels) - 1)]
    midpoint_levels = np.array([v for v in midpoint_levels if abs(v) >= 1e-6])

    plt.ioff()
    fig_tmp = plt.figure()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = plt.contour(phi2D, z2D, streamF, levels=midpoint_levels)

    wire_paths = []
    for i, level in enumerate(cs.levels):
        segs = cs.allsegs[i] if hasattr(cs, 'allsegs') else []
        for seg in segs:
            if len(seg) > 5:
                wire_paths.append((float(level), np.array(seg)))

    # Keep only the longest paths when there are too many short fragments
    expected = num_wires * 4
    if len(wire_paths) > expected:
        wire_paths.sort(
            key=lambda item: float(np.sum(np.linalg.norm(np.diff(item[1], axis=0), axis=1))),
            reverse=True,
        )
        wire_paths = wire_paths[:expected]

    plt.close(fig_tmp)
    plt.ion()
    return wire_paths, cs


# ─────────────────────────────────────────────────────────────────────────────
# Stream function computations
# ─────────────────────────────────────────────────────────────────────────────

def compute_gx_stream(
    coil_radius_mm,
    linear_region_mm=55,
    coil_length_mm=70,
    num_wires=10,
    num_high_orders=10,
    linearity_order=16,
    apodization=0.05,
    resolution=1e-3,
):
    """Gx (dBy/dx) stream function — axial saddle, cos(phi) symmetry."""
    a = coil_radius_mm * 1e-3
    b = 0.001 * a
    d = linear_region_mm * 1e-3
    Zmax = coil_length_mm * 1e-3

    Nsamp = int(2 * Zmax / resolution)
    z = np.linspace(-Zmax, Zmax, Nsamp)
    phi = np.linspace(-1.5 * np.pi, 0.5 * np.pi, Nsamp)
    k = np.linspace(1e-5, 1 / resolution, Nsamp)

    grad_shape = z / (1 + (z / d) ** linearity_order)
    grad_shape_k = np.fft.fft(grad_shape)
    t_k = np.exp(-2 * (apodization * k) ** 2)

    n_0 = 2 * np.pi * grad_shape_k * 1e-3 / (_calcQ(1, a, b, k) + _calcP(1, a, b, k))
    streamF = np.zeros((Nsamp, Nsamp))

    for n in range(num_high_orders + 1):
        sign = (-1) ** (n + 1)
        scale = np.ones_like(k)
        for m in range(n + 1):
            scale *= (_calcP(2 * m - 1, a, b, k) - _calcQ(2 * m - 1, a, b, k)) / \
                     (_calcP(2 * m + 1, a, b, k) + _calcQ(2 * m + 1, a, b, k))
        B_apo = sign * np.fft.ifft(n_0 * t_k * scale / k)
        streamF += np.outer(np.real(B_apo), np.cos((2 * n + 1) * phi))

    return compute_contour_paths(np.real(streamF), num_wires, phi, z)


def compute_gy_stream(
    coil_radius_mm,
    linear_region_mm=45,
    coil_length_mm=150,
    num_wires=10,
    linearity_order=16,
    apodization=0.05,
    resolution=1e-3,
):
    """Gy (dBy/dy) stream function — 45°-rotated quadrupole, cos(2*phi) symmetry."""
    a = coil_radius_mm * 1e-3
    b = 0.001 * a
    d = linear_region_mm * 1e-3
    Zmax = 2 * coil_length_mm * 1e-3

    Nsamp = int(2 * Zmax / resolution)
    z = np.linspace(-Zmax, Zmax, Nsamp)
    phi = np.linspace(-np.pi, np.pi, Nsamp) + np.pi / 4
    k = np.linspace(1e-5, 1 / resolution, Nsamp)

    grad_shape = (1 / (1 + (z / d) ** linearity_order)
                  - 1 / (1 + ((z + 3.5 * d) / (0.5 * d)) ** linearity_order)
                  - 1 / (1 + ((z - 3.5 * d) / (0.5 * d)) ** linearity_order))
    grad_shape_k = np.fft.fft(grad_shape)
    t_k = np.exp(-2 * (apodization * k) ** 2)

    n_0 = b * 2 * np.pi * grad_shape_k * 1e-3 / (_calcQ(2, a, b, k) + _calcP(2, a, b, k))
    B_apo = np.fft.ifft((2 / np.pi) * (-1) * n_0 * t_k / k)
    streamF = np.outer(np.real(B_apo), np.cos(2 * phi))

    d_samp = d / resolution
    i_lo = int(Nsamp / 2 - 1.5 * d_samp)
    i_hi = int(Nsamp / 2 + 1.5 * d_samp)
    streamF = streamF[i_lo:i_hi]
    z = z[i_lo:i_hi]

    return compute_contour_paths(streamF, num_wires, phi, z)


def compute_gz_stream(
    coil_radius_mm,
    linear_region_mm=45,
    coil_length_mm=150,
    num_wires=10,
    linearity_order=16,
    apodization=0.05,
    resolution=1e-3,
):
    """Gz (dBy/dz) stream function — quadrupole, sin(2*phi) symmetry."""
    a = coil_radius_mm * 1e-3
    b = 0.001 * a
    d = linear_region_mm * 1e-3
    Zmax = 2 * coil_length_mm * 1e-3

    Nsamp = int(2 * Zmax / resolution)
    z = np.linspace(-Zmax, Zmax, Nsamp)
    phi = np.linspace(-np.pi, np.pi, Nsamp)
    k = np.linspace(1e-5, 1 / resolution, Nsamp)

    grad_shape = (1 / (1 + (z / d) ** linearity_order)
                  - 1 / (1 + ((z + 3.5 * d) / (0.5 * d)) ** linearity_order)
                  - 1 / (1 + ((z - 3.5 * d) / (0.5 * d)) ** linearity_order))
    grad_shape_k = np.fft.fft(grad_shape)
    t_k = np.exp(-2 * (apodization * k) ** 2)

    n_0 = b * 2 * np.pi * grad_shape_k * 1e-3 / (_calcQ(2, a, b, k) + _calcP(2, a, b, k))
    B_apo = np.fft.ifft((2 / np.pi) * (-1) * n_0 * t_k / k)
    streamF = np.outer(np.real(B_apo), np.sin(2 * phi))

    d_samp = d / resolution
    i_lo = int(Nsamp / 2 - 1.5 * d_samp)
    i_hi = int(Nsamp / 2 + 1.5 * d_samp)
    streamF = streamF[i_lo:i_hi]
    streamF[0] = 0
    streamF[-1] = 0
    z = z[i_lo:i_hi]

    return compute_contour_paths(streamF, num_wires, phi, z)


# ─────────────────────────────────────────────────────────────────────────────
# 3D wire path conversion
# ─────────────────────────────────────────────────────────────────────────────

def wire_paths_to_3d(wire_paths, coil_radius_mm):
    """Convert (phi, z_axial) contour paths → 3D Cartesian coordinates (meters).

    Mapping (Halbach convention):
      x = z_axial   (bore axis)
      y = R·cos(phi)
      z = R·sin(phi)

    Returns
    -------
    list of dict with keys: 'level', 'xyz' (Nx3 array, meters), 'sign' (+1/-1)
    """
    R = coil_radius_mm * 1e-3
    wires = []
    for level, path in wire_paths:
        phi_arr = path[:, 0]
        x_axial = path[:, 1]
        xyz = np.column_stack([
            x_axial,
            R * np.cos(phi_arr),
            R * np.sin(phi_arr),
        ])
        # Current direction is encoded in the path direction by the stream function.
        # Sign is kept only for visualization coloring (positive vs negative level).
        wires.append({'level': level, 'xyz': xyz, 'sign': 1})
    return wires


def wire_length_m(wires_3d):
    """Total wire length in meters."""
    total = 0.0
    for w in wires_3d:
        d = np.diff(w['xyz'], axis=0)
        total += np.sum(np.linalg.norm(d, axis=1))
    return total


def wire_axial_extent_mm(wires_3d):
    """Returns (min_x, max_x) axial extent in mm."""
    xs = np.concatenate([w['xyz'][:, 0] for w in wires_3d])
    return xs.min() * 1000, xs.max() * 1000


def gradient_efficiency(contour_set):
    """Estimate gradient efficiency (mT/m/A) from contour level spacing."""
    lvls = contour_set.levels
    if len(lvls) > 1:
        return abs(1.0 / (lvls[1] - lvls[0]))
    return 0.0
