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
# k-space axis
# ─────────────────────────────────────────────────────────────────────────────

# The k=0 bin is evaluated at this instead of exactly zero.  Every kernel used
# below enters as 1/((P+Q)*k), and P+Q ~ C/k as k->0, so the product has a
# finite limit; it is flat to 6 significant figures for k <= 1e-3, while k=0
# itself would divide by zero in _calcQ (abs(k)/k) and in the explicit 1/k.
_K_EPS = 1e-6


def _fft_wavenumbers(z):
    """Angular wavenumbers (rad/m) matching the bin order of ``np.fft.fft(f(z))``.

    ``np.fft.fft`` returns bins in the order 0, +dk, ..., +k_nyq, -k_nyq, ..., -dk.
    Any k axis paired with it must use that same ordering and the same scale,
    ``2*pi*fftfreq(N, dz)``; a monotonically increasing array is not the
    frequency axis of an FFT and silently mislabels every negative frequency as
    a high positive one.

    P and Q are odd in k and the apodization is even, so with this axis the
    integrand is Hermitian and the inverse transform is real by construction.
    """
    dz = float(z[1] - z[0])
    k = 2 * np.pi * np.fft.fftfreq(len(z), d=dz)
    k[0] = _K_EPS
    return k


def _check_conditioning(mm, aa, bb, kk, apodization, label):
    """Warn when the apodization does not tame the ill-posed inverse kernel.

    1/(P+Q) grows like exp(|k|*a) -- roughly 1e100 between DC and Nyquist for a
    typical bore -- so the Gaussian apodization is what makes the inversion
    stable, not a cosmetic smoother.  Too small and high-k noise is amplified;
    too large (or a domain too short to resolve it) and only a couple of
    spectral bins survive, collapsing the stream function to a single sinusoid.
    """
    t_k = np.exp(-2 * (apodization * kk) ** 2)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        kernel = np.abs(1.0 / ((_calcP(mm, aa, bb, kk) + _calcQ(mm, aa, bb, kk)) * kk))
        gain = kernel * t_k
    gain = np.nan_to_num(gain, nan=0.0, posinf=0.0)

    amplification = gain.max() / gain[0] if gain[0] > 0 else np.inf
    if amplification > 10.0:
        warnings.warn(
            f'{label}: apodization={apodization} is too small for a coil of radius '
            f'{aa*1e3:.0f} mm -- the inverse kernel is amplified {amplification:.1e}x '
            f'at high k. Increase it (0.012-0.020 is the usable range here).',
            RuntimeWarning, stacklevel=3,
        )

    n_live = int(np.count_nonzero(t_k[:len(kk) // 2] > 1e-3))
    if n_live < 6:
        warnings.warn(
            f'{label}: only {n_live} spectral bins survive apodization={apodization}. '
            f'The computation domain is too short for that smoothing length; '
            f'lengthen it or reduce the apodization.',
            RuntimeWarning, stacklevel=3,
        )


def _close_axial_ends(streamF, taper_frac=0.04, label=''):
    """Force the stream function to zero at both axial ends of the former.

    No current can flow off the end of a finite former, so psi must vanish
    there; if it does not, the contours run off the edge of the domain and come
    back as open arcs, which are not closed current loops and make any
    Biot-Savart result meaningless.  The ends are tapered with a raised cosine
    rather than hard-clamped so the truncation does not inject a step in the
    current density.

    A warning is raised when a lot of psi is being thrown away, which means the
    computation domain is too short for the requested linear region.
    """
    n = streamF.shape[0]
    peak = np.abs(streamF).max()
    edge = max(np.abs(streamF[0]).max(), np.abs(streamF[-1]).max())
    if peak > 0 and edge / peak > 0.05:
        warnings.warn(
            f'{label}: stream function is still {edge/peak*100:.0f}% of peak at the '
            f'axial ends; the coil is being truncated. Lengthen the coil or shrink '
            f'the linear region.',
            RuntimeWarning, stacklevel=3,
        )

    n_taper = max(2, int(round(taper_frac * n)))
    window = np.ones(n)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / (n_taper - 1)))
    window[:n_taper] = ramp
    window[-n_taper:] = ramp[::-1]
    return streamF * window[:, None]


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
    apodization=0.015,
    resolution=1e-3,
    pad_factor=4.0,
):
    """Gx (dBy/dx) stream function — axial saddle, cos(phi) symmetry."""
    a = coil_radius_mm * 1e-3
    b = 0.001 * a
    d = linear_region_mm * 1e-3
    Zmax = coil_length_mm * 1e-3

    # Zero-pad the transform domain: the FFT bin spacing is 2*pi/L, and Gx's
    # physical domain (~+/-70 mm) is too short to resolve the apodization
    # length on its own.  The stream function is cropped back to +/-Zmax below.
    Zfft = pad_factor * Zmax
    Nsamp = int(2 * Zfft / resolution)
    n_phi = int(2 * Zmax / resolution)
    z = np.linspace(-Zfft, Zfft, Nsamp)
    phi = np.linspace(-1.5 * np.pi, 0.5 * np.pi, n_phi)
    k = _fft_wavenumbers(z)
    _check_conditioning(1, a, b, k, apodization, 'Gx')

    grad_shape = z / (1 + (z / d) ** linearity_order)
    grad_shape_k = np.fft.fft(grad_shape)
    t_k = np.exp(-2 * (apodization * k) ** 2)

    n_0 = 2 * np.pi * grad_shape_k * 1e-3 / (_calcQ(1, a, b, k) + _calcP(1, a, b, k))
    streamF = np.zeros((Nsamp, n_phi))

    for n in range(num_high_orders + 1):
        sign = (-1) ** (n + 1)
        scale = np.ones_like(k)
        for m in range(n + 1):
            scale *= (_calcP(2 * m - 1, a, b, k) - _calcQ(2 * m - 1, a, b, k)) / \
                     (_calcP(2 * m + 1, a, b, k) + _calcQ(2 * m + 1, a, b, k))
        B_apo = sign * np.fft.ifft(n_0 * t_k * scale / k)
        streamF += np.outer(np.real(B_apo), np.cos((2 * n + 1) * phi))

    # Crop the padding back off, then force the stream function to zero at the
    # axial ends so every contour closes (see _close_axial_ends).
    keep = np.abs(z) <= Zmax
    streamF = streamF[keep]
    z = z[keep]
    streamF = _close_axial_ends(streamF, label='Gx')

    return compute_contour_paths(streamF, num_wires, phi, z)


def compute_gy_stream(
    coil_radius_mm,
    linear_region_mm=45,
    coil_length_mm=150,
    num_wires=10,
    linearity_order=16,
    apodization=0.015,
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
    k = _fft_wavenumbers(z)
    _check_conditioning(2, a, b, k, apodization, 'Gy')

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
    streamF = _close_axial_ends(streamF, label='Gy')

    return compute_contour_paths(streamF, num_wires, phi, z)


def compute_gz_stream(
    coil_radius_mm,
    linear_region_mm=45,
    coil_length_mm=150,
    num_wires=10,
    linearity_order=16,
    apodization=0.015,
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
    k = _fft_wavenumbers(z)
    _check_conditioning(2, a, b, k, apodization, 'Gz')

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
    z = z[i_lo:i_hi]
    streamF = _close_axial_ends(streamF, label='Gz')

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


def amperes_per_turn(contour_set):
    """Current carried by each turn, in amperes, for a unit stream function.

    Adjacent contours of a stream function bound one turn's worth of current,
    so the level spacing *is* the per-turn current.  This is not the gradient
    efficiency -- 1/(level spacing) has units of 1/A, not mT/m/A.  Efficiency
    has to come from the field the wires actually produce; use
    ``physics.dsv_metrics``.
    """
    lvls = contour_set.levels
    if len(lvls) > 1:
        return abs(float(lvls[1] - lvls[0]))
    return 0.0


def open_path_count(wire_paths, tol=1e-9):
    """Number of contours that do not close on themselves.

    Open paths are not valid current loops (they violate div J = 0), so any
    field computed from them is meaningless.  Should always be zero.
    """
    return sum(1 for _, p in wire_paths
               if float(np.linalg.norm(p[0] - p[-1])) > tol)
