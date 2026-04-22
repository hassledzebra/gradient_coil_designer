"""
3D visualization tools for gradient coils and holders.

All drawing functions accept an optional `ax` argument.  Pass ax=None to
create a new figure automatically.  Each function returns the axes object so
callers can chain or further customize.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (needed for projection='3d')
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
import matplotlib.cm as cm


_AXIS_COLORS = {
    'Gx': ('#CC3333', '#6699CC'),   # (positive, negative)
    'Gy': ('#228B22', '#90EE90'),
    'Gz': ('#CC6600', '#FFD700'),
}

_HOLDER_COLORS = {
    'Gx': '#FF8C00',
    'Gy': '#228B22',
    'Gz': '#4169E1',
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: transparent cylinder shell
# ─────────────────────────────────────────────────────────────────────────────

def _draw_cylinder(ax, R, x_half, color='gray', alpha=0.08, n_phi=60):
    phi = np.linspace(0, 2 * np.pi, n_phi)
    x_ends = np.linspace(-x_half, x_half, 3)
    Tc, Xc = np.meshgrid(phi, x_ends)
    Yc = R * np.cos(Tc)
    Zc = R * np.sin(Tc)
    ax.plot_surface(Xc * 100, Yc * 100, Zc * 100,
                    alpha=alpha, color=color, linewidth=0, antialiased=False)


# ─────────────────────────────────────────────────────────────────────────────
# Wire path plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_wires_3d(wires_3d, ax=None, axis_label='', lw=0.9, alpha=0.9,
                 show_cylinder=True, R_cyl=None):
    """Plot wire paths in 3D.  Positive-level wires are warm, negative cool.

    Parameters
    ----------
    wires_3d : list of dict
        Wire paths from tfm.wire_paths_to_3d.
    axis_label : str, e.g. 'Gx'
    show_cylinder : bool
        Draw a translucent reference cylinder at R_cyl.
    R_cyl : float or None
        Cylinder radius in meters.  Auto-detected from wire data if None.
    """
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')

    pos_col, neg_col = _AXIS_COLORS.get(axis_label, ('#CC3333', '#6699CC'))

    for w in wires_3d:
        xyz = w['xyz'] * 100   # meters → cm
        col = pos_col if w.get('level', 1) > 0 else neg_col
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=col, lw=lw, alpha=alpha)

    if show_cylinder and wires_3d:
        if R_cyl is None:
            rs = [np.sqrt(w['xyz'][:, 1]**2 + w['xyz'][:, 2]**2).mean() for w in wires_3d]
            R_cyl = np.mean(rs)
        if wires_3d:
            xs = np.concatenate([w['xyz'][:, 0] for w in wires_3d])
            x_half = max(abs(xs.min()), abs(xs.max()))
            _draw_cylinder(ax, R_cyl, x_half)

    ax.set_xlabel('X (cm) — bore axis')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title(f'{axis_label} wire paths (red=+, blue=−)')
    _equal_aspect_3d(ax)
    return ax


def plot_holder_3d(holder, ax=None, axis_label='', alpha=0.55, wire_lw=1.0):
    """Draw a holder cylinder with recessed wire grooves on the outer surface."""
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')

    col = _HOLDER_COLORS.get(axis_label or holder.label, '#888888')
    pos_col, neg_col = _AXIS_COLORS.get(axis_label or holder.label, ('#CC3333', '#6699CC'))

    # Grooved outer surface (mm → cm)
    if holder._wire_paths_phi_x:
        X_g, Y_g, Z_g, _ = holder.groove_surface(n_phi=1200, n_x=800, groove_scale=3.0)
        ax.plot_surface(X_g / 10, Y_g / 10, Z_g / 10,
                        alpha=alpha, color=col, linewidth=0, antialiased=True,
                        rcount=800, ccount=1200)
    else:
        # Fallback: smooth cylinder
        n_phi = 60
        phi = np.linspace(-np.pi, np.pi, n_phi)
        Ro_cm = holder.R_outer / 10.0
        xh_cm = holder.x_half / 10.0
        Tc2, Xc2 = np.meshgrid(phi, np.array([-xh_cm, xh_cm]))
        ax.plot_surface(Xc2, Ro_cm * np.cos(Tc2), Ro_cm * np.sin(Tc2),
                        alpha=alpha, color=col, linewidth=0, antialiased=False)

    # Inner bore (smooth, lighter)
    n_phi = 60
    phi = np.linspace(-np.pi, np.pi, n_phi)
    Ri_cm = holder.R_inner / 10.0
    xh_cm = holder.x_half / 10.0
    Tc2, Xc2 = np.meshgrid(phi, np.array([-xh_cm, xh_cm]))
    ax.plot_surface(Xc2, Ri_cm * np.cos(Tc2), Ri_cm * np.sin(Tc2),
                    alpha=0.15, color='#CCCCCC', linewidth=0, antialiased=False)

    # Wire groove centerlines (colored)
    for k, (x_arr, phi_arr) in enumerate(holder.groove_path_on_surface()):
        col_w = pos_col if k % 4 < 2 else neg_col
        x_cm = x_arr / 10.0
        Rg_cm = (holder.R_outer - holder.groove_radius * 3.0) / 10.0
        y_cm = Rg_cm * np.cos(phi_arr)
        z_cm = Rg_cm * np.sin(phi_arr)
        ax.plot(x_cm, y_cm, z_cm, color=col_w, lw=wire_lw, alpha=0.9)

    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title(f'{axis_label or holder.label} holder  '
                 f'R={holder.R:.0f}mm  Ø{holder.R_outer*2:.0f}mm OD')
    _equal_aspect_3d(ax)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Full assembly view
# ─────────────────────────────────────────────────────────────────────────────

def plot_assembly(coil_data_list, ax=None, show_holders=True,
                 show_wires=True, show_bore=True, bore_radius_cm=None):
    """Plot all coil layers and holders in a single 3D view.

    Parameters
    ----------
    coil_data_list : list of dict
        Each entry: {'label': 'Gx', 'wires_3d': [...], 'holder': CoilHolder}
    bore_radius_cm : float or None
        Draw a reference bore cylinder at this radius (cm).  Auto from data.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    for item in coil_data_list:
        label = item.get('label', '')
        wires = item.get('wires_3d', [])
        holder = item.get('holder', None)

        if show_wires and wires:
            pos_col, neg_col = _AXIS_COLORS.get(label, ('#CC3333', '#6699CC'))
            for w in wires:
                xyz = w['xyz'] * 100
                col = pos_col if w.get('level', 1) > 0 else neg_col
                ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                        color=col, lw=0.7, alpha=0.75)

        if show_holders and holder is not None:
            _draw_cylinder(ax, holder.R_outer / 1000.0,
                           holder.x_half / 1000.0,
                           color=_HOLDER_COLORS.get(label, '#888888'),
                           alpha=0.08)

    if show_bore and bore_radius_cm is not None:
        x_half = max(
            [item['holder'].x_half / 1000 for item in coil_data_list
             if item.get('holder')], default=0.1
        )
        _draw_cylinder(ax, bore_radius_cm / 100.0, x_half,
                       color='silver', alpha=0.04)

    ax.set_xlabel('X (cm) — bore axis')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title('Gradient Coil Assembly\n(Gx outer → Gz inner)')
    _equal_aspect_3d(ax)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Unwrapped (phi vs x) view
# ─────────────────────────────────────────────────────────────────────────────

def plot_unwrapped(coil_data_list, axes=None):
    """Unwrapped cylinder view: phi (degrees) vs X (mm) for each coil axis.

    Parameters
    ----------
    coil_data_list : list of dict  with 'label', 'wires_3d'
    axes : list of matplotlib Axes or None
    """
    n = len(coil_data_list)
    if axes is None:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
        if n == 1:
            axes = [axes]

    for ax, item in zip(axes, coil_data_list):
        label = item.get('label', '')
        wires = item.get('wires_3d', [])
        pos_col, neg_col = _AXIS_COLORS.get(label, ('#CC3333', '#6699CC'))

        for w in wires:
            phi = np.degrees(np.arctan2(w['xyz'][:, 2], w['xyz'][:, 1]))
            x_mm = w['xyz'][:, 0] * 1000
            col = pos_col if w.get('level', 1) > 0 else neg_col
            ax.plot(phi, x_mm, color=col, lw=0.8, alpha=0.85)

        ax.set_xlabel('φ (degrees)')
        ax.set_ylabel('X (mm) — bore axis')
        ax.set_title(f'{label} unwrapped')
        ax.set_xlim(-180, 180)
        ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
        ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.4)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    return axes


# ─────────────────────────────────────────────────────────────────────────────
# Field profile / gradient linearity
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradient_profile(coords_mm, By_T, slope_mT_m_A, linearity_err,
                          axis='x', label='', ax=None):
    """Plot By vs position with linear fit overlay."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    By_mT = By_T * 1000
    ax.plot(coords_mm, By_mT, 'o-', ms=4, label='Biot-Savart', color='#1f77b4')
    t = np.linspace(coords_mm[0], coords_mm[-1], 200)
    fit = slope_mT_m_A * t  # linear through origin
    ax.plot(t, fit, '--', lw=1.5, color='red',
            label=f'Linear fit: {slope_mT_m_A:.3f} mT/m/A')

    ax.set_xlabel(f'{axis.upper()} (mm)')
    ax.set_ylabel('By (mT)  @  1 A')
    ax.set_title(f'{label}  gradient — linearity error {linearity_err:.1f}%')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_field_map(A_mm, B_mm, By_T, axis_labels=('X', 'Y'), label='', ax=None):
    """Colour map of By on a 2D plane."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    vmax = np.abs(By_T).max()
    pcm = ax.pcolormesh(A_mm, B_mm, By_T * 1000,
                        cmap='RdBu_r', vmin=-vmax * 1000, vmax=vmax * 1000,
                        shading='auto')
    plt.colorbar(pcm, ax=ax, label='By (mT) @ 1 A')
    ax.set_xlabel(f'{axis_labels[0]} (mm)')
    ax.set_ylabel(f'{axis_labels[1]} (mm)')
    ax.set_title(f'{label} field map')
    ax.set_aspect('equal')
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Combined dashboard
# ─────────────────────────────────────────────────────────────────────────────

def dashboard(designer_result, show=True, save_path=None):
    """Render a multi-panel dashboard from a GradientCoilDesigner.run() result.

    Parameters
    ----------
    designer_result : dict
        Output from GradientCoilDesigner.run().
    show : bool
        Call plt.show() after rendering.
    save_path : str or None
        If given, save the figure to this path.
    """
    from matplotlib.gridspec import GridSpec

    coils = designer_result.get('coils', {})
    axes_order = list(coils.keys())
    n = len(axes_order)

    fig = plt.figure(figsize=(6 * n, 14))
    fig.suptitle(designer_result.get('title', 'Gradient Coil Designer'),
                 fontsize=14, fontweight='bold')

    gs = GridSpec(4, n, figure=fig, hspace=0.45, wspace=0.35)

    for col, lbl in enumerate(axes_order):
        info = coils[lbl]
        wires = info['wires_3d']
        holder = info.get('holder')

        # Row 0: 3D wire paths
        ax3d = fig.add_subplot(gs[0, col], projection='3d')
        plot_wires_3d(wires, ax=ax3d, axis_label=lbl, show_cylinder=True)

        # Row 1: holder with grooves
        if holder is not None:
            ax_h = fig.add_subplot(gs[1, col], projection='3d')
            plot_holder_3d(holder, ax=ax_h, axis_label=lbl)

        # Row 2: unwrapped
        ax_u = fig.add_subplot(gs[2, col])
        plot_unwrapped([{'label': lbl, 'wires_3d': wires}], axes=[ax_u])

        # Row 3: gradient profile
        profile = info.get('profile')
        if profile is not None:
            ax_p = fig.add_subplot(gs[3, col])
            plot_gradient_profile(
                profile['coords_mm'], profile['By_T'],
                profile['gradient_mT_m_A'], profile['linearity_pct'],
                axis=lbl[-1], label=lbl, ax=ax_p
            )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Dashboard saved → {save_path}')
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _equal_aspect_3d(ax):
    """Force equal aspect ratio on a 3D axes."""
    try:
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        spans = limits[:, 1] - limits[:, 0]
        mid = limits.mean(axis=1)
        max_span = spans.max() / 2
        ax.set_xlim3d([mid[0] - max_span, mid[0] + max_span])
        ax.set_ylim3d([mid[1] - max_span, mid[1] + max_span])
        ax.set_zlim3d([mid[2] - max_span, mid[2] + max_span])
    except Exception:
        pass
