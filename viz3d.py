"""
PyVista 3D visualization for gradient coils and holders.

Provides interactive 3D rendering with GPU acceleration via VTK.
All functions return the pyvista.Plotter so callers can add more actors or call .show().
"""

import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

_WIRE_COLORS = {
    "Gx": {"pos": "#CC3333", "neg": "#6699CC"},
    "Gy": {"pos": "#228B22", "neg": "#90EE90"},
    "Gz": {"pos": "#CC6600", "neg": "#FFD700"},
}

_HOLDER_COLORS = {
    "Gx": "#FF8C00",
    "Gy": "#228B22",
    "Gz": "#4169E1",
}


def _require_pyvista():
    if not HAS_PYVISTA:
        raise ImportError(
            "pyvista is required for 3D GPU rendering.  "
            "Install with:  pip install pyvista"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Wire path → pyvista tube mesh
# ─────────────────────────────────────────────────────────────────────────────

def _wire_to_tube(xyz_m, tube_radius_m=0.0007):
    """Convert (N,3) wire path in metres to a pyvista tube PolyData."""
    points = np.array(xyz_m, dtype=float)
    if len(points) < 2:
        return None
    n = len(points)
    lines = np.hstack([[2, i, i+1] for i in range(n-1)])
    pd = pv.PolyData()
    pd.points = points
    pd.lines = lines
    return pd.tube(radius=tube_radius_m, n_sides=8)


# ─────────────────────────────────────────────────────────────────────────────
# Holder → pyvista mesh
# ─────────────────────────────────────────────────────────────────────────────

def _holder_to_pv(holder, n_phi=1200, n_x=800, with_grooves=True):
    """Build pyvista PolyData for the holder outer surface (with/without grooves)."""
    if with_grooves and holder._wire_paths_phi_x:
        X, Y, Z, _ = holder.groove_surface(n_phi=n_phi, n_x=n_x, groove_scale=3.0)
    else:
        phi = np.linspace(-np.pi, np.pi, n_phi)
        x = np.linspace(-holder.x_half, holder.x_half, n_x)
        PHI, XX = np.meshgrid(phi, x)
        X = XX
        Y = holder.R_outer * np.cos(PHI)
        Z = holder.R_outer * np.sin(PHI)

    # Convert grid to structured surface
    nr, nc = X.shape
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = []
    for i in range(nr - 1):
        for j in range(nc - 1):
            v00 = i * nc + j
            v10 = (i+1) * nc + j
            v01 = i * nc + (j+1)
            v11 = (i+1) * nc + (j+1)
            faces += [3, v00, v10, v01, 3, v11, v01, v10]
    mesh = pv.PolyData(points, np.array(faces))
    return mesh


def _inner_bore_pv(holder, n_phi=60, n_x=30):
    """Build pyvista mesh for the inner bore surface."""
    phi = np.linspace(-np.pi, np.pi, n_phi)
    x = np.linspace(-holder.x_half, holder.x_half, n_x)
    PHI, XX = np.meshgrid(phi, x)
    X = XX
    Y = holder.R_inner * np.cos(PHI)
    Z = holder.R_inner * np.sin(PHI)
    nr, nc = X.shape
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = []
    for i in range(nr - 1):
        for j in range(nc - 1):
            v00 = i * nc + j; v10 = (i+1)*nc+j; v01 = i*nc+(j+1); v11 = (i+1)*nc+(j+1)
            faces += [3, v00, v01, v10, 3, v11, v10, v01]  # reversed normals (inward)
    return pv.PolyData(points, np.array(faces))


# ─────────────────────────────────────────────────────────────────────────────
# Public plotting API
# ─────────────────────────────────────────────────────────────────────────────

def plot_wires(wires_3d, label="", tube_radius_mm=0.7,
               plotter=None, show=True, window_size=(900, 700)):
    """Plot wire paths as 3D tubes.

    Parameters
    ----------
    wires_3d : list of dict from tfm.wire_paths_to_3d
    label : str  e.g. 'Gx'
    tube_radius_mm : float  visual tube radius
    plotter : pyvista.Plotter or None
    show : bool  call .show() at the end
    """
    _require_pyvista()
    if plotter is None:
        plotter = pv.Plotter(window_size=window_size,
                             title=f"Gradient Coil Wires — {label}")

    colors = _WIRE_COLORS.get(label, {"pos": "#CC3333", "neg": "#6699CC"})
    tube_r = tube_radius_mm * 1e-3

    for w in wires_3d:
        mesh = _wire_to_tube(w["xyz"], tube_radius_m=tube_r)
        if mesh is None:
            continue
        color = colors["pos"] if w.get("level", 1) > 0 else colors["neg"]
        plotter.add_mesh(mesh, color=color, smooth_shading=True)

    plotter.add_axes()
    plotter.add_text(f"{label} wire paths", font_size=10)
    if show:
        plotter.show()
    return plotter


def plot_holder(holder, label="", tube_radius_mm=0.7,
                plotter=None, show=True, window_size=(900, 700),
                with_grooves=True, opacity=0.55):
    """Plot holder cylinder with wire groove channels.

    Parameters
    ----------
    holder : CoilHolder
    opacity : float  transparency of the holder shell (0=transparent, 1=solid)
    with_grooves : bool  show groove indentations
    """
    _require_pyvista()
    if plotter is None:
        plotter = pv.Plotter(window_size=window_size,
                             title=f"Coil Holder — {label or holder.label}")

    hcol = _HOLDER_COLORS.get(label or holder.label, "#888888")

    # Outer surface with grooves
    outer_mesh = _holder_to_pv(holder, with_grooves=with_grooves)
    plotter.add_mesh(outer_mesh, color=hcol, opacity=opacity, smooth_shading=True)

    # Inner bore
    inner_mesh = _inner_bore_pv(holder)
    plotter.add_mesh(inner_mesh, color="#CCCCCC", opacity=0.2, smooth_shading=True)

    # Wire groove paths highlighted on surface
    wcolors = _WIRE_COLORS.get(label or holder.label, {"pos": "#FF4444", "neg": "#4444FF"})
    tube_r = tube_radius_mm * 1e-3
    for k, (x_arr, phi_arr) in enumerate(holder.groove_path_on_surface()):
        # Convert mm + radians → metres Cartesian
        xyz = np.column_stack([
            x_arr * 1e-3,
            holder.R_outer * 1e-3 * np.cos(phi_arr),
            holder.R_outer * 1e-3 * np.sin(phi_arr),
        ])
        mesh = _wire_to_tube(xyz, tube_radius_m=tube_r * 0.6)
        if mesh:
            col = wcolors["pos"] if k % 4 < 2 else wcolors["neg"]
            plotter.add_mesh(mesh, color=col, smooth_shading=True)

    plotter.add_axes()
    plotter.add_text(f"{label or holder.label} holder  Ø{holder.R_outer*2:.0f}mm OD",
                     font_size=9)
    if show:
        plotter.show()
    return plotter


def plot_assembly(coil_data_list, bore_radius_mm=None,
                  show_wires=True, show_holders=True,
                  tube_radius_mm=0.5, holder_opacity=0.35,
                  plotter=None, show=True, window_size=(1100, 800)):
    """Interactive 3D assembly view of all coil layers and holders.

    Parameters
    ----------
    coil_data_list : list of dict  {'label', 'wires_3d', 'holder'}
    bore_radius_mm : float or None   draw reference bore cylinder
    """
    _require_pyvista()
    if plotter is None:
        plotter = pv.Plotter(window_size=window_size,
                             title="Gradient Coil Assembly")

    tube_r = tube_radius_mm * 1e-3

    for item in coil_data_list:
        lbl = item.get("label", "")
        wires = item.get("wires_3d", [])
        holder = item.get("holder")
        colors = _WIRE_COLORS.get(lbl, {"pos": "#CC3333", "neg": "#6699CC"})
        hcol = _HOLDER_COLORS.get(lbl, "#888888")

        if show_wires and wires:
            for w in wires:
                mesh = _wire_to_tube(w["xyz"], tube_radius_m=tube_r)
                if mesh:
                    col = colors["pos"] if w.get("level", 1) > 0 else colors["neg"]
                    plotter.add_mesh(mesh, color=col, smooth_shading=True)

        if show_holders and holder is not None:
            outer = _holder_to_pv(holder, n_phi=1200, n_x=800, with_grooves=True)
            plotter.add_mesh(outer, color=hcol, opacity=holder_opacity,
                             smooth_shading=True)

    # Reference bore cylinder
    if bore_radius_mm is not None:
        cyl = pv.Cylinder(radius=bore_radius_mm * 1e-3,
                          height=0.16, direction=(1, 0, 0),
                          center=(0, 0, 0), resolution=80)
        plotter.add_mesh(cyl, color="silver", opacity=0.06)

    # Coordinate axes label
    plotter.add_axes()
    plotter.add_text("Gradient Coil Assembly  (X = bore axis, Y = B\u2080)", font_size=10)

    if show:
        plotter.show()
    return plotter


def plot_field_volume(wires_3d, label="", current=1.0,
                      half_range_mm=35, n_pts=20,
                      isosurface_levels=8,
                      plotter=None, show=True, window_size=(900, 700)):
    """Render volumetric By field with isosurfaces.

    Parameters
    ----------
    isosurface_levels : int  number of isosurface levels to display
    """
    _require_pyvista()
    from .physics import biot_savart_B

    r = half_range_mm * 1e-3
    t = np.linspace(-r, r, n_pts)
    X3, Y3, Z3 = np.meshgrid(t, t, t, indexing="ij")
    pts = np.column_stack([X3.ravel(), Y3.ravel(), Z3.ravel()])

    print(f"Computing By field on {n_pts}³ grid ({len(pts)} points) ...")
    B = biot_savart_B(wires_3d, current=current, obs_points=pts)
    By = B[:, 1].reshape(n_pts, n_pts, n_pts)

    grid = pv.ImageData()
    grid.dimensions = np.array([n_pts, n_pts, n_pts])
    grid.origin = (-r, -r, -r)
    grid.spacing = (2*r/(n_pts-1),) * 3
    grid.point_data["By"] = By.ravel(order="F")

    if plotter is None:
        plotter = pv.Plotter(window_size=window_size,
                             title=f"{label} By field isosurfaces")

    vmax = np.abs(By).max()
    levels = np.linspace(-vmax * 0.85, vmax * 0.85, isosurface_levels)
    iso = grid.contour(levels, scalars="By")
    plotter.add_mesh(iso, scalars="By", cmap="RdBu_r",
                     clim=[-vmax, vmax], opacity=0.6,
                     scalar_bar_args={"title": "By (T) @ 1A"})
    plotter.add_axes()
    plotter.add_text(f"{label} field isosurfaces  ({n_pts}³ grid)", font_size=9)

    if show:
        plotter.show()
    return plotter
