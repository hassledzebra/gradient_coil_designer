"""Regression tests: holder grooves must be continuous in the exported mesh.

The groove is a height field sampled on a (phi, x) grid.  If the grid spacing is
comparable to the groove width, a wire passing between nodes leaves no groove
and the channel prints as a dotted line (the old GUI export used 120 x 80,
~2-3 mm spacing, and left ~5-7% of the wire path with no groove at all).

Run:  python -m pytest tests -q        (from the folder that contains the package)
"""
import io
import os
import sys
import warnings

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gradient_coil_designer import GradientCoilDesigner  # noqa: E402
from gradient_coil_designer import export as ex  # noqa: E402
from gradient_coil_designer.holder import GROOVE_SAMPLES_MIN  # noqa: E402


@pytest.fixture(scope="module")
def designs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = GradientCoilDesigner(bore_radius_mm=57.0, linear_region_mm=21.0,
                                 holder_length_mm=152.4, wire_diameter_mm=1.5,
                                 n_wires=10, verify_field=False)
        return {ax: d.design_axis(ax) for ax in ("Gx", "Gz")}


def centreline_depth(holder, n_phi, n_x, step_mm=0.05):
    """Groove depth (fraction of full) along every wire, as the mesh represents it."""
    X, _, _, dm = holder.groove_surface(n_phi=n_phi, n_x=n_x)
    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    f = RegularGridInterpolator((X[:, 0], np.r_[phi, np.pi]), np.c_[dm, dm[:, :1]])
    pts = []
    for ph, x in holder._wire_paths_phi_x:
        arc = holder.R_outer * np.unwrap(ph)
        s = np.r_[0.0, np.cumsum(np.hypot(np.diff(arc), np.diff(x)))]
        su = np.arange(0.0, s[-1], step_mm)
        a = np.interp(su, s, arc) / holder.R_outer
        pts.append(np.c_[np.interp(su, s, x), np.angle(np.exp(1j * a))])
    pts = np.vstack(pts)
    pts = pts[np.abs(pts[:, 0]) < holder.x_half - 1.0]
    return f(pts) / holder.groove_radius


@pytest.mark.parametrize("axis", ["Gx", "Gz"])
def test_contours_closed(designs, axis):
    assert designs[axis]["open_paths"] == 0


@pytest.mark.parametrize("axis", ["Gx", "Gz"])
def test_default_grid_is_groove_sized(designs, axis):
    h = designs[axis]["holder"]
    n_phi, n_x = h.export_grid()
    assert 2 * np.pi * h.R_outer / n_phi <= h.groove_radius / GROOVE_SAMPLES_MIN
    assert 2 * h.x_half / (n_x - 1) <= h.groove_radius / GROOVE_SAMPLES_MIN


@pytest.mark.parametrize("axis", ["Gx", "Gz"])
def test_default_export_groove_is_continuous(designs, axis):
    h = designs[axis]["holder"]
    depth = centreline_depth(h, *h.export_grid())
    assert depth.min() > 0.9, f"shallowest point on the wire path is {depth.min():.0%} deep"


def test_old_coarse_grid_was_broken(designs):
    """Documents the bug: the old 120 x 80 GUI grid leaves gaps in the groove."""
    h = designs["Gx"]["holder"]
    depth = centreline_depth(h, 120, 80)
    assert np.mean(depth < 0.1) > 0.01


def test_coarse_request_is_raised_with_warning(designs):
    h = designs["Gz"]["holder"]
    with pytest.warns(RuntimeWarning, match="too coarse"):
        n_phi, n_x = h.export_grid(120, 80)
    assert centreline_depth(h, n_phi, n_x).min() > 0.5


def test_obj_mesh_uses_safe_grid(designs):
    h = designs["Gz"]["holder"]
    verts, _ = ex._holder_mesh_data(h)
    n_phi, n_x = h.export_grid()
    assert len(verts) >= n_phi * n_x


def test_stl_is_single_watertight_body(designs):
    trimesh = pytest.importorskip("trimesh")
    h = designs["Gz"]["holder"]
    m = trimesh.load(io.BytesIO(h.to_stl_bytes()), file_type="stl", process=False)
    m.merge_vertices(digits_vertex=5)
    assert m.is_watertight
    assert m.body_count == 1
