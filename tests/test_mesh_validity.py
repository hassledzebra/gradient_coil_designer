"""Regression tests: exported holder meshes must be closed, consistently wound solids.

Before this fix the two step faces between the body and each flange were wound
backwards in the STL (watertight, but not a valid volume: a slicer or a boolean
tool had to repair it first), and the OBJ / Blender mesh was open at the flanges.

Run:  python -m pytest tests -q        (from the folder that contains the package)
"""
import io
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gradient_coil_designer import GradientCoilDesigner  # noqa: E402
from gradient_coil_designer import export as ex  # noqa: E402

trimesh = pytest.importorskip("trimesh")

KW = dict(bore_radius_mm=40.0, linear_region_mm=20.0, holder_length_mm=126.0,
          wire_diameter_mm=1.55, holder_wall_mm=4.0, layer_gap_mm=0.5,
          n_wires=4, verify_field=False,
          winding="loops", auto_fit=False)      # the plain holder; routed holders are in test_winding_route.py


def design(axis="Gz", **extra):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GradientCoilDesigner(**{**KW, **extra}).design_axis(axis)


@pytest.fixture(scope="module")
def holder():
    return design()["holder"]


def solid_volume_mm3(h, flanges):
    """Volume of the holder as a plain tube (plus flanges); grooves remove a little."""
    v = np.pi * (h.R_outer**2 - h.R_inner**2) * h.length
    if flanges:
        v += 2 * np.pi * (h.R_flange**2 - h.R_inner**2) * h.flange_w
    return v


@pytest.mark.parametrize("flanges", [True, False])
def test_stl_is_a_valid_solid(holder, flanges):
    m = trimesh.load(io.BytesIO(holder.to_stl_bytes(include_flanges=flanges)),
                     file_type="stl", process=False)
    m.merge_vertices(digits_vertex=5)
    assert m.is_watertight
    assert m.is_winding_consistent, "some faces are wound backwards"
    assert m.body_count == 1
    full = solid_volume_mm3(holder, flanges)
    assert 0.9 * full < m.volume < full, f"volume {m.volume:.0f} vs tube {full:.0f} mm^3"


@pytest.mark.parametrize("flange_w", [5.0, 0.0])
def test_obj_mesh_is_a_valid_solid(flange_w):
    h = design(flange_width_mm=flange_w)["holder"]
    verts, faces = ex._holder_mesh_data(h)
    tris = [[q[0], q[1], q[2]] for q in faces] + [[q[0], q[2], q[3]] for q in faces]
    m = trimesh.Trimesh(np.asarray(verts, float), np.asarray(tris), process=False)
    m.merge_vertices(digits_vertex=3)
    assert m.is_watertight
    assert m.is_winding_consistent
    assert m.body_count == 1
    full = solid_volume_mm3(h, flange_w > 0)
    assert 0.9 * full < m.volume < full


def test_stream_resolution_reaches_the_solver():
    coarse = design(stream_resolution_mm=1.0)
    fine = design(stream_resolution_mm=0.5)
    n = lambda r: sum(len(w["xyz"]) for w in r["wires_3d"])
    assert len(fine["wires_3d"]) == len(coarse["wires_3d"])
    assert n(fine) > 1.5 * n(coarse)
    assert fine["open_paths"] == 0


def test_stream_resolution_must_be_positive():
    with pytest.raises(ValueError):
        GradientCoilDesigner(stream_resolution_mm=0.0)
