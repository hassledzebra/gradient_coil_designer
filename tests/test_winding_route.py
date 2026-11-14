"""Tests for the one-wire winding route (winding='spiral', the default) and for
auto_fit, which adjusts a design until that route can be built and wound.

Run:  python -m pytest tests -q        (from the folder that contains the package)
"""
import io
import os
import sys
import warnings

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gradient_coil_designer import GradientCoilDesigner, physics as phys  # noqa: E402
from gradient_coil_designer import export as ex  # noqa: E402
from gradient_coil_designer import winding as wd  # noqa: E402

trimesh = pytest.importorskip("trimesh")

# Small former, 4 turns per lobe (the BME 4223 lab size) and a larger one with
# 10 turns per lobe whose Gx lobes have a pair of innermost loops side by side.
SMALL = dict(bore_radius_mm=40.0, linear_region_mm=20.0, holder_length_mm=126.0,
             wire_diameter_mm=1.55, holder_wall_mm=4.0, layer_gap_mm=0.5,
             flange_thick_mm=1.75, n_wires=4, verify_field=False)
LARGE = dict(bore_radius_mm=57.0, linear_region_mm=21.0, holder_length_mm=152.4,
             wire_diameter_mm=1.5, holder_wall_mm=3.6, n_wires=10, verify_field=False,
             stream_resolution_mm=0.5)


def design(axis, base=SMALL, **extra):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GradientCoilDesigner(**{**base, **extra}).design_axis(axis)


@pytest.fixture(scope="module")
def small():
    return {ax: design(ax) for ax in ("Gx", "Gy", "Gz")}


def settings(r):
    h = r["holder"]
    return wd.RouteSettings(h.wire_d, h.clearance)


def pitch(r):
    h = r["holder"]
    return h.wire_d + h.clearance


def route_points(r):
    """Route points on the unrolled cylinder, wrapped to one circumference."""
    C = 2 * np.pi * r["coil_radius_mm"]
    p = np.vstack([rt["path"][:, :2] for rt in r["routes"]])
    return np.c_[np.mod(p[:, 0], C), p[:, 1]], C


def assert_windable(r):
    """What 'a continuous winding path' means here."""
    assert r["routes"], "no route"
    assert r["open_paths"] == 0
    assert sum(rt["n_turns"] for rt in r["routes"]) == len(r["loops_3d"]), "a loop is not on the route"
    assert r["min_loop_spacing_mm"] >= pitch(r) - 1e-6, "two turns are closer than a wire"
    gap, touching = wd.route_clearance(r["routes"], r["coil_radius_mm"], settings(r))
    assert touching == 0 and gap >= pitch(r) - 0.1, f"the wire comes within {gap:.2f} mm of itself"
    m = trimesh.load(io.BytesIO(r["holder"].to_stl_bytes()), file_type="stl", process=False)
    m.merge_vertices(digits_vertex=5)
    assert m.is_watertight and m.is_winding_consistent and m.body_count == 1 and m.volume > 0


# ── the route ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("axis,wires,lobes", [("Gx", 2, 2), ("Gy", 1, 4), ("Gz", 1, 4)])
def test_one_wire_per_group_of_lobes(small, axis, wires, lobes):
    r = small[axis]
    assert len(r["routes"]) == wires
    assert all(rt["n_lobes"] == lobes for rt in r["routes"])
    assert sum(rt["n_turns"] for rt in r["routes"]) == len(r["loops_3d"])
    # each wire has one START and one END, both at the same end of the former
    for rt in r["routes"]:
        t = [q for q in r["holder"].tails() if q["wire"] == rt["name"]]
        assert sorted(q["label"] for q in t) == ["END", "START"]
        assert {q["end"] for q in t} == {rt["end"]}


@pytest.mark.parametrize("axis", ["Gx", "Gy", "Gz"])
def test_route_follows_every_loop(small, axis):
    r = small[axis]
    R = r["coil_radius_mm"]
    pts, C = route_points(r)
    tree = cKDTree(np.vstack([pts + [k * C, 0] for k in (-1, 0, 1)]))
    for w in r["loops_3d"]:
        xyz = w["xyz"] * 1e3
        loop = np.c_[np.mod(R * np.arctan2(xyz[:, 2], xyz[:, 1]), C), xyz[:, 0]]
        d, _ = tree.query(loop)
        # all of a loop is wound except the short stretch where the wire steps to the next turn
        assert np.mean(d < 0.6) > 0.8


@pytest.mark.parametrize("axis", ["Gx", "Gy", "Gz"])
def test_small_former_is_windable_with_all_its_turns(small, axis):
    r = small[axis]
    assert_windable(r)
    assert r["n_wires_used"] == 4, r["adjustments"]


@pytest.mark.parametrize("axis", ["Gx", "Gy", "Gz"])
def test_wound_coil_gives_the_designed_field(small, axis):
    r = small[axis]
    a = phys.dsv_metrics(r["loops_3d"], axis[-1].lower(), 10.0)["efficiency_mT_m_A"]
    b = phys.dsv_metrics(r["wires_3d"], axis[-1].lower(), 10.0)["efficiency_mT_m_A"]
    assert a * b > 0, "the wound coil has the opposite sign"
    assert abs(b - a) < 0.05 * abs(a), f"loops {a:.3f}, wound {b:.3f} mT/m/A"


@pytest.mark.parametrize("axis", ["Gx", "Gy", "Gz"])
def test_grooves_follow_the_route_at_full_depth(small, axis):
    r = small[axis]
    h = r["holder"]
    n_phi, n_x = h.export_grid()
    X, _, _, dm = h.groove_surface(n_phi=n_phi, n_x=n_x, seal_ends=False)
    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    f = RegularGridInterpolator((X[:, 0], np.r_[phi, np.pi]), np.c_[dm, dm[:, :1]])
    deep = wd.RouteSettings(h.wire_d, h.clearance).deep
    for rt in r["routes"]:
        w = rt["path"][np.abs(rt["path"][:, 1]) < h.x_half - 0.3]
        got = f(np.c_[w[:, 1], np.angle(np.exp(1j * w[:, 0] / h.R))])
        assert (got / (h.groove_radius + w[:, 2])).min() > 0.9
        assert w[:, 2].max() == pytest.approx(deep), "the lead-in is one wire deeper"


@pytest.mark.parametrize("axis", ["Gx", "Gz"])
def test_routed_stl_has_a_notch_for_every_tail(small, axis):
    h = small[axis]["holder"]
    m = trimesh.load(io.BytesIO(h.to_stl_bytes()), file_type="stl", process=False)
    m.merge_vertices(digits_vertex=5)
    v = m.vertices
    r = np.hypot(v[:, 1], v[:, 2])
    ang = np.arctan2(v[:, 2], v[:, 1])
    for t in h.tails():
        in_flange = t["end"] * v[:, 0] > h.x_half + 1.0
        near = np.abs(np.angle(np.exp(1j * (ang - np.radians(t["angle_deg"]))))) < 0.3 * h.wire_d / h.R_outer
        cut = r[in_flange & near & (r > h.R_inner + 0.2)]
        assert len(cut) and cut.max() < h.R_outer, f"no notch for {t['wire']} {t['label']}"
    # away from the tails the flange is whole
    assert r[np.abs(v[:, 0]) > h.x_half + 1.0].max() == pytest.approx(h.R_flange, abs=1e-3)


def test_routed_obj_file_is_a_valid_solid(small):
    h = small["Gz"]["holder"]
    m = trimesh.load(io.StringIO(ex.holder_to_obj(h)), file_type="obj", process=False)
    m.merge_vertices(digits_vertex=3)
    assert m.is_watertight and m.is_winding_consistent and m.body_count == 1 and m.volume > 0


def test_pair_of_innermost_loops_is_wound_too():
    r = design("Gx", base=LARGE, auto_fit=False)
    paths = [(w["level"], np.c_[np.unwrap(np.arctan2(w["xyz"][:, 2], w["xyz"][:, 1])), w["xyz"][:, 0]])
             for w in r["loops_3d"]]
    lobes = wd.group_lobes(paths, r["coil_radius_mm"])
    if not any(lb["eyes"] for lb in lobes):
        pytest.skip("this design has no lobe with a pair of innermost loops")
    assert sum(rt["n_turns"] for rt in r["routes"]) == len(r["loops_3d"])
    gap, touching = wd.route_clearance(r["routes"], r["coil_radius_mm"], settings(r))
    assert touching == 0
    assert gap >= min(r["min_loop_spacing_mm"], pitch(r)) - 0.1
    # mirror-image loops are kept together, so the two ends carry the same number of turns
    assert len({rt["n_turns"] for rt in r["routes"]}) == 1


# ── every design comes out windable (auto_fit) ───────────────────────────────

@pytest.mark.parametrize("axis", ["Gx", "Gy", "Gz"])
def test_designer_defaults_give_a_windable_route(axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = GradientCoilDesigner(verify_field=False)
        r = d.design_axis(axis)
    assert d.winding == "spiral" and d.auto_fit
    assert_windable(r)
    assert any("wall raised" in n for n in r["adjustments"])      # the 3.0 mm default wall is too thin


@pytest.mark.parametrize("axis", ["Gy", "Gz"])
def test_crowded_ends_are_fixed_without_giving_up_turns(axis):
    plain = design(axis, base=LARGE, winding="loops", auto_fit=False)
    fitted = design(axis, base=LARGE)
    assert plain["min_loop_spacing_mm"] < 1.0                     # the unadjusted design: turns on top of each other
    assert_windable(fitted)
    assert fitted["n_wires_used"] == 10
    assert any("return over" in n for n in fitted["adjustments"])


def test_turns_are_given_up_only_when_the_middle_is_too_dense():
    r = design("Gx", base=LARGE)
    assert_windable(r)
    assert 5 <= r["n_wires_used"] < 10
    assert any("turns per quadrant 10 ->" in n and "middle" in n for n in r["adjustments"])


def test_pattern_longer_than_the_holder_is_cut_to_fit():
    r = design("Gz", holder_length_mm=58.0)                       # the pattern asks for +/-30 mm
    assert_windable(r)
    assert any("cut to" in n for n in r["adjustments"])
    assert max(abs(v) for v in r["axial_extent_mm"]) < 29.0 - 4.0


def test_secondary_lobes_beyond_the_main_lobes_are_left_out():
    r = design("Gx", base=dict(bore_radius_mm=65.7, linear_region_mm=33.1, holder_length_mm=158.1,
                               wire_diameter_mm=0.8, n_wires=11, apodization=0.012, verify_field=False))
    assert_windable(r)
    assert r["n_wires_used"] == 11
    assert any("left out" in n for n in r["adjustments"])


def test_random_designs_always_give_a_windable_route():
    rng = np.random.default_rng(3)
    for _ in range(4):
        bore = float(rng.uniform(35, 90))
        lin = float(rng.uniform(0.25, 0.5) * bore)
        kw = dict(bore_radius_mm=round(bore, 1), linear_region_mm=round(lin, 1),
                  holder_length_mm=round(float(rng.uniform(2.8, 5.0) * lin), 1),
                  wire_diameter_mm=float(rng.choice([0.8, 1.0, 1.5, 2.0])), n_wires=int(rng.integers(3, 9)),
                  flange_width_mm=float(rng.choice([0.0, 5.0])), verify_field=False)
        for axis in ("Gx", "Gy", "Gz"):
            try:
                assert_windable(design(axis, base=kw))
            except Exception as e:
                raise AssertionError(f"{axis} with {kw}: {e}") from e


# ── opting out ───────────────────────────────────────────────────────────────

def test_wall_too_thin_is_raised_or_refused():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = GradientCoilDesigner(**{**SMALL, "holder_wall_mm": 3.0})
        assert d.wall == pytest.approx(3.4) and any("wall raised" in n for n in d.adjustments)
        with pytest.raises(wd.RouteError, match="wall"):
            GradientCoilDesigner(**{**SMALL, "holder_wall_mm": 3.0, "auto_fit": False}).design_axis("Gz")


def test_loops_mode_without_auto_fit_is_the_old_behaviour():
    r = design("Gz", winding="loops", auto_fit=False)
    assert r["routes"] is None
    assert r["holder"].tails() == []
    assert r["wires_3d"] is r["loops_3d"]
    assert r["adjustments"] == []
    assert len(r["loops_3d"]) == 16


def test_merging_grooves_are_reported_when_nothing_is_adjusted():
    with pytest.warns(RuntimeWarning, match="grooves merge"):
        GradientCoilDesigner(**{**SMALL, "winding": "loops", "auto_fit": False}).design_axis("Gy")


def test_unknown_winding_is_refused():
    with pytest.raises(ValueError):
        GradientCoilDesigner(winding="zigzag")
