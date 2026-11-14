"""
Cylindrical coil holder with recessed wire grooves.

Each holder is a thin-walled cylinder with:
  - Wire grooves recessed on the outer surface as a circular half-pipe that
    hugs the round wire (radial displacement height-field; no boolean, so the
    mesh stays smooth and watertight)
  - The land (outer surface) sits a hair above the wire centerline, so the
    wire seats at the coil radius and is captured by the groove walls
  - Optional end flanges for stacking/alignment

Units: millimeters throughout (unlike tfm.py which uses meters).
"""

import numpy as np
import struct
import io
import math
import warnings
from scipy.spatial import cKDTree

# Groove continuity rule (the v11 holders' "trick").
# The groove is a height field sampled on a (phi, x) grid, so the STL can only
# show it where grid nodes fall inside the half-pipe.  If the node spacing is
# comparable to the groove width, a wire running between two nodes leaves no
# groove at all and the channel prints as a dotted line.  v11
# (cooley/build_clean_holders.py) samples at 0.62 x 0.65 mm for a 0.85 mm groove
# radius and its grooves are continuous; the old GUI export used 2-3 mm and
# left ~5-7% of the wire path with no groove.  So the grid is tied to the
# groove: at least GROOVE_SAMPLES_MIN nodes per groove radius, both around the
# circumference and along the axis.
GROOVE_SAMPLES_DEFAULT = 2.5     # default export: spacing = groove_radius / 2.5
GROOVE_SAMPLES_MIN = 2.0         # coarsest allowed for a printable export

# Holders that carry a winding route (see winding.py) get a notch through the
# flange for every wire tail.  The flange is then part of the height field, and
# the step from body to flange is this wide instead of exactly vertical, so no
# two grid rows share a position.
FLANGE_STEP_MM = 0.05
NOTCH_EXTRA_WIDTH_MM = 0.7       # notch width = wire diameter + this
NOTCH_EXTRA_DEPTH_MM = 0.15      # notch floor is this far below the groove floor


class CoilHolder:
    """Cylindrical holder with recessed outer-surface wire grooves.

    Parameters
    ----------
    coil_radius_mm : float
        Radius at which the wire centerline sits (groove center radius).
    wall_mm : float
        Holder wall thickness, measured inward from the land (outer surface).
        The land sits at coil_radius_mm + groove_clearance_mm; the inner bore
        is wall_mm below the land.
    length_mm : float
        Total axial length of the holder (not including flanges).
    flange_width_mm : float
        Width of the end flanges (0 = no flanges).
    flange_thick_mm : float
        Radial thickness added to flanges beyond outer surface.
    wire_diameter_mm : float
        Wire/tube outer diameter, sets groove width and depth.
    groove_clearance_mm : float
        Extra clearance added to groove radius on each side.
    label : str
        Short label, e.g. 'Gx', 'Gy', 'Gz'.
    """

    def __init__(
        self,
        coil_radius_mm: float,
        wall_mm: float = 3.0,
        length_mm: float = 145.0,
        flange_width_mm: float = 5.0,
        flange_thick_mm: float = 5.0,
        wire_diameter_mm: float = 1.5,
        groove_clearance_mm: float = 0.1,
        label: str = '',
    ):
        self.R = coil_radius_mm
        self.wall = wall_mm
        self.length = length_mm
        self.flange_w = flange_width_mm
        self.flange_t = flange_thick_mm
        self.wire_d = wire_diameter_mm
        self.clearance = groove_clearance_mm
        self.label = label

        # Half-pipe groove convention: the land (outer surface) sits a hair
        # above the wire centerline so the wire nests in a circular groove that
        # hugs it; the wall extends inward from the land.
        self.R_outer = coil_radius_mm + groove_clearance_mm   # land / outer surface
        self.R_inner = self.R_outer - wall_mm                 # inner bore
        self.groove_radius = wire_diameter_mm / 2 + groove_clearance_mm
        self.x_half = length_mm / 2          # half-length of main cylinder body

        self._wire_paths_phi_x = []           # list of (phi_arr, x_arr) in radians/mm
        self._deep_pts = None                 # (arc at R_outer, x, extra depth): lead-in channels of a route
        self._tails = []                      # wire tails of a route: dicts with phi, end, extra, label, wire

    @property
    def total_length_mm(self):
        return self.length + 2 * self.flange_w

    @property
    def R_flange(self):
        return self.R_outer + self.flange_t

    def add_wire_paths(self, wires_3d):
        """Register 3D wire paths (from tfm.wire_paths_to_3d) for groove generation.

        Internally stores wire positions as (phi, x) pairs on the cylinder surface.
        """
        for w in wires_3d:
            xyz = w['xyz'] * 1000.0   # meters → mm
            x_arr = xyz[:, 0]
            phi_arr = np.arctan2(xyz[:, 2], xyz[:, 1])   # atan2(z, y)
            self._wire_paths_phi_x.append((phi_arr, x_arr))

    def add_route(self, routes):
        """Register winding routes from ``winding.plan_route`` instead of closed loops.

        The grooves follow the routes.  Where a route carries extra depth (the
        lead-in that passes under the turns) the groove is that much deeper,
        with the same half-pipe floor.  Each wire tail gets a notch through the
        end flange in the STL / OBJ export.
        """
        deep = [] if self._deep_pts is None else [self._deep_pts]
        for rt in routes:
            w = np.asarray(rt['path'], float)
            self._wire_paths_phi_x.append((w[:, 0] / self.R, w[:, 1]))
            d = w[w[:, 2] > 0.02]
            if len(d):
                deep.append(np.c_[d[:, 0] / self.R * self.R_outer, d[:, 1], d[:, 2]])
            for label, (u, x) in rt['marks']:
                if label in ('START', 'END'):
                    k = 0 if label == 'START' else -1
                    self._tails.append(dict(phi=float(u / self.R), end=int(np.sign(x)) or -1,
                                            extra=float(w[k, 2]), label=label, wire=rt.get('name', '')))
        self._deep_pts = np.vstack(deep) if deep else None

    def groove_surface(self, n_phi=300, n_x=200, groove_scale=1.0, max_pts_per_wire=400, seal_ends=True):
        """Compute the outer surface mesh with grooves indented.

        The groove is a circular half-pipe of radius ``groove_radius`` whose
        floor follows the wire: at in-surface distance ``d`` from the nearest
        wire the surface is pushed in by ``sqrt(gr**2 - d**2)``.  This hugs the
        round wire (depth = gr at the wire, smoothly back to the land at d=gr),
        unlike a linear V-groove, and seats the wire at the coil radius.

        Parameters
        ----------
        n_phi, n_x : int
            Grid resolution.  Higher = smoother grooves.
        groove_scale : float
            Multiplier on groove_radius (half-pipe radius).  1.0 = a channel
            that exactly hugs the wire (depth = width/2 = groove_radius).
        max_pts_per_wire : int
            Deprecated / ignored.  Wire paths are now arc-length resampled to
            ds = groove_radius/3 (densified, not capped) so the groove stays
            continuous regardless of the TFM contour point spacing.
        seal_ends : bool
            True (default) forces the first and last grid rows to the land
            radius so plain end caps close the mesh.  The routed export passes
            False, because its tail grooves run off the end of the body.

        Returns
        -------
        X, Y, Z : ndarray shape (n_x, n_phi)
            Cartesian coordinates of the grooved outer surface (mm).
        depth_map : ndarray shape (n_x, n_phi)
            Groove depth at each surface point (0 = no groove).
        """
        # endpoint=False: n_phi unique angles, no duplicate seam vertex
        phi_grid = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
        x_grid   = np.linspace(-self.x_half, self.x_half, n_x)
        PHI, XX  = np.meshgrid(phi_grid, x_grid)   # (n_x, n_phi)

        gr = self.groove_radius * groove_scale

        if not self._wire_paths_phi_x:
            depth_map = np.zeros_like(PHI)
        else:
            # Build a KD-tree in (arc, x) space from all wire paths.
            # The groove is the distance field to discrete wire POINTS, so the
            # points must be spaced well below the groove radius or the channel
            # beads up ("dotted") between samples.  TFM contours are often
            # coarse (1-4 mm spacing), so DENSIFY each path by arc-length
            # resampling to ds = gr/3 before building the tree.  Duplicate with
            # ±2π offsets so wrap-around is handled correctly.
            ds = max(gr / 3.0, 1e-3)
            arc_pts, x_pts = [], []
            for phi_arr, x_arr in self._wire_paths_phi_x:
                arc = self.R_outer * np.unwrap(phi_arr)   # unwrap to avoid seam jumps
                seg = np.hypot(np.diff(arc), np.diff(x_arr))
                s = np.concatenate([[0.0], np.cumsum(seg)])
                if s[-1] <= 0:
                    continue
                n = max(int(s[-1] / ds) + 1, len(arc))    # never thinner than input
                su = np.linspace(0.0, s[-1], n)
                arc_r = np.interp(su, s, arc)
                x_r   = np.interp(su, s, x_arr)
                phi_r = arc_r / self.R_outer
                # A route can run more than once round the former, so copies
                # two turns either side are needed to cover every wrap.
                for offset in (0.0, 2 * np.pi, -2 * np.pi, 4 * np.pi, -4 * np.pi):
                    arc_pts.append(self.R_outer * (phi_r + offset))
                    x_pts.append(x_r)

            wire_pts = np.column_stack([np.concatenate(arc_pts),
                                        np.concatenate(x_pts)])   # (N, 2)
            tree = cKDTree(wire_pts)

            # Query every grid point
            arc_flat  = (self.R_outer * PHI).ravel()
            grid_pts  = np.column_stack([arc_flat, XX.ravel()])    # (M, 2)
            min_dist, _ = tree.query(grid_pts, k=1, workers=-1)

            # Circular half-pipe profile (hugs the round wire), not a V-cone.
            dm_flat = np.where(min_dist < gr,
                               np.sqrt(np.clip(gr**2 - min_dist**2, 0.0, None)),
                               0.0)
            depth_map = dm_flat.reshape(PHI.shape)

            if self._deep_pts is not None and len(self._deep_pts):
                # Lead-in channels: same half-pipe floor, lower by the extra
                # depth each route point carries.
                C = 2 * np.pi * self.R_outer
                dp = np.vstack([self._deep_pts + [k * C, 0.0, 0.0] for k in (-2, -1, 0, 1, 2)])
                # Deepest of the half-pipes within reach, not just the nearest
                # point: where two parts of a route at different depths meet (the
                # wire rising out of the channel), the nearest one alone would
                # leave a step.
                near = np.where(min_dist < 2 * gr)[0]
                dist, j = cKDTree(dp[:, :2]).query(grid_pts[near], k=24,
                                                   distance_upper_bound=gr, workers=-1)
                ok = np.isfinite(dist)
                cand = np.where(ok, dp[np.where(ok, j, 0), 2]
                                + np.sqrt(np.clip(gr**2 - np.where(ok, dist, gr)**2, 0.0, None)), 0.0)
                extra = np.zeros(len(grid_pts))
                extra[near] = cand.max(axis=1)
                depth_map = np.maximum(depth_map, extra.reshape(PHI.shape))

        R_surface = self.R_outer - depth_map
        if seal_ends:
            # Force boundary rows to exact R_outer so end-cap annuli seal flush
            R_surface[0, :]  = self.R_outer
            R_surface[-1, :] = self.R_outer
        X = XX
        Y = R_surface * np.cos(PHI)
        Z = R_surface * np.sin(PHI)
        return X, Y, Z, depth_map

    def export_grid(self, n_phi=None, n_x=None, samples_per_radius=GROOVE_SAMPLES_DEFAULT):
        """Grid size for a fabrication export, tied to the groove radius.

        ``None`` picks a grid at ``groove_radius / samples_per_radius`` spacing.
        An explicit value coarser than ``groove_radius / GROOVE_SAMPLES_MIN`` is
        raised to that minimum with a warning, because a coarser grid cannot
        represent a continuous groove (see GROOVE_SAMPLES_MIN above).
        """
        def need(span_mm, k):
            return int(math.ceil(span_mm * k / self.groove_radius))

        circ, axial = 2 * np.pi * self.R_outer, 2 * self.x_half
        auto_phi, auto_x = need(circ, samples_per_radius), need(axial, samples_per_radius) + 1
        min_phi, min_x = need(circ, GROOVE_SAMPLES_MIN), need(axial, GROOVE_SAMPLES_MIN) + 1
        n_phi = auto_phi if n_phi is None else int(n_phi)
        n_x = auto_x if n_x is None else int(n_x)
        if n_phi < min_phi or n_x < min_x:
            warnings.warn(
                f'{self.label}: a {n_phi} x {n_x} grid is too coarse for a '
                f'{self.groove_radius:.2f} mm groove radius ({circ / n_phi:.2f} x '
                f'{axial / max(n_x - 1, 1):.2f} mm spacing) and would print broken, '
                f'dotted grooves. Using {max(n_phi, min_phi)} x {max(n_x, min_x)}.',
                RuntimeWarning, stacklevel=3,
            )
            n_phi, n_x = max(n_phi, min_phi), max(n_x, min_x)
        return n_phi, n_x

    def groove_path_on_surface(self):
        """Return wire groove centerlines as (x, phi) lists for visualization.

        Returns list of (x_arr, phi_arr) both in mm / radians.
        """
        return [(x_arr, phi_arr) for phi_arr, x_arr in self._wire_paths_phi_x]

    # ─────────────────────────────────────────────────────────────────────────
    # STL export
    # ─────────────────────────────────────────────────────────────────────────

    def to_stl_bytes(self, n_phi=None, n_x=None, include_flanges=True):
        """Generate binary STL of the holder as bytes.

        n_phi, n_x : grid size.  None (default) sizes it from the groove radius
        so the grooves stay continuous; coarser explicit values are raised to
        the minimum (see ``export_grid``).

        Water-tight topology (every edge shared by exactly 2 triangles):

        Without flanges:
          outer grooved body  ←→  end caps (R_inner–R_outer)  ←→  inner bore

        With flanges:
          outer grooved body  ←→  step face (R_outer–R_flange) at ±x_half
                                  flange OD band at R_flange
                                  flange end face (R_inner–R_flange) at ±x_flange
          inner bore (full length through flanges)  ←→  flange end face inner ring
        """
        tri_arrays = []
        n_phi, n_x = self.export_grid(n_phi, n_x)
        if self._tails:
            return _write_stl_binary(f"holder_{self.label}",
                                     self._routed_tri_arrays(n_phi, n_x, include_flanges))
        phi_g = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
        has_flanges = include_flanges and self.flange_w > 0
        x_flange = self.x_half + self.flange_w  # total half-length

        # ── Outer grooved body (boundary rows forced to R_outer) ───────────
        # flip=True → normals point outward (+R), grooves appear recessed
        X_o, Y_o, Z_o, _ = self.groove_surface(n_phi=n_phi, n_x=n_x, groove_scale=1.0)
        tri_arrays.append(_surface_to_triangles(X_o, Y_o, Z_o, flip=True))

        # ── Inner bore (extends to ±x_flange when flanges present) ─────────
        # flip=False → normals point inward (−R, toward bore axis)
        # A plain cylinder is ruled, so 2 rings suffice (keeps file size sane).
        x_inner_end = x_flange if has_flanges else self.x_half
        PHI_i, XX_i = np.meshgrid(phi_g, np.linspace(-x_inner_end, x_inner_end, 2))
        Y_i = self.R_inner * np.cos(PHI_i)
        Z_i = self.R_inner * np.sin(PHI_i)
        tri_arrays.append(_surface_to_triangles(XX_i, Y_i, Z_i, flip=False))

        if not has_flanges:
            # ── Simple end caps: R_inner → R_outer at ±x_half ──────────────
            for x_sign in [-1, 1]:
                tri_arrays.append(_annulus_triangles(
                    x_sign * self.x_half, self.R_inner, self.R_outer, n_phi,
                    flip=(x_sign > 0)
                ))
        else:
            for x_sign in [-1, 1]:
                face_x = x_sign * self.x_half
                flange_x = x_sign * x_flange

                # Step face at ±x_half: R_outer → R_flange.  It faces the body
                # (normal −x_sign), the opposite way from the flange end face, so
                # its flip is the opposite too.  With flip=(x_sign > 0) these two
                # rings were wound backwards: the STL was watertight but not a
                # consistently oriented solid (negative-volume patches).
                tri_arrays.append(_annulus_triangles(
                    face_x, self.R_outer, self.R_flange, n_phi,
                    flip=(x_sign < 0)
                ))
                # Flange OD band at R_flange from face_x to flange_x
                # flip=(x_sign > 0): x increases for right flange → outward +R normals
                x_band = np.array([face_x, flange_x])
                PHI_b, XX_b = np.meshgrid(phi_g, x_band)
                Y_b = self.R_flange * np.cos(PHI_b)
                Z_b = self.R_flange * np.sin(PHI_b)
                tri_arrays.append(_surface_to_triangles(XX_b, Y_b, Z_b,
                                                        flip=(x_sign > 0)))
                # Flange end face at ±x_flange: R_inner → R_flange
                # flip=(x_sign > 0) → +X normal at +x_flange, −X at −x_flange
                tri_arrays.append(_annulus_triangles(
                    flange_x, self.R_inner, self.R_flange, n_phi,
                    flip=(x_sign > 0)
                ))

        return _write_stl_binary(f"holder_{self.label}", tri_arrays)

    def _flange_profile(self, phi, end):
        """Outer radius of the flange at one end as a function of angle: the
        flange radius, with a notch at every wire tail on that end."""
        prof = np.full(phi.shape, self.R_flange, dtype=float)
        half = (self.wire_d + NOTCH_EXTRA_WIDTH_MM) / 2.0 / self.R_outer
        for t in self._tails:
            if t['end'] != end:
                continue
            d = np.abs(np.angle(np.exp(1j * (phi - t['phi']))))
            floor = self.R_outer - self.groove_radius - t['extra'] - NOTCH_EXTRA_DEPTH_MM
            prof = np.where(d <= half, np.minimum(prof, floor), prof)
        return prof

    def _routed_tri_arrays(self, n_phi, n_x, include_flanges=True):
        """Triangles of a holder that carries a winding route.

        One height field covers the body and both flanges, so the tail grooves
        can run off the body and through a notch in the flange without any
        boolean cut.  The end faces follow the notched outline.  Closed and
        wound outward like the plain export.
        """
        phi_g = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
        has_flanges = include_flanges and self.flange_w > 0
        X, _, _, depth = self.groove_surface(n_phi=n_phi, n_x=n_x, seal_ends=False)
        x_rows, r_rows = [X[:, 0]], [self.R_outer - depth]
        if has_flanges:
            dx = 2 * self.x_half / max(n_x - 1, 1)
            n_f = max(2, int(math.ceil(self.flange_w / dx)) + 1)
            for end in (-1, 1):
                prof = self._flange_profile(phi_g, end)
                xs = end * np.linspace(self.x_half + FLANGE_STEP_MM, self.x_half + self.flange_w, n_f)
                rr = np.tile(prof, (n_f, 1))
                if end < 0:
                    x_rows.insert(0, xs[::-1]); r_rows.insert(0, rr)
                else:
                    x_rows.append(xs); r_rows.append(rr)
        x_all, r_all = np.concatenate(x_rows), np.vstack(r_rows)
        PHI, XX = np.meshgrid(phi_g, x_all)
        tri = [_surface_to_triangles(XX, r_all * np.cos(PHI), r_all * np.sin(PHI), flip=True)]
        # Inner bore, full length
        PHI_i, XX_i = np.meshgrid(phi_g, np.array([x_all[0], x_all[-1]]))
        tri.append(_surface_to_triangles(XX_i, self.R_inner * np.cos(PHI_i),
                                         self.R_inner * np.sin(PHI_i), flip=False))
        # End faces: from the bore to the (notched) outline of the last row
        tri.append(_annulus_triangles(x_all[0], self.R_inner, r_all[0], n_phi, flip=False))
        tri.append(_annulus_triangles(x_all[-1], self.R_inner, r_all[-1], n_phi, flip=True))
        return tri

    def tails(self):
        """Wire tails of the winding route: list of dicts with 'wire', 'label'
        (START / END), 'end' (-1 or +1) and 'angle_deg' from +Y toward +Z."""
        return [dict(wire=t['wire'], label=t['label'], end=t['end'],
                     angle_deg=float(np.degrees(np.angle(np.exp(1j * t['phi'])))))
                for t in self._tails]

    def save_stl(self, filepath, **kwargs):
        """Write holder STL to file."""
        data = self.to_stl_bytes(**kwargs)
        with open(filepath, 'wb') as f:
            f.write(data)

    # ─────────────────────────────────────────────────────────────────────────
    # Info
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self):
        lines = [
            f"  CoilHolder [{self.label}]",
            f"    Coil radius (groove center): {self.R:.1f} mm",
            f"    Inner / Outer surface:       {self.R_inner:.1f} / {self.R_outer:.1f} mm",
            f"    Length (body / total):       {self.length:.0f} / {self.total_length_mm:.0f} mm",
            f"    Groove radius:               {self.groove_radius:.2f} mm  (wire Ø{self.wire_d} + {self.clearance:.2f} clearance)",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Mesh helpers  (fully vectorised — no Python loops)
# ─────────────────────────────────────────────────────────────────────────────

def _surface_to_triangles(X, Y, Z, flip=False):
    """Tessellate a cylindrical surface grid → (N, 3, 3) float32 triangle array.

    Expects an endpoint=False phi grid (n_phi unique angles).  Adds a
    wrap-around strip that connects the last phi column back to the first,
    closing the cylinder and eliminating non-manifold seam edges.
    """
    # Interior strips: column j → j+1  (shape n_x-1, n_phi-1, 3)
    v00 = np.stack([X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1]], axis=-1)
    v10 = np.stack([X[1:,  :-1], Y[1:,  :-1], Z[1:,  :-1]], axis=-1)
    v01 = np.stack([X[:-1,  1:], Y[:-1,  1:], Z[:-1,  1:]], axis=-1)
    v11 = np.stack([X[1:,   1:], Y[1:,   1:], Z[1:,   1:]], axis=-1)
    # Wrap-around strip: last column → first column  (shape n_x-1, 1, 3)
    w00 = np.stack([X[:-1, -1:], Y[:-1, -1:], Z[:-1, -1:]], axis=-1)
    w10 = np.stack([X[1:,  -1:], Y[1:,  -1:], Z[1:,  -1:]], axis=-1)
    w01 = np.stack([X[:-1,  :1], Y[:-1,  :1], Z[:-1,  :1]], axis=-1)
    w11 = np.stack([X[1:,   :1], Y[1:,   :1], Z[1:,   :1]], axis=-1)
    if not flip:
        t1  = np.stack([v00, v10, v01], axis=2)
        t2  = np.stack([v11, v01, v10], axis=2)
        tw1 = np.stack([w00, w10, w01], axis=2)
        tw2 = np.stack([w11, w01, w10], axis=2)
    else:
        t1  = np.stack([v00, v01, v10], axis=2)
        t2  = np.stack([v11, v10, v01], axis=2)
        tw1 = np.stack([w00, w01, w10], axis=2)
        tw2 = np.stack([w11, w10, w01], axis=2)
    return np.concatenate([
        t1.reshape(-1, 3, 3),  t2.reshape(-1, 3, 3),
        tw1.reshape(-1, 3, 3), tw2.reshape(-1, 3, 3),
    ], axis=0).astype(np.float32)


def _annulus_triangles(x_pos, r_inner, r_outer, n_phi, flip=False):
    """Triangulated annulus at a fixed x position → (N, 3, 3) float32.

    r_outer may be an array of n_phi radii (an outline that varies with angle).
    """
    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    phi_next = np.roll(phi, -1)
    ro = np.broadcast_to(np.asarray(r_outer, dtype=float), phi.shape)
    ro_next = np.roll(ro, -1)
    x = np.full(n_phi, x_pos, dtype=np.float32)
    vi = np.stack([x, r_inner * np.cos(phi),      r_inner * np.sin(phi)],      axis=1)
    vj = np.stack([x, r_inner * np.cos(phi_next), r_inner * np.sin(phi_next)], axis=1)
    vk = np.stack([x, ro * np.cos(phi),           ro * np.sin(phi)],           axis=1)
    vl = np.stack([x, ro_next * np.cos(phi_next), ro_next * np.sin(phi_next)], axis=1)
    if not flip:
        t1 = np.stack([vi, vj, vk], axis=1)
        t2 = np.stack([vl, vk, vj], axis=1)
    else:
        t1 = np.stack([vi, vk, vj], axis=1)
        t2 = np.stack([vl, vj, vk], axis=1)
    return np.concatenate([t1, t2], axis=0).astype(np.float32)


def _write_stl_binary(name: str, tri_arrays) -> bytes:
    """Pack triangle arrays as binary STL.

    tri_arrays : list of (N, 3, 3) float32 ndarrays returned by the helpers above.
    """
    all_tris = np.concatenate(tri_arrays, axis=0)   # (N, 3, 3) float32
    N = len(all_tris)

    # Vectorised normals
    e1 = all_tris[:, 1] - all_tris[:, 0]
    e2 = all_tris[:, 2] - all_tris[:, 0]
    normals = np.cross(e1, e2).astype(np.float32)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    normals /= lens

    # Binary STL record: 12 × float32 + 1 × uint16  = 50 bytes each
    dt = np.dtype([
        ('n',  '<f4', (3,)),
        ('v0', '<f4', (3,)),
        ('v1', '<f4', (3,)),
        ('v2', '<f4', (3,)),
        ('attr', '<u2'),
    ])
    recs = np.zeros(N, dtype=dt)
    recs['n']  = normals
    recs['v0'] = all_tris[:, 0]
    recs['v1'] = all_tris[:, 1]
    recs['v2'] = all_tris[:, 2]

    buf = io.BytesIO()
    buf.write(name.encode()[:80].ljust(80, b'\x00'))
    buf.write(struct.pack('<I', N))
    buf.write(recs.tobytes())
    return buf.getvalue()
