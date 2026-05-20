"""
Cylindrical coil holder with recessed wire grooves.

Each holder is a thin-walled cylinder with:
  - Wire grooves recessed on the outer surface (wires sit flush or proud)
  - Optional end flanges for stacking/alignment
  - Compatible with Blender Boolean workflow (groove cutter meshes)

Units: millimeters throughout (unlike tfm.py which uses meters).
"""

import numpy as np
import struct
import io
from scipy.spatial import cKDTree


class CoilHolder:
    """Cylindrical holder with recessed outer-surface wire grooves.

    Parameters
    ----------
    coil_radius_mm : float
        Radius at which the wire centerline sits (groove center radius).
    wall_mm : float
        Holder wall thickness. The outer surface is at coil_radius_mm + wall_mm/2,
        inner surface at coil_radius_mm - wall_mm/2.
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

        self.R_inner = coil_radius_mm - wall_mm / 2
        self.R_outer = coil_radius_mm + wall_mm / 2
        self.groove_radius = wire_diameter_mm / 2 + groove_clearance_mm
        self.x_half = length_mm / 2          # half-length of main cylinder body

        self._wire_paths_phi_x = []           # list of (phi_arr, x_arr) in radians/mm

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

    def groove_surface(self, n_phi=300, n_x=200, groove_scale=3.0, max_pts_per_wire=400):
        """Compute the outer surface mesh with grooves indented.

        Parameters
        ----------
        n_phi, n_x : int
            Grid resolution.  Higher = smoother grooves.
        groove_scale : float
            Multiplier on groove_radius for visual depth/width.
        max_pts_per_wire : int
            Max samples per wire path fed to the KD-tree.

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
            # Duplicate with ±2π offsets so wrap-around is handled correctly.
            arc_pts, x_pts = [], []
            for phi_arr, x_arr in self._wire_paths_phi_x:
                step = max(1, len(phi_arr) // max_pts_per_wire)
                phi_s = phi_arr[::step]
                x_s   = x_arr[::step]
                for offset in (0.0, 2 * np.pi, -2 * np.pi):
                    arc_pts.append(self.R_outer * (phi_s + offset))
                    x_pts.append(x_s)

            wire_pts = np.column_stack([np.concatenate(arc_pts),
                                        np.concatenate(x_pts)])   # (N, 2)
            tree = cKDTree(wire_pts)

            # Query every grid point
            arc_flat  = (self.R_outer * PHI).ravel()
            grid_pts  = np.column_stack([arc_flat, XX.ravel()])    # (M, 2)
            min_dist, _ = tree.query(grid_pts, k=1, workers=-1)

            dm_flat = np.where(min_dist < gr, gr - min_dist, 0.0)
            depth_map = dm_flat.reshape(PHI.shape)

        R_surface = self.R_outer - depth_map
        # Force boundary rows to exact R_outer so end-cap annuli seal flush
        R_surface[0, :]  = self.R_outer
        R_surface[-1, :] = self.R_outer
        X = XX
        Y = R_surface * np.cos(PHI)
        Z = R_surface * np.sin(PHI)
        return X, Y, Z, depth_map

    def groove_path_on_surface(self):
        """Return wire groove centerlines as (x, phi) lists for visualization.

        Returns list of (x_arr, phi_arr) both in mm / radians.
        """
        return [(x_arr, phi_arr) for phi_arr, x_arr in self._wire_paths_phi_x]

    # ─────────────────────────────────────────────────────────────────────────
    # STL export
    # ─────────────────────────────────────────────────────────────────────────

    def to_stl_bytes(self, n_phi=1200, n_x=800, include_flanges=True):
        """Generate binary STL of the holder as bytes.

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
        phi_g = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
        has_flanges = include_flanges and self.flange_w > 0
        x_flange = self.x_half + self.flange_w  # total half-length

        # ── Outer grooved body (boundary rows forced to R_outer) ───────────
        # flip=True → normals point outward (+R), grooves appear recessed
        X_o, Y_o, Z_o, _ = self.groove_surface(n_phi=n_phi, n_x=n_x, groove_scale=3.0)
        tri_arrays.append(_surface_to_triangles(X_o, Y_o, Z_o, flip=True))

        # ── Inner bore (extends to ±x_flange when flanges present) ─────────
        # flip=False → normals point inward (−R, toward bore axis)
        x_inner_end = x_flange if has_flanges else self.x_half
        PHI_i, XX_i = np.meshgrid(phi_g, np.linspace(-x_inner_end, x_inner_end, n_x))
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

                # Step face at ±x_half: R_outer → R_flange
                tri_arrays.append(_annulus_triangles(
                    face_x, self.R_outer, self.R_flange, n_phi,
                    flip=(x_sign > 0)
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
    """Triangulated annulus at a fixed x position → (N, 3, 3) float32."""
    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    phi_next = np.roll(phi, -1)
    x = np.full(n_phi, x_pos, dtype=np.float32)
    vi = np.stack([x, r_inner * np.cos(phi),      r_inner * np.sin(phi)],      axis=1)
    vj = np.stack([x, r_inner * np.cos(phi_next), r_inner * np.sin(phi_next)], axis=1)
    vk = np.stack([x, r_outer * np.cos(phi),      r_outer * np.sin(phi)],      axis=1)
    vl = np.stack([x, r_outer * np.cos(phi_next), r_outer * np.sin(phi_next)], axis=1)
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
