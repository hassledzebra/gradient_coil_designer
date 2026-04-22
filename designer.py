"""
GradientCoilDesigner -- universal parameterized gradient coil designer.

Coordinate convention (Halbach MRI):
  X = bore axis (axial)
  Y = B0 direction
  Z = perpendicular transverse

Usage
-----
>>> from gradient_coil_designer import GradientCoilDesigner
>>> d = GradientCoilDesigner(bore_radius_mm=78, linear_region_mm=40)
>>> result = d.run(['Gx', 'Gy', 'Gz'])
>>> d.report(result)
>>> d.save_all(result, output_dir='./output')
"""

import os
import json
import numpy as np

from . import tfm
from .holder import CoilHolder
from . import physics as phys


class GradientCoilDesigner:
    """Universal gradient coil designer for cylindrical bore MRI/NMR magnets.

    The designer uses the Turner Target Field Method (1986) to compute
    wire winding paths for Gx, Gy, Gz gradient coils.  It then wraps each
    coil in a cylindrical holder with recessed outer-surface wire grooves.
    Holders are nested concentrically with a user-specified radial gap.

    Parameters
    ----------
    bore_radius_mm : float
        Inner bore radius (mm).  The outermost coil/holder must fit inside.
    linear_region_mm : float
        Target DSV diameter (mm) for the linear imaging region.
    holder_length_mm : float
        Total axial length of the holders (mm).
    wire_diameter_mm : float
        Wire/Litz cable outer diameter (mm).
    holder_wall_mm : float
        Radial wall thickness of each holder cylinder (mm).
    layer_gap_mm : float
        Radial clearance between adjacent holder outer/inner surfaces (mm).
    flange_width_mm : float
        Width of end flanges on each holder (mm).  Set 0 to omit.
    flange_thick_mm : float
        Radial thickness of the end flanges beyond the outer wall (mm).
    n_wires : int
        Number of wire turns per quadrant (affects gradient efficiency).
    n_high_orders : int
        TFM higher-order correction terms (Gx only; 8–12 is typical).
    linearity_order : int
        Order of the linearity shaping polynomial (8–20).
    apodization : float
        Apodization coefficient (0.02–0.10).  Higher = smoother but lower G.
    coil_length_factor : float
        coil_length_mm = coil_length_factor * linear_region_mm (Gy/Gz).
    groove_clearance_mm : float
        Extra clearance added to groove radius for wire fit.
    verify_field : bool
        If True, run Biot-Savart verification after design.
    verify_half_range_mm : float
        Sampling half-range for Biot-Savart profile (mm).
    verify_n_pts : int
        Number of sample points for the profile.
    """

    def __init__(
        self,
        bore_radius_mm: float = 78.0,
        linear_region_mm: float = 40.0,
        holder_length_mm: float = 145.0,
        wire_diameter_mm: float = 1.5,
        holder_wall_mm: float = 3.0,
        layer_gap_mm: float = 3.0,
        flange_width_mm: float = 5.0,
        flange_thick_mm: float = 5.0,
        n_wires: int = 10,
        n_high_orders: int = 10,
        linearity_order: int = 16,
        apodization: float = 0.05,
        coil_length_factor: float = 3.0,
        groove_clearance_mm: float = 0.1,
        verify_field: bool = True,
        verify_half_range_mm: float = 30.0,
        verify_n_pts: int = 21,
    ):
        self.bore_radius = bore_radius_mm
        self.linear_region = linear_region_mm
        self.holder_length = holder_length_mm
        self.wire_d = wire_diameter_mm
        self.wall = holder_wall_mm
        self.gap = layer_gap_mm
        self.flange_w = flange_width_mm
        self.flange_t = flange_thick_mm
        self.n_wires = n_wires
        self.n_high_orders = n_high_orders
        self.linearity_order = linearity_order
        self.apodization = apodization
        self.coil_length_factor = coil_length_factor
        self.groove_clearance = groove_clearance_mm
        self.do_verify = verify_field
        self.verify_range = verify_half_range_mm
        self.verify_n_pts = verify_n_pts

        # Compute coil radii based on bore, stacking from outside in
        # Stack order: Gx (outermost) -> Gy -> Gz (innermost)
        # R_coil[k] = bore_radius - k*(wall + gap + wire_d)
        self._coil_radii = self._compute_layer_radii(['Gx', 'Gy', 'Gz'])

    def _compute_layer_radii(self, axes):
        """Compute coil wire-centerline radii from outside in."""
        radii = {}
        R = self.bore_radius - self.wall / 2   # outermost coil radius = bore - half wall
        for i, ax in enumerate(axes):
            radii[ax] = R - i * (self.wall + self.gap + self.wire_d)
        return radii

    def _coil_length_mm(self, axis):
        """FFT computation domain length (Gy/Gz need longer domain for sidelobes)."""
        if axis == 'Gx':
            # TFM Gx: coilLength ≈ linear_region + margin; wire extent = ±coilLength
            return self.holder_length / 2 * 0.95
        else:
            return self.coil_length_factor * self.linear_region

    def design_axis(self, axis: str):
        """Compute wire paths for a single gradient axis.

        Parameters
        ----------
        axis : 'Gx', 'Gy', or 'Gz'

        Returns
        -------
        dict with keys:
          'wires_3d'          -- list of wire dicts (from tfm.wire_paths_to_3d)
          'contour_set'       -- raw matplotlib contour object
          'coil_radius_mm'    -- radius used
          'efficiency_mT_m_A' -- gradient efficiency
          'wire_length_m'     -- total wire length
          'axial_extent_mm'   -- (min_x, max_x) tuple
          'profile'           -- Biot-Savart profile dict (if verify_field=True)
          'holder'            -- CoilHolder instance
        """
        R = self._coil_radii[axis]
        coil_len = self._coil_length_mm(axis)
        common = dict(
            coil_radius_mm=R,
            linear_region_mm=self.linear_region,
            coil_length_mm=coil_len,
            num_wires=self.n_wires,
            linearity_order=self.linearity_order,
            apodization=self.apodization,
        )

        print(f'  Computing {axis} (R={R:.1f}mm, linear={self.linear_region}mm)...',
              end='', flush=True)

        if axis == 'Gx':
            wire_paths, cs = tfm.compute_gx_stream(
                **common, num_high_orders=self.n_high_orders
            )
        elif axis == 'Gy':
            wire_paths, cs = tfm.compute_gy_stream(**common)
        elif axis == 'Gz':
            wire_paths, cs = tfm.compute_gz_stream(**common)
        else:
            raise ValueError(f'Unknown axis: {axis!r}. Use Gx, Gy, or Gz.')

        wires_3d = tfm.wire_paths_to_3d(wire_paths, R)
        eff = tfm.gradient_efficiency(cs)
        wl = tfm.wire_length_m(wires_3d)
        ext = tfm.wire_axial_extent_mm(wires_3d)

        print(f' {len(wires_3d)} segments, +/-{max(abs(ext[0]),abs(ext[1])):.0f}mm extent, '
              f'{eff:.2f} mT/m/A', flush=True)

        # Build holder
        holder = CoilHolder(
            coil_radius_mm=R,
            wall_mm=self.wall,
            length_mm=self.holder_length,
            flange_width_mm=self.flange_w,
            flange_thick_mm=self.flange_t,
            wire_diameter_mm=self.wire_d,
            groove_clearance_mm=self.groove_clearance,
            label=axis,
        )
        holder.add_wire_paths(wires_3d)

        # Biot-Savart verification
        # Symmetry rules (B0=Y, bore=X):
        #   Gx: By != 0 on x-axis (y=z=0)  -> sample along x
        #   Gy: By != 0 on y-axis (x=z=0)  -> sample along y
        #   Gz: By = 0 on y=0 plane         -> offset to y = R/3
        profile = None
        if self.do_verify:
            profile_axis = axis[-1].lower()   # 'x', 'y', 'z'
            offset = np.zeros(3)
            if axis == 'Gz':
                offset[1] = R * 1e-3 / 3.0   # y-offset so By != 0 by symmetry
            print(f'    Verifying field along {profile_axis.upper()} axis...', end='', flush=True)
            coords, By, slope, lin_err = phys.compute_gradient_profile(
                wires_3d,
                axis=profile_axis,
                half_range_mm=self.verify_range,
                n_pts=self.verify_n_pts,
                offset=offset,
            )
            profile = {
                'coords_mm': coords,
                'By_T': By,
                'gradient_mT_m_A': slope,
                'linearity_pct': lin_err,
            }
            print(f' {slope:.3f} mT/m/A, linearity error {lin_err:.1f}%', flush=True)

        return {
            'wires_3d': wires_3d,
            'contour_set': cs,
            'coil_radius_mm': R,
            'efficiency_mT_m_A': eff,
            'wire_length_m': wl,
            'axial_extent_mm': ext,
            'profile': profile,
            'holder': holder,
        }

    def run(self, axes=None):
        """Design all requested gradient axes.

        Parameters
        ----------
        axes : list of str, default ['Gx', 'Gy', 'Gz']

        Returns
        -------
        dict with:
          'coils'   -- {axis: design_dict}
          'title'   -- summary title
          'params'  -- designer parameters
        """
        if axes is None:
            axes = ['Gx', 'Gy', 'Gz']

        print('=' * 65)
        print('  GRADIENT COIL DESIGNER -- Target Field Method (Turner 1986)')
        print('=' * 65)
        print(f'  Bore radius:      {self.bore_radius:.1f} mm')
        print(f'  Linear region:    {self.linear_region:.1f} mm (target DSV)')
        print(f'  Holder length:    {self.holder_length:.0f} mm')
        print(f'  Wire diameter:    {self.wire_d} mm')
        print(f'  Wall / Gap:       {self.wall} / {self.gap} mm')
        print(f'  Wires/quadrant:   {self.n_wires}')
        print()

        coils = {}
        for ax in axes:
            coils[ax] = self.design_axis(ax)

        result = {
            'coils': coils,
            'title': f'Gradient Coils  bore={self.bore_radius:.0f}mm  DSV={self.linear_region:.0f}mm',
            'params': self._params_dict(),
        }
        self.report(result)
        return result

    def report(self, result):
        """Print a formatted summary table."""
        coils = result['coils']
        print()
        print('-' * 65)
        print('  DESIGN SUMMARY')
        print('-' * 65)
        print(f"  {'Axis':<6}{'R (mm)':<10}{'Efficiency':>14}{'Wire (m)':>12}"
              f"{'Extent mm':>14}{'Linearity':>12}")
        print('  ' + '-' * 63)

        copper_rho = 1.68e-8
        litz_fill = 0.55
        wire_area = np.pi * (self.wire_d * 1e-3 / 2)**2 * litz_fill

        for lbl, info in coils.items():
            R = info['coil_radius_mm']
            eff = info['efficiency_mT_m_A']
            wl = info['wire_length_m']
            ext = info['axial_extent_mm']
            R_ohm = copper_rho * wl / wire_area
            profile = info.get('profile')
            lin_str = f"{profile['linearity_pct']:.1f}%" if profile else '-'
            extent_str = f"+/-{max(abs(ext[0]),abs(ext[1])):.0f}"
            print(f"  {lbl:<6}{R:<10.1f}{eff:>13.3f}{'mT/m/A':>1}"
                  f"{wl:>11.2f}m{extent_str:>14}mm{lin_str:>10}")
            print(f"  {'':6}holder OD/ID: {info['holder'].R_outer*2:.0f}/{info['holder'].R_inner*2:.0f} mm"
                  f"  R={R_ohm:.4f} Ohm")

        print()
        # Nesting gaps
        ax_list = list(coils.keys())
        for i in range(len(ax_list) - 1):
            outer = coils[ax_list[i]]['holder']
            inner = coils[ax_list[i+1]]['holder']
            gap = outer.R_inner - inner.R_outer
            print(f"  Radial gap {ax_list[i]}->{ax_list[i+1]}: {gap:.1f} mm")
        print('=' * 65)

    def save_all(self, result, output_dir='.'):
        """Export wire CSVs, holder STLs, and a JSON summary.

        Files written:
          {axis}_wires.csv    -- wire path points
          {axis}_holder.stl   -- holder 3D mesh
          design_summary.json -- all parameters and metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        summary = {'params': result['params'], 'coils': {}}

        for lbl, info in result['coils'].items():
            # Wire CSV
            csv_path = os.path.join(output_dir, f'{lbl}_wires.csv')
            rows = []
            for w in info['wires_3d']:
                for pt in w['xyz']:
                    rows.append([*pt * 1000, w['sign']])   # mm, sign
            np.savetxt(csv_path,
                       np.array(rows),
                       delimiter=',',
                       header='x_mm,y_mm,z_mm,sign',
                       comments='',
                       fmt=['%.4f', '%.4f', '%.4f', '%d'])

            # Holder STL
            stl_path = os.path.join(output_dir, f'{lbl}_holder.stl')
            try:
                info['holder'].save_stl(stl_path)
                stl_note = stl_path
            except Exception as e:
                stl_note = f'FAILED: {e}'

            profile = info.get('profile')
            summary['coils'][lbl] = {
                'coil_radius_mm': info['coil_radius_mm'],
                'efficiency_mT_m_A': info['efficiency_mT_m_A'],
                'wire_length_m': info['wire_length_m'],
                'axial_extent_mm': list(info['axial_extent_mm']),
                'gradient_mT_m_A_BS': profile['gradient_mT_m_A'] if profile else None,
                'linearity_pct': profile['linearity_pct'] if profile else None,
                'stl': stl_note,
                'csv': csv_path,
            }
            print(f'  {lbl}: wires -> {csv_path}')
            print(f'  {lbl}: holder -> {stl_note}')

        json_path = os.path.join(output_dir, 'design_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'  Summary -> {json_path}')

    def _params_dict(self):
        return {
            'bore_radius_mm': self.bore_radius,
            'linear_region_mm': self.linear_region,
            'holder_length_mm': self.holder_length,
            'wire_diameter_mm': self.wire_d,
            'holder_wall_mm': self.wall,
            'layer_gap_mm': self.gap,
            'flange_width_mm': self.flange_w,
            'n_wires': self.n_wires,
            'n_high_orders': self.n_high_orders,
            'linearity_order': self.linearity_order,
            'apodization': self.apodization,
        }
