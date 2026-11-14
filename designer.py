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
import warnings

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
        Half-extent (radius) of the target linear region, in mm -- NOT the
        diameter.  This is the ``d`` that shapes the axial target profile in
        ``tfm``, so a value of 40 asks for linearity over |x| <= 40 mm, an
        80 mm DSV.  It also sets the default radius of the DSV over which
        linearity is scored.
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
        Gaussian k-space smoothing length in metres.  This is the
        regularization: the target-field inverse kernel grows like exp(k*a)
        (~1e100 across the band), so too small a value amplifies high-k noise
        and too large a value collapses the pattern to a single sinusoid.
        0.012-0.020 is the usable range for a bore of this size; ``tfm`` warns
        outside it.  (Values around 0.05 were right for the pre-fix code, which
        used a mis-scaled k axis -- they over-smooth badly now.)
    coil_length_factor : float
        coil_length_mm = coil_length_factor * linear_region_mm (Gy/Gz).
    groove_clearance_mm : float
        Extra clearance added to groove radius for wire fit.
    verify_field : bool
        If True, run Biot-Savart verification after design.
    dsv_radius_mm : float or None
        Radius of the sphere over which gradient efficiency and linearity are
        scored.  Defaults to ``linear_region_mm``.
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
        apodization: float = 0.015,
        coil_length_factor: float = 3.0,
        groove_clearance_mm: float = 0.1,
        verify_field: bool = True,
        dsv_radius_mm: float = None,
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
        self.dsv_radius = (linear_region_mm if dsv_radius_mm is None
                           else dsv_radius_mm)
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
          'efficiency_mT_m_A' -- gradient efficiency from the 3-D DSV fit
          'dsv'               -- full DSV metric dict (physics.dsv_metrics)
          'amperes_per_turn'  -- current per turn for a unit stream function
          'open_paths'        -- contours that failed to close (should be 0)
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

        n_open = tfm.open_path_count(wire_paths)
        if n_open:
            warnings.warn(
                f'{axis}: {n_open} of {len(wire_paths)} contours do not close. '
                f'Open paths are not valid current loops, so the field results '
                f'below are not trustworthy.',
                RuntimeWarning, stacklevel=2,
            )

        wires_3d = tfm.wire_paths_to_3d(wire_paths, R)
        i_turn = tfm.amperes_per_turn(cs)
        wl = tfm.wire_length_m(wires_3d)
        ext = tfm.wire_axial_extent_mm(wires_3d)

        print(f' {len(wires_3d)} loops, +/-{max(abs(ext[0]),abs(ext[1])):.0f}mm extent',
              flush=True)

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
        dsv = None
        if self.do_verify:
            profile_axis = axis[-1].lower()   # 'x', 'y', 'z'

            # Volume metric first: this is the number that decides whether the
            # design is good.  A line profile can look linear while the field
            # bows off-axis, so efficiency is taken from the 3-D fit, not from
            # contour spacing (which is amperes per turn, not mT/m/A).
            print(f'    Scoring over {self.dsv_radius:.0f}mm-radius DSV...',
                  end='', flush=True)
            dsv = phys.dsv_metrics(wires_3d, profile_axis, self.dsv_radius)
            print(f" {dsv['efficiency_mT_m_A']:.3f} mT/m/A, "
                  f"{dsv['max_deviation_pct']:.1f}% max deviation", flush=True)

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
            'efficiency_mT_m_A': (dsv['efficiency_mT_m_A'] if dsv else float('nan')),
            'amperes_per_turn': i_turn,
            'open_paths': n_open,
            'dsv': dsv,
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
        print(f"  {'Axis':<6}{'R mm':>7}{'mT/m/A':>10}{'DSV max':>10}{'DSV rms':>10}"
              f"{'Wire m':>9}{'Extent mm':>12}{'Line':>8}")
        print('  ' + '-' * 63)

        copper_rho = 1.68e-8
        litz_fill = 0.55
        wire_area = np.pi * (self.wire_d * 1e-3 / 2)**2 * litz_fill

        for lbl, info in coils.items():
            R = info['coil_radius_mm']
            wl = info['wire_length_m']
            ext = info['axial_extent_mm']
            R_ohm = copper_rho * wl / wire_area
            dsv = info.get('dsv')
            profile = info.get('profile')

            eff_str = f"{dsv['efficiency_mT_m_A']:.3f}" if dsv else '-'
            max_str = f"{dsv['max_deviation_pct']:.1f}%" if dsv else '-'
            rms_str = f"{dsv['rms_deviation_pct']:.1f}%" if dsv else '-'
            # Kept only to show how much the single-line metric flatters the
            # design relative to the volume metric next to it.
            lin_str = f"{profile['linearity_pct']:.1f}%" if profile else '-'
            extent_str = f"+/-{max(abs(ext[0]),abs(ext[1])):.0f}"

            print(f"  {lbl:<6}{R:>7.1f}{eff_str:>10}{max_str:>10}{rms_str:>10}"
                  f"{wl:>9.2f}{extent_str:>12}{lin_str:>8}")
            flag = f"  !! {info['open_paths']} OPEN LOOPS" if info.get('open_paths') else ''
            print(f"  {'':6}holder OD/ID: {info['holder'].R_outer*2:.0f}/{info['holder'].R_inner*2:.0f} mm"
                  f"  R={R_ohm:.4f} Ohm{flag}")

        dsv_r = next((c['dsv']['dsv_radius_mm'] for c in coils.values() if c.get('dsv')), None)
        if dsv_r is not None:
            print()
            print(f"  DSV max/rms: worst and rms departure from a perfect linear field")
            print(f"  over a {dsv_r:.0f} mm-radius sphere, as % of the ideal field at its edge.")
            print(f"  'Line' is the old single-axis metric, shown for comparison only.")

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
                'amperes_per_turn': info.get('amperes_per_turn'),
                'open_paths': info.get('open_paths'),
                'dsv': info.get('dsv'),
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
            'dsv_radius_mm': self.dsv_radius,
        }
