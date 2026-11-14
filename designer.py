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
from . import winding as wd


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
    winding : 'spiral' or 'loops'
        'spiral' (default): the loops are joined into a route one wire can
        follow without leaving a groove (see ``winding.py``): one spiral per
        lobe, a lead-in channel one wire deeper that passes under the turns,
        jumper grooves between lobes, and a flange notch for each wire tail.
        Gx becomes two wires (one per end), Gy and Gz one wire each.  The
        field check, wire length and CSV export then use the route, i.e. the
        coil as it will be wound.  'loops': one closed ring groove per contour,
        the earlier behaviour.
    auto_fit : bool
        True (default): the designer changes what it must so that every holder
        comes out with grooves a wire can follow, and says what it changed
        (printed, in ``result['coils'][axis]['adjustments']`` and in the JSON
        summary).  In this order:
          - wall: with winding='spiral' a wall too thin for the lead-in channel
            is raised to 1.5 * wire + 2 * clearance + 0.9 mm;
          - coil ends: where turns that are cut off at the end of the coil
            return closer together than a wire needs (wire + clearance), Gy and
            Gz first use spare holder length for the return, which keeps the
            field; then the band the turns return in is widened inward, up to
            30 % of the coil length (costs up to about 20 % of the gradient
            strength and 2 points of linearity); a Gy / Gz pattern longer than
            the holder is cut to fit it;
          - stray slivers and small secondary lobes that the route cannot reach
            are left out;
          - Gx coil length: shortened when the jumper grooves have no room
            beyond the loops;
          - turns: only if turns are still too close (for instance at the
            middle of a dense Gx), or the loops cannot be joined into a route,
            the number of turns per quadrant is reduced one at a time for that
            axis.  This lowers the gradient per ampere in proportion.
        False: nothing is changed; turns that are too close give a warning and
        an impossible route raises ``winding.RouteError``.
    stream_resolution_mm : float
        Grid spacing of the stream function, both along the axis and round the
        circumference (mm).  The wire contours are traced on this grid.  The
        default of 1.0 is fine for a first look; on coils under about 50 mm
        radius it moves the contours by up to 0.6-0.7 mm compared with a
        0.25 mm grid, so use 0.25-0.5 for a holder you intend to print.
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
        stream_resolution_mm: float = 1.0,
        winding: str = 'spiral',
        auto_fit: bool = True,
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
        if not stream_resolution_mm > 0:
            raise ValueError('stream_resolution_mm must be positive')
        self.stream_resolution = stream_resolution_mm
        if winding not in ('loops', 'spiral'):
            raise ValueError("winding must be 'loops' or 'spiral'")
        self.winding = winding
        self.auto_fit = bool(auto_fit)
        self.adjustments = []
        if winding == 'spiral' and self.auto_fit:
            st = wd.RouteSettings(wire_diameter_mm, groove_clearance_mm)
            need = float(np.ceil((st.groove_radius + st.deep + 0.8) * 10 - 1e-6) / 10)
            if self.wall < need - 1e-9:
                self.adjustments.append(
                    f'wall raised from {self.wall:.2f} to {need:.2f} mm for the lead-in channel')
                self.wall = need

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
          'wires_3d'          -- list of wire dicts: the closed loops, or with
                                 winding='spiral' the routed wires
          'loops_3d'          -- the closed loops (always)
          'routes'            -- winding routes (winding='spiral'), else None
          'min_loop_spacing_mm' -- closest approach of two loops (inf if > 4 mm)
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
        if axis not in ('Gx', 'Gy', 'Gz'):
            raise ValueError(f'Unknown axis: {axis!r}. Use Gx, Gy, or Gz.')
        R = self._coil_radii[axis]
        print(f'  Computing {axis} (R={R:.1f}mm, linear={self.linear_region}mm)...',
              end='', flush=True)

        fit = self._fit_axis(axis, R)
        wire_paths, cs, routes = fit['wire_paths'], fit['cs'], fit['routes']
        n_open, spacing, notes = fit['n_open'], fit['spacing'], fit['notes']
        pitch = self.min_pitch_mm()

        if n_open:
            warnings.warn(
                f'{axis}: {n_open} of {len(wire_paths)} contours do not close. '
                f'Open paths are not valid current loops, so the field results '
                f'below are not trustworthy.',
                RuntimeWarning, stacklevel=2,
            )
        # Design rule: two turns need a wire and a clearance between their centres.
        if spacing < pitch:
            warnings.warn(
                f'{axis}: two loops come within {spacing:.2f} mm of each other, less than '
                f'the {pitch:.2f} mm a {self.wire_d} mm wire needs, so their grooves merge '
                f'there and the wires cannot lie side by side. Use fewer wires, a thinner '
                f'wire or a longer coil (auto_fit=True does this for you).',
                RuntimeWarning, stacklevel=2,
            )

        loops_3d = tfm.wire_paths_to_3d(wire_paths, R)
        i_turn = tfm.amperes_per_turn(cs)
        ext = tfm.wire_axial_extent_mm(loops_3d)
        wires_3d = wd.route_to_wires_3d(routes, R) if routes else loops_3d
        wl = tfm.wire_length_m(wires_3d)

        joined = f' joined into {len(routes)} wire(s)' if routes else ''
        print(f' {len(loops_3d)} loops{joined}, +/-{max(abs(ext[0]),abs(ext[1])):.0f}mm extent',
              flush=True)
        for note in notes:
            print(f'    adjusted: {note}', flush=True)

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
        if routes:
            holder.add_route(routes)
        else:
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
            'loops_3d': loops_3d,
            'routes': routes,
            'min_loop_spacing_mm': spacing,
            'n_wires_used': fit['n_wires'],
            'adjustments': list(self.adjustments) + notes,
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

    def min_pitch_mm(self):
        """Closest two turns may lie: a wire and one clearance.  Their grooves
        then overlap by one clearance and a ridge is left between the wires."""
        return self.wire_d + self.groove_clearance

    def _contours(self, axis, R, coil_len, n_wires, crop, taper):
        common = dict(
            coil_radius_mm=R,
            linear_region_mm=self.linear_region,
            coil_length_mm=coil_len,
            num_wires=n_wires,
            linearity_order=self.linearity_order,
            apodization=self.apodization,
            resolution=self.stream_resolution * 1e-3,
            taper_frac=taper,
        )
        if axis == 'Gx':
            return tfm.compute_gx_stream(**common, num_high_orders=self.n_high_orders)
        if axis == 'Gy':
            return tfm.compute_gy_stream(**common, crop_factor=crop)
        return tfm.compute_gz_stream(**common, crop_factor=crop)

    # End-taper widths tried after the plain 4 %, as fractions of the coil length.
    # A wider taper lets the cut-off turns return further apart.  It costs some
    # gradient strength and a little linearity (about 20 % and 2 points at
    # 30 %), which is far less than giving up turns costs.
    _TAPERS = (0.07, 0.10, 0.15, 0.20, 0.25, 0.30)

    def _end_stages(self, axis, st):
        """Ways of closing the turns at the ends of the coil, mildest first, as
        (crop_factor, taper_frac, description).  The first is the plain design.

        Gy and Gz are cut off at 1.5 x the linear region, where the stream
        function is still large, so many turns have to return in the end
        taper.  If the former is longer than that, the pattern is kept further
        out and the taper covers the added length: the turns return over a
        wider band and nothing else about the design changes.  After that
        (and for Gx from the start) the taper is widened inward.  If the former
        is shorter than the pattern, the pattern is cut to fit it.
        """
        d = self.linear_region
        if axis == 'Gx':
            return [(1.5, 0.04, '')] + [(1.5, t, f'end taper widened to {t*100:.0f} % of the coil length')
                                        for t in self._TAPERS]
        reach = self.holder_length / 2 - (st.m2 + st.groove_radius + st.end_land + 0.5)
        c_max = min(reach / d, 0.98 * 2 * self._coil_length_mm(axis) / d)
        band = lambda c: (c - 1.44) / (2 * c)
        if c_max < 1.5:
            c = max(c_max, 0.3)
            fit = f'pattern cut to +/-{c * d:.0f} mm to fit the holder (the design asks for +/-{1.5 * d:.0f} mm)'
            return [(c, 0.04, fit)] + [(c, t, fit + f'; end taper widened to {t*100:.0f} % of the coil length')
                                        for t in self._TAPERS]
        stages = [(1.5, 0.04, '')]
        crops = sorted({min(c, c_max) for c in (1.75, 2.0, 2.5, 3.0)}) if c_max >= 1.6 else []
        for c in crops:
            stages.append((c, band(c), f'turns return over +/-{c * d:.0f} mm instead of +/-{1.5 * d:.0f} mm'))
        for c in ([crops[0], crops[-1]] if len(crops) > 1 else crops or [1.5]):
            t0 = band(c) if crops else 0.04
            for extra in (0.03, 0.06, 0.11, 0.16, 0.21):
                if t0 + extra > 0.36:
                    continue
                note = f'end taper widened to {(t0 + extra)*100:.0f} % of the coil length'
                if crops:
                    note = f'turns return over +/-{c * d:.0f} mm; ' + note
                stages.append((c, t0 + extra, note))
        return stages

    def _try(self, axis, R, st, n_w, stage, coil_len):
        """One candidate design.  Returns (design dict, problem) where problem is
        None, or (kind, text) with kind 'ends' (another way of closing the ends
        may help), 'middle' (only fewer turns can help) or 'length' (Gx: text
        is how much to shorten the coil by)."""
        crop, taper, _ = stage
        pitch = self.min_pitch_mm()
        wire_paths, cs = self._contours(axis, R, coil_len, n_w, crop, taper)
        n_open = tfm.open_path_count(wire_paths)
        n_stray = 0
        if self.auto_fit and not n_open:
            wire_paths, n_stray = wd.drop_stray_loops(wire_paths, R)
        spacing, where = ((float('nan'), 0.0) if n_open or not wire_paths
                          else wd.loop_spacing_mm(wire_paths, R, where=True))
        out = dict(wire_paths=wire_paths, cs=cs, routes=None, n_open=n_open, spacing=spacing,
                   n_wires=n_w, crop=crop, taper=taper, coil_length_mm=coil_len, n_stray=n_stray,
                   n_stacked=0)
        if not wire_paths:
            return out, ('middle', 'no loops were found')
        if n_open:
            return out, (('ends', f'{n_open} loops did not close') if self.auto_fit else None)
        if self.auto_fit and spacing < pitch:
            x_far = max(np.abs(p_[:, 1]).max() for _, p_ in wire_paths) * 1e3
            if abs(where) > 0.7 * x_far:
                return out, ('ends', f'turns were {spacing:.2f} mm apart at the end of the coil')
            return out, ('middle', f'turns were {spacing:.2f} mm apart near the middle of the coil')
        if self.winding != 'spiral':
            return out, None
        floor = self.wall - (st.groove_radius + st.deep)
        if floor < 0.8 - 1e-6:
            raise wd.RouteError(
                f'{axis}: the lead-in channel is {st.groove_radius + st.deep:.2f} mm deep and the '
                f'wall is {self.wall:.2f} mm, which leaves {floor:.2f} mm under it. Use a wall of at '
                f'least {np.ceil((st.groove_radius + st.deep + 0.8) * 10 - 1e-6) / 10:.1f} mm for '
                f'winding="spiral".', kind='wall')
        try:
            if self.auto_fit:
                wire_paths, out['n_stacked'] = wd.drop_stacked_lobes(wire_paths, R, self.holder_length)
                if out['n_stacked']:
                    out['wire_paths'] = wire_paths
                    out['spacing'] = spacing = wd.loop_spacing_mm(wire_paths, R)
            routes = wd.plan_route(wire_paths, R, self.holder_length, self.flange_w,
                                   self.wire_d, self.groove_clearance, settings=st)
            gap, touching = wd.route_clearance(routes, R, st)
            if touching or gap < min(spacing, pitch) - 0.1:
                raise wd.RouteError(f'the wire would come within {gap:.2f} mm of itself', kind='topology')
        except wd.RouteError as e:
            if not self.auto_fit:
                raise
            out['error'] = e
            if e.kind == 'room' and axis == 'Gx':
                return out, ('length', e.shortfall_mm / 2 + 0.5)
            if e.kind in ('room', 'topology'):
                return out, ('ends', str(e).split(';')[0])
            raise
        out['routes'] = routes
        return out, None

    def _fit_axis(self, axis, R):
        """Contours and winding route for one axis.

        With ``auto_fit`` the design is adjusted until every turn has room for
        its wire and (winding='spiral') the loops can be joined into a route
        that never comes too close to itself.  For the number of turns asked
        for, every way of closing the coil ends is tried, mildest first; only
        if none works is one turn per quadrant given up and the search
        repeated.  Each change is recorded in 'notes'.
        """
        st = wd.RouteSettings(self.wire_d, self.groove_clearance)
        x_half = self.holder_length / 2
        coil_len0 = self._coil_length_mm(axis)
        stages = self._end_stages(axis, st)
        if not self.auto_fit:
            stages = stages[:1] if stages[0][0] == 1.5 else [(1.5, 0.04, '')]
        found, reason, last_error = None, '', None

        for n_w in range(int(self.n_wires), 0, -1):
            for k, stage in enumerate(stages):
                coil_len = coil_len0
                for _ in range(6):                       # Gx: shorten the coil until the jumpers fit
                    out, problem = self._try(axis, R, st, n_w, stage, coil_len)
                    if problem and problem[0] == 'length' and coil_len - problem[1] > 0.55 * x_half:
                        coil_len -= problem[1]
                        continue
                    break
                last_error = out.get('error', last_error)
                if problem is None or not self.auto_fit:
                    found = (out, k)
                    break
                reason = problem[1] if isinstance(problem[1], str) else 'there was no room for the jumper grooves'
                if problem[0] == 'middle':
                    break                                # closing the ends differently cannot help
            if found:
                break
        if not found:
            if last_error is not None:
                raise last_error
            raise wd.RouteError(f'{axis}: no number of turns gives grooves that fit; use a thinner wire '
                                f'or a larger former ({reason})')

        out, k = found
        notes = []
        if out['n_wires'] != self.n_wires:
            notes.append(f'turns per quadrant {self.n_wires} -> {out["n_wires"]} ({reason})')
        if stages[k][2]:
            notes.append(stages[k][2])
        if out['n_stray']:
            notes.append(f'{out["n_stray"]} small stray loop(s) left out')
        if out['n_stacked']:
            notes.append(f'{out["n_stacked"]} turn(s) in small lobes beyond the main lobes left out '
                         f'(the route cannot reach them)')
        if out['coil_length_mm'] != coil_len0:
            notes.append(f'Gx coil shortened by {2 * (coil_len0 - out["coil_length_mm"]):.1f} mm '
                         f'to make room for the jumper grooves')
        out['notes'] = notes
        return out

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
        print(f'  Winding:          {"one wire per lobe group (spiral)" if self.winding == "spiral" else "closed loops"}')
        for note in self.adjustments:
            print(f'  Adjusted:         {note}')
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
                'min_loop_spacing_mm': (None if not np.isfinite(info.get('min_loop_spacing_mm', np.nan))
                                        else info['min_loop_spacing_mm']),
                'wire_tails': info['holder'].tails(),
                'turns_per_quadrant': info.get('n_wires_used'),
                'adjustments': info.get('adjustments', []),
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
            'stream_resolution_mm': self.stream_resolution,
            'winding': self.winding,
            'auto_fit': self.auto_fit,
            'holder_wall_mm_used': self.wall,
            'dsv_radius_mm': self.dsv_radius,
        }
