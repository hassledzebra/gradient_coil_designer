"""
Winding route: one continuous wire through every loop of a coil.

The target field method returns closed contour loops.  A coil wound from them
needs a wire that goes round each loop once and then gets to the next loop,
and with separate ring grooves the wire has to leave its groove to do that.
This module joins the loops into a route a single wire can follow without ever
leaving a groove:

  lobe      a set of nested loops (one "fingerprint").  Its loops are joined
            into ONE spiral: each turn stops a little short of a seam line and
            steps out to the start of the next turn.
  lead-in   the wire reaches the centre of a lobe in a channel one wire deeper
            than the normal groove, so it passes UNDER the turns instead of
            climbing over them.
  jumper    after the outermost turn the wire runs along a line just beyond the
            end of the lobes to the next lobe on the same wire.
  tails     each wire starts and ends at the end of the former, where the
            holder gets a notch through its flange.

Lobes that sit toward one end of the former (Gx: two at each end) are wound in
one wire per end; lobes centred on the former (Gy, Gz: four round the
circumference) are all on one wire, joined at the -x end.

Every loop is followed in the direction the contour has, which is the current
direction the stream function intends, so the wound coil gives the designed
field (check it with ``physics.dsv_metrics`` on ``route_to_wires_3d``).

Units: millimetres.  A route point is (arc, x, extra): arc length round the
cylinder at the coil radius (arc = R * phi, not wrapped), axial position, and
how far the groove floor is below a normal groove at that point.

The route was first built and checked for the BME 4223 gradient lab formers
(1.55 mm wire); every length below scales with the wire diameter.
"""

import numpy as np
from matplotlib.path import Path as _Path
from scipy.spatial import cKDTree

_REF_WIRE_MM = 1.55      # wire the lengths below were worked out for


class RouteError(ValueError):
    """The loops cannot be joined into a route with this holder geometry.

    ``kind`` says why, so the designer can adjust the design and try again:
    'room' (no space for the jumper grooves; ``shortfall_mm`` is how much
    longer the holder body would have to be), 'topology' (the loops do not
    nest in a way that can be wound), 'wall', or '' for anything else.
    """

    def __init__(self, msg, kind='', shortfall_mm=0.0):
        super().__init__(msg)
        self.kind = kind
        self.shortfall_mm = float(shortfall_mm)


class RouteSettings:
    """Lengths that shape the route, all in mm.

    ds        point spacing along the route
    jog       length along a turn over which the wire steps out to the next turn
    deep      extra depth of the lead-in channel (one wire and a little)
    smooth    moving-average window that rounds the corners of the route
    m1, m2    distance of the two jumper lines beyond the end of the lobes
    ramp      length over which a groove dives into, or rises out of, the channel
    bend      radius of the U-turn where a lead-in would otherwise meet its
              first turn head-on
    end_land  solid material kept between the outer jumper groove and the body end
    """

    def __init__(self, wire_diameter_mm, groove_clearance_mm=0.1, ds_mm=0.2):
        k = wire_diameter_mm / _REF_WIRE_MM
        self.wire_d = float(wire_diameter_mm)
        self.clear = float(groove_clearance_mm)
        self.ds = float(ds_mm)
        self.jog = 6.0 * k
        self.deep = self.wire_d + 0.1
        self.smooth = 5.0 * k
        self.m1 = 3.2 * k
        self.m2 = 5.8 * k
        self.ramp = 6.0 * k
        self.detour_a = 7.0 * k
        self.detour_b = 3.0 * k
        self.bend = 2.0 * self.wire_d
        self.end_land = 1.0

    @property
    def groove_radius(self):
        return self.wire_d / 2 + self.clear


# ─────────────────────────────────────────────────────────────────────────────
# Lobes
# ─────────────────────────────────────────────────────────────────────────────

def _signed_area_centroid(q):
    x, y = q[:, 0], q[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2.0
    if abs(a) < 1e-12:
        return 0.0, q.mean(axis=0)
    return a, np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (6.0 * a)


def _loop_table(wire_paths, coil_radius_mm):
    R = float(coil_radius_mm)
    loops = []
    for level, p in wire_paths:
        q = np.column_stack([R * p[:, 0], p[:, 1] * 1e3])
        if np.linalg.norm(q[0] - q[-1]) > 1e-6:
            raise RouteError('a contour is not closed; a route needs closed loops')
        area, cen = _signed_area_centroid(q)
        loops.append(dict(level=level, q=q, area=abs(area), turn=np.sign(area), cen=cen, path=_Path(q)))
    loops.sort(key=lambda d: d['area'])
    return loops


def group_lobes(wire_paths, coil_radius_mm):
    """Sort closed contour loops into lobes (sets of nested loops).

    wire_paths : list of (level, Nx2 array of (phi, x_metres)) from ``tfm``.

    Returns a list of dicts:
      'c'      centre of the lobe (arc, x) in mm; the seam line runs from it
      'loops'  list of Nx2 arrays in (arc, x) mm, innermost first
      'eyes'   None, or two single loops that sit side by side inside the
               innermost loop of 'loops' (a stream function with a double peak
               gives such a pair)
      'level'  stream-function level of the innermost loop
    """
    loops = _loop_table(wire_paths, coil_radius_mm)
    n = len(loops)
    parent = [-1] * n
    for i in range(n):                               # the smallest larger loop that contains this one
        for j in range(i + 1, n):
            if loops[j]['path'].contains_point(loops[i]['cen']):
                parent[i] = j
                break
    kids = {j: [i for i in range(n) if parent[i] == j] for j in range(n)}
    lobes = []
    for root in (i for i in range(n) if parent[i] < 0):
        chain, eyes, node = [root], None, root
        while kids[node]:
            ch = kids[node]
            if len(ch) == 1:
                node = ch[0]
                chain.append(node)
            elif len(ch) == 2 and not kids[ch[0]] and not kids[ch[1]]:
                eyes = ch
                break
            else:
                raise RouteError('a lobe branches into several nested groups of loops; only a pair of single '
                                 'innermost loops is supported. Changing n_wires by one usually removes the branch.',
                                 kind='topology')
        members = chain + (eyes or [])
        if len(set(loops[j]['turn'] for j in members)) != 1:
            raise RouteError('nested loops of one lobe run in opposite directions', kind='topology')
        inner = loops[chain[-1]]
        c = (loops[eyes[0]]['cen'] + loops[eyes[1]]['cen']) / 2 if eyes else inner['cen']
        lobes.append(dict(c=np.array(c, float), loops=[loops[j]['q'] for j in reversed(chain)],
                          eyes=[loops[j]['q'] for j in eyes] if eyes else None,
                          level=loops[(eyes or chain)[-1]]['level']))
    lobes.sort(key=lambda lb: (round(lb['c'][1], 3), lb['c'][0]))
    return lobes


def drop_stray_loops(wire_paths, coil_radius_mm, max_area_fraction=0.10):
    """Remove small loops that stand alone.

    A loop that neither encloses another loop nor lies inside one, and whose
    area is under ``max_area_fraction`` of the largest loop, is a by-product of
    where the contour levels happen to fall (typically a sliver in the band
    where the coil is closed at its ends).  It carries one turn round a small
    area, so it adds little field, and it cannot be reached by a route.

    Returns (kept wire_paths, number dropped).
    """
    loops = _loop_table(wire_paths, coil_radius_mm)
    if len(loops) < 2:
        return list(wire_paths), 0
    biggest = loops[-1]['area']
    nested = set()
    for i in range(len(loops)):
        for j in range(i + 1, len(loops)):
            if loops[j]['path'].contains_point(loops[i]['cen']):
                nested.update((i, j))
    keep = [lp for i, lp in enumerate(loops) if i in nested or lp['area'] >= max_area_fraction * biggest]
    R = float(coil_radius_mm)
    kept = [(lp['level'], np.column_stack([lp['q'][:, 0] / R, lp['q'][:, 1] * 1e-3])) for lp in keep]
    return kept, len(loops) - len(keep)


def loop_spacing_mm(wire_paths, coil_radius_mm, within_mm=4.0, step_mm=0.25, where=False):
    """Smallest centre distance between two different loops (mm), searched up to
    ``within_mm``; ``inf`` when no two loops come that close.  Grooves merge
    where this is less than the groove width, whatever the winding route.
    With ``where=True`` returns (distance, axial position in mm of that spot)."""
    R = float(coil_radius_mm)
    C = 2 * np.pi * R
    dense = []
    for k, (_, p) in enumerate(wire_paths):
        q = np.column_stack([R * p[:, 0], p[:, 1] * 1e3])
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
        su = np.arange(0, s[-1], step_mm)
        dense.append(np.c_[np.interp(su, s, q[:, 0]), np.interp(su, s, q[:, 1]), np.full(len(su), k)])
    if len(dense) < 2:
        return (np.inf, 0.0) if where else np.inf
    D = np.vstack(dense)
    u = np.mod(D[:, 0], C)
    u[u >= C] = 0.0
    lo = D[:, 1].min() - 1.0
    span = D[:, 1].max() - lo + 100.0
    P = np.c_[u, D[:, 1] - lo]
    tree = cKDTree(P, boxsize=[C, span])
    pairs = tree.query_pairs(within_mm, output_type='ndarray')
    pairs = pairs[D[pairs[:, 0], 2] != D[pairs[:, 1], 2]]
    if not len(pairs):
        return (np.inf, 0.0) if where else np.inf
    du = np.abs(P[pairs[:, 0], 0] - P[pairs[:, 1], 0])
    du = np.minimum(du, C - du)
    d = np.hypot(du, P[pairs[:, 0], 1] - P[pairs[:, 1], 1])
    k = int(np.argmin(d))
    return (float(d[k]), float(D[pairs[k, 0], 1])) if where else float(d[k])


# ─────────────────────────────────────────────────────────────────────────────
# One spiral per lobe
# ─────────────────────────────────────────────────────────────────────────────

def _line(a, b, ds):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(int(np.ceil(np.linalg.norm(b - a) / ds)), 1)
    return a + (b - a) * np.linspace(0, 1, n + 1)[:, None]


def _roll_to_ray(q, c, d, ds):
    """The closed loop q as an open polyline that starts where it crosses the
    ray from c in direction d, and runs once round in its own direction."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    e = np.array([d[1], -d[0]])                      # across the ray
    if np.allclose(q[0], q[-1]):
        q = q[:-1]
    qq = np.vstack([q, q[:1]])
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(qq, axis=0).T))]
    n = max(int(np.ceil(s[-1] / ds)), 8)
    su = np.arange(n) * (s[-1] / n)
    r = np.c_[np.interp(su, s, qq[:, 0]), np.interp(su, s, qq[:, 1])]
    a = (r - c) @ e                                  # side of the ray
    b = (r - c) @ d                                  # distance along it
    an = np.roll(a, -1)
    bm = (b + np.roll(b, -1)) / 2
    idx = np.where((a * an <= 0) & (a != an) & (bm > 0))[0]
    if len(idx) != 1:
        raise RouteError(f'a loop crosses the seam line of its lobe {len(idx)} times; '
                         f'the loops are too irregular to join into a spiral', kind='topology')
    i = idx[0]
    j = (i + 1) % n
    f = a[i] / (a[i] - a[j])
    p = np.asarray(c, float) + (b[i] + f * (b[j] - b[i])) * d
    return np.vstack([p, np.roll(r, -j, axis=0)])


def _open_turn(r, st, tip=False):
    """A full turn that stops ``jog`` short of where it started.

    ``tip=True`` is for a turn that starts at a sharp tip of its loop (the thin
    lens-shaped innermost pair): there the two sides of the loop run closer
    together than a wire is wide, so the turn stops where they are still a
    wire and a bit apart.  The stretch left out carries two nearly opposite
    currents side by side, which add almost nothing to the field.
    """
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(r, axis=0).T))]
    n_keep = int(np.searchsorted(s, s[-1] - min(st.jog, 0.25 * s[-1]), side='right'))
    if tip and len(r) > 20:
        half = len(r) // 2
        d, _ = cKDTree(r[:half]).query(r[half:])
        ok = np.where(d >= st.wire_d + st.clear + 0.3)[0]
        if len(ok):
            n_keep = min(n_keep, half + int(ok[-1]) + 1)
    return r[:max(n_keep, 2)]


def _eye_turns(lobe, sx, st):
    """The two innermost loops that sit side by side: wind one, hop across to
    the other along the line that joins them, wind it.  The order is chosen so
    the second one ends on the seam side, clear of the hop."""
    best = None
    for qa, qb in (lobe['eyes'], lobe['eyes'][::-1]):
        ca, cb = _signed_area_centroid(qa)[1], _signed_area_centroid(qb)[1]
        first = _open_turn(_roll_to_ray(qa, ca, cb - ca, st.ds), st, tip=True)
        second = _open_turn(_roll_to_ray(qb, cb, ca - cb, st.ds), st, tip=True)
        # The hop from the end of the first to the tip of the second runs beside
        # the line that joins the two tips, a wire and a bit to the side the
        # first turn ended on.  That keeps it clear of the first one's own tip
        # (where its turn began) and of the step from the second one out to the
        # enclosing turn, which leaves on the other side.
        ridge = (cb - ca) / np.linalg.norm(cb - ca)
        off = (first[-1] - first[0]) - ((first[-1] - first[0]) @ ridge) * ridge
        if np.linalg.norm(off) > 1e-6:
            off = off / np.linalg.norm(off)
            need = st.wire_d + st.clear + 0.3
            # ...but not so far that it runs into the enclosing turn, which
            # bulges in between the pair on that side.
            try:
                room = np.linalg.norm(_roll_to_ray(lobe['loops'][0], (ca + cb) / 2, off, st.ds)[0] - (ca + cb) / 2)
            except RouteError:
                room = np.inf
            off = min(need, max(room - need, 0.35 * room)) * off
            first = np.vstack([first, _line(first[-1], first[0] + off, st.ds)[1:],
                               _line(first[0] + off, second[0] + off, st.ds)[1:]])
        score = sx * (second[-1][1] - cb[1])
        if best is None or score > best[0]:
            best = (score, [first, second])
    return best[1]


def spiral(lobe, sx, st):
    """Innermost turn first.  Each turn stops ``jog`` short of the seam and steps
    out to the start of the next turn.  Returns (points, start, end)."""
    parts = _eye_turns(lobe, sx, st) if lobe.get('eyes') else []
    for q in lobe['loops']:
        parts.append(_open_turn(_roll_to_ray(q, lobe['c'], (0.0, float(sx)), st.ds), st))
    if lobe.get('eyes'):
        # From the second of the pair out to the turn that encloses both: come up
        # the seam line and meet that turn square-on.  A straight line from the
        # end of the second one would run alongside the enclosing turn, closer
        # than a wire, where that turn dips between the pair.
        s_start = parts[2][0]
        h = sx * (s_start[1] - lobe['c'][1])
        v = np.array([s_start[0], lobe['c'][1] + sx * max(h - 2.5 * (st.wire_d + st.clear + 0.3), 0.3 * h)])
        parts[1] = np.vstack([parts[1], _line(parts[1][-1], v, st.ds)[1:]])
    pts = [parts[0]]
    for a, b in zip(parts[:-1], parts[1:]):
        pts.append(_line(a[-1], b[0], st.ds)[1:-1])
        pts.append(b)
    return np.vstack(pts), parts[0][0], parts[-1][-1], parts[0]


def _lead_in(P, s1, first, st):
    """The deep channel from P to the start s1 of the first turn.

    Normally a straight line: the wire meets the turn about square-on and the
    corner is rounded later.  If the first turn sets off back toward where the
    lead-in comes from (the pair of side-by-side innermost loops does this),
    the wire would have to fold back on itself, so the channel runs past the
    start on the outer side of the turn and comes round in a U of radius
    ``bend`` to join it head-to-tail.  The U is in the deep channel, so it may
    pass under other turns."""
    P, s1 = np.asarray(P, float), np.asarray(s1, float)
    k = min(max(int(1.0 / st.ds), 1), len(first) - 1)
    T = first[k] - first[0]
    T = T / np.linalg.norm(T)
    D = (s1 - P) / np.linalg.norm(s1 - P)
    if D @ T > -0.2:
        return _line(P, s1, st.ds)
    n = np.array([T[1], -T[0]])
    if n @ (s1 - first.mean(axis=0)) < 0:            # outward from the first turn
        n = -n
    r = st.bend
    O, Q = s1 + r * n, s1 + 2 * r * n
    th = np.linspace(0.0, np.pi, max(int(np.ceil(np.pi * r / st.ds)), 8) + 1)
    arc = O + r * (np.cos(th)[:, None] * n - np.sin(th)[:, None] * T)
    return np.vstack([_line(P, Q, st.ds), arc[1:]])


def _shifted(lobe, du):
    eyes = [q + [du, 0] for q in lobe['eyes']] if lobe.get('eyes') else None
    return dict(c=lobe['c'] + [du, 0], loops=[q + [du, 0] for q in lobe['loops']], eyes=eyes, level=lobe['level'])


def _exit_side(lobe, sx, st):
    side = np.sign(spiral(lobe, sx, st)[2][0] - lobe['c'][0])
    return 1.0 if side == 0 else side


# ─────────────────────────────────────────────────────────────────────────────
# One wire through a group of lobes
# ─────────────────────────────────────────────────────────────────────────────

def _build_wire(lobes, sx, t, st, x_half, x_end):
    """One continuous wire through ``lobes`` in the order given, travelling round
    the former in direction t (+1 or -1 in arc).  Returns the (arc, x, extra)
    points and a list of labelled points."""
    ds, deep = st.ds, st.deep
    x_far = max(abs(q[:, 1]).max() for lb in lobes for q in lb['loops'] + (lb.get('eyes') or []))
    m1, m2 = sx * (x_far + st.m1), sx * (x_far + st.m2)
    xe = sx * x_end
    room = x_half - st.groove_radius - st.end_land
    if x_far + st.m1 >= room:
        raise RouteError(f'no room for the jumper groove beyond the loops: lengthen the holder by at least '
                         f'{2 * (x_far + st.m1 - room):.1f} mm', kind='room',
                         shortfall_mm=2 * (x_far + st.m1 - room))
    segs, marks, prev = [], [], None
    n_ramp = max(int(st.ramp / ds), 2)

    def add(pts, e0, e1=None):
        e = np.full(len(pts), e0, float) if e1 is None else np.linspace(e0, e1, len(pts))
        segs.append(np.c_[pts, e])

    for i, lb in enumerate(lobes):
        sp, s1, en, first = spiral(lb, sx, st)
        uc = lb['c'][0]
        side = np.sign(en[0] - uc) or 1.0
        if i == 0:                                   # from the end of the former, under the turns
            add(_lead_in((uc, xe), s1, first, st), deep)
            marks.append(('START', (uc, xe)))
        elif side * t > 0:                           # this lobe's exit is on its far side
            corner = np.array([uc, m1])
            run = _line(prev, corner, ds)
            k = max(len(run) - n_ramp, 1)
            add(run[:k], 0.0)
            add(run[k:], 0.0, deep)
            add(_lead_in(corner, s1, first, st)[1:], deep)
        else:                                        # exit on the near side: go round it on the outer line
            if x_far + st.m2 >= room:
                raise RouteError(f'no room for the outer jumper groove: lengthen the holder by at least '
                                 f'{2 * (x_far + st.m2 - room):.1f} mm', kind='room',
                                 shortfall_mm=2 * (x_far + st.m2 - room))
            a = np.array([en[0] - t * st.detour_a, m1])
            b = np.array([en[0] - t * st.detour_b, m2])
            corner = np.array([uc, m2])
            add(_line(prev, a, ds), 0.0)
            add(_line(a, b, ds)[1:], 0.0)
            run = _line(b, corner, ds)[1:]
            k = max(len(run) - max(int(5.0 / 6.0 * n_ramp), 2), 1)
            add(run[:k], 0.0)
            add(run[k:], 0.0, deep)
            add(_lead_in(corner, s1, first, st)[1:], deep)
        k = min(n_ramp, len(sp) - 1)                 # the first turn rises out of the deep channel
        add(sp[:k], deep, 0.0)
        add(sp[k:], 0.0)
        marks.append((str(i + 1), tuple(lb['c'])))
        if i == len(lobes) - 1:
            add(_line(en, (en[0], xe), ds)[1:], 0.0)
            marks.append(('END', (en[0], xe)))
        else:
            prev = np.array([en[0], m1])
            add(_line(en, prev, ds)[1:], 0.0)
    w = np.vstack(segs)
    # resample evenly and round the corners with a moving average (ends kept)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(w[:, :2], axis=0).T))]
    keep = np.r_[True, np.diff(s) > 1e-9]
    w, s = w[keep], s[keep]
    su = np.arange(0, s[-1], ds)
    w = np.c_[[np.interp(su, s, w[:, k]) for k in range(3)]].T
    n = int(st.smooth / ds) | 1
    pad = n // 2
    ker = np.ones(n) / n
    sm = np.c_[[np.convolve(np.r_[np.full(pad, w[0, k]), w[:, k], np.full(pad, w[-1, k])], ker, 'valid')
                for k in range(3)]].T
    sm[:, 2] = np.clip(sm[:, 2], 0, deep)
    return sm, marks


def _order_group(lobes, sx, st, circumference):
    """Order in which one wire visits the lobes of a group, and its direction.

    The wire goes round the former lobe by lobe.  The last lobe must have its
    exit on the far side (otherwise its tail would cross its own approach), and
    as many of the others as possible should, because a near-side exit needs
    the second, outer jumper line."""
    n = len(lobes)
    grp = sorted(lobes, key=lambda lb: lb['c'][0])
    if n == 1:
        return [grp[0]], 1
    best = None
    for t in (1, -1):
        sides = [_exit_side(lb, sx, st) * t for lb in grp]
        for k0 in range(n):
            seq = [(k0 + t * j) % n for j in range(n)]
            far = [sides[k] > 0 for k in seq]
            if not far[-1]:
                continue
            score = (sum(far[1:]), t, -k0)
            if best is None or score > best[0]:
                best = (score, t, seq)
    if best is None:
        raise RouteError('no visiting order found for the lobes of one wire')
    _, t, seq = best
    order, last_u = [], None
    for k in seq:
        lb = grp[k]
        if last_u is not None:
            while t * (lb['c'][0] - last_u) < 0:
                lb = _shifted(lb, t * circumference)
        last_u = lb['c'][0]
        order.append(lb)
    return order, t


def plan_route(wire_paths, coil_radius_mm, holder_length_mm, flange_width_mm,
               wire_diameter_mm, groove_clearance_mm=0.1, settings=None):
    """Join the closed loops of one coil into windable wires.

    Returns a list of route dicts, one per wire:
      'name'    'A', 'B', ...
      'end'     -1 or +1: the end of the former where this wire's tails are
      'path'    (N, 3) array of (arc mm at the coil radius, x mm, extra depth mm)
      'marks'   [(label, (arc, x))]: 'START', 'END' and the lobe numbers in winding order
      'n_lobes', 'n_turns'
    """
    st = settings or RouteSettings(wire_diameter_mm, groove_clearance_mm)
    R = float(coil_radius_mm)
    x_half = holder_length_mm / 2.0
    x_end = x_half + max(flange_width_mm, 0.0)
    lobes = group_lobes(wire_paths, R)
    if not lobes:
        raise RouteError('no loops to route')
    tol = 0.05 * x_half
    groups = {}
    for lb in lobes:
        key = 0 if abs(lb['c'][1]) < tol else int(np.sign(lb['c'][1]))
        groups.setdefault(key, []).append(lb)
    if 0 in groups and len(groups) > 1:
        raise RouteError('lobes both at the centre and toward the ends of the former: this layout is not supported',
                         kind='topology')
    routes = []
    for key in sorted(groups):
        sx = -1 if key == 0 else key
        order, t = _order_group(groups[key], sx, st, 2 * np.pi * R)
        path, marks = _build_wire(order, sx, t, st, x_half, x_end)
        routes.append(dict(name='ABCDEFGH'[len(routes)], end=sx, direction=t, path=path, marks=marks,
                           n_lobes=len(order),
                           n_turns=sum(len(lb['loops']) + len(lb.get('eyes') or []) for lb in order)))
    return routes


# ─────────────────────────────────────────────────────────────────────────────
# Use and checks
# ─────────────────────────────────────────────────────────────────────────────

def route_to_wires_3d(routes, coil_radius_mm):
    """Routes as wire dicts for ``physics`` and ``viz`` (xyz in metres).  The
    lead-in sections lie deeper, at their true radius."""
    R = float(coil_radius_mm)
    wires = []
    for rt in routes:
        w = rt['path']
        r = R - w[:, 2]
        phi = w[:, 0] / R
        xyz = np.c_[w[:, 1], r * np.cos(phi), r * np.sin(phi)] * 1e-3
        wires.append({'level': 1.0, 'xyz': xyz, 'sign': 1, 'name': rt['name']})
    return wires


def route_clearance(routes, coil_radius_mm, settings):
    """How close the route comes to itself.

    Returns (min_distance_mm, n_touching): the smallest distance between the
    centres of two parts of the wire that are not neighbours along it, measured
    in 3-D (distance on the surface and difference in depth), and the number of
    places where that distance is under three quarters of a wire.  Where the
    wire passes under itself in the deep channel the distance is the channel's
    extra depth, one wire and a little, which is fine.
    """
    st = settings
    C = 2 * np.pi * float(coil_radius_mm)
    pts, wid, idx = [], [], []
    for k, rt in enumerate(routes):
        w = rt['path']
        u = np.mod(w[:, 0], C)
        u[u >= C] = 0.0
        pts.append(np.c_[u, w[:, 1], w[:, 2]])
        wid.append(np.full(len(w), k))
        idx.append(np.arange(len(w)))
    pts, wid, idx = np.vstack(pts), np.concatenate(wid), np.concatenate(idx)
    lo = pts[:, 1].min() - 1.0
    span = pts[:, 1].max() - lo + 1.0 + 10 * st.wire_d
    tree = cKDTree(np.c_[pts[:, 0], pts[:, 1] - lo], boxsize=[C, span])
    pairs = tree.query_pairs(st.wire_d + 2 * st.clear + 1.2, output_type='ndarray')
    if not len(pairs):
        return np.inf, 0
    apart = (wid[pairs[:, 0]] != wid[pairs[:, 1]]) | (np.abs(idx[pairs[:, 0]] - idx[pairs[:, 1]]) > int(2 * st.ramp / st.ds))
    pairs = pairs[apart]
    if not len(pairs):
        return np.inf, 0
    du = np.abs(pts[pairs[:, 0], 0] - pts[pairs[:, 1], 0])
    du = np.minimum(du, C - du)
    d = np.sqrt(du**2 + (pts[pairs[:, 0], 1] - pts[pairs[:, 1], 1])**2
                + (pts[pairs[:, 0], 2] - pts[pairs[:, 1], 2])**2)
    touching = d < 0.75 * st.wire_d
    cells = set(map(tuple, np.round(pts[pairs[touching, 0], :2] / (3 * st.wire_d)).astype(int)))
    return float(d.min()), len(cells)


def drop_stacked_lobes(wire_paths, coil_radius_mm, holder_length_mm):
    """Leave out small lobes that sit beyond a main lobe at the same angle.

    Some designs have, between a main lobe and the end of the former, a small
    secondary lobe of one to three turns (and Gy / Gz kept far out can show
    their return lobes).  The route takes the wire from lobe to lobe round the
    former, so it cannot reach a lobe stacked behind another one.  Of lobes at
    the same angle on the same side, the one with the most turns is kept.

    Returns (kept wire_paths, number of loops left out).
    """
    R = float(coil_radius_mm)
    C = 2 * np.pi * R
    lobes = group_lobes(wire_paths, R)
    tol = 0.05 * holder_length_mm / 2

    def turns(lb):
        return len(lb['loops']) + len(lb.get('eyes') or [])

    def side(lb):
        return 0 if abs(lb['c'][1]) < tol else int(np.sign(lb['c'][1]))

    keep = [True] * len(lobes)
    if any(side(lb) == 0 for lb in lobes):           # central lobes: anything toward the ends is secondary
        keep = [side(lb) == 0 for lb in lobes]
    else:
        width = [np.ptp(lb['loops'][-1][:, 0]) for lb in lobes]
        for i, a in enumerate(lobes):
            for j, b in enumerate(lobes):
                if i >= j or side(a) != side(b) or not (keep[i] and keep[j]):
                    continue
                du = abs((a['c'][0] - b['c'][0] + C / 2) % C - C / 2)
                if du < 0.5 * min(width[i], width[j]):
                    ka = (turns(a), -abs(a['c'][1]))
                    kb = (turns(b), -abs(b['c'][1]))
                    keep[j if ka >= kb else i] = False
    kept, dropped = [], 0
    for lb, k in zip(lobes, keep):
        loops = lb['loops'] + (lb.get('eyes') or [])
        if k:
            kept += [(lb['level'], np.column_stack([q[:, 0] / R, q[:, 1] * 1e-3])) for q in loops]
        else:
            dropped += len(loops)
    return kept, dropped


def route_length_m(routes):
    return float(sum(np.hypot(*np.diff(rt['path'][:, :2], axis=0).T).sum() for rt in routes)) * 1e-3
