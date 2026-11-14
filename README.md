# Gradient Coil Designer

A universal Python toolkit for designing and fabricating MRI/NMR gradient coils using the **Target Field Method** (Turner, 1986). Given a bore size and gradient requirements, it computes wire winding patterns, generates 3D-printable holder models with recessed wire grooves, evaluates gradient linearity, and produces Blender / CAD exports.

---

## Features

- **Target Field Method** stream-function computation for Gx, Gy, and Gz gradient axes
- **Biot-Savart field verification** with linearity error reporting
- **Concentric cylindrical holders** that stack inside each other — Gx (outermost) → Gy → Gz (innermost)
- **Smooth half-pipe wire grooves** carved on the outer surface as a radial displacement / KD-tree distance field (no booleans → watertight, no slivers). The wire seats at the coil radius and is cradled by the groove walls. Export grids are sized from the groove radius (≥2 samples per groove radius, default 2.5, about 0.34 mm for 1.5 mm wire), so grooves stay continuous instead of printing as dotted lines; any coarser `n_phi`/`n_x` passed in is raised to that minimum with a warning
- **One-wire winding route, every time** (the default): the closed loops are joined into a route a single wire follows without leaving a groove, with a lead-in channel that passes under the turns and flange notches for the wire tails. `auto_fit` adjusts the design until that route can be built and wound, and lists what it changed
- **Interactive GUI** (CustomTkinter dark theme) with tabbed matplotlib visualisation
- **PyVista 3D** interactive GPU-accelerated rendering (optional)
- **Exports**: binary STL, OBJ, CSV wire paths, Blender script, `.blend` file (headless)
- Works with any cylindrical bore magnet — designed around the Cooley Halbach MRI at University of Central Oklahoma

---

## Installation

```bash
# Clone
git clone https://github.com/hassledzebra/gradient_coil_designer.git
cd gradient_coil_designer

# Install dependencies
pip install -r requirements.txt

# Optional: GPU-accelerated 3D viewer
pip install pyvista
```

### Requirements

- Python ≥ 3.9
- numpy, scipy, matplotlib
- customtkinter (GUI)
- pyvista (optional, for interactive 3D)

---

## Quick Start

### Command-line / script

```python
from gradient_coil_designer import GradientCoilDesigner, viz

designer = GradientCoilDesigner(
    bore_radius_mm=78.0,       # inner bore radius
    linear_region_mm=40.0,     # target DSV diameter (±20 mm)
    holder_length_mm=145.0,
    wire_diameter_mm=1.5,
    n_wires=10,                # wire turns per quadrant
)

result = designer.run(['Gx', 'Gy', 'Gz'])

# Visualise
viz.dashboard(result, save_path='dashboard.png')

# Export STL for 3D printing
designer.save_all(result, output_dir='output/')
```

### GUI

```bash
python -m gradient_coil_designer.ui
```

The GUI lets you adjust all design parameters, run the solver, inspect plots, and export files interactively.

---

## GUI Overview

| Panel | Description |
|-------|-------------|
| **Left sidebar** | Bore geometry, DSV, wire size, winding (one wire by default, or closed loops), auto-fit, layer gaps, axes to design, export buttons |
| **Assembly 3D** | All three coil axes rendered together |
| **Coil Wires** | Per-axis wire paths (positive = warm colour, negative = cool) |
| **Holders** | Transparent cylindrical holders with recessed groove channels |
| **Unwrapped** | φ vs X unrolled wire pattern for each axis |
| **Field Profile** | Biot-Savart By vs position with linear fit and linearity error |
| **Log** | Full solver output |

**Export buttons** (bottom of sidebar):

| Button | Output |
|--------|--------|
| Export CSV | Wire path coordinates for each axis |
| Export STL | Binary STL of each holder (mesh grid sized from the groove radius; half-pipe groove sized to the wire) |
| Export OBJ | Wavefront OBJ mesh of each holder |
| Export Blender script | Self-contained `.py` that rebuilds the scene in Blender |
| Generate .blend | Runs Blender headlessly and saves a `.blend` file directly |

**PyVista buttons** launch separate interactive windows (requires `pip install pyvista`).

---

## Winding route: one wire instead of separate loops

The target field method gives closed loops. With one ring groove per loop the wire has to leave its groove to get from one turn to the next. By default the designer joins the loops into a route (`winding='spiral'`):

```python
designer = GradientCoilDesigner(bore_radius_mm=57, linear_region_mm=21, holder_length_mm=152.4,
                                wire_diameter_mm=1.5, n_wires=10)
result = designer.run(['Gx', 'Gy', 'Gz'])
for axis, coil in result['coils'].items():
    print(axis, coil['n_wires_used'], coil['adjustments'])
    for t in coil['holder'].tails():
        print('  ', t)    # wire, START/END, which end of the former, angle from +Y toward +Z
```

| Part of the route | What it is |
|---|---|
| Spiral | The nested loops of one lobe, wound from the innermost turn outward. Each turn stops short of a seam line and steps out to the next turn |
| Lead-in | The wire reaches the centre of a lobe in a channel one wire deeper than a normal groove, so it passes under the turns |
| Jumper | After the outermost turn the wire runs along a line just beyond the end of the lobes to the next lobe |
| Tails | Each wire starts and ends at one end of the former; the flange has a notch there |

- **Wires per coil**: Gx is two wires, one per end of the former (join them outside the former). Gy and Gz are one wire each, with both tails at the -x end.
- **Current direction**: each loop is followed in the direction of its contour, so the wound coil gives the designed field. Efficiency, linearity, wire length and the CSV export are computed from the route, not from the ideal loops (`result['coils'][axis]['loops_3d']` keeps those).
- **Pair of innermost loops**: when the innermost contour of a lobe splits in two (a stream function with a double peak), both are wound, one after the other, before the turns that enclose them.

### auto_fit: a windable holder every time

A route is only useful if a wire fits in it everywhere. With `auto_fit=True` (the default; *Auto-fit so the wire always fits* in the GUI) the designer changes what it must, in this order, and reports each change (printed, in the GUI log, in `coil['adjustments']` and in `design_summary.json`):

| Problem | What auto_fit does | Cost |
|---|---|---|
| Wall too thin for the lead-in channel | Raises the wall to 1.5 × wire + 2 × clearance + 0.9 mm (3.3 mm for 1.5 mm wire) | Coils sit slightly further in |
| Turns cut off at the end of the coil return closer together than a wire needs (wire + clearance) | Gy, Gz: uses spare holder length for the return. Then widens the band the turns return in, up to 30 % of the coil length | None for the first; up to about 20 % of the gradient strength and 2 points of linearity for the second |
| Gy / Gz pattern longer than the holder | Cuts the pattern to fit | Smaller linear region |
| No room for the jumper grooves beyond a Gx coil | Shortens the Gx coil | Small |
| Stray slivers, and small secondary lobes behind a main lobe | Leaves them out | Small; the reported field is that of what is wound |
| Turns still too close (usually the middle of a dense Gx), or loops that cannot be joined | Gives up one turn per quadrant at a time, for that axis only | Gradient strength in proportion |

`coil['n_wires_used']` is the number of turns per quadrant actually used. If turns were given up and you need them, use a thinner wire or a larger former.

`auto_fit=False` changes nothing: turns that are too close give a warning, and a route that cannot be built raises `winding.RouteError`. `winding='loops', auto_fit=False` is the earlier behaviour (one closed ring groove per loop).

---

## Coordinate Convention (Halbach MRI)

```
X = bore axis (axial)
Y = B₀ direction (transverse, vertical for Halbach)
Z = perpendicular transverse
```

Gradient definitions:
- **Gx** = dBy/dx — saddle coil, cos(φ) stream function
- **Gy** = dBy/dy — 45°-rotated quadrupole, cos(2φ) stream function
- **Gz** = dBy/dz — quadrupole, sin(2φ) stream function

---

## Cooley Halbach Example

Reproduces the three-axis gradient coil set designed for the Cooley portable Halbach MRI scanner:

```bash
python gradient_coil_designer/examples/cooley_halbach.py
```

Parameters: bore = 78 mm, DSV = ±20 mm, holders fit inside 78 mm bore radius with 4 mm radial clearance between layers.

Expected results (10 wires/quadrant):

| Axis | R (mm) | OD (mm) | Efficiency | Linearity |
|------|--------|---------|------------|-----------|
| Gx   | 76.5   | 153     | ~3.9 mT/m/A | ~4 %    |
| Gy   | 69.5   | 139     | ~3.0 mT/m/A | ~2 %    |
| Gz   | 62.5   | 125     | ~3.8 mT/m/A | ~1 %    |

(OD = 2 × land radius = 2 × (coil_radius + groove_clearance); flanges add `flange_thick_mm` beyond that.)

---

## Package Structure

```
gradient_coil_designer/
├── __init__.py        # Package entry point
├── designer.py        # GradientCoilDesigner — top-level orchestrator
├── tfm.py             # Target Field Method stream functions (Turner 1986)
├── physics.py         # Biot-Savart integration, field verification
├── holder.py          # CoilHolder — grooved cylindrical shell, STL export
├── winding.py         # One-wire winding route: spirals, lead-in channel, jumpers, tails
├── viz.py             # Matplotlib 3D visualisation
├── viz3d.py           # PyVista GPU visualisation (optional)
├── export.py          # OBJ, Blender script, .blend headless export
├── ui.py              # CustomTkinter GUI application
└── examples/
    ├── cooley_halbach.py   # Cooley MRI example
    └── generic_example.py  # Configurable generic example
```

---

## Physical Design Notes

- **Groove profile**: a circular half-pipe of radius `wire_radius + clearance`, carved as a radial displacement that hugs the round wire (depth = that radius). Wire paths are arc-length resampled below the groove radius so channels stay continuous regardless of TFM contour spacing. The land sits just above the wire centerline, so the wire seats at the coil radius and is captured by the groove walls
- **Layer stacking**: each inner holder clears the previous by `layer_gap_mm` radially
- **End flanges**: extend axially beyond the coil for alignment and retention
- **STL units**: millimetres (import into slicer at 1:1 scale)
- **Mesh validity**: the STL and the OBJ / Blender mesh are closed, consistently wound solids (one body, positive volume), so they slice and take boolean cuts without a repair step
- **Stream-function grid**: `stream_resolution_mm` (default 1.0) sets the grid the wire contours are traced on. For coils under about 50 mm radius use 0.25-0.5; at 1.0 the contours sit up to 0.6-0.7 mm away from where a 0.25 mm grid puts them

---

## Citation

If you use this tool in research, please cite:

> Turner, R. (1986). A target field approach to optimal coil design. *Journal of Physics D: Applied Physics*, 19(8), L147.

---

## License

MIT License — see `LICENSE` file.
