# Gradient Coil Designer

A universal Python toolkit for designing and fabricating MRI/NMR gradient coils using the **Target Field Method** (Turner, 1986). Given a bore size and gradient requirements, it computes wire winding patterns, generates 3D-printable holder models with recessed wire grooves, evaluates gradient linearity, and produces Blender / CAD exports.

---

## Features

- **Target Field Method** stream-function computation for Gx, Gy, and Gz gradient axes
- **Biot-Savart field verification** with linearity error reporting
- **Concentric cylindrical holders** that stack inside each other — Gx (outermost) → Gy → Gz (innermost)
- **Smooth recessed wire grooves** on the outer surface of each holder (KD-tree, vectorised)
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
| **Left sidebar** | Bore geometry, DSV, wire size, layer gaps, axes to design, export buttons |
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
| Export STL | Binary STL of each holder (1200 × 800 mesh, 2 mm groove depth) |
| Export OBJ | Wavefront OBJ mesh of each holder |
| Export Blender script | Self-contained `.py` that rebuilds the scene in Blender |
| Generate .blend | Runs Blender headlessly and saves a `.blend` file directly |

**PyVista buttons** launch separate interactive windows (requires `pip install pyvista`).

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
| Gx   | 76.5   | 156     | ~3.9 mT/m/A | ~4 %    |
| Gy   | 69.5   | 142     | ~3.0 mT/m/A | ~2 %    |
| Gz   | 62.5   | 128     | ~3.8 mT/m/A | ~1 %    |

---

## Package Structure

```
gradient_coil_designer/
├── __init__.py        # Package entry point
├── designer.py        # GradientCoilDesigner — top-level orchestrator
├── tfm.py             # Target Field Method stream functions (Turner 1986)
├── physics.py         # Biot-Savart integration, field verification
├── holder.py          # CoilHolder — grooved cylindrical shell, STL export
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

- **Groove depth**: 2 × (wire_radius + clearance) — wires sit flush and are held in place
- **Layer stacking**: each inner holder clears the previous by `layer_gap_mm` radially
- **End flanges**: extend axially beyond the coil for alignment and retention
- **STL units**: millimetres (import into slicer at 1:1 scale)

---

## Citation

If you use this tool in research, please cite:

> Turner, R. (1986). A target field approach to optimal coil design. *Journal of Physics D: Applied Physics*, 19(8), L147.

---

## License

MIT License — see `LICENSE` file.
