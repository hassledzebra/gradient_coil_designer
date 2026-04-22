"""
Generic gradient coil design example.

Shows how to set up the designer for any bore size, DSV, and wire gauge,
then visualize and export results.

Run from the repo root:
  python -m gradient_coil_designer.examples.generic_example
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('TkAgg')   # change to 'Qt5Agg' or 'Agg' as needed
import matplotlib.pyplot as plt

from gradient_coil_designer import GradientCoilDesigner, viz


# ─────────────────────────────────────────────────────────────────────────────
# USER PARAMETERS  ← edit these
# ─────────────────────────────────────────────────────────────────────────────

BORE_RADIUS_MM     = 50.0    # inner bore radius (mm)
DSV_MM             = 30.0    # target linear region diameter (mm)
HOLDER_LENGTH_MM   = 120.0   # holder axial length (mm)
WIRE_DIAMETER_MM   = 1.5     # wire / Litz OD (mm)
HOLDER_WALL_MM     = 2.5     # holder wall thickness (mm)
LAYER_GAP_MM       = 2.0     # radial gap between layers (mm)
N_WIRES            = 8       # wire turns per quadrant (more → higher gradient)
AXES               = ['Gx', 'Gy', 'Gz']   # which gradients to design

OUTPUT_DIR         = './output'


# ─────────────────────────────────────────────────────────────────────────────
# Design
# ─────────────────────────────────────────────────────────────────────────────

d = GradientCoilDesigner(
    bore_radius_mm   = BORE_RADIUS_MM,
    linear_region_mm = DSV_MM / 2,        # TFM uses radius, not diameter
    holder_length_mm = HOLDER_LENGTH_MM,
    wire_diameter_mm = WIRE_DIAMETER_MM,
    holder_wall_mm   = HOLDER_WALL_MM,
    layer_gap_mm     = LAYER_GAP_MM,
    flange_width_mm  = 4.0,
    flange_thick_mm  = 4.0,
    n_wires          = N_WIRES,
    verify_field     = True,
    verify_half_range_mm = DSV_MM / 2,
)

result = d.run(AXES)

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# (1) Per-axis 3D wire paths
for lbl, info in result['coils'].items():
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    viz.plot_wires_3d(info['wires_3d'], ax=ax, axis_label=lbl, show_cylinder=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{lbl}_wires.png'), dpi=120, bbox_inches='tight')
    plt.close()

# (2) Holder with grooves
for lbl, info in result['coils'].items():
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    viz.plot_holder_3d(info['holder'], ax=ax, axis_label=lbl)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{lbl}_holder.png'), dpi=120, bbox_inches='tight')
    plt.close()

# (3) Full stacked assembly
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
coil_items = [{'label': lbl, 'wires_3d': info['wires_3d'], 'holder': info['holder']}
              for lbl, info in result['coils'].items()]
viz.plot_assembly(coil_items, ax=ax, bore_radius_cm=BORE_RADIUS_MM / 10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'assembly.png'), dpi=150, bbox_inches='tight')
plt.close()

# (4) Unwrapped patterns
n = len(AXES)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
if n == 1:
    axes = [axes]
coil_list = [{'label': lbl, 'wires_3d': result['coils'][lbl]['wires_3d']} for lbl in AXES]
viz.plot_unwrapped(coil_list, axes=axes)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'unwrapped.png'), dpi=120, bbox_inches='tight')
plt.close()

# (5) Gradient linearity profiles
if any(result['coils'][lbl].get('profile') for lbl in AXES):
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax_plot, lbl in zip(axes, AXES):
        profile = result['coils'][lbl].get('profile')
        if profile:
            viz.plot_gradient_profile(
                profile['coords_mm'], profile['By_T'],
                profile['gradient_mT_m_A'], profile['linearity_pct'],
                axis=lbl[-1], label=lbl, ax=ax_plot,
            )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'profiles.png'), dpi=120, bbox_inches='tight')
    plt.close()

# (6) Export STL + CSV
d.save_all(result, output_dir=OUTPUT_DIR)

print(f'\nDone.  Outputs: {OUTPUT_DIR}/')
