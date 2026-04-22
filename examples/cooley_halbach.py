"""
Example: Cooley Halbach MRI gradient coil design.

Bore radius ≈ 78 mm (ring inner face), target DSV 40 mm diameter (±20 mm).
Three concentric holders: Gx (outermost) → Gy → Gz (innermost).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from gradient_coil_designer import GradientCoilDesigner, viz
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'cooley', 'output_gradient_design')


def main():
    d = GradientCoilDesigner(
        bore_radius_mm=78.0,           # Cooley ring bore inner face
        linear_region_mm=40.0,         # ±20 mm DSV
        holder_length_mm=145.0,        # fits inside Halbach assembly (~204 mm)
        wire_diameter_mm=1.5,          # Litz wire OD
        holder_wall_mm=3.0,
        layer_gap_mm=2.5,
        flange_width_mm=5.0,
        flange_thick_mm=5.0,
        n_wires=10,                    # wires per quadrant
        n_high_orders=10,
        linearity_order=16,
        apodization=0.05,
        verify_field=True,
        verify_half_range_mm=20.0,     # ±20 mm = full DSV radius
        verify_n_pts=21,
    )

    result = d.run(['Gx', 'Gy', 'Gz'])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Assembly 3D view ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    coil_items = [
        {'label': lbl, 'wires_3d': info['wires_3d'], 'holder': info['holder']}
        for lbl, info in result['coils'].items()
    ]
    viz.plot_assembly(coil_items, ax=ax, bore_radius_cm=7.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'assembly_3d.png'), dpi=150, bbox_inches='tight')
    print(f"Saved assembly_3d.png")
    plt.close()

    # ── Per-axis dashboard ───────────────────────────────────────────────────
    viz.dashboard(result, show=False,
                  save_path=os.path.join(OUTPUT_DIR, 'dashboard.png'))
    plt.close('all')

    # ── Unwrapped patterns ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    coil_list = [{'label': lbl, 'wires_3d': info['wires_3d']}
                 for lbl, info in result['coils'].items()]
    viz.plot_unwrapped(coil_list, axes=list(axes))
    plt.suptitle('Cooley Halbach Gradient Coils — Unwrapped Wire Patterns', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'unwrapped.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Save STL and CSV ────────────────────────────────────────────────────
    d.save_all(result, output_dir=OUTPUT_DIR)

    print(f"\nAll outputs written to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
