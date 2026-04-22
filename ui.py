"""
Gradient Coil Designer — CustomTkinter GUI.

Run:
  python -m gradient_coil_designer.ui
  python gradient_coil_designer/ui.py
"""

import os
import sys
import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401


# ── package imports ──────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _here not in sys.path:
    sys.path.insert(0, _here)

from gradient_coil_designer.designer import GradientCoilDesigner
from gradient_coil_designer import viz, viz3d, export as exp

# ── app theme ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_ACCENT  = "#2979FF"
_PANEL   = "#1e1e2e"
_CARD    = "#282840"
_FG      = "#e0e0e0"
_MUTED   = "#888899"
_GREEN   = "#26C281"
_ORANGE  = "#FF8C00"
_RED     = "#E74C3C"


# ─────────────────────────────────────────────────────────────────────────────
# Helper widgets
# ─────────────────────────────────────────────────────────────────────────────

class LabeledEntry(ctk.CTkFrame):
    """Label + Entry pair with optional unit annotation."""
    def __init__(self, parent, label, default, unit="", width=90, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        ctk.CTkLabel(self, text=label, anchor="w", width=200, text_color=_FG).pack(side="left")
        self.var = tk.StringVar(value=str(default))
        ctk.CTkEntry(self, textvariable=self.var, width=width,
                     justify="right").pack(side="left", padx=4)
        if unit:
            ctk.CTkLabel(self, text=unit, text_color=_MUTED, width=40).pack(side="left")

    def get(self):
        return self.var.get()

    def get_float(self, fallback=0.0):
        try:
            return float(self.var.get())
        except ValueError:
            return fallback

    def get_int(self, fallback=0):
        try:
            return int(self.var.get())
        except ValueError:
            return fallback


class LabeledCombo(ctk.CTkFrame):
    def __init__(self, parent, label, values, default, width=130, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        ctk.CTkLabel(self, text=label, anchor="w", width=200, text_color=_FG).pack(side="left")
        self.var = tk.StringVar(value=str(default))
        ctk.CTkComboBox(self, variable=self.var, values=values,
                         width=width).pack(side="left", padx=4)

    def get(self):
        return self.var.get()


class SectionLabel(ctk.CTkLabel):
    def __init__(self, parent, text, **kwargs):
        super().__init__(parent, text=f"  {text}", anchor="w",
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=_ACCENT,
                         fg_color=_CARD, corner_radius=4,
                         height=26, **kwargs)


class StatusBar(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, height=28, fg_color=_CARD, corner_radius=0)
        self._lbl = ctk.CTkLabel(self, text="Ready.", anchor="w",
                                  text_color=_FG, font=ctk.CTkFont(size=11))
        self._lbl.pack(side="left", padx=10)
        self._progress = ctk.CTkProgressBar(self, width=140, height=8)
        self._progress.set(0)
        self._progress.pack(side="right", padx=10, pady=8)

    def set(self, msg, color=_FG):
        self._lbl.configure(text=msg, text_color=color)

    def progress(self, value):   # 0.0 – 1.0
        self._progress.set(value)


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gradient Coil Designer")
        self.geometry("1400x860")
        self.configure(fg_color=_PANEL)
        self.minsize(1100, 700)

        self._result = None
        self._designing = False

        self._build_layout()

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_layout(self):
        # ── top bar ──
        topbar = ctk.CTkFrame(self, height=46, fg_color=_CARD, corner_radius=0)
        topbar.pack(fill="x", side="top")
        ctk.CTkLabel(topbar, text="  Gradient Coil Designer",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=_FG).pack(side="left", pady=6)
        ctk.CTkLabel(topbar, text="Target Field Method (Turner 1986)  |  Gx · Gy · Gz",
                     text_color=_MUTED, font=ctk.CTkFont(size=11)).pack(side="left", padx=20)

        # ── status bar ──
        self._status = StatusBar(self)
        self._status.pack(fill="x", side="bottom")

        # ── main split ──
        main = ctk.CTkFrame(self, fg_color=_PANEL)
        main.pack(fill="both", expand=True)

        # Left sidebar (parameters)
        self._sidebar = ctk.CTkScrollableFrame(main, width=310, fg_color=_CARD,
                                                corner_radius=0)
        self._sidebar.pack(side="left", fill="y", padx=0, pady=0)

        # Center (notebook with views)
        center = ctk.CTkFrame(main, fg_color=_PANEL)
        center.pack(side="left", fill="both", expand=True)

        self._tabs = ctk.CTkTabview(center, fg_color=_PANEL,
                                     segmented_button_selected_color=_ACCENT,
                                     segmented_button_unselected_color=_CARD)
        self._tabs.pack(fill="both", expand=True, padx=4, pady=4)

        for tab in ("Assembly 3D", "Coil Wires", "Holders", "Unwrapped", "Field Profile", "Log"):
            self._tabs.add(tab)

        self._build_sidebar()
        self._build_tabs()

    # ── Sidebar ──────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        sb = self._sidebar
        pad = {"padx": 8, "pady": 3, "fill": "x"}

        def section(text):
            SectionLabel(sb, text).pack(padx=8, pady=(10, 2), fill="x")

        def entry(label, default, unit="", width=90):
            e = LabeledEntry(sb, label, default, unit=unit, width=width)
            e.pack(**pad)
            return e

        def combo(label, values, default, width=130):
            c = LabeledCombo(sb, label, values, default, width=width)
            c.pack(**pad)
            return c

        # ── Geometry ──
        section("Bore & Coil Geometry")
        self._bore     = entry("Bore radius",        78.0, "mm")
        self._dsv      = entry("Linear region (DSV)", 40.0, "mm")
        self._hlength  = entry("Holder length",      145.0, "mm")
        self._wall     = entry("Wall thickness",       3.0, "mm")
        self._gap      = entry("Layer gap",            2.5, "mm")
        self._flange_w = entry("Flange width",         5.0, "mm")
        self._flange_t = entry("Flange thickness",     5.0, "mm")

        # ── Wire ──
        section("Wire & Winding")
        self._wire_d   = entry("Wire diameter",  1.5, "mm")
        self._n_wires  = entry("Wires/quadrant", 10)

        # ── Axes ──
        section("Gradient Axes")
        self._ax_vars = {}
        for ax, default in [("Gx", True), ("Gy", True), ("Gz", True)]:
            var = tk.BooleanVar(value=default)
            cb = ctk.CTkCheckBox(sb, text=ax, variable=var,
                                  checkmark_color=_FG, fg_color=_ACCENT)
            cb.pack(**pad)
            self._ax_vars[ax] = var

        # ── Advanced ──
        section("Advanced (TFM)")
        self._n_orders  = entry("Higher order terms", 10)
        self._lin_order = entry("Linearity order",    16)
        self._apo       = entry("Apodization",       0.05)

        # ── Verification ──
        section("Field Verification")
        self._verify_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(sb, text="Run Biot-Savart check", variable=self._verify_var,
                         checkmark_color=_FG, fg_color=_ACCENT).pack(**pad)
        self._verify_pts = entry("Sample points", 15)

        # ── Buttons ──
        ctk.CTkFrame(sb, height=1, fg_color=_MUTED).pack(fill="x", pady=10, padx=8)

        self._btn_design = ctk.CTkButton(
            sb, text="  Design Coils", command=self._start_design,
            fg_color=_ACCENT, hover_color="#1565C0", font=ctk.CTkFont(size=13, weight="bold"),
            height=36,
        )
        self._btn_design.pack(padx=8, pady=4, fill="x")

        section("Export")
        for text, cmd in [
            ("Export CSV (wire paths)", self._export_csv),
            ("Export STL (holders)",    self._export_stl),
            ("Export OBJ (holders)",    self._export_obj),
            ("Export Blender script",   self._export_bscript),
            ("Generate .blend file",    self._export_blend),
        ]:
            ctk.CTkButton(sb, text=text, command=cmd, height=30,
                           fg_color=_CARD, hover_color="#3a3a5a",
                           border_width=1, border_color="#555570",
                           text_color=_FG,
                           font=ctk.CTkFont(size=11)).pack(padx=8, pady=2, fill="x")

        # PyVista launch
        ctk.CTkFrame(sb, height=1, fg_color=_MUTED).pack(fill="x", pady=8, padx=8)
        section("PyVista (interactive GPU)")
        for text, cmd in [
            ("3D Assembly view",    self._pv_assembly),
            ("Holder with grooves", self._pv_holder),
            ("Field isosurfaces",   self._pv_field),
        ]:
            ctk.CTkButton(sb, text=text, command=cmd, height=30,
                           fg_color="#1a1a3a", hover_color="#2a2a5a",
                           border_width=1, border_color=_ACCENT,
                           text_color=_FG,
                           font=ctk.CTkFont(size=11)).pack(padx=8, pady=2, fill="x")

    # ── Tabs ─────────────────────────────────────────────────────────────────

    def _build_tabs(self):
        # Assembly 3D
        self._fig_asm   = self._make_mpl_tab("Assembly 3D",   projection="3d")
        # Coil wires
        self._fig_wires = self._make_mpl_tab("Coil Wires",    projection="3d")
        # Holders
        self._fig_hold  = self._make_mpl_tab("Holders",       projection="3d")
        # Unwrapped
        self._fig_unwr  = self._make_mpl_tab("Unwrapped",     projection=None)
        # Field profile
        self._fig_prof  = self._make_mpl_tab("Field Profile", projection=None)

        # Log tab
        log_tab = self._tabs.tab("Log")
        self._log_text = ctk.CTkTextbox(log_tab, font=ctk.CTkFont(family="Courier", size=11),
                                         fg_color="#111122", text_color="#B0FFB0",
                                         wrap="word")
        self._log_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _make_mpl_tab(self, tab_name, projection=None):
        tab = self._tabs.tab(tab_name)
        fig = Figure(figsize=(8, 6), facecolor=_PANEL)
        if projection:
            ax = fig.add_subplot(111, projection=projection)
        else:
            ax = fig.add_subplot(111)
        ax.set_facecolor(_CARD)
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
        return fig, ax, canvas

    # ── Design ───────────────────────────────────────────────────────────────

    def _start_design(self):
        if self._designing:
            return
        axes = [ax for ax, var in self._ax_vars.items() if var.get()]
        if not axes:
            messagebox.showwarning("No axes", "Select at least one gradient axis.")
            return
        self._designing = True
        self._btn_design.configure(state="disabled", text="  Designing...")
        self._status.set("Computing wire patterns ...", color=_ORANGE)
        self._status.progress(0.05)
        self._log("=" * 60)
        self._log(f"Design request: {', '.join(axes)}")
        threading.Thread(target=self._run_design, args=(axes,), daemon=True).start()

    def _run_design(self, axes):
        try:
            d = GradientCoilDesigner(
                bore_radius_mm    = float(self._bore.get()),
                linear_region_mm  = float(self._dsv.get()),
                holder_length_mm  = float(self._hlength.get()),
                wire_diameter_mm  = float(self._wire_d.get()),
                holder_wall_mm    = float(self._wall.get()),
                layer_gap_mm      = float(self._gap.get()),
                flange_width_mm   = float(self._flange_w.get()),
                flange_thick_mm   = float(self._flange_t.get()),
                n_wires           = int(self._n_wires.get()),
                n_high_orders     = int(self._n_orders.get()),
                linearity_order   = int(self._lin_order.get()),
                apodization       = float(self._apo.get()),
                verify_field      = self._verify_var.get(),
                verify_half_range_mm = float(self._dsv.get()) / 2,
                verify_n_pts      = int(self._verify_pts.get()),
            )
            result = d.run(axes)
            self._result = result
            self.after(0, self._on_design_done, result)
        except Exception as e:
            import traceback
            msg = traceback.format_exc()
            self.after(0, self._on_design_error, str(e), msg)

    def _on_design_done(self, result):
        self._designing = False
        self._btn_design.configure(state="normal", text="  Design Coils")
        self._status.set("Design complete.", color=_GREEN)
        self._status.progress(1.0)

        # Log summary
        self._log_result(result)

        # Update all matplotlib views
        self._draw_assembly(result)
        self._draw_wires(result)
        self._draw_holders(result)
        self._draw_unwrapped(result)
        self._draw_profiles(result)

        self.after(3000, lambda: self._status.progress(0))

    def _on_design_error(self, msg, traceback_str):
        self._designing = False
        self._btn_design.configure(state="normal", text="  Design Coils")
        self._status.set(f"Error: {msg}", color=_RED)
        self._log(f"ERROR: {traceback_str}")
        messagebox.showerror("Design Error", msg)

    # ── Matplotlib drawing ────────────────────────────────────────────────────

    def _draw_assembly(self, result):
        fig, ax, canvas = self._fig_asm
        ax.clear()
        ax.set_facecolor(_CARD)
        items = [{"label": lbl, "wires_3d": info["wires_3d"], "holder": info["holder"]}
                 for lbl, info in result["coils"].items()]
        viz.plot_assembly(items, ax=ax, show_wires=True, show_holders=True)
        fig.tight_layout()
        canvas.draw()

    def _draw_wires(self, result):
        fig, ax, canvas = self._fig_wires
        ax.clear()
        ax.set_facecolor(_CARD)
        for lbl, info in result["coils"].items():
            viz.plot_wires_3d(info["wires_3d"], ax=ax, axis_label=lbl,
                              show_cylinder=True, lw=0.7)
        fig.tight_layout()
        canvas.draw()

    def _draw_holders(self, result):
        fig, ax, canvas = self._fig_hold
        ax.clear()
        ax.set_facecolor(_CARD)
        for lbl, info in result["coils"].items():
            viz.plot_holder_3d(info["holder"], ax=ax, axis_label=lbl, alpha=0.14)
        fig.tight_layout()
        canvas.draw()

    def _draw_unwrapped(self, result):
        fig, ax, canvas = self._fig_unwr
        fig.clear()
        axes_list = list(result["coils"].keys())
        n = len(axes_list)
        sub_axes = [fig.add_subplot(1, n, i+1) for i in range(n)]
        for sub_ax in sub_axes:
            sub_ax.set_facecolor(_CARD)
        coil_list = [{"label": lbl, "wires_3d": result["coils"][lbl]["wires_3d"]}
                     for lbl in axes_list]
        viz.plot_unwrapped(coil_list, axes=sub_axes)
        fig.set_facecolor(_PANEL)
        fig.tight_layout()
        canvas.draw()

    def _draw_profiles(self, result):
        fig, ax, canvas = self._fig_prof
        fig.clear()
        axes_list = [lbl for lbl, info in result["coils"].items()
                     if info.get("profile")]
        if not axes_list:
            ax = fig.add_subplot(111)
            ax.set_facecolor(_CARD)
            ax.text(0.5, 0.5, "Verification disabled", transform=ax.transAxes,
                    ha="center", color=_MUTED, fontsize=12)
            canvas.draw()
            return
        n = len(axes_list)
        for i, lbl in enumerate(axes_list):
            sub = fig.add_subplot(1, n, i+1)
            sub.set_facecolor(_CARD)
            p = result["coils"][lbl]["profile"]
            viz.plot_gradient_profile(
                p["coords_mm"], p["By_T"],
                p["gradient_mT_m_A"], p["linearity_pct"],
                axis=lbl[-1], label=lbl, ax=sub,
            )
            sub.set_facecolor(_CARD)
            sub.tick_params(colors=_MUTED)
        fig.set_facecolor(_PANEL)
        fig.tight_layout()
        canvas.draw()

    # ── Log ──────────────────────────────────────────────────────────────────

    def _log(self, msg):
        def _append():
            self._log_text.configure(state="normal")
            self._log_text.insert("end", msg + "\n")
            self._log_text.see("end")
            self._log_text.configure(state="disabled")
        self.after(0, _append)

    def _log_result(self, result):
        self._log("\nDESIGN SUMMARY")
        self._log("-" * 50)
        for lbl, info in result["coils"].items():
            h = info["holder"]
            prof = info.get("profile")
            grad_str = f"{prof['gradient_mT_m_A']:.3f} mT/m/A  ({prof['linearity_pct']:.1f}% error)" \
                       if prof else "not computed"
            self._log(f"\n{lbl}:")
            self._log(f"  Coil radius:    {info['coil_radius_mm']:.1f} mm")
            self._log(f"  Holder OD/ID:   {h.R_outer*2:.1f} / {h.R_inner*2:.1f} mm")
            self._log(f"  TFM efficiency: {info['efficiency_mT_m_A']:.3f} mT/m/A")
            self._log(f"  B-S gradient:   {grad_str}")
            self._log(f"  Wire length:    {info['wire_length_m']:.2f} m")
            self._log(f"  Wire segments:  {len(info['wires_3d'])}")
        self._log("")

    # ── Export helpers ────────────────────────────────────────────────────────

    def _need_result(self):
        if self._result is None:
            messagebox.showwarning("No design", "Run 'Design Coils' first.")
            return False
        return True

    def _export_csv(self):
        if not self._need_result():
            return
        d = filedialog.askdirectory(title="Save CSV wire paths to folder")
        if not d:
            return
        from gradient_coil_designer.designer import GradientCoilDesigner as _GCD
        import tempfile
        # Reuse save_all for CSV export
        designer = GradientCoilDesigner.__new__(GradientCoilDesigner)
        designer.wire_d = float(self._wire_d.get())
        try:
            for lbl, info in self._result["coils"].items():
                rows = []
                for w in info["wires_3d"]:
                    for pt in w["xyz"]:
                        rows.append([*pt * 1000, 1])
                p = os.path.join(d, f"{lbl}_wires.csv")
                np.savetxt(p, np.array(rows), delimiter=",",
                           header="x_mm,y_mm,z_mm,sign", comments="",
                           fmt=["%.4f", "%.4f", "%.4f", "%d"])
            self._status.set(f"CSV saved to {d}", color=_GREEN)
            self._log(f"CSV files written to: {d}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _export_stl(self):
        if not self._need_result():
            return
        d = filedialog.askdirectory(title="Save STL holder files to folder")
        if not d:
            return
        try:
            for lbl, info in self._result["coils"].items():
                p = os.path.join(d, f"{lbl}_holder.stl")
                info["holder"].save_stl(p, n_phi=120, n_x=80)
                self._log(f"STL: {p}")
            self._status.set(f"STL saved to {d}", color=_GREEN)
        except Exception as e:
            messagebox.showerror("STL export error", str(e))

    def _export_obj(self):
        if not self._need_result():
            return
        d = filedialog.askdirectory(title="Save OBJ holder files to folder")
        if not d:
            return
        try:
            paths = exp.save_all_obj(self._result, output_dir=d)
            for lbl, p in paths.items():
                self._log(f"OBJ [{lbl}]: {p}")
            self._status.set(f"OBJ saved to {d}", color=_GREEN)
        except Exception as e:
            messagebox.showerror("OBJ export error", str(e))

    def _export_bscript(self):
        if not self._need_result():
            return
        p = filedialog.asksaveasfilename(
            title="Save Blender Python script",
            defaultextension=".py",
            filetypes=[("Python script", "*.py"), ("All files", "*.*")],
        )
        if not p:
            return
        try:
            exp.save_blender_script(self._result, p)
            self._log(f"Blender script saved: {p}")
            self._status.set(f"Script saved: {os.path.basename(p)}", color=_GREEN)
        except Exception as e:
            messagebox.showerror("Script export error", str(e))

    def _export_blend(self):
        if not self._need_result():
            return
        p = filedialog.asksaveasfilename(
            title="Save Blender .blend file",
            defaultextension=".blend",
            filetypes=[("Blender file", "*.blend"), ("All files", "*.*")],
        )
        if not p:
            return

        blender_exe = exp.find_blender()
        if blender_exe is None:
            # Ask user
            blender_exe = filedialog.askopenfilename(
                title="Locate blender.exe",
                filetypes=[("Blender", "blender.exe blender"), ("All files", "*.*")],
            )
            if not blender_exe:
                return

        self._status.set("Running Blender headlessly ...", color=_ORANGE)
        self._log(f"Generating .blend: {p}")
        self._log(f"Using Blender: {blender_exe}")

        def _run():
            try:
                exp.export_blend(
                    self._result, p,
                    blender_exe=blender_exe,
                    progress_callback=self._log,
                )
                self.after(0, lambda: self._status.set(
                    f".blend saved: {os.path.basename(p)}", color=_GREEN))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Blend export error", str(e)))
                self.after(0, lambda: self._status.set("Blend export failed.", color=_RED))

        threading.Thread(target=_run, daemon=True).start()

    # ── PyVista launchers ────────────────────────────────────────────────────

    def _pv_assembly(self):
        if not self._need_result():
            return
        items = [{"label": lbl, "wires_3d": info["wires_3d"], "holder": info["holder"]}
                 for lbl, info in self._result["coils"].items()]
        bore_r = float(self._bore.get())
        threading.Thread(
            target=lambda: viz3d.plot_assembly(
                items, bore_radius_mm=bore_r, show=True),
            daemon=True,
        ).start()

    def _pv_holder(self):
        if not self._need_result():
            return
        # Show all holders in one plotter
        def _run():
            pl = viz3d.pv.Plotter(shape=(1, len(self._result["coils"])),
                                   window_size=(1200, 700),
                                   title="Holders with Wire Grooves")
            for idx, (lbl, info) in enumerate(self._result["coils"].items()):
                pl.subplot(0, idx)
                viz3d.plot_holder(info["holder"], label=lbl, plotter=pl, show=False)
            pl.show()
        threading.Thread(target=_run, daemon=True).start()

    def _pv_field(self):
        if not self._need_result():
            return
        # Pick first axis
        lbl = next(iter(self._result["coils"]))
        wires = self._result["coils"][lbl]["wires_3d"]
        n_pts_str = tk.simpledialog.askstring(
            "Grid resolution",
            "Grid points per axis (e.g. 12; more = slower):",
            initialvalue="12", parent=self,
        ) if hasattr(tk, "simpledialog") else "12"
        try:
            n_pts = int(n_pts_str or "12")
        except (ValueError, TypeError):
            n_pts = 12

        dsv_r = float(self._dsv.get()) / 2
        threading.Thread(
            target=lambda: viz3d.plot_field_volume(
                wires, label=lbl,
                half_range_mm=dsv_r, n_pts=n_pts, show=True),
            daemon=True,
        ).start()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Patch tk.simpledialog for the field dialog
    import tkinter.simpledialog
    tk.simpledialog = tkinter.simpledialog

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
