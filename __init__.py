"""
gradient_coil_designer
======================
Universal gradient coil designer for MRI/NMR cylindrical bore magnets.

Quick start
-----------
>>> from gradient_coil_designer import GradientCoilDesigner, viz
>>> d = GradientCoilDesigner(bore_radius_mm=78, linear_region_mm=40)
>>> result = d.run(['Gx', 'Gy', 'Gz'])
>>> viz.dashboard(result, save_path='output/dashboard.png')
>>> d.save_all(result, output_dir='output/')
"""

from .designer import GradientCoilDesigner
from .holder import CoilHolder
from . import tfm, physics, viz, export

try:
    from . import viz3d
    __all__ = ['GradientCoilDesigner', 'CoilHolder', 'tfm', 'physics', 'viz', 'viz3d', 'export']
except ImportError:
    __all__ = ['GradientCoilDesigner', 'CoilHolder', 'tfm', 'physics', 'viz', 'export']
