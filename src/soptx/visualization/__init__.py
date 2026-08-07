"""Optional visualization namespace.

Concrete helpers live in the submodules:

* :mod:`soptx.visualization.vtk_export` — write FE fields to VTU
  (:func:`soptx.visualization.vtk_export.write_vtu` and the
  displacement convenience wrapper
  :func:`soptx.visualization.vtk_export.export_vtu`);
* :mod:`soptx.visualization.vtk_render` — VTK off-screen warped-field
  rendering (:func:`soptx.visualization.vtk_render.load_vtu`,
  :func:`soptx.visualization.vtk_render.create_warped_actor` and
  :func:`soptx.visualization.vtk_render.render_and_save`).

This namespace remains lightweight so importing SOPTX does not require ``viz``:
the submodules import ``pyevtk``/``vtk`` only when the caller explicitly
imports them.
"""

__all__: list[str] = []
