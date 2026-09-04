"""Optional post-processing namespace: everything that happens after a solve.

Concrete helpers live in the submodules:

* :mod:`soptx.postprocess.vtk_export` -- serialize FE fields to VTU and read
  them back (:func:`soptx.postprocess.vtk_export.write_vtu`, the displacement
  convenience wrapper :func:`soptx.postprocess.vtk_export.export_vtu` and the
  bit-exact cell-field reader
  :func:`soptx.postprocess.vtk_export.read_vtu_cell_data`);
* :mod:`soptx.postprocess.vtk_render` -- VTK off-screen warped-field rendering
  (:func:`soptx.postprocess.vtk_render.load_vtu`,
  :func:`soptx.postprocess.vtk_render.create_warped_actor` and
  :func:`soptx.postprocess.vtk_render.render_and_save`);
* :mod:`soptx.postprocess.optimization_history` -- dump a density history to
  per-iteration VTU, persist and reload it as JSON, and plot the convergence
  curves;
* :mod:`soptx.postprocess.stress_report` -- post-optimization stress-constraint
  checking, solid-element statistics and von Mises yield-surface plots.

The package is named for what it does, not for one of the things it does: only
``vtk_render`` and the plotting helpers are visualization.  ``vtk_export``
serializes, ``optimization_history`` also persists and reloads, and
``stress_report`` mostly computes.  "Post-processing" is the term that covers
all four.

The two reporting modules read
:class:`soptx.topology.optimizers.history.OptimizationHistory` -- layer 4
consuming layer 3, which is why they belong here rather than under
``soptx.topology``: they produce output artefacts, they are not part of the
optimization algorithm.

This namespace stays lightweight so importing SOPTX does not require ``viz``:
the submodules import ``pyevtk``/``vtk``/``matplotlib`` only when the caller
explicitly imports them.
"""

__all__: list[str] = []
