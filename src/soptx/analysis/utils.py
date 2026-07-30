"""Compatibility imports for FEM utility functions."""

from soptx.fem.utils import (
    _scalar_disp_to_tensor_disp,
    calculate_multiresolution_gphi_eg,
    map_bcs_to_sub_elements,
    project_solution_to_finer_mesh,
    reshape_multiresolution_data,
    reshape_multiresolution_data_bcakup,
    reshape_multiresolution_data_inverse,
    reshape_multiresolution_data_inverse_backup,
)

__all__ = [
    "calculate_multiresolution_gphi_eg",
    "map_bcs_to_sub_elements",
    "project_solution_to_finer_mesh",
    "reshape_multiresolution_data",
    "reshape_multiresolution_data_bcakup",
    "reshape_multiresolution_data_inverse",
    "reshape_multiresolution_data_inverse_backup",
]
