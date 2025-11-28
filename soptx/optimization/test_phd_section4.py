from typing import Optional, Union
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike

from soptx.utils.base_logged import BaseLogged
from soptx.optimization.tools import save_optimization_history, plot_optimization_history
from soptx.optimization.tools import OptimizationHistory

class DensityTopOptTest(BaseLogged):
    def __init__(self, 
                enable_logging: bool = False, 
                logger_name: Optional[str] = None) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)


    @variantmethod('test')
    def run(self) -> Union[TensorLike, OptimizationHistory]:
        domain = [0, 40.0, 0, 20.0]
        E, nu = 1.0, 0.3
        plane_type = 'plane_stress' 

        nx, ny = 40, 20
        # nx, ny = 60, 30
        # nx, ny = 100, 50
        mesh_type = 'uniform_quad'

        space_degree = 1
        integration_order = space_degree + 1 # 张量网格
        # integration_order = space_degree**2 + 2  # 单纯形网格

        volume_fraction = 0.3
        penalty_factor = 3.0

        # 'element', 'element_multiresolution', 'node', 'node_multiresolution'
        density_location = 'element_multiresolution'
        sub_density_element = 4

        relative_density = volume_fraction

        # 'standard', 'standard_multiresolution', 'voigt', 'voigt_multiresolution'
        assembly_method = 'standard_multiresolution'

        optimizer_algorithm = 'mma'  # 'oc', 'mma'
        max_iterations = 500
        tolerance = 1e-2
        use_penalty_continuation = True

        filter_type = 'none' # 'none', 'sensitivity', 'density'
        rmin = 1.2