from typing import Optional, Union

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.typing import TensorLike

from soptx.utils.base_logged import BaseLogged
from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer

from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace, Function


class ComplianceObjectiveTester(BaseLogged):
    def __init__(self,
                enable_logging: bool = True,
                logger_name: Optional[str] = None
            ) -> None:

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self.compliance_objective = None

    @variantmethod('test_lfem_hzmfem_compliance')
    def run(self, model_type: str) -> None:
        """测试考虑密度时, 位移法和混合法下计算的设计变量值是否一致; 目标函数是否一致"""
        if model_type == 'half_clamped_beam_2d':
            # 集中载荷的算例            
            E = 30.0
            nu = 0.4  # 可压缩
            plane_type = 'plane_stress'  # 'plane_stress' or 'plane_strain'
            from soptx.model.clamped_beam_2d import HalfClampedBeam2D
            pde = HalfClampedBeam2D(
                    domain=[0, 80, 0, 20],
                    p=-1.5,
                    E=E, nu=nu,
                    plane_type=plane_type,
                )
            volume_fraction = 0.4
            nx, ny = 80, 20

        elif model_type == 'bearing_device_2d':
            # 分布载荷的算例
            E = 100.0
            nu = 0.4 # 可压缩
            plane_type = 'plane_stress'  # 'plane_stress' or 'plane_strain'
            from soptx.model.bearing_device_2d import HalfBearingDevice2D
            pde = HalfBearingDevice2D(
                                domain=[0, 0.6, 0, 0.4],
                                t=-1.8,
                                E=E, nu=nu,
                                plane_type=plane_type,
                            )
            volume_fraction = 0.35
            nx, ny = 60, 40
            
        pde.init_mesh.set('uniform_aligned_tri')
        displacement_mesh = pde.init_mesh(nx=nx, ny=ny)
    
        from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
        material = IsotropicLinearElasticMaterial(
                                            youngs_modulus=pde.E, 
                                            poisson_ratio=pde.nu, 
                                            plane_type=pde.plane_type,
                                            enable_logging=False
                                        )
        density_location = 'element'
        penalty_factor = 3.0
        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme = MaterialInterpolationScheme(
                                    density_location=density_location,
                                    interpolation_method='msimp',
                                    options={
                                        'penalty_factor': penalty_factor,
                                        'void_youngs_modulus': 1e-9,
                                        'target_variables': ['E']
                                    },
                                )
        
        relative_density = volume_fraction
        if density_location in ['element']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
                                                ) 
        elif density_location in ['node']:
            design_variable_mesh = displacement_mesh
            d, rho = interpolation_scheme.setup_density_distribution(
                                                    design_variable_mesh=design_variable_mesh,
                                                    displacement_mesh=displacement_mesh,
                                                    relative_density=relative_density,
                                                    integration_order=integration_order,
                                                )
        else:
            raise ValueError(f"不支持的密度位置类型: {density_location}")

        assembly_method = 'voigt'
        state_variable = 'u'
        space_degree = 3
        integration_order = space_degree + 1
        from soptx.analysis.lagrange_fem_analyzer import LagrangeFEMAnalyzer
        lagrange_fem_analyzer = LagrangeFEMAnalyzer(
                                    mesh=displacement_mesh,
                                    pde=pde,
                                    material=material,
                                    space_degree=space_degree,
                                    integration_order=integration_order,
                                    assembly_method=assembly_method,
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                    interpolation_scheme=interpolation_scheme,
                                )
        from soptx.optimization.compliance_objective import ComplianceObjective
        co_lfem = ComplianceObjective(analyzer=lagrange_fem_analyzer, state_variable=state_variable)
        c_lfem = co_lfem.fun(density=rho)
        
        state_variable = 'sigma'  # 'u', 'sigma'
        huzhang_space_degree = 3
        integration_order = huzhang_space_degree + 1
        from soptx.analysis.huzhang_mfem_analyzer import HuZhangMFEMAnalyzer
        huzhang_mfem_analyzer = HuZhangMFEMAnalyzer(
                                    mesh=displacement_mesh,
                                    pde=pde,
                                    material=material,
                                    space_degree=huzhang_space_degree,
                                    integration_order=integration_order,
                                    solve_method='mumps',
                                    topopt_algorithm='density_based',
                                    interpolation_scheme=interpolation_scheme,
                                )
        co_hzmfem = ComplianceObjective(analyzer=huzhang_mfem_analyzer, state_variable=state_variable)
        c_hzmfem = co_hzmfem.fun(density=rho)

        print(f"--------------")


if __name__ == '__main__':

    compliance_objective = ComplianceObjectiveTester(enable_logging=True)
    compliance_objective.run.set('test_lfem_hzmfem_compliance')
    compliance_objective.run(model_type='bearing_device_2d')