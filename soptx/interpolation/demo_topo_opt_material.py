"""
TopologyOptimizationMaterial 使用演示
展示拓扑优化材料的基本用法和关键功能
"""
from fealpy.backend import backend_manager as bm

# 1. 设置基础线弹性材料
from soptx.interpolation.linear_elastic_material import IsotropicLinearElasticMaterial
base_material = IsotropicLinearElasticMaterial(
                                    youngs_modulus=1.0, 
                                    poisson_ratio=0.3, 
                                    plane_type='plane_stress',
                                    enable_logging=False
                                )
print(f"基础材料参数: E={base_material.youngs_modulus}, ν={base_material.poisson_ratio}")

# 2. 设置材料插值方案
from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
interpolation_scheme = MaterialInterpolationScheme(
                                penalty_factor=3.0, 
                                void_youngs_modulus=1e-12, 
                                interpolation_method='simp', 
                                enable_logging=False
                            )
print(f"插值方案: {interpolation_scheme.interpolation_method}, p={interpolation_scheme.penalty_factor}")

# 3. 创建简单网格
from soptx.pde.mbb_beam_2d import HalfMBBBeam2dData1
pde = HalfMBBBeam2dData1(domain=[0, 2, 0, 1], T=-1.0, E=1.0, nu=0.3)
pde.init_mesh.set('uniform_quad')
mesh = pde.init_mesh(nx=4, ny=2)
print(f"网格信息: {mesh.number_of_cells()} 个单元")

# 4. TopologyOptimizationMaterial 基本用法演示
print("\n=== TopologyOptimizationMaterial 基本用法 ===")

from soptx.interpolation.topology_optimization_material import TopologyOptimizationMaterial

# 方式1: 单元密度分布 (最简单)
print("\n[方式1] 单元密度分布:")
topm1 = TopologyOptimizationMaterial(
    mesh=mesh,
    base_material=base_material,
    interpolation_scheme=interpolation_scheme,
    relative_density=0.5,
    density_location='element',
    enable_logging=False
)
print(f"  密度分布形状: {topm1.density_distribution.shape}")
print(f"  相对密度: {topm1.relative_density}")

# 方式2: 高斯积分点密度分布
print("\n[方式2] 高斯积分点密度分布:")
topm2 = TopologyOptimizationMaterial(
    mesh=mesh,
    base_material=base_material,
    interpolation_scheme=interpolation_scheme,
    relative_density=0.8,
    density_location='element_gauss_integrate_point',
    quadrature_order=1,
    enable_logging=False
)
print(f"  密度分布形状: {topm2.density_distribution.shape}")
print(f"  积分次数: {topm2.quadrature_order}")

# 5. 关键方法演示
print("\n=== 关键方法演示 ===")

# 更新相对密度
print("\n[密度更新]")
old_shape = topm1.density_distribution.shape
topm1.set_relative_density(0.3)
print(f"  密度从 0.5 更新到 {topm1.relative_density}")
print(f"  密度分布形状保持: {topm1.density_distribution.shape}")

# 切换密度位置
print("\n[密度位置切换]")
old_shape = topm1.density_distribution.shape
topm1.set_density_location('element_gauss_integrate_point', quadrature_order=3)
print(f"  从 'element' 切换到 'element_gauss_integrate_point'")
print(f"  密度分布形状: {old_shape} -> {topm1.density_distribution.shape}")

# 更新积分次数
print("\n[积分次数更新]")
old_shape = topm2.density_distribution.shape
topm2.set_quadrature_order(4)
print(f"  积分次数: 2 -> {topm2.quadrature_order}")
print(f"  密度分布形状: {old_shape} -> {topm2.density_distribution.shape}")

# 6. 代理方法演示（通过拓扑材料修改基础材料和插值方案）
print("\n=== 代理方法演示 ===")

# 修改插值方法
print(f"\n[插值方法] 原方法: {topm1.interpolation_method}")
topm1.set_interpolation_method('modified_simp')
print(f"           新方法: {topm1.interpolation_method}")

# 修改惩罚因子
print(f"\n[惩罚因子] 原因子: {topm1.penalty_factor}")
topm1.set_penalty_factor(5.0)
print(f"          新因子: {topm1.penalty_factor}")

# 7. 弹性矩阵计算演示
print("\n=== 弹性矩阵计算 ===")

# 计算插值后的弹性矩阵
D_interpolated = topm1.elastic_matrix()
print(f"插值弹性矩阵形状: {D_interpolated.shape}")
print(f"基础弹性矩阵形状: {base_material.elastic_matrix().shape}")

# 8. 材料信息显示
print("\n=== 材料信息 ===")
topm1.display_material_info()

print("\n=== 演示完成 ===")