"""机器精度交叉比对脚本 (compare_lagrange.py).

只回答两个问题:

1. **矩阵结果是否一致** —— EA 的算子作用与 FA 的全局 CSR 矩阵乘, 在同一个随机
   探针下差多少. 分裸算子与施加 Dirichlet 条件后两条: EA 侧是作用时现场置零,
   FA 侧是装配后对称消元, 代码路径不同, 不能并成一条. 量级应在 1e-16.
2. **求解结果是否一致** —— EA 的 matrix-free CG 解与 FA 的 Scipy 直接解差多少.
   这一条的量级由 CG 的停机准则决定, 不是机器精度.

本脚本判 PASS/FAIL 用的阈值来自 ``tools.matrix_free_evidence.contract``, 与
``validate.py`` 是同一份, 所以这里的结论和正式门禁不会各说各话.

本脚本只能串行运行: FA 在多 rank 下不存在 (对称消元发生在全局装配之后), 因此
"EA 对得上 FA" 这个问题本身只在单 rank 下成立. 跨 rank 的一致性由
``tools/matrix_free_evidence/validate.py --include-parallel`` 负责.

使用方法:
    python examples/matrix_free_elasticity/compare_lagrange.py           # 2D 平面应变
    python examples/matrix_free_elasticity/compare_lagrange.py --dim 3   # 3D 多项式无散场
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 门禁阈值住在仓库根下的 tools/, 先把仓库根放上 sys.path 才导得到
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import TetrahedronMesh, TriangleMesh

from soptx.fem.solvers import solve_ea_system
from soptx.fem.verification import relative_difference, serial_references
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems.elasticity import (
    DivergenceFreePolynomialElasticity3D,
    SinusoidalPlaneStrainElasticity2D,
)
from tools.matrix_free_evidence import contract


# 维数决定这三样, 其余一律向对象本身要: 区域与弹性常数由制造解自带, 网格实体名由
# 网格自带. 平面降维假设不属于制造解, 所以只能在这里定
PROBLEM_FACTORIES = {
    2: SinusoidalPlaneStrainElasticity2D,
    3: DivergenceFreePolynomialElasticity3D,
}
MESH_FACTORIES = {2: TriangleMesh, 3: TetrahedronMesh}
MATERIAL_HYPOTHESES = {2: "plane_strain", 3: "3D"}


def verdict(passed: bool) -> str:
    """门禁布尔值转成表格里那两个词"""

    return "PASS" if passed else "FAIL"


def run_cross_comparison(dimension: int = 2, resolution_n: int = 8) -> bool:
    """跑一轮 EA/FA 交叉比对, 打印结果表并返回是否全部通过门禁"""

    degree = contract.DEFAULT_DEGREE
    resolution = (resolution_n,) * dimension

    # 1. 制造解, 网格, 空间与材料: 区域取自制造解, 弹性常数取自制造解的 lam/mu,
    #    保证刚度算子和精确解建立在同一组参数上
    problem = PROBLEM_FACTORIES[dimension]()
    mesh = MESH_FACTORIES[dimension].from_box(
        list(problem.domain),
        **dict(zip(("nx", "ny", "nz"), resolution)),
    )
    scalar_space = LagrangeFESpace(mesh, p=degree, ctype="C")
    vector_space = TensorFunctionSpace(scalar_space, shape=(-1, dimension))
    num_dofs = vector_space.number_of_global_dofs()
    material = IsotropicLinearElasticMaterial(
        hypothesis=MATERIAL_HYPOTHESES[dimension],
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        device=bm.get_device(mesh),
    )

    # 2. 构造 EA/FA 分析器与全部参照量: 随机探针下的 MatVec 相对差, 正定性探针,
    #    以及 FA 施加边界条件后的 Scipy 直接解
    matvec_ref, direct_sol = serial_references(
        vector_space,
        problem,
        material,
        degree=degree,
        seed=contract.REFERENCE_RANDOM_SEED,
    )

    # 3. 用 EA 算子装配 matrix-free 线性系统并以 CG 求解, 容差走缺省, 与流水线一致
    cg_sol, _ = solve_ea_system(
        vector_space,
        problem,
        material,
        degree=degree,
        dof_comm=None,  # 串行执行, 无跨 rank 归约
    )

    # 4. 与 Scipy 直接解比相对差, 量级由 CG 的停机准则决定, 不是机器精度
    _, sol_rel_err = relative_difference(cg_sol, direct_sol)

    # 5. 施加门禁: 判据函数和 validate.py 是同一个, 不只是阈值相同
    gates = contract.matvec_reference_gates(matvec_ref)
    passed_sol = contract.explicit_solution_gate(sol_rel_err)
    all_passed = all(gates.values()) and passed_sol

    # 6. 打印对比表. 只有两个板块: 矩阵结果是否一致, 求解结果是否一致. 正定性
    #    探针参与判定但不单独占行 —— 它不是 EA/FA 之间的比对, 只是一条兜底断言
    print("\n" + "=" * 76)
    print(f" EA vs FA / Scipy Cross-Comparison [{dimension}D - {type(problem).__name__}]")
    print("=" * 76)
    print(f" Grid Resolution        : {'x'.join(str(r) for r in resolution)}")
    print(f" Total Global DOFs      : {num_dofs}")
    print("-" * 76)
    print(" [1] Matrix Agreement (EA vs FA, machine precision)")
    print(f"   Raw Operator MatVec  : {matvec_ref['raw_relative_error']:.5e}  (Tol: {contract.MATVEC_RELATIVE_TOL:g}) -> [{verdict(gates['raw_matvec'])}]")
    print(f"   With Dirichlet BC    : {matvec_ref['dirichlet_relative_error']:.5e}  (Tol: {contract.MATVEC_RELATIVE_TOL:g}) -> [{verdict(gates['dirichlet_matvec'])}]")
    print(f"   (positive-definite probe: energy={matvec_ref['random_vector_energy']:.5e} -> [{verdict(gates['positive_definite'])}])")
    print("-" * 76)
    print(" [2] Solution Agreement (EA-CG vs FA Scipy direct solve)")
    print(f"   Relative Difference  : {sol_rel_err:.5e}  (Tol: {contract.EXPLICIT_SOLUTION_RELATIVE_TOL:g}) -> [{verdict(passed_sol)}]")
    print("-" * 76)
    print(f" Overall Verification Result : [{'PASSED' if all_passed else 'FAILED'}]")
    print("=" * 76 + "\n")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Matrix-Free (EA) vs Full Assembly (FA) Precision Comparison"
    )
    parser.add_argument(
        "--dim",
        type=int,
        choices=(2, 3),
        default=2,
        help="Spatial dimension (2 or 3, default: 2)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="Mesh resolution along each axis (default: 8)",
    )
    args = parser.parse_args()
    success = run_cross_comparison(dimension=args.dim, resolution_n=args.n)
    sys.exit(0 if success else 1)
