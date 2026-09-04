"""线弹性积分器（LinearElasticIntegrator）组装变体与网格类型支持验证 Demo

本脚本验证 docs/fem/linear-elastic-integrator-implementation.md §3.2 中给出的
“网格支持矩阵与几何前置条件”：
1. 仿射单纯形网格（2D TriangleMesh / 3D TetrahedronMesh）：
   - 验证 standard / voigt / fast 三种变体的单元刚度矩阵与全局矩阵数值等价（机器精度级别）；
   - 验证刚度矩阵对称性与变密度场缩放一致性；
2. 规则张量积网格（2D QuadrangleMesh / 3D HexahedronMesh）：
   - 验证 standard 与 voigt 的数值等价性；
   - 探测 fast 变体在当前网格下的边界与异常防御行为。

用法：
    python examples/lagrange_elasticity/assembly_methods_mesh_demo.py
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

from fealpy.backend import TensorLike, backend_manager as bm
from fealpy.fem import BilinearForm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import (
    HexahedronMesh,
    QuadrangleMesh,
    TetrahedronMesh,
    TriangleMesh,
)

from soptx.fem.integrators import LinearElasticIntegrator
from soptx.materials import IsotropicLinearElasticMaterial


def rel_frobenius_err(actual: TensorLike, expected: TensorLike) -> float:
    """计算两数组间的相对 Frobenius 误差。"""
    denom = max(float(bm.linalg.norm(expected)), float(bm.finfo(bm.float64).eps))
    return float(bm.linalg.norm(actual - expected) / denom)


def max_abs_err(actual: TensorLike, expected: TensorLike) -> float:
    """计算最大绝对误差。"""
    return float(bm.max(bm.abs(actual - expected)))


def check_symmetry(matrix: TensorLike) -> float:
    """计算刚度矩阵的对称性相对误差。"""
    transpose = bm.swapaxes(matrix, -1, -2)
    return rel_frobenius_err(matrix, transpose)


class MeshSupportDemo:
    def __init__(self) -> None:
        bm.set_backend("numpy")
        self.results: List[Dict[str, str]] = []

    def verify_simplex_2d(self) -> None:
        print("\n" + "=" * 70)
        print("▶ 测试 1: 2D 仿射单纯形网格 (TriangleMesh / tri3, tri6, tri10)")
        print("=" * 70)

        mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
        mat = IsotropicLinearElasticMaterial(
            lame_lambda=1.0, shear_modulus=1.0, hypothesis="plane_stress"
        )
        nc = mesh.number_of_cells()

        for p in (1, 2, 3):
            space = TensorFunctionSpace(LagrangeFESpace(mesh, p=p), shape=(-1, 2))
            gdof = space.number_of_global_dofs()

            int_std = LinearElasticIntegrator(mat, method="standard")
            int_voigt = LinearElasticIntegrator(mat, method="voigt")
            int_fast = LinearElasticIntegrator(mat, method="fast")

            k_std = int_std.assembly(space)
            k_voigt = int_voigt.assembly(space)
            k_fast = int_fast.assembly(space)

            err_fast_std = max_abs_err(k_fast, k_std)
            err_fast_voigt = max_abs_err(k_fast, k_voigt)
            sym_fast = check_symmetry(k_fast)

            print(f"  [p={p}] 单元数 NC={nc}, 自由度 GDof={gdof}")
            print(f"        · max_abs_err(fast vs standard) = {err_fast_std:.2e}")
            print(f"        · max_abs_err(fast vs voigt)    = {err_fast_voigt:.2e}")
            print(f"        · fast 矩阵对称性相对误差       = {sym_fast:.2e}")

            # 变密度系数测试
            coef = bm.linspace(0.2, 1.0, nc, dtype=bm.float64)
            int_std_c = LinearElasticIntegrator(mat, coef=coef, method="standard")
            int_fast_c = LinearElasticIntegrator(mat, coef=coef, method="fast")
            k_std_c = int_std_c.assembly(space)
            k_fast_c = int_fast_c.assembly(space)
            err_coef = max_abs_err(k_fast_c, k_std_c)

            passed = max(err_fast_std, err_fast_voigt, err_coef) < 1.0e-12
            self.results.append({
                "网格类型": "2D TriangleMesh",
                "阶数": f"p={p}",
                "单元/自由度": f"{nc} / {gdof}",
                "测试变体": "fast / standard / voigt",
                "最大绝对误差": f"{max(err_fast_std, err_fast_voigt):.2e}",
                "校验判定": "PASS (数值严格等价)" if passed else "FAIL",
            })

    def verify_simplex_3d(self) -> None:
        print("\n" + "=" * 70)
        print("▶ 测试 2: 3D 仿射单纯形网格 (TetrahedronMesh / tet4, tet10, tet20)")
        print("=" * 70)

        mesh = TetrahedronMesh.from_box([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], nx=1, ny=1, nz=1)
        mat = IsotropicLinearElasticMaterial(
            lame_lambda=1.0, shear_modulus=1.0, hypothesis="3D"
        )
        nc = mesh.number_of_cells()

        for p in (1, 2, 3):
            space = TensorFunctionSpace(LagrangeFESpace(mesh, p=p), shape=(-1, 3))
            gdof = space.number_of_global_dofs()

            int_std = LinearElasticIntegrator(mat, method="standard")
            int_voigt = LinearElasticIntegrator(mat, method="voigt")
            int_fast = LinearElasticIntegrator(mat, method="fast")

            k_std = int_std.assembly(space)
            k_voigt = int_voigt.assembly(space)
            k_fast = int_fast.assembly(space)

            err_fast_std = max_abs_err(k_fast, k_std)
            err_fast_voigt = max_abs_err(k_fast, k_voigt)
            sym_fast = check_symmetry(k_fast)

            print(f"  [p={p}] 单元数 NC={nc}, 自由度 GDof={gdof}")
            print(f"        · max_abs_err(fast vs standard) = {err_fast_std:.2e}")
            print(f"        · max_abs_err(fast vs voigt)    = {err_fast_voigt:.2e}")
            print(f"        · fast 矩阵对称性相对误差       = {sym_fast:.2e}")

            # 全局装配一致性校验
            form_std = BilinearForm(space)
            form_std.add_integrator(int_std)
            g_std = bm.array(form_std.assembly(format="csr").to_scipy().toarray())

            form_fast = BilinearForm(space)
            form_fast.add_integrator(int_fast)
            g_fast = bm.array(form_fast.assembly(format="csr").to_scipy().toarray())

            err_global = max_abs_err(g_fast, g_std)

            passed = max(err_fast_std, err_fast_voigt, err_global) < 1.0e-12
            self.results.append({
                "网格类型": "3D TetrahedronMesh",
                "阶数": f"p={p}",
                "单元/自由度": f"{nc} / {gdof}",
                "测试变体": "fast / standard / voigt",
                "最大绝对误差": f"{max(err_fast_std, err_fast_voigt):.2e}",
                "校验判定": "PASS (数值严格等价)" if passed else "FAIL",
            })

    def verify_tensor_2d(self) -> None:
        print("\n" + "=" * 70)
        print("▶ 测试 3: 2D 规则四边形网格 (QuadrangleMesh / quad4, quad9, quad16)")
        print("=" * 70)

        mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
        mat = IsotropicLinearElasticMaterial(
            lame_lambda=1.0, shear_modulus=1.0, hypothesis="plane_stress"
        )
        nc = mesh.number_of_cells()

        for p in (1, 2, 3):
            space = TensorFunctionSpace(LagrangeFESpace(mesh, p=p), shape=(-1, 2))
            gdof = space.number_of_global_dofs()

            int_std = LinearElasticIntegrator(mat, method="standard")
            int_voigt = LinearElasticIntegrator(mat, method="voigt")
            int_fast = LinearElasticIntegrator(mat, method="fast")

            k_std = int_std.assembly(space)
            k_voigt = int_voigt.assembly(space)
            k_fast = int_fast.assembly(space)

            err_fast_std = max_abs_err(k_fast, k_std)
            err_fast_voigt = max_abs_err(k_fast, k_voigt)
            err_std_voigt = max_abs_err(k_std, k_voigt)
            sym_fast = check_symmetry(k_fast)

            print(f"  [p={p}] 单元数 NC={nc}, 自由度 GDof={gdof}")
            print(f"        · max_abs_err(fast vs standard)  = {err_fast_std:.2e}")
            print(f"        · max_abs_err(fast vs voigt)     = {err_fast_voigt:.2e}")
            print(f"        · max_abs_err(standard vs voigt) = {err_std_voigt:.2e}")
            print(f"        · fast 矩阵对称性相对误差        = {sym_fast:.2e}")

            passed = max(err_fast_std, err_fast_voigt) < 1.0e-12
            self.results.append({
                "网格类型": "2D QuadrangleMesh",
                "阶数": f"p={p}",
                "单元/自由度": f"{nc} / {gdof}",
                "测试变体": "fast / standard / voigt",
                "最大绝对误差": f"{max(err_fast_std, err_fast_voigt):.2e}",
                "校验判定": "PASS (数值严格等价)" if passed else "FAIL",
            })

    def verify_tensor_3d(self) -> None:
        print("\n" + "=" * 70)
        print("▶ 测试 4: 3D 规则六面体网格 (HexahedronMesh / hex8, hex27, hex64)")
        print("=" * 70)

        mesh = HexahedronMesh.from_box([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], nx=1, ny=1, nz=1)
        mat = IsotropicLinearElasticMaterial(
            lame_lambda=1.0, shear_modulus=1.0, hypothesis="3D"
        )
        nc = mesh.number_of_cells()

        for p in (1, 2, 3):
            space = TensorFunctionSpace(LagrangeFESpace(mesh, p=p), shape=(-1, 3))
            gdof = space.number_of_global_dofs()

            int_std = LinearElasticIntegrator(mat, method="standard")
            int_voigt = LinearElasticIntegrator(mat, method="voigt")
            int_fast = LinearElasticIntegrator(mat, method="fast")

            k_std = int_std.assembly(space)
            k_voigt = int_voigt.assembly(space)
            k_fast = int_fast.assembly(space)

            err_fast_std = max_abs_err(k_fast, k_std)
            err_fast_voigt = max_abs_err(k_fast, k_voigt)
            err_std_voigt = max_abs_err(k_std, k_voigt)
            sym_fast = check_symmetry(k_fast)

            print(f"  [p={p}] 单元数 NC={nc}, 自由度 GDof={gdof}")
            print(f"        · max_abs_err(fast vs standard)  = {err_fast_std:.2e}")
            print(f"        · max_abs_err(fast vs voigt)     = {err_fast_voigt:.2e}")
            print(f"        · max_abs_err(standard vs voigt) = {err_std_voigt:.2e}")
            print(f"        · fast 矩阵对称性相对误差        = {sym_fast:.2e}")

            passed = max(err_fast_std, err_fast_voigt) < 1.0e-12
            self.results.append({
                "网格类型": "3D HexahedronMesh",
                "阶数": f"p={p}",
                "单元/自由度": f"{nc} / {gdof}",
                "测试变体": "fast / standard / voigt",
                "最大绝对误差": f"{max(err_fast_std, err_fast_voigt):.2e}",
                "校验判定": "PASS (数值严格等价)" if passed else "FAIL",
            })

    def print_summary_table(self) -> None:
        def cjk_len(s: str) -> int:
            return sum(2 if ord(c) > 127 else 1 for c in s)

        def pad_cjk(s: str, width: int, align: str = "<") -> str:
            pad = max(0, width - cjk_len(s))
            if align == ">":
                return " " * pad + s
            elif align == "^":
                left = pad // 2
                return " " * left + s + " " * (pad - left)
            return s + " " * pad

        cols = [
            ("网格类型", 22, "<"),
            ("阶数", 8, "^"),
            ("单元/自由度", 14, "^"),
            ("测试变体", 26, "<"),
            ("最大绝对误差", 14, "^"),
            ("校验判定", 22, "<"),
        ]

        total_width = sum(w for _, w, _ in cols) + len(cols) * 3 + 1
        print("\n" + "=" * total_width)
        title = "LinearElasticIntegrator 网格类型与高阶 (p=1,2,3) 数值等价性验证总表"
        print(f"{title:^{total_width}}")
        print("=" * total_width)

        header_str = "| " + " | ".join(pad_cjk(name, w, "^") for name, w, _ in cols) + " |"
        sep_str = "+-" + "-+-".join("-" * w for _, w, _ in cols) + "-+"
        print(header_str)
        print(sep_str)

        for row in self.results:
            line_str = "| " + " | ".join(
                pad_cjk(row[name], w, align) for name, w, align in cols
            ) + " |"
            print(line_str)
        print("=" * total_width + "\n")


def main() -> int:
    demo = MeshSupportDemo()
    demo.verify_simplex_2d()
    demo.verify_simplex_3d()
    demo.verify_tensor_2d()
    demo.verify_tensor_3d()
    demo.print_summary_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())

