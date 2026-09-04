"""体力向量组装的分步峰值内存诊断.

用途: 确认 ``experiments/matrix_free_capability`` 表 b-2 里 ``operator`` 之后那段
``3.25 GiB`` 瞬态究竟由哪一步产生, 并量化"按单元分块"能把它压到多少。

它**不装配刚度算子**, 因此进程里没有更高的历史高水位来遮盖后面的步骤 ——
这正是主 benchmark 里 FA 的 ``bc`` 阶段显示 ``+0.000`` 的原因。

口径与 ``benchmark_cpu_ea.py --mode serial-peak-rss`` 一致: 逐步读
``resource.getrusage(RUSAGE_SELF).ru_maxrss``, 得到的是**累积**高水位, 相邻两步
之差才是"这一步额外抬高了多少"。

两种模式必须各自独占一个进程 (高水位不可回落):

    python diag_body_force.py --mode stepwise --n 32
    python diag_body_force.py --mode chunked  --n 32 --chunk 65536

用法示例 (由粗到细, n=64 那档约需 3-4 GiB):

    for N in 16 32 64; do
        python diag_body_force.py --mode stepwise --n $N
    done
"""

from __future__ import annotations

import argparse
import resource
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "examples" / "matrix_free_elasticity"))

from fealpy.backend import backend_manager as bm
from fealpy.functional import linear_integral
from fealpy.utils import process_coef_func

import benchmark_cpu_ea as bench

GIB = 2 ** 30


def body_force_value(problem):
    """从统一载荷对象序列中取出唯一体力值函数."""
    load = next(load for load in problem.loads() if load.kind == "body_force")
    return load.body_force


def peak_bytes() -> int:
    """当前进程的驻留集高水位, 单位字节 (Linux 下 ru_maxrss 以 KiB 计)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class Trace:
    """按顺序记录累积高水位, 打印时换算成逐步增量."""

    def __init__(self) -> None:
        self._marks: list[tuple[str, int]] = [("baseline", peak_bytes())]

    def stamp(self, name: str) -> None:
        self._marks.append((name, peak_bytes()))

    def report(self) -> None:
        print(f"{'步骤':<28}{'累积 GiB':>12}{'增量 GiB':>12}")
        print("-" * 52)
        previous = self._marks[0][1]
        for name, value in self._marks:
            print(f"{name:<28}{value / GIB:>12.3f}{(value - previous) / GIB:>12.3f}")
            previous = value


def build(resolution: int):
    """构造 3D tet P1 的网格、张量空间与求积数据, 不碰刚度算子."""
    _, problem, vector_space, _, mesh = bench.build_context(
        "polynomial", "tet", resolution
    )
    # 与 LagrangeFEMAnalyzer 的默认一致: integration_order = degree + 3。
    order = bench.DEGREE + 3
    quadrature = mesh.quadrature_formula(order, "cell")
    bcs, ws = quadrature.get_quadrature_points_and_weights()
    return problem, vector_space, mesh, bcs, ws


def theory(mesh, vector_space, n_quadrature: int) -> None:
    """打印几个关键中间张量的理论字节数, 供与实测增量对照."""
    n_cells = int(mesh.number_of_cells())
    tldof = int(vector_space.cell_to_dof().shape[1])
    print(f"NC = {n_cells:,}  NQ = {n_quadrature}  tldof = {tldof}  "
          f"TGDOF = {vector_space.number_of_global_dofs():,}")
    for label, shape in (
        ("(NC, NQ)", (n_cells, n_quadrature)),
        ("(NC, NQ, 3)", (n_cells, n_quadrature, 3)),
        ("(NC, NQ, tldof)", (n_cells, n_quadrature, tldof)),
        ("(NC, NQ, tldof, 3)", (n_cells, n_quadrature, tldof, 3)),
    ):
        size = 8
        for axis in shape:
            size *= axis
        print(f"  {label:<22}{size / GIB:>10.3f} GiB")
    print()


def run_stepwise(resolution: int) -> None:
    """按 SourceIntegrator.assembly 的真实调用顺序逐步打点."""
    trace = Trace()
    problem, vector_space, mesh, bcs, ws = build(resolution)
    trace.stamp("mesh + space")

    theory(mesh, vector_space, int(ws.shape[0]))

    index = bm.arange(mesh.number_of_cells())
    cell_measure = mesh.entity_measure("cell", index=index)
    phi = vector_space.basis(bcs, index=index)
    trace.stamp(f"phi {tuple(phi.shape)} + cm")

    # process_coef_func 内部先算 ps 再调 coef(ps), 两者同时存活 —— 这里拆开打点。
    points = mesh.bc_to_point(bcs, index=index)
    trace.stamp("bc_to_point(ps)")

    body_force = body_force_value(problem)
    val = body_force(points)
    trace.stamp("body_force(ps)")

    del points
    trace.stamp("del ps (高水位不回落)")

    local = linear_integral(phi, ws, cell_measure, val, batched=False)
    trace.stamp(f"linear_integral -> {tuple(local.shape)}")

    # local 是逐单元的 (NC, tldof), 与 chunked 模式的全局向量不是同一个对象;
    # 必须散射到全局自由度后才能比对范数。
    F = bm.zeros((vector_space.number_of_global_dofs(),), dtype=bm.float64)
    bm.index_add(F, vector_space.cell_to_dof().reshape(-1), local.reshape(-1))
    trace.stamp(f"index_add -> {tuple(F.shape)}")

    norm = float(bm.linalg.norm(F))
    del local
    del val
    print()
    trace.report()
    print(f"\n单元数 {mesh.number_of_cells():,}, 峰值 {peak_bytes() / GIB:.3f} GiB")
    print(f"全局 F 的 L2 范数 = {norm:.15e}  (与 chunked 模式逐位可比)")


def run_chunked(resolution: int, chunk: int, verify: bool) -> None:
    """按单元分块流式累加的原型, 用于量化"分块能省多少".

    参数:
        resolution: 各轴剖分数.
        chunk: 每块的单元数.
        verify: 为 ``True`` 时额外走一遍未分块的官方路径并比对相对差. 该路径会
            物化全部中间张量, 只应在小 ``n`` 上开启.
    """
    trace = Trace()
    problem, vector_space, mesh, bcs, ws = build(resolution)
    trace.stamp("mesh + space")

    n_cells = int(mesh.number_of_cells())
    cell_to_dof = vector_space.cell_to_dof()
    body_force = body_force_value(problem)
    F = bm.zeros((vector_space.number_of_global_dofs(),), dtype=bm.float64)
    trace.stamp("F 全零向量")

    for start in range(0, n_cells, chunk):
        block = bm.arange(start, min(start + chunk, n_cells))
        cell_measure = mesh.entity_measure("cell", index=block)
        # phi 与单元无关, 形状 (1, NQ, tldof, 3), 每块重取的代价可忽略。
        phi = vector_space.basis(bcs, index=block)
        val = process_coef_func(
            body_force, bcs=bcs, mesh=mesh, etype="cell", index=block
        )
        local = linear_integral(phi, ws, cell_measure, val, batched=False)
        bm.index_add(F, cell_to_dof[block].reshape(-1), local.reshape(-1))
    trace.stamp(f"分块累加 (chunk={chunk:,})")

    print()
    trace.report()
    print(f"\n单元数 {n_cells:,}, 块数 {(n_cells + chunk - 1) // chunk}, "
          f"峰值 {peak_bytes() / GIB:.3f} GiB")
    print(f"全局 F 的 L2 范数 = {float(bm.linalg.norm(F)):.15e}  (与 stepwise 逐位可比)")

    if verify:
        # 分块只改变求和顺序, 不改变数值; 相对差应在浮点累加误差量级 (1e-16~1e-14)。
        reference = _reference_load(problem, vector_space, mesh, bcs, ws)
        difference = float(bm.linalg.norm(F - reference))
        scale = max(float(bm.linalg.norm(reference)), 1.0e-30)
        print(f"与未分块官方路径的相对差 = {difference / scale:.3e}")


def _reference_load(problem, vector_space, mesh, bcs, ws):
    """未分块的参考实现, 与 SourceIntegrator.assembly 逐行一致."""
    index = bm.arange(mesh.number_of_cells())
    cell_measure = mesh.entity_measure("cell", index=index)
    phi = vector_space.basis(bcs, index=index)
    body_force = body_force_value(problem)
    val = process_coef_func(
        body_force, bcs=bcs, mesh=mesh, etype="cell", index=index
    )
    local = linear_integral(phi, ws, cell_measure, val, batched=False)
    reference = bm.zeros((vector_space.number_of_global_dofs(),), dtype=bm.float64)
    bm.index_add(reference, vector_space.cell_to_dof().reshape(-1), local.reshape(-1))
    return reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("stepwise", "chunked"), default="stepwise")
    parser.add_argument("--n", type=int, default=32, help="各轴剖分数")
    parser.add_argument("--chunk", type=int, default=65536, help="chunked 模式的单元块大小")
    parser.add_argument("--verify", action="store_true",
                        help="chunked 模式下额外跑一遍未分块路径并比对数值 (只在小 n 上用)")
    arguments = parser.parse_args()

    bm.set_backend("numpy")
    print(f"=== mode={arguments.mode} n={arguments.n} ===")
    if arguments.mode == "stepwise":
        run_stepwise(arguments.n)
    else:
        run_chunked(arguments.n, arguments.chunk, arguments.verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
