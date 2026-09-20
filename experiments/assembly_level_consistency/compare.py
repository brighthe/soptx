# -*- coding: utf-8 -*-
"""矩阵组装层级正确性证据的汇总报表.

读制造解求解链的 JSON, 打印逐档 L2 误差与实测阶, 判末档阶是否够门禁。每个算子层级
(fa / ea / pa / ua) 各占一条工况, 一条工况下每种网格各一条链; 迭代解法的链另打印逐档迭代数。

判读顺序是先横向后纵向: 同一档上 ea 与 pa 的迭代数应逐项相等 —— 二者是同一离散算子的
两种存法, 对同一初值与同一右端张成同一个 Krylov 子空间, 迭代序列因而逐步相同, 只差舍入,
与用哪种 Krylov 方法无关。这比 L2 阶门禁灵敏得多; 确认迭代数对齐之后, 再看各条链
自己的末档阶。ua 对 pa 的判据更硬: 两者的几何量同出 levels/_quadrature.py 的
quadrature_geometry, 之后走同一串 einsum, L2 误差列应当逐位相同。各条链逐档的 L2
误差也应彼此吻合到求解容差量级, 那就是跨层级一致性 ——
它原先由一个独立的 correctness 面板承担, 已随该面板删除, 理由见 cases.toml 头部注意 0。

内存与 MatVec 耗时的对比报表在 ``experiments/assembly_level_capability/compare.py``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _THIS_DIR / "outputs"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import config  # noqa: E402

# 与 examples/lagrange_elasticity/manufactured_convergence_demo.py 的同名常量对齐:
# 只有这些求解器的产物里 niter / 预条件子才有意义
ITERATIVE_SOLVERS = ("cg",)

def load_artifact(filename: str, output_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """安全读取单点 JSON 产物."""
    p = (output_dir or _OUTPUT_DIR) / filename
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def show_convergence_table(output_dir: Optional[Path] = None) -> bool:
    """逐条打印制造解求解链的 L2 误差与实测收敛阶.

    工况与产物名从 ``cases.toml`` 读, 不在本文件写死: 收敛链的文件名由上游脚本按
    (dim, mesh_type, model, degree, solver, assembly_method) 拼成, 在两处各抄一份
    迟早对不上。

    Parameters
    ----------
    output_dir : Path, optional
        自定义产物目录, 默认本目录 ``outputs/``。

    Returns
    -------
    bool
        全部 convergence 工况都有产物且末档 L2 阶达门禁时为 True。
    """
    print("\n" + "=" * 100)
    print("【convergence: 各算子层级求解链的制造解收敛阶】")
    print("=" * 100)

    try:
        _, cases = config.load_cases()
    except config.ConfigError as error:
        print(f"  [cases.toml 有误: {error}]")
        print("=" * 100 + "\n")
        return False

    targets = [c for c in cases if c.panel == "convergence"]
    if not targets:
        print("  [cases.toml 未注册 convergence 工况]")
        print("=" * 100 + "\n")
        return False

    all_passed = True
    for case in targets:
        data = load_artifact(case.artifact, output_dir)
        print(f"\n  {case.ref} [{case.scheme.upper()}]: {case.summary}")
        if not data:
            print(f"    [未生成产物: {case.artifact}] 先运行: python run.py --case {case.id} --mesh {case.mesh}")
            all_passed = False
            continue

        # 迭代解法才有 niter; 直接解法那几档是 None, 整列省掉免得排一列空。
        has_niter = any(level.get("niter") is not None for level in data["levels"])
        head = f"    {'剖分':>6} {'单元数':>10} {'自由度':>10} {'L2 误差':>12} {'实测阶':>8} {'用时/s':>9}"
        if has_niter:
            head += f" {'niter':>7} {'conv':>6}"
        print(head)
        for level in data["levels"]:
            order = level.get("l2_order")
            order_text = "  —" if order is None else f"{order:8.3f}"
            line = (
                f"    {level['subdivisions']:>6d} {level['cells']:>10,d} {level['dofs']:>10,d} "
                f"{level['l2_error']:>12.3e} {order_text:>8} {level['seconds']:>9.1f}"
            )
            if has_niter:
                niter = level.get("niter")
                line += f" {'—' if niter is None else niter:>7} {str(level.get('converged')):>6}"
            print(line)

        gate = data.get("minimum_l2_order_gate", case.min_l2_order)
        final = data.get("final_l2_order")
        ok = final is not None and final >= gate and bool(data.get("l2_error_decreasing"))
        all_passed = all_passed and ok
        print(
            f"    末档 L2 阶 = {final:.3f} (理论 {data.get('theoretical_order')}, 门禁 {gate}), "
            f"误差单调下降 = {data.get('l2_error_decreasing')}, 判定 = {'PASS' if ok else 'FAIL'}"
        )
        line = (
            f"    算子层级 = {data.get('operator_level', 'fa')}, 求解器 = {data.get('solver')}, "
            f"装配路径 = {data.get('assembly_method')}"
        )
        # 迭代链才有预条件子可言。这里不能用 has_niter 判: 直接解法的产物也把 niter
        # 记成 1。旧产物没有 preconditioner 字段, 缺了就明说"未记录", 不默认成 none
        # —— niter 列的判读全靠它, 猜一个反而比空着更危险。
        if data.get("solver") in ITERATIVE_SOLVERS:
            if "preconditioner" in data:
                pc = data["preconditioner"] or "none"
                if data.get("preconditioner_level"):
                    pc += f" @ {data['preconditioner_level']}"
            else:
                pc = "未记录 (产物早于该字段, 需重跑)"
            line += f", 预条件子 = {pc}"
        print(line + f", 最大残差 = {data.get('max_residual'):.2e}")

    print("\n" + "-" * 100)
    print(f"  all_passed = {all_passed}")
    print("=" * 100 + "\n")
    return all_passed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="compare.py", description="assembly_level_consistency 结果对比报表")
    parser.add_argument("--list", action="store_true", help="列出可打印的报表")
    parser.add_argument("--case", choices=["convergence", "all"], default="all",
                        help="指定展示的报表 (默认 all, 眼下与 convergence 等价)")
    parser.add_argument("--output-dir", type=Path, default=None, help="自定义产物目录")
    parser.add_argument("--strict", action="store_true", help="有数据点缺产物或未过门禁时返回非零退出码")
    args = parser.parse_args(argv)

    if args.list:
        print("\ncase          description")
        print("------------  ------------------------------------------------------------")
        print("convergence   各算子层级 (fa / ea / pa) 求解链的制造解 L2 收敛阶")
        print("all           以上全部\n")
        return 0

    all_passed = show_convergence_table(args.output_dir)
    return 0 if (all_passed or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
