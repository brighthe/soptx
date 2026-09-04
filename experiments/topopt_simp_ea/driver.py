# -*- coding: utf-8 -*-
"""EA 变密度拓扑优化的执行驱动: 跑一次运行, 写这一次运行的全部产物.

可以直接执行, 也被 ``run.py`` 就地调用 (同进程 import, argv 是两层之间唯一的
接口, 口径同 experiments/huzhang_topopt_paper)::

    python driver.py --case half_mbb_2d_concentrated
    python driver.py --case half_mbb_2d_concentrated --override simp_penalty=4.0

``--case`` 只认注册工况 id; 带 override 的运行不在注册表里, 由 id + override 推导
出 run_id (= 产物目录 outputs/<工况 id>/<参数标签>/), 一条命令一次运行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
from fealpy.backend import backend_manager as bm
from soptx.postprocess.vtk_export import write_vtu

import provenance
from collect import validate_result
from config import (
    CASES_FILE,
    ConfigError,
    config_values,
    MESH_LAYOUT,
    TopOptCase,
    build_overridden_case,
    load,
    parse_override_args,
)
from pipeline import FORMULATION, build_pipeline


def _join(values) -> str:
    """把 grid / domain 序列渲染成 160x100 这样的紧凑文本."""
    return "x".join(f"{value:g}" for value in values)


def _history_records(history: Any) -> list[dict[str, Any]]:
    records = []
    for position, iteration in enumerate(history.iter_indices):
        record = {
            "iter": int(iteration),
            "change": float(history.changes[position]),
            "iteration_time": float(history.iteration_times[position]),
        }
        record.update(
            {
                name: float(values[position])
                for name, values in history.scalar_histories.items()
            }
        )
        records.append(record)
    return records


def _stage_dir(output_dir: Path) -> Path:
    """腾出一个空的暂存目录; 上次崩溃遗留的残骸在这里清掉."""
    staging = output_dir.with_name(output_dir.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def _publish(staging: Path, target: Path) -> None:
    """把写满的暂存目录换成正式产物目录.

    两次 rename 各自原子, 所以 target 在任一时刻要么是上一次的完整结果,
    要么是这一次的, 不会出现 "新 VTU 配旧 summary.json" 这种混合目录 ——
    collect 判完成只看 summary.json 在不在, 混合目录会被它当成有效结果。
    崩在两次 rename 之间只会留下 .partial/ 与 .previous/, 都不是注册的
    产物目录, collect 视而不见, 下次运行开头清掉。
    """
    previous = target.with_name(target.name + ".previous")
    if previous.exists():
        shutil.rmtree(previous)
    if target.exists():
        target.rename(previous)
    staging.rename(target)
    shutil.rmtree(previous, ignore_errors=True)


def _write_density_vtu_history(
    mesh: Any,
    raw_history: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """把全部迭代密度写为 VTU，并生成 ParaView PVD 时间序列.

    output_dir 是 _stage_dir 给的空暂存目录, 不必清理旧帧。
    """
    vtu_dir = output_dir / "vtu"
    vtu_dir.mkdir(parents=True, exist_ok=True)

    root = ET.Element(
        "VTKFile",
        type="Collection",
        version="0.1",
        byte_order="LittleEndian",
    )
    collection = ET.SubElement(root, "Collection")
    for iteration, physical_density in zip(
        raw_history.iter_indices,
        raw_history.physical_densities,
        strict=True,
    ):
        density = np.asarray(
            bm.to_numpy(physical_density[:]),
            dtype=np.float64,
        ).reshape(-1)
        stem = f"density_iter_{int(iteration):04d}"
        write_vtu(
            mesh=mesh,
            filepath=str(vtu_dir / stem),
            cell_data={"density": density},
        )
        ET.SubElement(
            collection,
            "DataSet",
            timestep=str(int(iteration)),
            group="",
            part="0",
            file=f"{stem}.vtu",
        )

    series_path = vtu_dir / "density_history.pvd"
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(series_path, encoding="utf-8", xml_declaration=True)
    return {
        "directory": "vtu",
        "series": "vtu/density_history.pvd",
        "file_count": len(raw_history.iter_indices),
    }


def run_case(
    case: TopOptCase,
    *,
    enable_timing: bool = False,
    quiet: bool = False,
) -> None:
    """运行一次 (基准或带 override 的) 运行并写入数值证据和完整 VTU 迭代历史."""
    note = ""
    if case.overrides:
        pairs = ", ".join(f"{name}={text}" for name, text in case.overrides)
        note = f" (override: {pairs})"
    # 头行报 run_id, 它就是 outputs/ 下的产物目录路径; 收尾的 [out] 报同一个名字。
    print(f"[run] {case.run_id}{note}: {FORMULATION}, {case.summary}")
    optimizer, design_variable, density_distribution, mesh, analyzer = (
        build_pipeline(case)
    )
    n_cells = int(mesh.number_of_cells())
    n_nodes = int(mesh.number_of_nodes())
    # 位移自由度按张量空间实取: P>1 时不等于 节点数 x 维数。
    n_dofs = int(analyzer.tensor_space.number_of_global_dofs())
    scalar_space = analyzer.scalar_space
    # 优化侧参数一律读活对象: cases.toml 未声明的 OC 高级项取库默认值,
    # 只有打出来才不会隐身。
    opts = optimizer.options
    rho0 = float(
        np.mean(np.asarray(bm.to_numpy(density_distribution[:]), dtype=np.float64))
    )
    # 以下分行与 cases.toml 同一分类轴: A 问题 / B 离散 / C 拓扑建模 / D 算法。
    # 各行里能用 --override 改的数值一律打 cases.toml 的字段名 (simp_penalty 而
    # 非 p): 回执上看到的名字就是命令行能用的名字。cases.toml 未声明、取库默认值的
    # OC 高级项打库属性全名; A 层只有 E / nu 例外, 保留连续问题的论文符号。
    print(
        f"[problem] {case.problem} ({analyzer.pde.plane_type}), "
        f"domain={_join(case.domain)}, "
        f"load={case.load:g}, "
        f"E={case.emax:g}, nu={case.nu:g}, volfrac <= {case.volfrac:g}"
    )
    print(
        f"[mesh] {case.cell_type} grid={_join(case.grid)} = {n_cells} 单元, "
        f"{n_nodes} 节点"
    )
    print(
        f"[space] analyzer=lfem 位移 Lagrange P{scalar_space.p} "
        f"(ctype={scalar_space.ctype}, "
        f"张量 {analyzer.tensor_space.shape}), 自由度 {n_dofs}, "
        f"integration_order={case.integration_order}"
    )
    filter_text = f"{case.filter_type} (filter_radius={case.filter_radius:g}"
    if case.filter_type == "projection":
        filter_text += (
            f", projection_beta={case.projection_beta:g}"
            f"->{case.projection_beta_max:g} 每 "
            f"projection_continuation_iter={case.projection_continuation_iter} 迭代, "
            f"projection_eta={case.projection_eta:g}"
        )
    filter_text += ")"
    print(
        f"[topopt] interpolation_method={case.interpolation_method} "
        f"(simp_penalty={case.simp_penalty:g}, "
        f"void_youngs_modulus={case.void_youngs_modulus:g}), "
        f"filter_type={filter_text}, "
        f"密度 {analyzer.interpolation_scheme.density_location} "
        f"(初值均匀 rho0={rho0:g})"
    )
    solver = case.solve_method
    if case.solve_method == "cg":
        solver += (
            f" (cg_rtol={case.cg_rtol:g}, cg_maxiter={case.cg_maxiter}, "
            f"cg_precond={case.cg_precond})"
        )
    print(
        f"[solve] 算子 EA (matrix-free, 逐单元 gather/scatter), "
        f"assembly_method={case.assembly_method}, "
        f"solve_method={solver}"
    )
    if case.optimizer == "oc":
        detail = (
            f"move={opts.move_limit:g}, "
            f"density_min={opts.design_variable_min:g}, "
            f"damping_coef={opts.damping_coef:g}, "
            f"initial_lambda={opts.initial_lambda:g}, "
            f"bisection_tol={opts.bisection_tol:g}"
        )
    else:
        detail = f"move={opts.move_limit:g}, density_min={case.density_min:g}"
    print(
        f"[optim] {case.optimizer.upper()} ({detail}), "
        f"终止准则 max_iter={opts.max_iterations} 或 "
        f"tol_change<={opts.change_tolerance:g}"
    )
    # 组件默认静默 (供 compare/测试等程序化调用)；CLI 交互运行打开逐迭代日志.
    if not quiet:
        optimizer.enable_logging(True)
    density, raw_history = optimizer.optimize(
        design_variable=design_variable,
        density_distribution=density_distribution,
        enable_timing=enable_timing,
    )
    density_array = np.asarray(bm.to_numpy(density[:]), dtype=np.float64)
    history = _history_records(raw_history)
    if not history:
        raise RuntimeError(f"运行 {case.run_id} 未生成优化历史.")
    # 全部产物先写进暂存目录, 最后一步整目录换上去 (见 _publish)。
    staging = _stage_dir(case.output_dir)
    # 密度只存 VTU: pyevtk 写的是 appended raw float64, read_vtu_cell_data
    # 读回逐位相同, 所以 collect/compare 直接以 VTU 为数据源, 不再另存 .npy。
    # density_final.vtu 与 vtu/ 下末次迭代内容相同, 单独写一份是为了在
    # ParaView 里一眼可见最终构型 (口径同 huzhang_topopt_paper)。
    write_vtu(
        mesh=mesh,
        filepath=str(staging / "density_final"),
        cell_data={"density": density_array},
    )
    vtu_history = _write_density_vtu_history(mesh, raw_history, staging)
    validation = validate_result(case, density_array, history)
    optimizer_options = {
        "initial_density": rho0,
        "move_limit": float(opts.move_limit),
        "max_iterations": int(opts.max_iterations),
        "change_tolerance": float(opts.change_tolerance),
    }
    if case.optimizer == "oc":
        optimizer_options.update(
            {
                "design_variable_min": float(opts.design_variable_min),
                "damping_coef": float(opts.damping_coef),
                "initial_lambda": float(opts.initial_lambda),
                "bisection_tol": float(opts.bisection_tol),
            }
        )
    last = history[-1]
    summary = {
        "case_id": case.id,
        "run_id": case.run_id,
        # 影响结果的参数一次性按字段名原样落盘, 与 cases.toml 出自同一份声明:
        # 这里不重新起名、不挑字段, 少一个就是门禁少查一个。
        "config": config_values(case),
        # override 原文按字段名排序落盘; collect 据此在注册表基准上重放出同一次
        # 运行, compare 据此在 FA 侧重放出配对运行。参数取值已在 config 块里。
        "overrides": dict(case.overrides) or None,
        "method": "EA-SIMP",
        "assembly_level": "element",
        "backend": "fealpy-numpy",
        # 以下为求解过程算出来的量, 不是注册表里的参数。
        "plane_type": analyzer.pde.plane_type,
        "mesh_layout": MESH_LAYOUT,
        "n_cells": int(density_array.size),
        "n_nodes": n_nodes,
        "space_ctype": scalar_space.ctype,
        "n_dofs": n_dofs,
        "optimizer_options": optimizer_options,
        "fa_reference": case.fa_reference or None,
        "iterations": len(history),
        "converged": validation["converged"],
        "final_compliance": float(last["compliance"]),
        "final_volume_fraction": float(last["volfrac"]),
        "final_change": float(last["change"]),
        "mean_iteration_time": float(np.mean(raw_history.iteration_times)),
        "vtu_history": vtu_history,
        "validation": validation,
        "provenance": provenance.capture((CASES_FILE,)),
    }
    for name, value in (("history.json", history), ("summary.json", summary)):
        (staging / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    _publish(staging, case.output_dir)
    print(f"[run] validation={'PASS' if validation['passed'] else 'FAIL'}")
    print(
        f"[out] {case.run_id}: summary.json, history.json, "
        f"density_final.vtu, vtu/ ({vtu_history['file_count']} 帧迭代密度 "
        f"+ density_history.pvd)"
    )


def build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: run.py 把它不认识的参数原样透传到这里, 前缀匹配会把
    # 写错的选项静默认成另一个, 报错比猜好。
    parser = argparse.ArgumentParser(
        prog="driver.py",
        description="EA 变密度拓扑优化的单次运行驱动",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--case",
        required=True,
        metavar="ID",
        help="注册工况 id (run.py --list 首列)",
    )
    parser.add_argument("--timing", action="store_true", help="输出迭代内部阶段计时")
    parser.add_argument(
        "--quiet", action="store_true", help="关闭逐迭代日志 (批量长跑时用)"
    )
    parser.add_argument(
        "--override",
        action="append",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="在 --case 的基准参数上改若干字段 (一个旗标可带多组, 旗标本身也可"
        "重复; 列表值用逗号, 如 grid=60,20)。结果写入 outputs/<id>/<字段>-<值>/, "
        "与基准运行同属一个工况目录; FA 侧用同一组 override 即可配对 compare",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """跑 --case 指定的那一次运行; argv 为 None 时取 sys.argv[1:]."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        overrides = parse_override_args(arguments.override)
    except ConfigError as error:
        parser.error(str(error))
    bm.set_backend("numpy")
    _, cases = load()
    # id 全表唯一由 config.load 保证, 这里最多命中一条。
    selected = tuple(case for case in cases if case.id == arguments.case)
    if not selected:
        parser.error(f"没有匹配的工况: {arguments.case}")
    case = selected[0]
    try:
        if overrides:
            case = build_overridden_case(case, overrides)
    except ConfigError as error:
        print(f"配置错误: {error}", file=sys.stderr)
        return 1
    run_case(case, enable_timing=arguments.timing, quiet=arguments.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
