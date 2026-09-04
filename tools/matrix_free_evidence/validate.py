from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

# 本进程只派发子进程, 自身不做任何 MPI 计算, 因此必须保持在 MPI 之外。
#
# 历史成因是 ``soptx`` 包根曾为 MUMPS 主动 ``from mpi4py import MPI``, 使得只
# 导入 ``contract`` 也会连带 ``MPI_Init``; 该副作用已收进
# ``soptx.core.mpi_runtime.ensure_mpi_initialized``, 本进程不再有这条路径。
# 保留本设置作兜底: 一旦本进程成为 MPI singleton, 从它内部再启动 ``mpiexec``
# 就是 "MPI 作业里套 MPI 作业", OpenMPI 的 ``orterun`` 会立刻以 1 退出且不打印
# 任何诊断, 表现为每个算例都 "process returned 1 / summary was not created",
# 极难定位。
#
# 关掉自动初始化即可 —— 子进程是各自独立的解释器, 照常初始化, 不受影响。
try:
    import mpi4py

    mpi4py.rc.initialize = False
except ImportError:
    pass

from tools.matrix_free_evidence import contract, layout, schema


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the 2D/3D Matrix-Free elasticity baseline."
    )
    parser.add_argument(
        "--dim",
        choices=("2", "3", "all"),
        default="all",
    )
    parser.add_argument(
        "--include-parallel",
        action="store_true",
        help=(
            "阶段 1b: 追加 2-rank EA 算例与 1/2-rank 一致性门禁。"
            "默认只跑阶段 1a 的 CPU 串行范围。"
        ),
    )
    return parser.parse_args()


def selected_dimensions(value: str) -> tuple[int, ...]:
    if value == "all":
        return contract.SUPPORTED_DIMENSIONS
    return (int(value),)


def validate_parameter_checks(
    mpiexec: str,
) -> tuple[dict[str, dict], list[str]]:
    """Require every unsupported CLI configuration to fail explicitly."""

    specifications = (
        (
            "invalid-dimension",
            1,
            ("--dim", "4"),
            "invalid choice",
        ),
        (
            "two-dimensional-nz",
            1,
            ("--dim", "2", "--nx", "1", "--ny", "1", "--nz", "1"),
            "--nz is only valid when --dim 3",
        ),
        (
            "non-positive-resolution",
            1,
            ("--dim", "2", "--nx", "0", "--ny", "1"),
            "all mesh resolution values must be positive",
        ),
        (
            "unsupported-degree",
            1,
            ("--dim", "2", "--p", "2"),
            "stage 1 currently supports only --p 1",
        ),
        (
            "fa-multiple-ranks",
            2,
            (
                "--dim",
                "2",
                "--operator-level",
                "fa",
                "--nx",
                "1",
                "--ny",
                "1",
            ),
            "FA operator level currently supports one MPI rank",
        ),
    )
    directory = layout.OUTPUT_DIR / "parameter-checks"
    directory.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    failures: list[str] = []

    for name, ranks, arguments, expected_message in specifications:
        summary = directory / f"{name}.json"
        solution = summary.with_suffix(".npy")
        output = directory / f"{name}.vtu"
        for artifact in (summary, solution, output):
            artifact.unlink(missing_ok=True)
        command = [
            mpiexec,
            "-n",
            str(ranks),
            sys.executable,
            str(layout.RUN_SCRIPT),
            *arguments,
            "--summary",
            str(summary),
            "--output",
            str(output),
        ]
        print(
            f"\n[parameter/{name}] {' '.join(command)}",
            flush=True,
        )
        completed = subprocess.run(
            command,
            cwd=layout.REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        combined_output = (completed.stdout or "") + (
            completed.stderr or ""
        )
        artifacts_written = [
            str(path)
            for path in (summary, solution, output)
            if path.exists()
        ]
        passed = (
            completed.returncode != 0
            and expected_message in combined_output
            and not artifacts_written
        )
        results[name] = {
            "ranks": ranks,
            "arguments": list(arguments),
            "expected_message": expected_message,
            "returncode": completed.returncode,
            "artifacts_written": artifacts_written,
            "passed": passed,
        }
        if passed:
            print(
                f"  expected failure confirmed: {expected_message}",
                flush=True,
            )
            continue

        if completed.returncode == 0:
            failures.append(
                f"parameter/{name}: command unexpectedly succeeded"
            )
        if expected_message not in combined_output:
            failures.append(
                f"parameter/{name}: expected error message not found: "
                f"{expected_message!r}"
            )
        if artifacts_written:
            failures.append(
                f"parameter/{name}: failing command wrote artifacts: "
                + ", ".join(artifacts_written)
            )
    return results, failures


def run_case(
    mpiexec: str,
    *,
    dimension: int,
    name: str,
    refinement: int,
    ranks: int,
    operator_level: str,
) -> tuple[dict | None, list[str]]:
    summary, solution, vtu = layout.validation_artifact_paths(
        dimension,
        name,
    )
    for artifact in (summary, solution, vtu):
        artifact.unlink(missing_ok=True)

    command = [
        mpiexec,
        "-n",
        str(ranks),
        sys.executable,
        str(layout.RUN_SCRIPT),
        "--dim",
        str(dimension),
        "--operator-level",
        operator_level,
        "--p",
        str(contract.DEFAULT_DEGREE),
        "--nx",
        str(refinement),
        "--ny",
        str(refinement),
    ]
    if dimension == 3:
        command.extend(("--nz", str(refinement)))
    command.extend(
        (
            "--maxit",
            str(contract.DEFAULT_MAX_ITERATIONS),
            "--rtol",
            str(contract.DEFAULT_RTOL),
            "--atol",
            str(contract.DEFAULT_ATOL),
            "--output",
            str(vtu),
            "--summary",
            str(summary),
        )
    )
    print(f"\n[{dimension}d/{name}] {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=layout.REPOSITORY_ROOT,
        check=False,
    )

    failures: list[str] = []
    label = f"{dimension}d/{name}"
    if completed.returncode != 0:
        failures.append(
            f"{label}: process returned {completed.returncode}"
        )
    if not summary.is_file():
        failures.append(
            f"{label}: summary was not created: {summary}"
        )
        return None, failures

    payload = json.loads(summary.read_text(encoding="utf-8"))
    if not payload.get("local_passed", False):
        failures.append(f"{label}: local gates failed")
    artifacts_missing = False
    if not solution.is_file():
        failures.append(
            f"{label}: solution was not created: {solution}"
        )
        artifacts_missing = True
    if not vtu.is_file():
        failures.append(f"{label}: VTU was not created: {vtu}")
        artifacts_missing = True
    if artifacts_missing:
        return None, failures
    return payload, failures


def check_case(
    dimension: int,
    name: str,
    payload: dict,
) -> list[str]:
    failures: list[str] = []
    label = f"{dimension}d/{name}"
    parameters = payload.get("parameters", {})
    if payload.get("schema_version") != schema.SCHEMA_VERSION:
        failures.append(
            f"{label}: expected schema_version="
            f"{schema.SCHEMA_VERSION}"
        )
    missing = schema.missing_summary_fields(payload)
    if missing:
        failures.append(
            f"{label}: summary is missing top-level fields: "
            + ", ".join(missing)
        )
        return failures
    if parameters.get("dimension") != dimension:
        failures.append(f"{label}: result dimension mismatch")

    solver = payload["solver"]
    residual_limit = contract.residual_limit(solver["rhs_norm"])
    if not solver["converged"]:
        failures.append(f"{label}: CG did not converge")
    if solver["breakdown"] is not None:
        failures.append(
            f"{label}: CG breakdown: {solver['breakdown']}"
        )
    if solver["true_absolute_residual"] > residual_limit:
        failures.append(
            f"{label}: true residual "
            f"{solver['true_absolute_residual']:.16e} > "
            f"{residual_limit:.16e}"
        )
    if solver["boundary_absolute_error"] > contract.BOUNDARY_ABSOLUTE_TOL:
        failures.append(
            f"{label}: boundary error "
            f"{solver['boundary_absolute_error']:.16e} > "
            f"{contract.BOUNDARY_ABSOLUTE_TOL:g}"
        )

    if payload["mpi_size"] != 1:
        return failures

    matvec = payload["matvec_reference"]
    if matvec is None:
        failures.append(f"{label}: missing serial MatVec reference")
    else:
        for key, description, tolerance in (
            (
                "raw_relative_error",
                "raw EA/FA MatVec",
                contract.MATVEC_RELATIVE_TOL,
            ),
            (
                "dirichlet_relative_error",
                "Dirichlet EA/FA MatVec",
                contract.MATVEC_RELATIVE_TOL,
            ),
        ):
            if matvec[key] > tolerance:
                failures.append(
                    f"{label}: {description} error "
                    f"{matvec[key]:.16e} > {tolerance:g}"
                )
        if matvec["random_vector_energy"] <= 0.0:
            failures.append(
                f"{label}: random-vector energy is not positive"
            )

    direct = payload["explicit_solution_reference"]
    if direct is None:
        failures.append(
            f"{label}: missing explicit solution reference"
        )
    elif (
        direct["relative_error"]
        > contract.EXPLICIT_SOLUTION_RELATIVE_TOL
    ):
        failures.append(
            f"{label}: CG/assembled solution error "
            f"{direct['relative_error']:.16e} > "
            f"{contract.EXPLICIT_SOLUTION_RELATIVE_TOL:g}"
        )
    return failures


def relative_solution_difference(left: dict, right: dict) -> float:
    left_solution = np.load(left["artifacts"]["solution_npy"])
    right_solution = np.load(right["artifacts"]["solution_npy"])
    return float(
        np.linalg.norm(left_solution - right_solution)
        / max(np.linalg.norm(left_solution), contract.NORM_FLOOR)
    )


def compare_cases(
    dimension: int,
    results: dict[str, dict],
    *,
    include_parallel: bool = False,
):
    # 档位由 layout.EA_EVIDENCE_ROLES 单点定义, 这里不再写死档数: 增减一档只
    # 需改那个元组与 contract.REFINEMENTS.
    ea_names = [
        layout.case_name(role, "ea", 1)
        for role in layout.EA_EVIDENCE_ROLES
    ]
    fine_serial_name = ea_names[-1]
    fine_parallel_name = layout.case_name(
        layout.EA_EVIDENCE_ROLES[-1],
        "ea",
        2,
    )
    fine_serial = results[fine_serial_name]

    # EA/FA 逐档比对: FA 与 EA 同档, 每个档位都取同一网格的两个算例,
    # 否则比的是两个不同的离散。``coarse_solution_...`` 字段保留向后兼容,
    # 值取自最粗档, 新消费方应读逐档字段。
    fa_differences = {
        role: relative_solution_difference(
            results[layout.case_name(role, "ea", 1)],
            results[layout.case_name(role, "fa", 1)],
        )
        for role in layout.EA_EVIDENCE_ROLES
    }
    errors = [
        float(results[name]["error"]["l2_relative"]) for name in ea_names
    ]
    orders = [
        math.log2(previous / current)
        for previous, current in zip(errors, errors[1:])
    ]

    failures: list[str] = []
    for role, difference in fa_differences.items():
        if difference > contract.EA_FA_SOLUTION_RELATIVE_TOL:
            failures.append(
                f"{dimension}d: {role} EA/FA solution difference "
                f"{difference:.16e} > "
                f"{contract.EA_FA_SOLUTION_RELATIVE_TOL:g}"
            )

    # 1b（CPU 并行 EA）的跨 rank 门禁; 1a 下这两条不参与判定, 也不记入
    # comparison, 以免串行证据里出现空占位而被误读为"已检验".
    parallel_difference: float | None = None
    parallel_l2_difference: float | None = None
    parallel_error: float | None = None
    if include_parallel:
        fine_parallel = results[fine_parallel_name]
        parallel_error = float(fine_parallel["error"]["l2_relative"])
        parallel_difference = relative_solution_difference(
            fine_serial,
            fine_parallel,
        )
        if parallel_difference > contract.PARALLEL_SOLUTION_RELATIVE_TOL:
            failures.append(
                f"{dimension}d: fine 1/2-rank solution difference "
                f"{parallel_difference:.16e} > "
                f"{contract.PARALLEL_SOLUTION_RELATIVE_TOL:g}"
            )
        parallel_l2_difference = abs(errors[-1] - parallel_error)
        if parallel_l2_difference > contract.PARALLEL_L2_DIFFERENCE_TOL:
            failures.append(
                f"{dimension}d: fine 1/2-rank L2-error difference "
                f"{parallel_l2_difference:.16e} > "
                f"{contract.PARALLEL_L2_DIFFERENCE_TOL:g}"
            )

    if not all(
        previous > current for previous, current in zip(errors, errors[1:])
    ):
        failures.append(
            f"{dimension}d: relative L2 error did not decrease: "
            + ", ".join(f"{value:.16e}" for value in errors)
        )
    if orders[-1] < contract.MINIMUM_FINAL_L2_ORDER:
        failures.append(
            f"{dimension}d: final relative L2 order "
            f"{orders[-1]:.8f} < {contract.MINIMUM_FINAL_L2_ORDER}"
        )

    relative_l2_errors = dict(zip(ea_names, errors))
    comparison = {
        "stage": "1b" if include_parallel else "1a",
        # 向后兼容: 旧消费方读单值, 新消费方读逐档字段。
        "coarse_solution_ea_fa_relative_difference": (
            fa_differences[layout.EA_EVIDENCE_ROLES[0]]
        ),
        "ea_fa_solution_relative_differences": fa_differences,
        "relative_l2_errors": relative_l2_errors,
        "observed_relative_l2_orders": orders,
        "gated_relative_l2_order": orders[-1],
        "minimum_gated_relative_l2_order": (
            contract.MINIMUM_FINAL_L2_ORDER
        ),
    }
    if include_parallel:
        relative_l2_errors[fine_parallel_name] = parallel_error
        comparison["fine_solution_1rank_2rank_relative_difference"] = (
            parallel_difference
        )
        comparison["fine_relative_l2_error_1rank_2rank_difference"] = (
            parallel_l2_difference
        )
    return comparison, failures


def validate_dimension(
    mpiexec: str,
    dimension: int,
    *,
    include_parallel: bool = False,
) -> tuple[dict, list[str]]:
    cases = layout.validation_case_specs(
        contract.REFINEMENTS[dimension],
        include_parallel=include_parallel,
    )
    layout.dimension_output_dir(dimension).mkdir(
        parents=True,
        exist_ok=True,
    )
    results: dict[str, dict] = {}
    failures: list[str] = []
    for name, refinement, ranks, operator_level in cases:
        payload, run_failures = run_case(
            mpiexec,
            dimension=dimension,
            name=name,
            refinement=refinement,
            ranks=ranks,
            operator_level=operator_level,
        )
        failures.extend(run_failures)
        if payload is not None:
            results[name] = payload
            failures.extend(check_case(dimension, name, payload))

    comparison = None
    if len(results) == len(cases):
        comparison, comparison_failures = compare_cases(
            dimension,
            results,
            include_parallel=include_parallel,
        )
        failures.extend(comparison_failures)
    else:
        failures.append(
            f"{dimension}d: cross-run checks skipped because "
            "one or more cases are missing"
        )
    return {
        "cases": {
            name: results.get(name)
            for name, _, _, _ in cases
        },
        "comparison": comparison,
        "passed": not failures,
    }, failures


def verify_launcher(mpiexec: str) -> str | None:
    """探测启动器与 ``mpi4py`` 是否同厂.

    以 ``-n 2`` 起一个只打印 ``COMM_WORLD`` 大小的探针。启动器与 ``mpi4py``
    链接的 MPI 不是同一实现时, 两个进程不会组成通信域, 而是各自成为
    ``size=1`` 的独立进程 —— 这不会报错, 只会让后续所有多 rank 断言得到无意义
    的结果(例如"多 rank 应触发的守卫"因为 ``size`` 恒为 1 而永不触发, 却被报成
    "expected error message not found", 指向完全错误的方向)。

    参数:
        mpiexec: 待探测的启动器路径.

    返回:
        message: 探测失败时的诊断文本; 启动器可用时返回 ``None``.
    """
    command = [
        mpiexec, "-n", "2", sys.executable, "-c",
        "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())",
    ]
    try:
        completed = subprocess.run(
            command, cwd=layout.REPOSITORY_ROOT, check=False,
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"启动探针无法运行: {error}"

    sizes = [line.strip() for line in completed.stdout.splitlines()
             if line.strip().isdigit()]
    if completed.returncode != 0 or len(sizes) != 2:
        return (f"启动探针异常退出(returncode={completed.returncode}):\n"
                f"{completed.stderr.strip()[:600]}")
    if any(size != "2" for size in sizes):
        try:
            from mpi4py import MPI
            name, version = MPI.get_vendor()[:2]
            vendor = f"{name} {'.'.join(str(part) for part in version)}"
        except Exception:  # noqa: BLE001 - 探测失败时不该再抛
            vendor = "未知"
        return (
            f"启动器与 mpi4py 不同厂: `{mpiexec} -n 2` 起出的进程各自报 "
            f"COMM_WORLD.size={sizes}, 期望 ['2', '2']。\n"
            f"两个进程没有组成通信域, 而是各自独立运行, 所有多 rank 断言都会失真。\n"
            f"mpi4py 链接的是 {vendor}; 请用 SOPTX_MPIEXEC 指定同厂的启动器, 例如\n"
            f"  SOPTX_MPIEXEC=/usr/bin/mpiexec.openmpi python {Path(__file__).name} ...")
    return None


def main() -> int:
    arguments = parse_arguments()
    # 优先解释器同目录的 mpiexec, 前提是它与 mpi4py 链接的那份 MPI 是同一实现。
    # 这个前提会被破坏: 环境里可能同时装着别的 MPI 运行时(例如 pypi 的
    # ``impi-rt`` 会把 Intel MPI 的 mpiexec 放进 ``bin/``), 而 mpi4py 链的是系统
    # OpenMPI. 用错启动器时 mpi4py 读不到启动器写入的进程信息, ``mpi_size`` 直接
    # 是 ``None``, 报错点会落在很远的下游, 因此留一个显式出口:
    # ``SOPTX_MPIEXEC=/usr/bin/mpiexec`` 指定与 mpi4py 匹配的那一份。
    override = os.environ.get("SOPTX_MPIEXEC")
    env_mpiexec = Path(sys.executable).parent / "mpiexec"
    if override:
        mpiexec = override
    elif env_mpiexec.is_file():
        mpiexec = str(env_mpiexec)
    else:
        mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        print(
            "mpiexec was not found; activate an environment containing "
            "MPI and mpi4py.",
            file=sys.stderr,
        )
        return 2

    # 先探测启动器再跑证据: 用错启动器时下游报的错会指向错误的方向, 这里挡住。
    launcher_problem = verify_launcher(mpiexec)
    if launcher_problem is not None:
        print(f"[mpiexec] {mpiexec}", file=sys.stderr)
        print(launcher_problem, file=sys.stderr)
        return 2
    print(f"[mpiexec] {mpiexec} (COMM_WORLD.size=2 探测通过)", flush=True)

    layout.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parameter_checks, parameter_failures = validate_parameter_checks(
        mpiexec
    )
    dimensions: dict[str, dict] = {}
    failures: list[str] = list(parameter_failures)
    for dimension in selected_dimensions(arguments.dim):
        result, dimension_failures = validate_dimension(
            mpiexec,
            dimension,
            include_parallel=arguments.include_parallel,
        )
        dimensions[str(dimension)] = result
        failures.extend(dimension_failures)

    evidence = {
        "schema_version": schema.SCHEMA_VERSION,
        "stage": contract.STAGE,
        "substage": "1b" if arguments.include_parallel else "1a",
        "selected_dimensions": list(
            selected_dimensions(arguments.dim)
        ),
        "parameter_checks": parameter_checks,
        "dimensions": dimensions,
        "passed": not failures,
        "failures": failures,
    }
    evidence_path = layout.validation_evidence_path(arguments.dim)
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if failures:
        substage = "1b" if arguments.include_parallel else "1a"
        print(
            f"\nStage {substage} elasticity validation: FAILED",
            flush=True,
        )
        for failure in failures:
            print(f"  - {failure}", flush=True)
        print(f"Evidence: {evidence_path}", flush=True)
        return 1

    print(
        f"\nStage {'1b' if arguments.include_parallel else '1a'} "
        "elasticity validation: PASSED",
        flush=True,
    )
    summary = {
        dimension: {
            "passed": result["passed"],
            "comparison": result["comparison"],
        }
        for dimension, result in dimensions.items()
    }
    print(
        json.dumps(summary, ensure_ascii=False, indent=2),
        flush=True,
    )
    print(f"Evidence: {evidence_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
