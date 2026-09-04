"""性能测量的独立进程协议, 不作为用户命令行入口.

父进程串行启动全新的解释器, 工作进程只执行一条分析路径.
峰值内存采用 Linux 当前进程的 VmHWM, 不包含父进程和其他路径.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def peak_rss_bytes() -> int:
    """读取 Linux 当前进程地址空间的峰值 RSS, 返回字节数."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("峰值 RSS 测量仅支持 Linux/WSL, 请在 Ubuntu 中运行性能比较脚本.")
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == "VmHWM:":
            if len(fields) != 3 or fields[2] != "kB" or int(fields[1]) <= 0:
                raise RuntimeError(f"无法解析峰值 RSS: {line}")
            return int(fields[1]) * 1024
    raise RuntimeError("/proc/self/status 未提供 VmHWM, 不能报告峰值内存.")


def run_worker(config: dict, directory: Path) -> dict:
    """启动并等待一个全新的分析进程, 只读取小型记录而不加载位移.

    Parameters
    ----------
    config : dict
        route（fa/full_trace/linear_corner）、dim、n_sub、n_fine 和 density_mode.
    directory : Path
        本样本专有临时目录, 生命周期由调用方管理.

    Returns
    -------
    dict
        工作进程输出的计时、峰值内存、精度和环境记录.

    Raises
    ------
    RuntimeError
        工作进程失败或输出的路径标识不匹配.
    """
    directory.mkdir(parents=True, exist_ok=True)
    request = directory / "request.json"
    record_path = directory / "record.json"
    displacement_path = directory / "displacement.npy"
    request.write_text(json.dumps(config, allow_nan=False), encoding="utf-8")
    command = [
        sys.executable, "-m", "examples.substructure_elasticity._performance_process",
        str(request.resolve()), str(record_path.resolve()), str(displacement_path.resolve()),
    ]
    process = subprocess.Popen(
        command, cwd=Path(__file__).resolve().parents[2],
        env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    try:
        stdout, stderr = process.communicate()
    except BaseException:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        raise
    if process.returncode != 0:
        hint = " (SIGKILL, 可能内存不足; 需查内核日志确认)" if process.returncode == -9 else ""
        raise RuntimeError(
            f"{config['route']} 子进程失败, exit code={process.returncode}{hint}.\n"
            f"{stderr[-8000:]}\n{stdout[-8000:]}"
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["route"] != config["route"]:
        raise RuntimeError("子进程结果的 route 与请求不一致.")
    record["worker_stdout"] = stdout[-8000:]
    record["worker_stderr"] = stderr[-8000:]
    return record


def worker_main(argv: list[str] | None = None) -> int:
    """执行父进程的私有请求, 在峰值采样之后才写出位移和记录."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 3:
        raise ValueError("内部 worker 需要 request.json, record.json 和 displacement.npy 三个路径.")
    # 先确认测量能力; 随后的导入和问题准备均计入最终 VmHWM.
    peak_rss_bytes()
    request, record_path, displacement_path = map(Path, arguments)
    config = json.loads(request.read_text(encoding="utf-8"))
    route = config.pop("route")
    if route not in ("fa", "full_trace", "linear_corner"):
        raise ValueError(f"未知性能路径: {route}")

    import numpy as np
    from examples.substructure_elasticity._comparison import (
        performance_environment, prepare_performance_problem, timed_full_analysis,
    )

    pde, reference, density, force, fixed, metadata, preparation_seconds = (
        prepare_performance_problem(**config)
    )
    displacement, record = timed_full_analysis(
        route, pde, reference, density, force, fixed,
        peak_memory_reader=peak_rss_bytes,
    )
    record.update(
        route=route, pid=os.getpid(), problem_data=metadata,
        preparation_seconds=preparation_seconds, environment=performance_environment(),
    )
    # 采样完成后才序列化, 文件用于父进程中两条路径的精度比较.
    with displacement_path.open("xb") as stream:
        np.save(stream, np.asarray(displacement), allow_pickle=False)
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(worker_main())
