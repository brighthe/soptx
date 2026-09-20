"""复用 SOPTX 训练循环的 CPU/CUDA 固定预算实验。

模块顶层只导入标准库，数值依赖在实际运行时加载。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from dataclasses import asdict
from pathlib import Path


class ExperimentError(RuntimeError):
    """实验条件或结果不满足比较契约。"""


def _dependencies(case):
    """在加载数值库前固定线程配置。"""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(case.cpu_threads)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import numpy as np
    import torch
    torch.set_num_threads(case.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return np, torch


def _hash_arrays(*arrays):
    """摘要同时覆盖形状、dtype 与数值。"""
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(_hash_arrays(tensor.detach().cpu().numpy()).encode())
    return digest.hexdigest()


def _prepare(case, np, torch):
    """生成角点迹变形分量标签；两端共享同一准备结果。"""
    if case.baseline == "example_full_trace":
        from baseline import prepare
        return prepare(case, np, torch)
    from fealpy.backend import backend_manager as bm
    from soptx.fem.substructure import (
        SubstructurePrototype, ExactSchurReduction, LinearCornerTraceBasis,
    )
    from soptx.ml.substructure import (
        DensitySamplingConfig,
        ShapeFunctionSurrogateNet,
        sample_density_fields,
    )
    t0 = time.perf_counter()
    bm.set_backend("numpy")
    proto = SubstructurePrototype(case.sub_size, case.n_fine, E_base=case.emax,
                                  nu=case.nu, penal=case.simp_penalty, rho_min=1e-6)
    trace = LinearCornerTraceBasis.from_prototype(proto)
    T = np.asarray(trace.matrix)
    full_rigid = np.asarray(proto.rigid_basis)
    q_rigid = np.linalg.lstsq(T, full_rigid, rcond=None)[0]
    if not np.allclose(T @ q_rigid, full_rigid, rtol=1e-10, atol=1e-12):
        raise ExperimentError("角点迹不能表示原型刚体模式。")
    Q, _ = np.linalg.qr(q_rigid, mode="complete")
    R, D = Q[:, :proto.n_rigid], Q[:, proto.n_rigid:]
    Phi = np.asarray(proto.rigid_interior_modes) @ (full_rigid.T @ T @ R)
    sampling = DensitySamplingConfig(shape=case.n_fine, design_min=case.design_min,
                                     seed=case.seed)
    samples = sample_density_fields(case.samples, sampling)
    density = samples.values
    K = proto.assemble_local_stiffness_batch(proto.grid_to_cell_field(density))
    exact = ExactSchurReduction(proto.i_dofs, proto.b_dofs).reduce_many(K, density)
    B = np.asarray(trace.reduce_recovery(exact.recovery))
    Kr = np.asarray(trace.project_stiffness(exact.stiffness))
    x = density.reshape(case.samples, -1).astype(np.float32)
    y = (B @ D).reshape(case.samples, -1).astype(np.float32)
    if not all(np.isfinite(a).all() for a in (x, y, Kr)):
        raise ExperimentError("精确标签中存在非有限值。")
    order = np.random.default_rng(case.seed + 1).permutation(case.samples)
    n_val = max(1, min(case.samples - 1, round(case.samples * case.validation_fraction)))
    vi, ti = order[:n_val], order[n_val:]
    arrays = tuple(np.ascontiguousarray(a) for a in (x[ti], y[ti], x[vi], y[vi]))
    torch.manual_seed(case.seed)
    model = ShapeFunctionSurrogateNet(
        x.shape[1], y.shape[1], (case.hidden_dim, case.hidden_dim)
    )
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    i, b = np.asarray(proto.i_dofs), np.asarray(proto.b_dofs)
    Kv = np.asarray(K)[vi]
    data = dict(arrays=arrays, state=state, R=R, D=D, Phi=Phi, T=T,
                Kii=Kv[:, i[:, None], i], Kib=Kv[:, i[:, None], b],
                Kbb=Kv[:, b[:, None], b], exact_B=B[vi], exact_K=Kr[vi],
                n_i=len(i), input_dim=x.shape[1], output_dim=y.shape[1])
    data["metadata"] = dict(
        data_sha256=_hash_arrays(*arrays), initial_state_sha256=_state_hash(state),
        split_sha256=_hash_arrays(ti, vi), train_samples=len(ti), validation_samples=len(vi),
        input_dim=x.shape[1], output_dim=y.shape[1], n_trace=T.shape[1],
        n_deformation=D.shape[1], sampler_version=samples.sampler_version,
        source_counts=dict(samples.source_counts), stiffness_floor_ratio=1e-6,
        label_dtype="float64", training_dtype="float32",
        target="M = (N_int T) D; B_hat = Phi R^T + M_hat D^T",
        preparation_s=time.perf_counter() - t0,
    )
    return data


def _sync(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()


def _metrics(np, prediction, data):
    """恢复形函数并评价无回退的变分刚度。"""
    M = prediction.astype(np.float64).reshape(-1, data["n_i"], data["D"].shape[1])
    B = data["Phi"] @ data["R"].T + M @ data["D"].T
    T = data["T"]
    cross = T.T @ data["Kib"].transpose(0, 2, 1) @ B
    Kr = T.T @ data["Kbb"] @ T + cross + cross.transpose(0, 2, 1)
    Kr += B.transpose(0, 2, 1) @ data["Kii"] @ B
    result = {}
    for name, actual, exact in (("shape", B, data["exact_B"]),
                                 ("raw_stiffness", Kr, data["exact_K"])):
        denominator = np.linalg.norm(exact, axis=(1, 2))
        if np.any(denominator <= 0):
            raise ExperimentError("相对误差的精确参照范数为零。")
        errors = np.linalg.norm(actual - exact, axis=(1, 2)) / denominator
        if not np.isfinite(errors).all():
            raise ExperimentError("训练输出或力学误差存在非有限值。")
        result[name] = dict(mean=float(errors.mean()), p95=float(np.quantile(errors, .95)),
                            maximum=float(errors.max()))
    return result


def _train(case, data, np, torch):
    """训练计时包括原有循环的验证和参数快照管理。"""
    if case.baseline == "example_full_trace":
        from baseline import train
        return train(case, data, np, torch)
    from soptx.ml.substructure import (
        ShapeFunctionSurrogateNet,
        TrainingConfig,
        train_surrogate,
    )
    _sync(torch, case.device)
    t0 = time.perf_counter()
    model = ShapeFunctionSurrogateNet(
        data["input_dim"], data["output_dim"], (case.hidden_dim, case.hidden_dim)
    )
    model.load_state_dict(data["state"])
    initial_hash = _state_hash(model.state_dict())
    model.to(case.device)
    arrays = tuple(torch.as_tensor(a, device=case.device) for a in data["arrays"])
    _sync(torch, case.device)
    setup_s = time.perf_counter() - t0
    config = TrainingConfig(epochs=case.epochs, batch_size=case.batch_size,
                            learning_rate=case.learning_rate, seed=case.seed,
                            patience=0, select_final_state=True)
    if case.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _sync(torch, case.device)
    t0 = time.perf_counter()
    result = train_surrogate(model, *arrays, config)
    _sync(torch, case.device)
    elapsed = time.perf_counter() - t0
    memory = dict(process_lifetime_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if case.device == "cuda":
        memory.update(training_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      training_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    t0 = time.perf_counter()
    with torch.no_grad():
        prediction = model(arrays[2]).detach().cpu().numpy()
    metrics = _metrics(np, prediction, data)
    steps = result.epochs_run * ((len(arrays[0]) + case.batch_size - 1) // case.batch_size)
    if not np.isfinite(result.training_losses + result.validation_losses).all():
        raise ExperimentError("训练 loss 存在非有限值。")
    return dict(case=asdict(case), contract=case.comparison_contract(),
                initial_state_sha256=initial_hash, data_sha256=data["metadata"]["data_sha256"],
                device_name=torch.cuda.get_device_name() if case.device == "cuda" else platform.processor(),
                setup_and_transfer_s=setup_s, training_pipeline_s=elapsed,
                pipeline_s_per_step=elapsed / steps, steps=steps, epochs_run=result.epochs_run,
                final_training_loss=result.final_training_loss,
                final_validation_loss=result.best_validation_loss,
                training_losses=result.training_losses, validation_losses=result.validation_losses,
                metrics=metrics, final_evaluation_s=time.perf_counter() - t0, memory=memory,
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())


def _provenance(torch, np):
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        completed = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return completed.stdout.strip() if completed.returncode == 0 else "unavailable"
    return dict(python=platform.python_version(), platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                git_head=git("rev-parse", "HEAD"), git_status=git("status", "--short"),
                threads={name: os.environ.get(name) for name in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                deterministic_algorithms=True, tf32=False,
                timing_scope="train_surrogate inclusive validation, loss.item, final-state snapshots; cold start, no warmup",
                memory_scope="CPU RSS lifetime high-water, not training-only; CUDA allocator peak during training")


def _execute(cases, output_dir):
    np, torch = _dependencies(cases[0])
    if any(c.device == "cuda" for c in cases) and not torch.cuda.is_available():
        raise ExperimentError("CUDA 不可用；未执行训练，也不生成加速比。")
    if len(cases) == 2 and cases[0].comparison_contract() != cases[1].comparison_contract():
        raise ExperimentError("CPU/CUDA 配置不一致。")
    output_dir = Path(output_dir)
    name = cases[0].pair_id if len(cases) == 2 else cases[0].id
    output = output_dir / (name + ".json")
    if output.exists():
        raise ExperimentError(f"结果已存在，请使用新的 --output-dir: {output}")
    data = _prepare(cases[0], np, torch)
    results = []
    for case in cases:
        print(f"开始 {case.id}: {case.epochs} epochs", flush=True)
        results.append(_train(case, data, np, torch))
    payload = dict(schema_version=2, stage="training", status="completed",
                   preparation=data["metadata"], provenance=_provenance(torch, np), results=results)
    if cases[0].baseline == "example_full_trace":
        payload["provenance"]["timing_scope"] = "examples shared full-batch Adam; includes optimizer creation and loss.item; excludes validation and global evaluation; no warmup"
        payload["accuracy_status"] = "requires_review_of_local_and_global_errors"
    else:
        payload["accuracy_status"] = "exploratory_not_validated"
    if len(results) == 2:
        cpu, gpu = results
        for key in ("contract", "initial_state_sha256", "data_sha256", "steps", "epochs_run"):
            if cpu[key] != gpu[key]:
                raise ExperimentError(f"配对契约不一致: {key}")
        payload["comparison"] = dict(training_pipeline_speedup=cpu["training_pipeline_s"] / gpu["training_pipeline_s"],
                                     scope="fixed budget, not time-to-target accuracy", repetitions=1,
                                     device_order=[c.device for c in cases])
    output_dir.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return output


def run_case(case, output_dir):
    """运行一个已注册训练工况。"""
    return _execute((case,), output_dir)


def run_pair(cpu, cuda, output_dir):
    """共享数据与初始权重，依次运行 CPU 与 CUDA。"""
    return _execute((cpu, cuda), output_dir)
