"""为 examples 的 full_trace 精度基线添加设备计时与溯源。

不另写采样、训练循环或力学评价公式。
"""
from __future__ import annotations

import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path


def _example():
    """按唯一目录定位已有精度验证模块，仅实际运行时导入。"""
    directory = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(directory))
    from examples.piml_substructure_elasticity import verify_shape_function_route as shared
    return shared


def prepare(case, np, torch):
    """复用示例标签并保存两端共用的初始参数。"""
    from training import _hash_arrays, _state_hash
    from fealpy.backend import backend_manager as bm
    from soptx.ml.substructure import ShapeFunctionSurrogateNet
    shared = _example()
    t0 = time.perf_counter()
    bm.set_backend("numpy")
    shared.set_random_seed(case.seed)
    ev = shared.Eq17Evaluator()
    X, Y = shared.prepare_training_arrays(ev, case.samples)
    rho = shared.sample_random_density(case.n_eval, seed_offset=555)
    _, _, exact_N = ev.exact_batch(rho)
    arrays = (X.numpy(), Y.numpy(),
              np.asarray(rho).reshape(case.n_eval, -1).astype(np.float32),
              ev.project_M(exact_N).reshape(case.n_eval, -1).astype(np.float32))
    net = ShapeFunctionSurrogateNet(
        X.shape[1], Y.shape[1], (case.hidden_dim, case.hidden_dim)
    )
    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    return dict(ev=ev, arrays=arrays, state=state,
                metadata=dict(data_sha256=_hash_arrays(*arrays),
                              initial_state_sha256=_state_hash(state),
                              train_samples=case.samples, validation_samples=case.n_eval,
                              input_dim=X.shape[1], output_dim=Y.shape[1],
                              n_trace=ev.n_b, n_deformation=ev.n_reduced,
                              trace_basis="full_trace", density_range=list(shared.DENSITY_RANGE),
                              sampling_seed=shared.SHAPE_SEED, validation_seed_offset=555,
                              sampler_version="historical_uniform_full_trace",
                              stiffness_floor_ratio=shared.SHAPE_RHO_MIN, label_dtype="float64", training_dtype="float32",
                              target="M = N_int R_perp",
                              source="examples/piml_substructure_elasticity/verify_shape_function_route.py",
                              preparation_s=time.perf_counter() - t0))


def train(case, data, np, torch):
    """只对共享全批量训练循环计时，解层复核在计时区间之外。"""
    from training import _sync, _state_hash, ExperimentError
    from soptx.ml.substructure import ShapeFunctionSurrogateNet
    shared = _example()
    _sync(torch, case.device)
    t0 = time.perf_counter()
    net = ShapeFunctionSurrogateNet(
        data["arrays"][0].shape[1],
        data["arrays"][1].shape[1],
        (case.hidden_dim, case.hidden_dim),
    )
    net.load_state_dict(data["state"])
    initial_hash = _state_hash(net.state_dict())
    net.to(case.device)
    X, Y = [torch.as_tensor(a, device=case.device) for a in data["arrays"][:2]]
    _sync(torch, case.device)
    setup_s = time.perf_counter() - t0
    if case.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _sync(torch, case.device)
    t0 = time.perf_counter()
    losses = shared.fit_full_batch(net, X, Y, case.epochs, case.learning_rate)
    _sync(torch, case.device)
    elapsed = time.perf_counter() - t0
    memory = dict(process_lifetime_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if case.device == "cuda":
        memory.update(training_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      training_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if not np.isfinite(losses).all():
        raise ExperimentError("训练损失存在非有限值。")
    t0 = time.perf_counter()
    # 在同一 CPU 评价器上复核两端的最终模型，避免把评价设备差异混入精度比较。
    net.cpu()
    local = shared.step3_trained_network(data["ev"], net, case.n_eval)
    from examples.piml_substructure_elasticity.verify_shape_function_route import step4_solution_layer
    solution = step4_solution_layer(data["ev"], net)
    with torch.no_grad():
        val_loss = float(torch.nn.functional.mse_loss(
            net(torch.as_tensor(data["arrays"][2])), torch.as_tensor(data["arrays"][3])).item())
    if not all(np.isfinite(v) for v in list(local.values()) + list(solution.values()) + [val_loss]):
        raise ExperimentError("局部或全局精度结果存在非有限值。")
    metrics = {}
    for name, key in (("shape", "eps_N"), ("raw_stiffness", "eps_K17")):
        metrics[name] = dict(mean=local[key + "_mean"], p95=local[key + "_p95"], maximum=local[key + "_max"])
    return dict(case=asdict(case), contract=case.comparison_contract(),
                initial_state_sha256=initial_hash, data_sha256=data["metadata"]["data_sha256"],
                setup_and_transfer_s=setup_s, training_pipeline_s=elapsed,
                pipeline_s_per_step=elapsed / case.epochs, steps=case.epochs, epochs_run=case.epochs,
                training_losses=losses, validation_losses=[], final_training_loss=losses[-1],
                final_validation_loss=val_loss, metrics=metrics,
                local_validation=local, solution_layer=solution,
                final_evaluation_s=time.perf_counter() - t0, memory=memory,
                device_name=torch.cuda.get_device_name() if case.device == "cuda" else "cpu",
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())

