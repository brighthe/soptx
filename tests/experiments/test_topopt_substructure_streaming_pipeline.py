# -*- coding: utf-8 -*-
"""精确子结构拓扑优化实验的流式 pipeline 回归测试."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from fealpy.backend import backend_manager as bm


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
    REPOSITORY_ROOT / "experiments" / "topopt_simp_substructure"
)


def _load_module(name: str, path: Path) -> Any:
    """以独立模块名加载实验文件, 避免与其他实验的 ``config`` 冲突."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载实验模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


experiment_config = _load_module(
    "_topopt_substructure_config",
    EXPERIMENT_ROOT / "config.py",
)
previous_config = sys.modules.get("config")
sys.modules["config"] = experiment_config
try:
    experiment_pipeline = _load_module(
        "_topopt_substructure_pipeline",
        EXPERIMENT_ROOT / "pipeline.py",
    )
finally:
    if previous_config is None:
        sys.modules.pop("config", None)
    else:
        sys.modules["config"] = previous_config


def _full_batch_linear_corner_forward(ctx: Any, rho: Any) -> tuple[float, Any]:
    """按第 4 步接入前的完整批量路径计算小工况基准."""
    rho_sub_grid = ctx.assembler.split_global_cell_field(rho)
    rho_sub_cell = ctx.prototype.grid_to_cell_field(rho_sub_grid)
    local_stiffness = ctx.prototype.assemble_local_stiffness_batch(rho_sub_cell)
    result = ctx.reduction.reduce_many(local_stiffness, rho_sub_grid)
    trace_stiffness = ctx.trace_basis.project_stiffness(result.stiffness)
    system = ctx.assembler.assemble_macro_system(
        ctx.sub_meshes,
        trace_stiffness,
    )
    displacement = experiment_pipeline.solve_interface_system(
        system,
        ctx.load,
        ctx.fixed_dofs,
    )

    trace_displacement = displacement[ctx.trace_indices]
    boundary_displacement = ctx.trace_basis.expand_displacement(
        trace_displacement
    )
    internal_displacement = result.recover(boundary_displacement)
    local_displacement = bm.zeros(
        (ctx.n_sub_total, ctx.prototype.n_total_dofs),
        dtype=bm.float64,
    )
    local_displacement = bm.set_at(
        local_displacement,
        (slice(None), ctx.prototype.b_dofs),
        boundary_displacement,
    )
    local_displacement = bm.set_at(
        local_displacement,
        (slice(None), ctx.prototype.i_dofs),
        internal_displacement,
    )
    element_displacement = local_displacement[:, ctx.prototype.cell2dof]
    energy_cell = bm.sum(
        (element_displacement @ ctx.prototype.KE_unit[0])
        * element_displacement,
        axis=-1,
    )
    energy_grid = ctx.prototype.cell_to_grid_field(energy_cell)
    energy_global = ctx.assembler.merge_substructure_cell_field(energy_grid)
    compliance = float(bm.dot(ctx.load, displacement))
    return compliance, bm.reshape(energy_global, (-1,))


def test_registered_cases_have_positive_chunk_size() -> None:
    """四条注册工况必须显式给出正的流式批大小."""
    _, cases = experiment_config.load()

    assert len(cases) == 4
    assert all(case.chunk_size > 0 for case in cases)
    assert all(case.integration_order > 0 for case in cases)

    case_3d = experiment_config.get_case("mbb_3d_lc")
    assert case_3d.domain == (0.0, 12.0, 0.0, 2.0, 0.0, 2.0)
    assert case_3d.n_sub == (12, 2, 2)
    assert case_3d.n_fine == (4, 4, 4)
    assert case_3d.filter_type == "sensitivity"


def test_linear_corner_pipeline_matches_full_batch_without_calling_batch_api(
    monkeypatch: Any,
) -> None:
    """实验流式路径应匹配旧批量结果且不调用完整局部刚度 API."""
    bm.set_backend("numpy")
    registered = experiment_config.get_case("cantilever_2d_lc")
    case = replace(
        registered,
        domain=(0.0, 2.0, 0.0, 1.0),
        n_sub=(2, 1),
        n_fine=(2, 2),
        chunk_size=1,
        max_iter=1,
    )
    ctx = experiment_pipeline.build_components(case)
    rho = bm.asarray(
        np.linspace(0.42, 0.78, ctx.n_elem_total),
        dtype=bm.float64,
    )

    expected_compliance, expected_energy = _full_batch_linear_corner_forward(
        ctx,
        rho,
    )

    def reject_full_batch(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("linear_corner 不应调用完整批量局部刚度装配接口")

    monkeypatch.setattr(
        ctx.prototype,
        "assemble_local_stiffness_batch",
        reject_full_batch,
    )
    actual = experiment_pipeline.solve_forward(ctx, rho)

    np.testing.assert_allclose(
        actual.compliance,
        expected_compliance,
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        bm.to_numpy(actual.energy),
        bm.to_numpy(expected_energy),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
