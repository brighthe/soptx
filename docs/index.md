---
title: "SOPTX Docs Index"
tags:
  - docs-index
  - soptx
status: "active"
date: 2026-06-29
---

# SOPTX docs 索引

本目录存放与 SOPTX 当前代码强绑定的架构备忘录与 AI 接续上下文。
研究计划与数学原则沉淀在 `dut-postdoc` 仓库的 `research/`。

当前围绕拓扑优化**状态方程求解的两条分析后端**各成一套文档体系（结构对称）：

## Matrix-Free 高性能求解

- [status.md](../ai/common/status.md) — **AI 工作线总入口**，先读这份
- [progress-frame8_matrix_free.md](../ai/common/progress-frame8_matrix_free.md) — **续接入口**（进度 + 决策），先读这份
- [frame8_matrix_free_pipeline_results.md](frame8_matrix_free_pipeline_results.md) — 帧 8 验证数值结果快照（deck 上游事实源）
- [matrix_free_architecture_notes.md](matrix_free_architecture_notes.md) — SOPTX 实现备忘录
- 帧级主入口：`dut-postdoc/research/postdoc-plan/defense-sprint/direction-1-piml-matrix-free/frame8_matrix_free_pipeline_guide.md`

## PIML 多尺度预测

- [progress-frame7_piml.md](../ai/common/progress-frame7_piml.md) — **续接入口**（进度 + 决策 + 阶段二/三接入设计），先读这份
- [frame7_piml_pipeline_results.md](frame7_piml_pipeline_results.md) — 帧 7 验证数值结果快照（deck 上游事实源）
- 帧级主入口：`dut-postdoc/research/postdoc-plan/defense-sprint/direction-1-piml-matrix-free/frame7_piml_pipeline_guide.md`；
  任务计划 / 数学原则：同目录 `soptx-piml-multiscale-integration-plan.md` / `piml_multiscale_math_principles.md`

> 原 `ai_piml_context.md`（旧续接入口）与 `piml_multiscale_architecture_notes.md`（接入前
> 架构备忘录）已于 2026-07-02 并入 `progress-frame7_piml.md` 后删除。

## MMC 显式几何高精度离散

- [progress-frame10_mmc.md](../ai/common/progress-frame10_mmc.md) — **续接入口**（进度 + 决策 + 复现命令），先读这份
- [frame10_mmc_pipeline_results.md](frame10_mmc_pipeline_results.md) — 帧 10 验证数值结果快照（deck 上游事实源）
- 帧级主入口 / 长期调研：`dut-postdoc/research/postdoc-plan/` 下
  `defense-sprint/direction-2-mmc-mmv/frame10_mmc_pipeline_guide.md` / `long-term/direction-2-mmc-mmv/mmc-mmv-numerical-discretization-survey.md`

> 两条后端共享 `LagrangeFEMAnalyzer` 状态方程求解的后端选择
> （`operator_backend = assembled | matrix_free | piml_multiscale`），
> 并在"PIML 预测局部等效刚度 → 全局 Matrix-Free 作用"处协同。
