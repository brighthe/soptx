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
- [progress-matrix-free.md](../ai/common/progress-matrix-free.md) — Matrix-Free 当前进度和续接入口
- [matrix_free_architecture_notes.md](matrix_free_architecture_notes.md) — SOPTX 实现备忘录
- 总体计划：`dut-postdoc/research/soptx-matrix-free-integration-plan.md`
- 数学原则：`dut-postdoc/research/matrix_free_math_principles.md`

## PIML 多尺度预测

- [ai_piml_context.md](ai_piml_context.md) — **接续上下文（总入口）**，先读这份
- [piml_multiscale_architecture_notes.md](piml_multiscale_architecture_notes.md) — SOPTX 实现备忘录
- 总体计划：`dut-postdoc/research/soptx-piml-multiscale-integration-plan.md`
- 数学原则：`dut-postdoc/research/piml_multiscale_math_principles.md`

> 两条后端共享 `LagrangeFEMAnalyzer` 状态方程求解的后端选择
> （`operator_backend = assembled | matrix_free | piml_multiscale`），
> 并在"PIML 预测局部等效刚度 → 全局 Matrix-Free 作用"处协同。
