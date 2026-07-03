---
title: "SOPTX AI Status"
tags:
  - ai-context
  - soptx
  - status
status: "active"
date: 2026-06-29
---

# SOPTX AI Status

本文件是 `soptx_heliang` 仓库的 AI 上下文 hub。新开 AI/Codex 窗口续接工作时，优先从这里定位当前工作线，再顺链阅读对应 progress 文档。

## 使用方式

推荐续接提示词：

```text
按 C:\workspace\soptx_heliang\ai\common\status.md 续接「Matrix-Free 结构分析原型」。
先复述当前进度、已定关键决策、下一步与实现计划，我确认后再继续。
```

AI 续接流程：

```text
ai/common/status.md
  -> 在工作线表中定位目标工作线
  -> 阅读对应 progress 文档
  -> 如需写代码，再阅读 progress 文档中列出的实现上下文、测试和代码位置
  -> 先复述当前进度、关键决策、下一步计划
  -> 等用户确认后再动手
```

## 工作线表

| 工作线 | 状态 | 续接文档 | 当前阶段 | 下一步 |
|---|---|---|---|---|
| Matrix-Free 结构分析原型 | active | `ai/common/progress-frame8_matrix_free.md` | 三档（NumPy/PyTorch CPU/CUDA）跑通且一致；GPU MatVec 到 ndof≈1.3e5 约 12x vs NumPy | 做 matrix-free 预条件器让大规模 CG 收敛，再补端到端 GPU 求解计时 |
| PIML 多尺度原型 | active | `ai/common/progress-frame7_piml.md` | 单步前向闭环打通（分支 codex/piml-multiscale-prototype）；V1 机器精度（L=5: 1.4e-15, L=10: 2.6e-15）；接口解 vs 全尺度直解 ~1e-12；T4b 极小 MLP TrainedPredictor 已训练（‖K̂_s−K_s‖/‖K_s‖ 均值 L=5: 1.6e-3、L=10: 8.2e-3，优于 Mock ~3e-2）；deck 帧 7 ④ 已回填 dut-postdoc | 阶段二：`operator_backend="piml_multiscale"` 接入 `LagrangeFEMAnalyzer`（V3 出图可选缓做） |
| MMC 显式几何高精度离散原型 | active | `ai/common/progress-frame10_mmc.md` | (T1-V1) $40\times20$ 切割管线跑通；边界精准重构，全域积分点成功从 3200 压缩至 1028 | 提供跑分结果给 PPT 修复；后续扩展 Matrix-Free 直连与 AD 敏度 |

## 全局约定

1. 本仓库开发和测试默认使用本地虚拟环境：

   ```text
   C:\workspace\soptx_heliang\.venv
   ```

2. 运行 Python/pytest 时优先使用：

   ```powershell
   .\.venv\Scripts\python.exe ...
   ```

3. `dut-postdoc` 保存研究计划、数学原则和总体路线；`soptx_heliang` 保存当前代码实现、测试、接口决策和与实现强绑定的上下文。两库文档的跨库映射见下「跨库映射」表。
4. 新窗口不要只依赖聊天历史；应按本文件顺链读取工作线文档。
5. 重要阶段结论应沉淀到对应 progress 文档；实现细节再同步到 `docs/*.md` 或 `ai/common/*.md`。

## 跨库映射（soptx_heliang ↔ dut-postdoc）

唯一的跨库映射入口：`soptx_heliang`（代码/实现/测试）里许多工作线的**上位依据**在
`dut-postdoc`（研究计划/数学原则/答辩口径）。各 progress 文档的「上位文档」节只给细粒度
指针，整体对照以本表为准。引用只用「文件名 + 章节号 + 计划项 ID」，**不写行号**（跨库最易漂）。
末次核对：2026-07-03。

| 工作线 | 上位 guide (dut-postdoc) | 数学原理锚点 | 计划项 ID | 代码实现 (soptx_heliang) |
|---|---|---|---|---|
| PIML 多尺度原型 | `frame7_piml_pipeline_guide.md` | 同 guide §3.3（静力缩聚 / Schur 补） | T1.3.2（长期计划 `piml-matrix-free-execution-plan.md`） | `soptx/analysis/multiscale/` + `tests/test_equivalent_stiffness_vs_fullscale.py` |
| Matrix-Free 结构分析原型 | `frame8_matrix_free_pipeline_guide.md` | 待核 | 待核 | `soptx/analysis/matrix_free/` |
| MMC 显式几何离散原型 | `frame10_mmc_pipeline_guide.md` | 待核 | 待核 | `mmc_cut_mesh_prototype.py` |

> **死链治理记录（2026-07-03）**：核对发现 PIML 工作线原「上位文档」曾指向 dut-postdoc
> 已删除的 `soptx-piml-multiscale-integration-plan.md` 与 `piml_multiscale_math_principles.md`
> （两者均被单帧 guide 归并接管），已更正为 `frame7_piml_pipeline_guide.md`（数学原理并入其 §3）。
> 非 PIML 两行的 guide 名取自 dut-postdoc `log.md`；数学原理锚点 / 计划项 ID 待各工作线接手时核对。
