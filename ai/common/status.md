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
| Matrix-Free 结构分析原型 | active | `ai/common/progress-matrix-free.md` | Python/NumPy 数学路径、接口闭环和正确性验证完成 | 单机 GPU/多后端验证与 benchmark |
| PIML 多尺度原型 | active | `ai/common/progress-frame7_piml.md` | 单步前向闭环打通（分支 codex/piml-multiscale-prototype）；V1 机器精度（L=5: 1.4e-15, L=10: 2.6e-15）；接口解 vs 全尺度直解 ~1e-12；T4b 极小 MLP TrainedPredictor 已训练（‖K̂_s−K_s‖/‖K_s‖ 均值 L=5: 1.6e-3、L=10: 8.2e-3，优于 Mock ~3e-2）；deck 帧 7 ④ 已回填 dut-postdoc | 阶段二：`operator_backend="piml_multiscale"` 接入 `LagrangeFEMAnalyzer`（V3 出图可选缓做） |

## 全局约定

1. 本仓库开发和测试默认使用本地虚拟环境：

   ```text
   C:\workspace\soptx_heliang\.venv
   ```

2. 运行 Python/pytest 时优先使用：

   ```powershell
   .\.venv\Scripts\python.exe ...
   ```

3. `dut-postdoc` 保存研究计划、数学原则和总体路线；`soptx_heliang` 保存当前代码实现、测试、接口决策和与实现强绑定的上下文。
4. 新窗口不要只依赖聊天历史；应按本文件顺链读取工作线文档。
5. 重要阶段结论应沉淀到对应 progress 文档；实现细节再同步到 `docs/*.md` 或 `ai/common/*.md`。
