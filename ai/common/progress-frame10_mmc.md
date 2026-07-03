---
title: "SOPTX MMC Cut-Mesh Progress"
tags:
  - ai-context
  - mmc
  - explicit-geometry
  - soptx
  - progress
status: "active"
date: 2026-07-02
---

# MMC 显式几何高精度离散原型 Progress

本文件是 SOPTX MMC 工作线的续接入口。新窗口续接时，先读 `ai/common/status.md`，再读本文件；帧 10 数值快照见 `docs/frame10_mmc_pipeline_results.md`。

## 上位文档（dut-postdoc）

- **帧级主入口**（数学原理 + 任务划分 + 答辩口径）：`research/postdoc-plan/defense-sprint/direction-2-mmc-mmv/frame10_mmc_pipeline_guide.md`
- **deck 侧进度与决策**：`ai/common/progress-part2-mmc.md`
- **长期调研**：`research/postdoc-plan/long-term/direction-2-mmc-mmv/mmc-mmv-numerical-discretization-survey.md`

## 当前结论

截至 2026-07-02，当前结论应表述为：

```text
MMC 显式几何前向切割管线 (T1-V1) 已跑通: 超椭圆 MMC 组件 TDF 隐函数
-> 40x20 背景网格单元三分类 (Solid 140 / Void 580 / Cut 80)
-> cut 单元实体侧多边形重构 + Delaunay 三角化布点
-> 高阶积分点从全域 Ersatz 3200 压缩至 1028, 边界精准截断、孔洞侧零渗入。
产出验证图 docs/mmc_integration_result.png (带局部放大)。
```

当前不应表述为：

```text
已接入 FEALPy/SOPTX 分析链路 / 已有切割网格上的刚度组装与求解 / 已做优化闭环。
```

（当前是仓库根目录的独立 numpy/matplotlib 原型脚本，未进 soptx 包、无测试；
积分点只生成了 `(x, y, weight)` 数据，尚未喂给任何组装/算子。）

## 已定关键决策

1. **答辩范围收敛**：只展示前向"显式几何 → 背景网格精确切割 → 高阶积分点布设"，
   不做 MMC 优化闭环（决策依据见 dut-postdoc `progress-part2-mmc.md`）。
2. **切割方案**：TDF 隐函数（超椭圆 $m=4$）对单元角点取值做 Solid/Void/Cut 三分类；
   Cut 单元沿 TDF 零水平集重构实体侧多边形，再三角化布设积分点。
3. **数据格式以直连 Matrix-Free 为目标**：积分点输出 `(x, y, weight)`，
   目标是后续直接喂给组装/无矩阵算子（与 Matrix-Free 线咬合）。
4. **原型暂为根目录独立脚本**：`mmc_cut_mesh_prototype.py` 未整理进 soptx 包，
   整理 + 最小测试属后续任务。

## 重要代码位置

```text
C:\workspace\soptx_heliang\mmc_cut_mesh_prototype.py          （原型脚本，仓库根目录）
C:\workspace\soptx_heliang\docs\frame10_mmc_pipeline_results.md（帧 10 数值快照）
C:\workspace\soptx_heliang\docs\mmc_integration_result.png     （验证图）
```

## 复现命令

在 `C:\workspace\soptx_heliang` 下运行：

```powershell
.\.venv\Scripts\python.exe mmc_cut_mesh_prototype.py
```

注意：脚本把 PNG 写到**仓库根目录**（`mmc_integration_result.png`），
`docs/` 下那份是手动移入的；重跑后需同步（或后续把脚本输出路径改到 docs/）。

## 下一步（按优先级）

1. **`mmc_baseline.pdf` 矢量图出图**，替换 deck 帧 10 的占位 TikZ
   （deck 侧状态见 dut-postdoc `progress-part2-mmc.md`）。
2. **脚本整理进 soptx 包 + 最小测试**（分类计数/积分点权重和等可固化断言）。
3. **扩展**：切割积分点直连 Matrix-Free 组装/作用；AD 敏度
   （TDF 参数 → 积分点 → 刚度 的可微路径）。

## 新窗口续接提示词

```text
按 C:\workspace\soptx_heliang\ai\common\status.md 续接「MMC 显式几何高精度离散原型」。
先复述当前进度、已定关键决策、下一步与实现计划，我确认后再继续。
```
