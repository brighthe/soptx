# 经典子结构有限元与静力缩聚范例 (Substructure FE Elasticity Demo)

> **理论事实源**：本范例对应的完整物理原理、变分推导与 Schur 补代数性质见知识库概念页：`C:\workspace\dut-postdoc\concepts\substructural-condensation.md`。

本目录提供 **100% 纯粹的经典子结构有限元 (Substructure FEM) 与静力缩聚 (Static Condensation / Schur Complement)** 端到端自包含代码范例（0 神经网络依赖）。

---

## 导航与文件职责

| 文件 | 职责说明 |
|---|---|
| **[README.md](README.md)** | 本文件：快速上手、架构设计与快速导航 |
| **[math_spec.md](math_spec.md)** | 代码映射规范：`substructure.py` 代码变量与数学符号 1-to-1 映射 |
| **[substructure.py](substructure.py)** | 纯有限元子结构网格剖分 (2D/3D)、SIMP 刚度装配与 Schur 补消元 |
| **[assembler.py](assembler.py)** | 全局接口 Scatter-Add 装配与全尺寸 FEA 直解基线 (2D/3D) |
| **[minimal_demo.py](minimal_demo.py)** | 100% 纯有限元自包含测试主程序 (2D & 3D 内部缩聚位移恢复测试) |
| **[compare_lagrange.py](compare_lagrange.py)** | 与 SOPTX 官方标准全装配拉格朗日有限元 (Lagrange FEM) 的交叉比对测试 |
| **[results_analysis.md](results_analysis.md)** | 实验诊断分析报告：包含 2D/3D 实测误差数据表格、柔度比对与降阶分析 |

---

## 快速上手

在 Shell 中运行纯有限元测试主程序与拉格朗日有限元对比程序：

```bash
# 1. 运行子结构内部缩聚位移恢复与与拉格朗日对比集成 Demo
python minimal_demo.py

# 2. 运行与拉格朗日有限元 (Lagrange FEM) 的 2D / 3D 独立比对
python compare_lagrange.py --dim 2
python compare_lagrange.py --dim 3
```

---

## 验证结果标尺

- **2D 子结构缩聚位移恢复误差**：`3.7618e-16`（机器精度，严格数学等价）
- **3D 子结构缩聚位移恢复误差**：`3.4338e-16`（机器精度，严格数学等价）
- **2D/3D 与拉格朗日有限元 (Lagrange FEM) 柔度及位移相对误差**：`< 1.0e-14`（机器精度级一致）

