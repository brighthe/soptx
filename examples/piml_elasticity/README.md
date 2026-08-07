# PIML 二维线弹性子结构静力缩聚示例 (PIML Elasticity Demo)

本目录提供 **Problem-Independent Machine Learning (PIML)** 范式在二维线弹性静力学结构的端到端自包含代码范例。

算例对标 `soptx/examples/pinn_elasticity`，展示了如何通过**子结构静力缩聚（Schur Complement）**将全尺度细网格位移求解降维至接口自由度求解，并利用神经网络代理秒级预测局部缩聚刚度。

---

## 目录结构

```text
piml_elasticity/
├── README.md           # 本说明文件
├── math_spec.md        # 静力缩聚原理与 PIML 代理代数规范
├── minimal_demo.py     # 100% 自包含的 PIML 端到端测试 Demo
└── outputs/            # 训练日志、误差数据与位移场对比可视化图
```

---

## 快速上手与运行方式

进入 `soptx` 目录或当前算例目录，直接运行自包含主程序：

```bash
python minimal_demo.py
```

---

## 核心算法流程

1. **几何与网格剖分**：构建悬臂梁结构，划分为 $N_x \times N_y$ 个粗子结构，每个子结构细分 $L \times L$ 个 Q4 有限元；
2. **局部静力缩聚 (Exact Baseline)**：提取各子结构分块刚度 $\mathbf{K}_{ii}, \mathbf{K}_{ib}, \mathbf{K}_{bb}$，计算精确 Schur 补刚度 $\mathbf{K}_s^j$ 与形函数 $\mathbf{N}^j$；
3. **全局接口求解与细尺度恢复**：将 $\mathbf{K}_s^j$ 装配至全局接口系统 $\mathbf{K}_{\text{global}} \boldsymbol{U}_b = \mathbf{F}_b$，求解接口位移后通过 $\boldsymbol{u}_i^j = \mathbf{N}^j \boldsymbol{u}_b^j$ 秒级恢复全场位移；
4. **PIML 代理网络演示**：以 PyTorch MLP 从局部材料密度 $\boldsymbol{\rho}^j$ 预测缩聚刚度 $\widehat{\mathbf{K}}_s^j$，组装求解并评估下游误差。

---

## 验收结果样例

- **子结构缩聚精确性 (vs 全尺度 Schur 补)**：`~ 1.38e-15`（机器精度，严格数学等价）
- **位移场恢复相对 $L_2$ 误差 (vs 全尺度 FEA 直解)**：`< 1.0e-12`
- **全局求解降维比**：`2.56× ~ 4.76×`
