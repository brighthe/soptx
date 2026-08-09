# PIML 二维线弹性子结构静力缩聚示例 (PIML Elasticity Demo)

本目录提供 **Problem-Independent Machine Learning (PIML)** 范式在二维线弹性静力学结构的端到端自包含代码范例。

> 当前状态：**早期原型**。PIML 预测器仅完成单个子结构的 K_s 预测与局部误差评估，
> 尚未将预测算子装配到全局接口系统。基线精确缩聚路径（`FEAStaticCondensation`）
> 可正常完成全流程。详细已知问题与后续计划见 [`results_analysis.md`](results_analysis.md)。

---

## 目录结构

```text
piml_elasticity/
├── README.md              # 本说明文件
├── results_analysis.md    # 数学—代码映射契约、验证范围与已知问题
├── substructure.py        # 2D 子结构网格、SIMP 局部刚度装配
├── assembler.py           # 全尺度 FE 参考解、全局接口装配
├── static_condensation.py # 精确缩聚 (FEAStaticCondensation) 与 PIML 代理 (PIMLStaticCondensation) 统一接口
├── minimal_demo.py        # 自包含端到端 Demo：精确基线 + MLP 训练 + 单子结构 K_s 误差评估
└── outputs/               # 训练日志、误差数据与位移场对比可视化图
```

---

## 快速上手与运行方式

```bash
python minimal_demo.py
```

---

## 核心算法流程

1. **几何与网格剖分**：构建悬臂梁结构，划分为 N_x × N_y 个粗子结构，每个子结构细分 L × L 个 Q4 有限元；
2. **局部静力缩聚 (Exact Baseline)**：提取各子结构分块刚度，计算精确 Schur 补刚度 K_s 与形函数 N；
3. **全尺度 FE 参考解与位移恢复验证**：全装配求解参考解，通过 N × u_b 回代验证机器精度一致性；
4. **PIML 代理网络演示**：在单个子结构上以 200 个随机密度样本训练 MLP，比较预测 K_s 与精确 K_s 的相对误差。

## 精确基线验证

精确缩聚路径（`FEAStaticCondensation`）的验收结果：

- 位移场恢复相对 L_2 误差 (vs 全尺度 FEA 直解)：`< 1.0e-12`
- 子结构缩聚精确性：机器精度

上述数字来自精确缩聚的 `minimal_demo.py` 输出；PIML 预测路径尚未达到同等验收水平。

## PIML 预测器接口

`static_condensation.py` 提供了 MFEM 风格的统一抽象：

- `StaticCondensationBase` — 抽象基类，定义 `condense()` 与 `recover()` 接口；
- `FEAStaticCondensation` — 精确 Schur 补消元（基线）；
- `PIMLStaticCondensation` — 神经网络代理，含自动精确回退。

PIML 代理在以下条件自动触发精确回退：模型未加载、密度输入缺失、预测异常、正定性检查失败。
