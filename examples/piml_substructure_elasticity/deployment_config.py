"""piml_substructure_elasticity 投产算例的单一配置来源.

本目录下的验证与采集脚本一律从这里取几何, 材料与训练分布参数, 不要各自复制字面量.
`verify_shape_function_route.py`, `verify_stiffness_route.py` 与
`collect_ood_probe_trajectory.py` 面对的必须是同一个 MBB 梁投产配置: 任何一处漂移都
不会报错, 只会静默产出彼此对不上的数, 因此集中在本文件.

配置对齐 Huang 2023 第 4.1 节 MBB 梁算例: 12x2 子结构划分, 单个子结构 5x5 Q4 细单元,
合计 24 个子结构与 60x10 全局网格.
"""

from __future__ import annotations

from pathlib import Path

# --- 几何 ---
DOMAIN = (0.0, 12.0, 0.0, 2.0)
N_SUB = (12, 2)
N_FINE = (5, 5)

# --- 材料与载荷 ---
P_LOAD = -1.0
E_BASE = 1.0
NU = 0.3

# --- 代理模型的训练密度区间 ---
# 离线采样与在线评估共用; 超出该区间即代理工作在外插区间.
DENSITY_RANGE = (0.3, 1.0)
TRAINING_DENSITY_RANGE = DENSITY_RANGE

# --- 产物落盘 ---
OUTPUT_DIR = Path(__file__).parent / "outputs"

# --- 派生量: 一律由上面的基准量算出, 不要手写字面量 ---
DOMAIN_SIZE = (DOMAIN[1] - DOMAIN[0], DOMAIN[3] - DOMAIN[2])
SUB_SIZE = (DOMAIN_SIZE[0] / N_SUB[0], DOMAIN_SIZE[1] / N_SUB[1])
N_SUB_TOTAL = N_SUB[0] * N_SUB[1]
TOTAL_FINE = (N_SUB[0] * N_FINE[0], N_SUB[1] * N_FINE[1])
# 全局细单元尺寸 (hx, hy); 滤波半径 rmin 与它同单位.
CELL_SIZE = (DOMAIN_SIZE[0] / TOTAL_FINE[0], DOMAIN_SIZE[1] / TOTAL_FINE[1])
