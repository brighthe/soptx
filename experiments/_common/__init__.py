# -*- coding: utf-8 -*-
"""experiments 各目录共用的测量、问题构建、注册表加载与子进程调度代码.

本包不是安装包 (pyproject 的 packages.find 不含 experiments), 各实验目录的 ``run.py`` 通过
``sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`` 后 ``from _common import ...`` 使用.

模块划分:

- ``metrology``   CPU RSS 分阶段测量 (``StageMeter``, 基于 VmHWM 重置) 与 CUDA 对应物 (``CudaStageMeter``),
                  以及全局内存事实常量 ``DEFAULT_MEMORY_TOTAL`` / ``MEMORY_BUDGET``.
- ``fe_problem``  制造解线弹性问题 + tet4 向量 P1 空间 + 材料的统一构建, CPU/CUDA 后端切换.
- ``casefile``    ``cases.toml`` 的加载、校验与 ``Case`` 对象.
- ``scheduler``   独立子进程调度: 实时资源监控、OOM 归因、失败侧车、算例表打印.
- ``baseline``    单核硬件基线 (memcpy 带宽、dgemm 算力), 供算子乘指标换算占比.

代码抽自 ``experiments/fa_assembly_capability/run.py`` 与 ``config.py``, 口径与其一致.
"""
