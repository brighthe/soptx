# 移植自 brighthe/fealpy ``fealpy/backend/__init__.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""SOPTX 计算后端.

提供全局后端管理器 ``backend_manager`` (别名 ``bm``), 默认后端为 numpy. 通过
``bm.set_backend('pytorch')`` 切换, 之后 ``bm.zeros``、``bm.einsum`` 等调用转发给
当前线程选定的后端.
"""
import logging

from .base import *
from .manager import BackendManager

# 与 FEALPy 的 ``logs.py`` 一致: 应用未配置日志时, 后端日志不输出.
logging.getLogger(__name__).addHandler(logging.NullHandler())

backend_manager = BackendManager(default_backend='numpy')
bm = backend_manager
Tensor = TensorLike
