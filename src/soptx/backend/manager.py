# 移植自 brighthe/fealpy ``fealpy/backend/manager.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""后端管理器: 按名加载后端, 为每个线程记录当前后端, 并转发属性访问."""

from typing import Dict, Optional
import importlib
import logging
import threading

from .base import BackendProxy

logger = logging.getLogger(__name__)


class BackendManager():
    """后端管理器.

    各后端代理只加载一次; 当前后端按线程记录. 对管理器的属性读写都转发给当前
    后端, 未设置时自动加载默认后端.

    Parameters
    ----------
    default_backend : str, optional
        首次访问后端属性而尚未设置后端时使用的默认后端名.
    """
    # _instance = None

    # def __new__(cls, *, default_backend: str):
    #     if cls._instance is None:
    #         cls._instance = super().__new__(cls)
    #     return cls._instance

    def __init__(self, *, default_backend: Optional[str]=None):
        self._backends: Dict[str, BackendProxy] = {}
        self._THREAD_LOCAL = threading.local()
        self._default_backend_name = default_backend

    def set_backend(self, name: str) -> None:
        """把当前线程的后端设为 ``name``, 未加载时先加载."""
        if name not in self._backends:
            self.load_backend(name)
        self._THREAD_LOCAL.__dict__['backend'] = self._backends[name]

    def load_backend(self, name: str) -> None:
        """按名加载后端: 导入 ``soptx.backend.<name>_backend`` 并实例化其代理.

        Raises
        ------
        RuntimeError
            找不到或无法加载该后端.
        """
        if name not in BackendProxy._available_backends:
            try:
                importlib.import_module(f"soptx.backend.{name}_backend")
            except ImportError:
                raise RuntimeError(f"Backend '{name}' is not found.")

        if name in BackendProxy._available_backends:
            if name in self._backends:
                logger.info(f"Backend '{name}' has already been loaded.")
                return
            # NOTE: 加载时实例化后端代理. 代理实例是单例, 无需加载两次.
            backend = BackendProxy._available_backends[name]()
            self._backends[name] = backend
        else:
            raise RuntimeError(f"Failed to load backend '{name}'.")

    def get_current_backend(self, logger_msg=None) -> BackendProxy:
        """返回当前线程的后端; 尚未设置时加载默认后端.

        Parameters
        ----------
        logger_msg : str, optional
            触发自动设置的访问说明, 写入日志与报错信息.

        Raises
        ------
        RuntimeError
            尚未设置后端且没有默认后端.
        """
        if 'backend' not in self._THREAD_LOCAL.__dict__:
            if self._default_backend_name is None:
                raise RuntimeError(
                    f"Backend properties were accessed ({logger_msg}) "
                    "before a backend was specified, "
                    "and no default backend was set in the backend manager."
                )
            self.set_backend(self._default_backend_name)
            logger.info(f"Backend auto-setting triggered by {logger_msg}."
                        "To reduce unnecessary backend loading, "
                        "get backend properties and methods after executing set_backend()")
        return self._THREAD_LOCAL.__dict__['backend']

    def __getattr__(self, item):
        """把属性读取转发给当前后端."""
        return getattr(self.get_current_backend("GET_ATTR: " + item), item)

    def __setattr__(self, key, value):
        """管理器自身的三个内部属性直接设置, 其余属性写到当前后端上."""
        if key in {'_backends', '_THREAD_LOCAL', '_default_backend_name'}:
            super().__setattr__(key, value)
        else:
            setattr(self.get_current_backend("SET_ATTR: " + key), key, value)
