"""Logging primitives with no numerical-package dependencies."""

from __future__ import annotations

from abc import ABC
import logging
from typing import NoReturn


class _MessageFormatter(logging.Formatter):
    """INFO 及以下只输出消息本体; WARNING 及以上保留来源与级别前缀."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno <= logging.INFO:
            return message
        return f"{record.name} - {record.levelname} - {message}"


class BaseLogged(ABC):
    """Compatibility logging base for maintained legacy components."""

    def __init__(
        self,
        enable_logging: bool = True,
        logger_name: str | None = None,
    ) -> None:
        self._enable_logging = enable_logging
        self._logger_name = logger_name or self.__class__.__name__
        self._setup_logging()

    def _setup_logging(self) -> None:
        if self._enable_logging:
            self.logger = logging.getLogger(self._logger_name)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(_MessageFormatter())
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = None

    def enable_logging(self, enable: bool = True) -> None:
        self._enable_logging = enable
        if enable:
            self._setup_logging()
            self._log_info("Logging enabled")
        else:
            if self.logger:
                for handler in self.logger.handlers[:]:
                    self.logger.removeHandler(handler)
            self.logger = None

    def set_log_level(self, level: int) -> None:
        if self.logger:
            self.logger.setLevel(level)
            self._log_info(
                f"Log level set to {logging.getLevelName(level)}"
            )

    def _log_debug(self, message: str) -> None:
        if self._enable_logging and self.logger:
            self.logger.debug(message)

    def _log_info(self, message: str, force_log: bool = False) -> None:
        # 先判 logger 是否存在再判是否该输出: 与原先的嵌套条件等价, 但类型检查器
        # 能沿这条分支收窄 self.logger, 不再把它当成可能的 None
        if self.logger is not None:
            if self._enable_logging or force_log:
                self.logger.info(message)
        elif force_log:
            print(message)

    def _log_warning(
        self,
        message: str,
        force_log: bool = True,
    ) -> None:
        if force_log or (self._enable_logging and self.logger):
            if self.logger:
                self.logger.warning(message)
            elif force_log:
                print(f"WARNING: {message}")

    def _log_error(
        self,
        message: str,
        force_log: bool = True,
    ) -> NoReturn:
        """记录错误并抛出 ``RuntimeError``, 永不正常返回.

        参数:
            message: 错误信息, 同时作为 ``RuntimeError`` 的内容.
            force_log: 为 ``True`` 时即使关闭日志也输出该错误.

        异常:
            RuntimeError: 恒抛出.

        返回类型标注为 ``NoReturn``: 调用点之后的代码不可达, 类型检查器
        因此不会把 ``if/elif`` 末尾调用本方法的分支推断为隐式返回 ``None``.
        """
        if force_log or (self._enable_logging and self.logger):
            if self.logger:
                self.logger.error(message)
            elif force_log:
                print(f"ERROR: {message}")
        raise RuntimeError(message)

    def _log_critical(self, message: str) -> None:
        if self._enable_logging and self.logger:
            self.logger.critical(message)

    def is_logging_enabled(self) -> bool:
        return self._enable_logging

    def get_logger_name(self) -> str:
        return self._logger_name
