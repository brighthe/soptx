"""Logging primitives with no numerical-package dependencies."""

from __future__ import annotations

from abc import ABC
import logging


class BaseLogged(ABC):
    """Compatibility logging base for maintained legacy components."""

    def __init__(
        self,
        enable_logging: bool = True,
        logger_name: str | None = None,
    ) -> None:
        self._enable_logging = enable_logging
        self._logger_name = logger_name or (
            f"{self.__class__.__name__}_{id(self)}"
        )
        self._setup_logging()

    def _setup_logging(self) -> None:
        if self._enable_logging:
            self.logger = logging.getLogger(self._logger_name)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(
                    logging.Formatter(
                        "%(name)s - %(levelname)s - %(message)s"
                    )
                )
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
        if (self._enable_logging and self.logger) or force_log:
            if force_log and not self.logger:
                print(f"{self._logger_name} - INFO - {message}")
            else:
                self.logger.info(message)

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
    ) -> None:
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
