import logging
from typing import Optional
from abc import ABC

class BaseLogged(ABC):
    """提供统一日志功能的基类"""
    
    def __init__(self, 
                 enable_logging: bool = True,
                 logger_name: Optional[str] = None):
        self._enable_logging = enable_logging
        self._logger_name = logger_name or f"{self.__class__.__name__}_{id(self)}"
        self._setup_logging()

    def _setup_logging(self):
        """设置统一的日志系统"""
        if self._enable_logging:
            self.logger = logging.getLogger(self._logger_name)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = None
    
    def enable_logging(self, enable: bool = True):
        """启用或禁用日志输出"""
        self._enable_logging = enable
        if enable:
            self._setup_logging()
            self._log_info("Logging enabled")
        else:
            if self.logger:
                for handler in self.logger.handlers[:]:
                    self.logger.removeHandler(handler)
            self.logger = None
    
    def set_log_level(self, level: int):
        """设置日志级别"""
        if self.logger:
            self.logger.setLevel(level)
            self._log_info(f"Log level set to {logging.getLevelName(level)}")
    
    def _log_debug(self, message: str):
        """输出调试信息"""
        if self._enable_logging and self.logger:
            self.logger.debug(message)
    
    def _log_info(self, message: str, force_log: bool = False):
        """输出一般信息"""
        if (self._enable_logging and self.logger) or force_log:
            if force_log and not self.logger:
                print(f"{self._logger_name} - INFO - {message}")
            else:
                self.logger.info(message)
    
    def _log_warning(self, message: str, force_log: bool = False):
        """输出警告信息"""
        if force_log or (self._enable_logging and self.logger):
            if self.logger:
                self.logger.warning(message)
            elif force_log:
                print(f"WARNING: {message}")
    
    def _log_error(self, message: str):
        """输出错误信息"""
        if self._enable_logging and self.logger:
            self.logger.error(message)
    
    def _log_critical(self, message: str):
        """输出严重错误信息"""
        if self._enable_logging and self.logger:
            self.logger.critical(message)
    
    def is_logging_enabled(self) -> bool:
        """检查日志是否启用"""
        return self._enable_logging
    
    def get_logger_name(self) -> str:
        """获取日志器名称"""
        return self._logger_name