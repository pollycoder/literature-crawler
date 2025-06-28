"""
统一异常处理系统
提供优雅的异常分类、处理和恢复机制
"""

import asyncio
import functools
import time
from typing import Dict, Any, Optional, Callable, List, Type, Union
from enum import Enum
from dataclasses import dataclass
from ..core.exceptions import APIException, RateLimitException, AuthenticationException
from .llm_logger import get_llm_logger, StatusCode


class ExceptionSeverity(Enum):
    """异常严重程度"""
    LOW = "low"           # 轻微错误，可恢复
    MEDIUM = "medium"     # 中等错误，需要重试
    HIGH = "high"         # 严重错误，需要降级
    CRITICAL = "critical" # 关键错误，服务中断


class ExceptionCategory(Enum):
    """异常分类"""
    AUTHENTICATION = "auth"       # 认证相关
    RATE_LIMIT = "rate_limit"    # 速率限制
    TIMEOUT = "timeout"          # 超时错误
    SERVER_ERROR = "server"      # 服务器错误
    CLIENT_ERROR = "client"      # 客户端错误  
    NETWORK = "network"          # 网络错误
    VALIDATION = "validation"    # 参数验证
    UNKNOWN = "unknown"          # 未知错误


@dataclass
class ExceptionInfo:
    """异常信息"""
    exception: Exception
    category: ExceptionCategory
    severity: ExceptionSeverity
    is_recoverable: bool
    suggested_action: str
    retry_delay: float = 0
    max_retries: int = 0
    status_code: int = 500


class ExceptionClassifier:
    """异常分类器"""
    
    # 异常类型映射表
    EXCEPTION_MAPPING = {
        # 认证类异常
        AuthenticationException: {
            "category": ExceptionCategory.AUTHENTICATION,
            "severity": ExceptionSeverity.HIGH,
            "recoverable": False,
            "action": "检查API密钥配置",
            "status_code": StatusCode.UNAUTHORIZED
        },
        
        # 速率限制异常
        RateLimitException: {
            "category": ExceptionCategory.RATE_LIMIT,
            "severity": ExceptionSeverity.MEDIUM,
            "recoverable": True,
            "action": "等待后重试",
            "retry_delay": 60,
            "max_retries": 3,
            "status_code": StatusCode.RATE_LIMITED
        },
        
        # 超时异常
        asyncio.TimeoutError: {
            "category": ExceptionCategory.TIMEOUT,
            "severity": ExceptionSeverity.MEDIUM,
            "recoverable": True,
            "action": "增加超时时间后重试",
            "retry_delay": 5,
            "max_retries": 2,
            "status_code": StatusCode.TIMEOUT
        },
        
        # API异常
        APIException: {
            "category": ExceptionCategory.SERVER_ERROR,
            "severity": ExceptionSeverity.HIGH,
            "recoverable": True,
            "action": "检查服务状态后重试",
            "retry_delay": 10,
            "max_retries": 2,
            "status_code": StatusCode.SERVER_ERROR
        },
        
        # 网络异常
        ConnectionError: {
            "category": ExceptionCategory.NETWORK,
            "severity": ExceptionSeverity.MEDIUM,
            "recoverable": True,
            "action": "检查网络连接后重试",
            "retry_delay": 5,
            "max_retries": 3,
            "status_code": StatusCode.BAD_GATEWAY
        },
        
        # 参数验证异常
        ValueError: {
            "category": ExceptionCategory.VALIDATION,
            "severity": ExceptionSeverity.LOW,
            "recoverable": False,
            "action": "检查输入参数",
            "status_code": StatusCode.BAD_REQUEST
        }
    }
    
    @classmethod
    def classify(cls, exception: Exception) -> ExceptionInfo:
        """分类异常"""
        exc_type = type(exception)
        
        # 查找精确匹配
        if exc_type in cls.EXCEPTION_MAPPING:
            mapping = cls.EXCEPTION_MAPPING[exc_type]
        else:
            # 查找父类匹配
            mapping = None
            for exc_class, exc_mapping in cls.EXCEPTION_MAPPING.items():
                if isinstance(exception, exc_class):
                    mapping = exc_mapping
                    break
            
            # 未知异常的默认处理
            if mapping is None:
                mapping = {
                    "category": ExceptionCategory.UNKNOWN,
                    "severity": ExceptionSeverity.HIGH,
                    "recoverable": False,
                    "action": "联系技术支持",
                    "status_code": StatusCode.SERVER_ERROR
                }
        
        return ExceptionInfo(
            exception=exception,
            category=mapping["category"],
            severity=mapping["severity"],
            is_recoverable=mapping["recoverable"],
            suggested_action=mapping["action"],
            retry_delay=mapping.get("retry_delay", 0),
            max_retries=mapping.get("max_retries", 0),
            status_code=mapping["status_code"]
        )


class ExceptionHandler:
    """统一异常处理器"""
    
    def __init__(self):
        self.classifier = ExceptionClassifier()
        self.logger = get_llm_logger()
        self.exception_stats = {}  # 异常统计
        
    def handle_exception(self, 
                        exception: Exception,
                        context: Dict[str, Any] = None) -> ExceptionInfo:
        """处理异常"""
        context = context or {}
        
        # 分类异常
        exc_info = self.classifier.classify(exception)
        
        # 记录异常统计
        self._update_stats(exc_info)
        
        # 记录日志
        self._log_exception(exc_info, context)
        
        # 执行处理动作
        self._execute_action(exc_info, context)
        
        return exc_info
    
    def _update_stats(self, exc_info: ExceptionInfo):
        """更新异常统计"""
        category = exc_info.category.value
        if category not in self.exception_stats:
            self.exception_stats[category] = {
                "count": 0,
                "last_occurrence": None,
                "severity_distribution": {}
            }
        
        self.exception_stats[category]["count"] += 1
        self.exception_stats[category]["last_occurrence"] = time.time()
        
        severity = exc_info.severity.value
        if severity not in self.exception_stats[category]["severity_distribution"]:
            self.exception_stats[category]["severity_distribution"][severity] = 0
        self.exception_stats[category]["severity_distribution"][severity] += 1
    
    def _log_exception(self, exc_info: ExceptionInfo, context: Dict[str, Any]):
        """记录异常日志"""
        request_id = context.get("request_id", "unknown")
        provider = context.get("provider", "unknown")
        model = context.get("model", "unknown")
        session_id = context.get("session_id")
        
        # 只有在有session_id的情况下才记录日志
        if session_id:
            self.logger.log_error(
                request_id=request_id,
                provider=provider,
                model=model,
                error_type=type(exc_info.exception).__name__,
                error_message=str(exc_info.exception),
                duration=context.get("duration", 0),
                status_code=exc_info.status_code,
                session_id=session_id
            )
    
    def _execute_action(self, exc_info: ExceptionInfo, context: Dict[str, Any]):
        """执行处理动作"""
        # 这里可以根据异常类型执行特定动作
        # 比如发送告警、清理资源、记录指标等
        
        if exc_info.severity == ExceptionSeverity.CRITICAL:
            self._handle_critical_exception(exc_info, context)
        elif exc_info.severity == ExceptionSeverity.HIGH:
            self._handle_high_severity_exception(exc_info, context)
    
    def _handle_critical_exception(self, exc_info: ExceptionInfo, context: Dict[str, Any]):
        """处理关键异常"""
        # 发送紧急告警
        print(f"CRITICAL: {exc_info.exception} - {exc_info.suggested_action}")
    
    def _handle_high_severity_exception(self, exc_info: ExceptionInfo, context: Dict[str, Any]):
        """处理高严重性异常"""
        # 发送告警
        print(f"HIGH: {exc_info.exception} - {exc_info.suggested_action}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取异常统计"""
        return self.exception_stats


def with_exception_handling(fallback_value: Any = None, 
                           context_extractor: Callable = None):
    """异常处理装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            handler = ExceptionHandler()
            start_time = time.time()
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # 提取上下文信息
                context = {"duration": time.time() - start_time}
                if context_extractor:
                    context.update(context_extractor(*args, **kwargs))
                
                # 处理异常
                exc_info = handler.handle_exception(e, context)
                
                # 如果异常可恢复且提供了回退值，返回回退值
                if exc_info.is_recoverable and fallback_value is not None:
                    return fallback_value
                
                # 否则重新抛出异常
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            handler = ExceptionHandler()
            start_time = time.time()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 提取上下文信息
                context = {"duration": time.time() - start_time}
                if context_extractor:
                    context.update(context_extractor(*args, **kwargs))
                
                # 处理异常
                exc_info = handler.handle_exception(e, context)
                
                # 如果异常可恢复且提供了回退值，返回回退值
                if exc_info.is_recoverable and fallback_value is not None:
                    return fallback_value
                
                # 否则重新抛出异常
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class RetryHandler:
    """智能重试处理器"""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.handler = ExceptionHandler()
    
    async def execute_with_retry(self, 
                               func: Callable,
                               *args,
                               context: Dict[str, Any] = None,
                               **kwargs) -> Any:
        """带重试的执行函数"""
        context = context or {}
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # 分类异常
                exc_info = self.handler.classifier.classify(e)
                
                # 如果不可恢复或达到最大重试次数，直接抛出
                if not exc_info.is_recoverable or attempt >= exc_info.max_retries:
                    self.handler.handle_exception(e, context)
                    raise
                
                # 记录重试
                if "request_id" in context and "session_id" in context:
                    self.handler.logger.log_retry(
                        request_id=context["request_id"],
                        provider=context.get("provider", "unknown"),
                        model=context.get("model", "unknown"),
                        attempt=attempt + 1,
                        error_type=type(e).__name__,
                        wait_time=exc_info.retry_delay,
                        session_id=context["session_id"]
                    )
                
                # 等待后重试
                if exc_info.retry_delay > 0:
                    await asyncio.sleep(exc_info.retry_delay)
        
        # 不应该到达这里，但以防万一
        if last_exception:
            raise last_exception


# 全局异常处理器实例
_global_handler: Optional[ExceptionHandler] = None


def get_exception_handler() -> ExceptionHandler:
    """获取全局异常处理器"""
    global _global_handler
    if _global_handler is None:
        _global_handler = ExceptionHandler()
    return _global_handler