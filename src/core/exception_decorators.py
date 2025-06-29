"""
统一的异常处理装饰器
消除各个模块中重复的异常处理逻辑
"""

import asyncio
import functools
from typing import Dict, Any, Callable, TypeVar, Optional, Type
from enum import Enum

from .exceptions import (
    APIException, AuthenticationException, RateLimitException, 
    ValidationException, ConfigurationException
)
from ..utils.exception_handler import ExceptionHandler
from ..utils.llm_logger import get_llm_logger

F = TypeVar('F', bound=Callable[..., Any])

class ProviderType(Enum):
    """提供商类型枚举"""
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"


class ExceptionMapper:
    """异常映射器"""
    
    # 各提供商的异常映射
    PROVIDER_EXCEPTIONS = {
        ProviderType.OPENAI: {
            "RateLimitError": RateLimitException,
            "AuthenticationError": AuthenticationException,
            "InvalidRequestError": ValidationException,
            "APIError": APIException,
            "APIConnectionError": APIException,
            "Timeout": APIException,
        },
        ProviderType.CLAUDE: {
            "RateLimitError": RateLimitException,
            "AuthenticationError": AuthenticationException,
            "BadRequestError": ValidationException,
            "APIError": APIException,
            "APIConnectionError": APIException,
        },
        ProviderType.GEMINI: {
            "ResourceExhausted": RateLimitException,
            "Unauthenticated": AuthenticationException,
            "InvalidArgument": ValidationException,
            "FailedPrecondition": ValidationException,
            "DeadlineExceeded": APIException,
        },
        ProviderType.DEEPSEEK: {
            "429": RateLimitException,
            "401": AuthenticationException,
            "403": AuthenticationException,
            "400": ValidationException,
            "500": APIException,
            "502": APIException,
            "503": APIException,
        }
    }
    
    @classmethod
    def map_exception(cls, exception: Exception, provider: ProviderType, 
                     provider_name: str) -> Exception:
        """
        将原始异常映射为统一的异常类型
        
        Args:
            exception: 原始异常
            provider: 提供商类型
            provider_name: 提供商名称
        
        Returns:
            映射后的异常
        """
        exception_name = type(exception).__name__
        exception_mapping = cls.PROVIDER_EXCEPTIONS.get(provider, {})
        
        # 首先尝试精确匹配异常类型名
        if exception_name in exception_mapping:
            exception_class = exception_mapping[exception_name]
            return exception_class(
                f"{provider_name} {exception_name}: {str(exception)}",
                status_code=getattr(exception, 'status_code', None),
                provider=provider_name
            )
        
        # 对于HTTP状态码异常（如DeepSeek）
        if hasattr(exception, 'status') or hasattr(exception, 'status_code'):
            status_code = getattr(exception, 'status', None) or getattr(exception, 'status_code', None)
            if status_code and str(status_code) in exception_mapping:
                exception_class = exception_mapping[str(status_code)]
                return exception_class(
                    f"{provider_name} HTTP {status_code}: {str(exception)}",
                    status_code=status_code,
                    provider=provider_name
                )
        
        # 对于字符串匹配（如Gemini）
        error_msg = str(exception).lower()
        for pattern, exception_class in exception_mapping.items():
            if pattern.lower() in error_msg:
                return exception_class(
                    f"{provider_name} {pattern}: {str(exception)}",
                    status_code=getattr(exception, 'status_code', None),
                    provider=provider_name
                )
        
        # 默认映射为API异常
        return APIException(
            f"{provider_name} unknown error: {str(exception)}",
            status_code=getattr(exception, 'status_code', None),
            provider=provider_name
        )


def handle_provider_exceptions(provider: ProviderType, provider_name: str):
    """
    处理提供商异常的装饰器
    
    Args:
        provider: 提供商类型
        provider_name: 提供商名称
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # 映射异常
                mapped_exception = ExceptionMapper.map_exception(e, provider, provider_name)
                
                # 记录异常
                logger = get_llm_logger()
                logger.log_error(
                    request_id=kwargs.get('request_id', 'unknown'),
                    error_type=type(mapped_exception).__name__,
                    error_message=str(mapped_exception),
                    provider=provider_name,
                    context={"original_exception": str(e)}
                )
                
                raise mapped_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 映射异常
                mapped_exception = ExceptionMapper.map_exception(e, provider, provider_name)
                
                # 记录异常
                logger = get_llm_logger()
                logger.log_error(
                    request_id=kwargs.get('request_id', 'unknown'),
                    error_type=type(mapped_exception).__name__,
                    error_message=str(mapped_exception),
                    provider=provider_name,
                    context={"original_exception": str(e)}
                )
                
                raise mapped_exception
        
        # 根据函数类型返回相应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore
    
    return decorator


def handle_api_exceptions(component_name: str, operation: str = "operation"):
    """
    处理通用API异常的装饰器
    
    Args:
        component_name: 组件名称
        operation: 操作名称
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except (APIException, AuthenticationException, RateLimitException, ValidationException):
                # 已知异常直接重新抛出
                raise
            except Exception as e:
                # 未知异常包装为API异常
                error_msg = f"{component_name} {operation} failed: {str(e)}"
                logger = get_llm_logger()
                logger.log_error(
                    request_id=kwargs.get('request_id', 'unknown'),
                    error_type=type(e).__name__,
                    error_message=error_msg,
                    context={"component": component_name, "operation": operation}
                )
                
                raise APIException(error_msg)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (APIException, AuthenticationException, RateLimitException, ValidationException):
                # 已知异常直接重新抛出
                raise
            except Exception as e:
                # 未知异常包装为API异常
                error_msg = f"{component_name} {operation} failed: {str(e)}"
                logger = get_llm_logger()
                logger.log_error(
                    request_id=kwargs.get('request_id', 'unknown'),
                    error_type=type(e).__name__,
                    error_message=error_msg,
                    context={"component": component_name, "operation": operation}
                )
                
                raise APIException(error_msg)
        
        # 根据函数类型返回相应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore
    
    return decorator


def retry_on_failure(max_retries: int = 3, backoff_factor: float = 2.0, 
                    retry_exceptions: tuple = (APIException,)):
    """
    失败重试装饰器
    
    Args:
        max_retries: 最大重试次数
        backoff_factor: 退避因子
        retry_exceptions: 需要重试的异常类型
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        # 最后一次重试失败
                        logger = get_llm_logger()
                        logger.log_error(
                            request_id=kwargs.get('request_id', 'unknown'),
                            error_type=type(e).__name__,
                            error_message=f"Max retries ({max_retries}) exceeded: {str(e)}",
                            context={"attempts": max_retries + 1}
                        )
                        raise
                    
                    # 计算等待时间
                    wait_time = backoff_factor ** attempt
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    # 非重试异常直接抛出
                    raise
            
            # 理论上不会到达这里
            if last_exception:
                raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        # 最后一次重试失败
                        logger = get_llm_logger()
                        logger.log_error(
                            request_id=kwargs.get('request_id', 'unknown'),
                            error_type=type(e).__name__,
                            error_message=f"Max retries ({max_retries}) exceeded: {str(e)}",
                            context={"attempts": max_retries + 1}
                        )
                        raise
                    
                    # 同步等待（不推荐在生产环境中使用）
                    import time
                    wait_time = backoff_factor ** attempt
                    time.sleep(wait_time)
                except Exception as e:
                    # 非重试异常直接抛出
                    raise
            
            # 理论上不会到达这里
            if last_exception:
                raise last_exception
        
        # 根据函数类型返回相应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore
    
    return decorator


def log_performance(component_name: str, operation: str = "operation"):
    """
    性能日志装饰器
    
    Args:
        component_name: 组件名称
        operation: 操作名称
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                
                logger = get_llm_logger()
                logger.log_request(
                    provider=component_name,
                    model=operation,
                    prompt="",  # 性能日志不记录具体内容
                    request_id=kwargs.get('request_id', 'unknown'),
                    context={"duration": duration, "status": "success"}
                )
                
                return result
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                
                logger = get_llm_logger()
                logger.log_error(
                    request_id=kwargs.get('request_id', 'unknown'),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"duration": duration, "component": component_name, "operation": operation}
                )
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                
                logger = get_llm_logger()
                logger.log_request(
                    provider=component_name,
                    model=operation,
                    prompt="",  # 性能日志不记录具体内容
                    request_id=kwargs.get('request_id', 'unknown'),
                    context={"duration": duration, "status": "success"}
                )
                
                return result
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                
                logger = get_llm_logger()
                logger.log_error(
                    request_id=kwargs.get('request_id', 'unknown'),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"duration": duration, "component": component_name, "operation": operation}
                )
                
                raise
        
        # 根据函数类型返回相应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore
    
    return decorator