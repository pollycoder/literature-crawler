"""
熔断器模式实现
提供优雅降级和服务保护机制
"""

import time
import asyncio
from typing import Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from .exception_handler import ExceptionSeverity, ExceptionCategory


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"       # 关闭状态，正常工作
    OPEN = "open"          # 开启状态，拒绝请求
    HALF_OPEN = "half_open" # 半开状态，试探性恢复


@dataclass
class CircuitConfig:
    """熔断器配置"""
    failure_threshold: int = 5          # 失败阈值
    success_threshold: int = 3          # 成功阈值（半开状态）
    timeout: float = 60.0              # 熔断超时时间（秒）
    reset_timeout: float = 300.0       # 重置超时时间（秒）


class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, name: str, config: CircuitConfig = None):
        self.name = name
        self.config = config or CircuitConfig()
        
        # 状态管理
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        
        # 统计信息
        self.total_requests = 0
        self.total_failures = 0
        self.total_timeouts = 0
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """通过熔断器调用函数"""
        self.total_requests += 1
        
        # 检查熔断器状态
        await self._check_state()
        
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    async def _check_state(self):
        """检查和更新熔断器状态"""
        current_time = time.time()
        
        if self.state == CircuitState.OPEN:
            # 检查是否可以转换到半开状态
            if current_time - self.last_failure_time >= self.config.timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                print(f"Circuit breaker {self.name} changed to HALF_OPEN")
        
        elif self.state == CircuitState.HALF_OPEN:
            # 在半开状态下，如果成功次数达到阈值，转换到关闭状态
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                print(f"Circuit breaker {self.name} changed to CLOSED")
    
    def _on_success(self):
        """处理成功调用"""
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
        elif self.state == CircuitState.CLOSED:
            # 重置失败计数
            self.failure_count = 0
    
    def _on_failure(self, exception: Exception):
        """处理失败调用"""
        self.total_failures += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            
            # 检查是否需要开启熔断器
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                print(f"Circuit breaker {self.name} changed to OPEN due to failures")
        
        elif self.state == CircuitState.HALF_OPEN:
            # 半开状态下任何失败都会导致重新开启
            self.state = CircuitState.OPEN
            self.failure_count += 1
            print(f"Circuit breaker {self.name} changed to OPEN from HALF_OPEN")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取熔断器统计信息"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "failure_rate": self.total_failures / self.total_requests if self.total_requests > 0 else 0,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time
        }
    
    def reset(self):
        """重置熔断器"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0


class CircuitBreakerManager:
    """熔断器管理器"""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.default_config = CircuitConfig()
    
    def get_breaker(self, name: str, config: CircuitConfig = None) -> CircuitBreaker:
        """获取或创建熔断器"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config or self.default_config)
        return self.breakers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有熔断器统计"""
        return {name: breaker.get_stats() for name, breaker in self.breakers.items()}
    
    def reset_all(self):
        """重置所有熔断器"""
        for breaker in self.breakers.values():
            breaker.reset()


class CircuitBreakerOpenException(Exception):
    """熔断器开启异常"""
    pass


# 全局熔断器管理器
_global_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """获取全局熔断器管理器"""
    global _global_manager
    if _global_manager is None:
        _global_manager = CircuitBreakerManager()
    return _global_manager


def with_circuit_breaker(name: str, config: CircuitConfig = None):
    """熔断器装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = get_circuit_breaker_manager()
            breaker = manager.get_breaker(name, config)
            return await breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator