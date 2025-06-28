from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import asyncio
import time
from ...core.config_models import ProviderConfig
from ...core.exceptions import APIException, RateLimitException
from ...utils.llm_logger import get_llm_logger, StatusCode
from ...utils.exception_handler import ExceptionHandler, RetryHandler

class BaseLLMProvider(ABC):
    """LLM提供商基类"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.timeout = config.timeout
        self.retry_count = config.retry_count
        
        # 性能统计
        self.request_count = 0
        self.success_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # 日志记录器
        self.logger = get_llm_logger()
        
        # 异常处理器
        self.exception_handler = ExceptionHandler()
        self.retry_handler = RetryHandler(max_retries=self.retry_count)
        
    @abstractmethod
    async def generate_text(self, prompt: str, model_name: str, **kwargs) -> str:
        """生成文本"""
        pass
    
    @abstractmethod
    async def generate_batch(self, prompts: List[str], model_name: str, **kwargs) -> List[str]:
        """批量生成文本"""
        pass
    
    @abstractmethod
    def estimate_cost(self, text: str, model_name: str = None) -> float:
        """估算成本"""
        pass
    
    def count_tokens(self, text: str) -> int:
        """计算token数量（通用实现）"""
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + english_chars / 4)
    
    def test_connection(self) -> bool:
        """测试连接（通用实现）"""
        try:
            if self.config.models:
                model_name = self.config.models[0].name
                
                async def _test():
                    return await self.generate_text("Hello", model_name=model_name, max_tokens=1000)
                
                try:
                    loop = asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _test())
                        future.result(timeout=20)
                    return True
                except RuntimeError:
                    asyncio.run(_test())
                    return True
            return False
        except Exception:
            return False
    
    async def generate_with_retry(self, prompt: str, model_name: str, **kwargs) -> str:
        """带统一异常处理的生成方法"""
        start_time = time.time()
        provider_name = self.config.provider_type
        
        # 记录请求，每次请求生成新的session_id
        request_id, session_id = self.logger.log_request(
            provider=provider_name,
            model=model_name,
            prompt=prompt,
            params=kwargs
        )
        
        # 构建上下文信息
        context = {
            "request_id": request_id,
            "session_id": session_id,
            "provider": provider_name,
            "model": model_name,
            "start_time": start_time
        }
        
        try:
            # 使用统一重试处理器
            result = await self.retry_handler.execute_with_retry(
                self._execute_single_request,
                prompt, model_name, request_id, context, **kwargs
            )
            
            return result
            
        except Exception as e:
            # 使用统一异常处理器
            exc_info = self.exception_handler.handle_exception(e, context)
            
            # 记录最终失败
            duration = time.time() - start_time
            context["duration"] = duration
            
            # 重新抛出异常
            raise
    
    async def _execute_single_request(self, prompt: str, model_name: str, 
                                    request_id: str, context: Dict[str, Any], 
                                    **kwargs) -> str:
        """执行单次请求"""
        try:
            self.request_count += 1
            result = await self.generate_text(prompt, model_name, **kwargs)
            self.success_count += 1
            
            # 更新统计
            tokens = self.count_tokens(prompt + result)
            self.total_tokens += tokens
            cost = self.estimate_cost(prompt + result, model_name)
            self.total_cost += cost
            
            # 记录成功响应
            duration = time.time() - context["start_time"]
            self.logger.log_response(
                request_id=request_id,
                response=result,
                tokens_used=tokens,
                cost=cost,
                duration=duration,
                status_code=StatusCode.SUCCESS,
                session_id=context["session_id"],
                model=model_name
            )
            
            return result
            
        except Exception as e:
            # 让重试处理器决定是否重试
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        success_rate = self.success_count / self.request_count if self.request_count > 0 else 0
        return {
            'provider': self.__class__.__name__,
            'provider_type': self.config.provider_type,
            'request_count': self.request_count,
            'success_count': self.success_count,
            'success_rate': success_rate,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'avg_cost_per_request': self.total_cost / self.request_count if self.request_count > 0 else 0
        }
    
    def reset_stats(self):
        """重置统计"""
        self.request_count = 0
        self.success_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0