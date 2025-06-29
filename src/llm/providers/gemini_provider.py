import asyncio
import google.generativeai as genai
from typing import List, Dict, Any
from .base_provider import BaseLLMProvider
from ...core.exceptions import APIException, RateLimitException, AuthenticationException


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API提供商"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # 配置Google SDK
        configure_kwargs = {
            "api_key": self.api_key,
            "transport": "rest"  # 必须使用REST协议，避免gRPC问题
        }
        
        # 如果提供了base_url，使用client_options配置自定义端点
        if self.base_url:
            configure_kwargs["client_options"] = {"api_endpoint": self.base_url}
        
        genai.configure(**configure_kwargs)
    
    async def generate_text(self, prompt: str, model_name: str, **kwargs) -> str:
        """生成文本"""
        try:
            # 创建模型实例
            model = genai.GenerativeModel(model_name)
            
            # 配置生成参数
            generation_config = genai.GenerationConfig(
                temperature=kwargs.get('temperature', 1.0),
                max_output_tokens=kwargs.get('max_tokens', 4000),
                top_p=kwargs.get('top_p', 0.95),
                top_k=kwargs.get('top_k', 64)
            )
            
            # 生成内容（异步调用需要在线程池中执行）
            import concurrent.futures
            import threading
            
            def _generate_sync():
                return model.generate_content(prompt, generation_config=generation_config)
            
            # 在线程池中执行同步调用
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                response = await loop.run_in_executor(executor, _generate_sync)
            
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            
            # 处理常见的Google API错误
            if "403" in error_msg or "Forbidden" in error_msg:
                raise AuthenticationException(f"Gemini authentication failed: {error_msg}", 
                                            status_code=403, provider="gemini")
            elif "429" in error_msg or "quota" in error_msg.lower():
                raise RateLimitException(f"Gemini rate limit exceeded: {error_msg}", 
                                       status_code=429, provider="gemini")
            elif "400" in error_msg or "Bad Request" in error_msg:
                raise APIException(f"Gemini API error: {error_msg}", 
                                 status_code=400, provider="gemini")
            else:
                raise APIException(f"Gemini API error: {error_msg}", 
                                 status_code=500, provider="gemini")
    
    async def generate_batch(self, prompts: List[str], model_name: str, **kwargs) -> List[str]:
        """批量生成文本"""
        # 使用信号量限制并发数
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def generate_single(prompt: str) -> str:
            async with semaphore:
                return await self.generate_text(prompt, model_name, **kwargs)
        
        tasks = [generate_single(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                raise result
            final_results.append(result)
        
        return final_results
    
    def estimate_cost(self, text: str, model_name: str = "gemini-1.5-flash") -> float:
        """估算成本 - Gemini Pro的定价"""
        tokens = self.count_tokens(text)
        # Gemini Pro 定价（示例，实际价格可能变化）
        # 输入: $0.0005/1K tokens, 输出: $0.0015/1K tokens
        return (tokens / 1000) * 0.001  # 简化计算
    
    
