import openai
import asyncio
from typing import List, Dict, Any
from .base_provider import BaseLLMProvider
from ...core.config_models import ProviderConfig
from ...core.exceptions import APIException, RateLimitException, AuthenticationException

class OpenAIProvider(BaseLLMProvider):
    """OpenAI模型提供商"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout
        }
        
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        self.client = openai.AsyncOpenAI(**client_kwargs)
        
        # OpenAI定价（示例价格，实际应从配置读取）
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }
    
    async def generate_text(self, prompt: str, model_name: str, **kwargs) -> str:
        """生成文本"""
        try:
            # 构建请求参数
            request_params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 4000),
                "timeout": self.timeout
            }
            
            # o1/o3/o4推理模型不支持temperature参数
            if not self._is_reasoning_model(model_name):
                request_params["temperature"] = kwargs.get("temperature", 0.7)
            
            response = await self.client.chat.completions.create(**request_params)
            
            return response.choices[0].message.content
            
        except openai.RateLimitError as e:
            raise RateLimitException(f"OpenAI rate limit exceeded: {str(e)}", provider="openai")
        except openai.AuthenticationError as e:
            raise AuthenticationException(f"OpenAI authentication failed: {str(e)}", provider="openai")
        except Exception as e:
            raise APIException(f"OpenAI API error: {str(e)}", provider="openai")
    
    async def generate_batch(self, prompts: List[str], model_name: str, **kwargs) -> List[str]:
        """批量生成文本"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def generate_single(prompt: str) -> str:
            async with semaphore:
                return await self.generate_text(prompt, model_name, **kwargs)
        
        tasks = [generate_single(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    
    def estimate_cost(self, text: str, model_name: str = "gpt-3.5-turbo") -> float:
        """估算成本"""
        tokens = self.count_tokens(text)
        model_pricing = self.pricing.get(model_name, {"input": 0.01, "output": 0.01})
        
        # 简化：假设输入输出各占一半
        input_tokens = tokens * 0.6
        output_tokens = tokens * 0.4
        
        cost = (input_tokens / 1000 * model_pricing["input"] + 
                output_tokens / 1000 * model_pricing["output"])
        return cost
    
    def _is_reasoning_model(self, model_name: str) -> bool:
        """判断是否为推理模型（不支持temperature参数）"""
        reasoning_prefixes = ('o')
        return model_name.startswith(reasoning_prefixes)
    
