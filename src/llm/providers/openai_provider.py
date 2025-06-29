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
                "timeout": self.timeout
            }
            
            # o1/o3/o4推理模型使用max_completion_tokens，其他模型使用max_tokens
            if self._is_reasoning_model(model_name):
                request_params["max_completion_tokens"] = kwargs.get("max_tokens", 4000)
            else:
                request_params["max_tokens"] = kwargs.get("max_tokens", 4000)
            
            # o1/o3/o4推理模型不支持temperature参数
            if not self._is_reasoning_model(model_name):
                request_params["temperature"] = kwargs.get("temperature", 1.0)
            
            response = await self.client.chat.completions.create(**request_params)
            
            return response.choices[0].message.content
            
        except openai.RateLimitError as e:
            status_code = getattr(e, 'status_code', None) or 429
            raise RateLimitException(f"OpenAI rate limit exceeded: {str(e)}", status_code=status_code, provider="openai")
        except openai.AuthenticationError as e:
            status_code = getattr(e, 'status_code', None) or 401
            raise AuthenticationException(f"OpenAI authentication failed: {str(e)}", status_code=status_code, provider="openai")
        except openai.BadRequestError as e:
            status_code = getattr(e, 'status_code', None) or 400
            raise APIException(f"OpenAI API error: {str(e)}", status_code=status_code, provider="openai")
        except openai.NotFoundError as e:
            status_code = getattr(e, 'status_code', None) or 404
            raise APIException(f"OpenAI API error: {str(e)}", status_code=status_code, provider="openai")
        except openai.PermissionDeniedError as e:
            status_code = getattr(e, 'status_code', None) or 403
            raise APIException(f"OpenAI API error: {str(e)}", status_code=status_code, provider="openai")
        except openai.UnprocessableEntityError as e:
            status_code = getattr(e, 'status_code', None) or 422
            raise APIException(f"OpenAI API error: {str(e)}", status_code=status_code, provider="openai")
        except openai.InternalServerError as e:
            status_code = getattr(e, 'status_code', None) or 500
            raise APIException(f"OpenAI API error: {str(e)}", status_code=status_code, provider="openai")
        except openai.APITimeoutError as e:
            status_code = getattr(e, 'status_code', None) or 504
            raise APIException(f"OpenAI API error: {str(e)}", status_code=status_code, provider="openai")
        except Exception as e:
            # 尝试从异常中提取状态码
            status_code = getattr(e, 'status_code', None) or 500
            raise APIException(f"OpenAI API error: {str(e)}", status_code=status_code, provider="openai")
    
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
    
