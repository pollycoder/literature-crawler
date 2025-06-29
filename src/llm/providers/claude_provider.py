import anthropic
import asyncio
from typing import List, Dict, Any
from .base_provider import BaseLLMProvider
from ...core.config_models import ProviderConfig
from ...core.exceptions import APIException, RateLimitException, AuthenticationException

class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude模型提供商"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout
        }
        
        # 如果有base_url配置，添加到客户端配置中
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        self.client = anthropic.AsyncAnthropic(**client_kwargs)
        
        # Claude定价
        self.pricing = {
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075}
        }
    
    async def generate_text(self, prompt: str, model_name: str, **kwargs) -> str:
        """生成文本"""
        try:
            response = await self.client.messages.create(
                model=model_name,
                max_tokens=kwargs.get("max_tokens", 4000),
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except anthropic.RateLimitError as e:
            status_code = getattr(e, 'status_code', None) or 429
            raise RateLimitException(f"Claude rate limit exceeded: {str(e)}", status_code=status_code, provider="claude")
        except anthropic.AuthenticationError as e:
            status_code = getattr(e, 'status_code', None) or 401
            raise AuthenticationException(f"Claude authentication failed: {str(e)}", status_code=status_code, provider="claude")
        except anthropic.BadRequestError as e:
            status_code = getattr(e, 'status_code', None) or 400
            raise APIException(f"Claude API error: {str(e)}", status_code=status_code, provider="claude")
        except anthropic.NotFoundError as e:
            status_code = getattr(e, 'status_code', None) or 404
            raise APIException(f"Claude API error: {str(e)}", status_code=status_code, provider="claude")
        except anthropic.PermissionDeniedError as e:
            status_code = getattr(e, 'status_code', None) or 403
            raise APIException(f"Claude API error: {str(e)}", status_code=status_code, provider="claude")
        except anthropic.InternalServerError as e:
            status_code = getattr(e, 'status_code', None) or 500
            raise APIException(f"Claude API error: {str(e)}", status_code=status_code, provider="claude")
        except Exception as e:
            status_code = getattr(e, 'status_code', None) or 500
            raise APIException(f"Claude API error: {str(e)}", status_code=status_code, provider="claude")
    
    async def generate_batch(self, prompts: List[str], model_name: str, **kwargs) -> List[str]:
        """批量生成文本"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def generate_single(prompt: str) -> str:
            async with semaphore:
                return await self.generate_text(prompt, model_name, **kwargs)
        
        tasks = [generate_single(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    
    def estimate_cost(self, text: str, model_name: str = "claude-3-haiku-20240307") -> float:
        """估算成本"""
        tokens = self.count_tokens(text)
        model_pricing = self.pricing.get(model_name, {"input": 0.003, "output": 0.015})
        
        input_tokens = tokens * 0.6
        output_tokens = tokens * 0.4
        
        cost = (input_tokens / 1000 * model_pricing["input"] + 
                output_tokens / 1000 * model_pricing["output"])
        return cost
    
