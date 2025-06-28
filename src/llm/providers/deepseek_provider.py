import asyncio
import aiohttp
import json
from typing import List, Dict, Any
from .base_provider import BaseLLMProvider
from ...core.exceptions import APIException, RateLimitException


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API提供商"""
    
    def __init__(self, config):
        super().__init__(config)
        # 如果没有指定base_url，使用默认的DeepSeek API
        if not self.base_url:
            self.base_url = "https://api.deepseek.com"
    
    async def generate_text(self, prompt: str, model_name: str, **kwargs) -> str:
        """生成文本"""
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 构建请求数据（使用OpenAI兼容格式）
        data = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": kwargs.get('max_tokens', 4000),
            "temperature": kwargs.get('temperature', 0.7),
            "top_p": kwargs.get('top_p', 1.0),
            "stream": False
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 429:
                    raise RateLimitException("Rate limit exceeded")
                elif response.status != 200:
                    error_text = await response.text()
                    raise APIException(f"DeepSeek API error {response.status}: {error_text}")
                
                result = await response.json()
                
                # 解析响应（OpenAI格式）
                if 'choices' in result and len(result['choices']) > 0:
                    choice = result['choices'][0]
                    if 'message' in choice and 'content' in choice['message']:
                        return choice['message']['content']
                
                raise APIException("Invalid response format from DeepSeek API")
    
    async def generate_batch(self, prompts: List[str], model_name: str, **kwargs) -> List[str]:
        """批量生成文本"""
        # DeepSeek API使用并发请求
        tasks = [self.generate_text(prompt, model_name, **kwargs) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                raise result
            final_results.append(result)
        
        return final_results
    
    def estimate_cost(self, text: str, model_name: str = None) -> float:
        """估算成本 - DeepSeek的定价"""
        tokens = self.count_tokens(text)
        # DeepSeek定价（示例，实际价格可能变化）
        # 通常比OpenAI便宜很多
        return (tokens / 1000) * 0.002  # 简化计算
    
    
