import asyncio
import aiohttp
import json
from typing import List, Dict, Any
from .base_provider import BaseLLMProvider
from ...core.exceptions import APIException, RateLimitException


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API提供商"""
    
    def __init__(self, config):
        super().__init__(config)
        # 如果没有指定base_url，使用默认的Google API
        if not self.base_url:
            self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    async def generate_text(self, prompt: str, model_name: str, **kwargs) -> str:
        """生成文本"""
        url = f"{self.base_url}/models/{model_name}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        # 构建请求数据
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": kwargs.get('temperature', 0.7),
                "maxOutputTokens": kwargs.get('max_tokens', 4000),
                "topP": kwargs.get('top_p', 0.95),
                "topK": kwargs.get('top_k', 64)
            }
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 429:
                    raise RateLimitException("Rate limit exceeded")
                elif response.status != 200:
                    error_text = await response.text()
                    raise APIException(f"Gemini API error {response.status}: {error_text}")
                
                result = await response.json()
                
                # 解析响应
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            return parts[0]['text']
                
                raise APIException("Invalid response format from Gemini API")
    
    async def generate_batch(self, prompts: List[str], model_name: str, **kwargs) -> List[str]:
        """批量生成文本"""
        # Gemini API没有原生的batch支持，使用并发请求
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
        """估算成本 - Gemini Pro的定价"""
        tokens = self.count_tokens(text)
        # Gemini Pro 定价（示例，实际价格可能变化）
        # 输入: $0.0005/1K tokens, 输出: $0.0015/1K tokens
        return (tokens / 1000) * 0.001  # 简化计算
    
    
