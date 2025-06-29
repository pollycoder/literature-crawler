#!/usr/bin/env python3
"""
简单的模型API连接测试
"""

import asyncio
from openai import AsyncOpenAI

async def test_api():
    client = AsyncOpenAI(
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
        base_url='https://api.openai-proxy.org/v1',
        api_key='sk-HoCIo0kiA6xoIWWfybIvu2T17TCtVhGM1HGh6kk9tQy5wMi7',
    )

    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "告诉我2025年有多少天？",
                }
            ],
            model="o3", # 如果是其他兼容模型，比如deepseek，直接这里改模型名即可，其他都不用动
        )
        
        print(chat_completion)
        print(f"\n回复内容: {chat_completion.choices[0].message.content}")
        
    except Exception as e:
        print(f"API调用失败: {e}")
    
    finally:
        await client.close()

if __name__ == '__main__':
    asyncio.run(test_api())
