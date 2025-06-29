#!/usr/bin/env python3
"""
测试self-supervised的不同写法
"""

import sys
import os
import asyncio

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.crawler.arxiv_crawler import ArxivCrawler
from src.core.models import SearchQuery


async def test_self_supervised():
    """测试self-supervised的不同写法"""
    
    test_queries = [
        'all:"self-supervised learning"',
        'all:"self supervised learning"',  # 没有连字符
        'all:"self-supervised"',
        'all:"self supervised"',
        'all:"unsupervised learning"',
        'all:"representation learning"'
    ]
    
    async with ArxivCrawler() as crawler:
        for query_string in test_queries:
            print(f"\n查询: {query_string}")
            
            query = SearchQuery(
                query_string=query_string,
                max_results=3
            )
            
            result = await crawler.fetch_papers(query)
            print(f"  结果: {result.crawled_count} 篇论文")
            print(f"  总计找到: {result.total_found}")
            
            await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(test_self_supervised())