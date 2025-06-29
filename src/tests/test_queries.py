#!/usr/bin/env python3
"""
测试具体的搜索查询
"""

import sys
import os
import asyncio

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.crawler.arxiv_crawler import ArxivCrawler
from src.core.models import SearchQuery


async def test_specific_queries():
    """测试具体的查询语句"""
    
    # 从需求分析生成的查询
    problematic_queries = [
        'all:"deep learning" AND cat:cs.LG',
        'all:"deep learning" AND abs:"survey" AND cat:cs.LG'
    ]
    
    # 简单有效的查询
    simple_queries = [
        'all:"deep learning"',
        'cat:cs.LG',
        'all:"neural network"',
        'all:"machine learning"'
    ]
    
    async with ArxivCrawler() as crawler:
        print("🔍 测试问题查询...")
        for query_string in problematic_queries:
            print(f"\n查询: {query_string}")
            
            query = SearchQuery(
                query_string=query_string,
                max_results=5
            )
            
            try:
                result = await crawler.fetch_papers(query)
                print(f"  结果: {result.crawled_count} 篇论文")
                print(f"  总计找到: {result.total_found}")
                if result.errors:
                    print(f"  错误: {result.errors}")
                    
            except Exception as e:
                print(f"  ❌ 失败: {e}")
        
        print("\n✅ 测试简单查询...")
        for query_string in simple_queries:
            print(f"\n查询: {query_string}")
            
            query = SearchQuery(
                query_string=query_string,
                max_results=5
            )
            
            try:
                result = await crawler.fetch_papers(query)
                print(f"  结果: {result.crawled_count} 篇论文")
                print(f"  总计找到: {result.total_found}")
                if result.errors:
                    print(f"  错误: {result.errors}")
                    
            except Exception as e:
                print(f"  ❌ 失败: {e}")


if __name__ == "__main__":
    asyncio.run(test_specific_queries())