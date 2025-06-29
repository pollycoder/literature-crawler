#!/usr/bin/env python3
"""
测试实际生成的查询
"""

import sys
import os
import asyncio

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.crawler.arxiv_crawler import ArxivCrawler
from src.core.models import SearchQuery


async def test_actual_queries():
    """测试实际生成的查询语句"""
    
    # 从papers.json中的实际查询
    actual_queries = [
        '(abs:"deep learning" AND cat:cs.LG)',
        '(ti:"transformer" AND abs:"self-supervised" AND (cat:cs.CL OR cat:cs.CV))'
    ]
    
    # 修复版本
    fixed_queries = [
        'abs:"deep learning" AND cat:cs.LG',
        'ti:"transformer" AND abs:"self-supervised"'
    ]
    
    async with ArxivCrawler() as crawler:
        print("🔍 测试实际生成的查询...")
        for i, query_string in enumerate(actual_queries):
            print(f"\n查询 {i+1}: {query_string}")
            
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
                
            await asyncio.sleep(0.5)
        
        print("\n✅ 测试修复版本...")
        for i, query_string in enumerate(fixed_queries):
            print(f"\n修复查询 {i+1}: {query_string}")
            
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
                
            await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(test_actual_queries())