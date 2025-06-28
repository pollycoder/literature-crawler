import asyncio
import aiohttp
import feedparser
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import urlencode
import re
import json
from ..core.models import Paper, SearchQuery, CrawlResult
from ..core.exceptions import CrawlerException

class ArxivCrawler:
    """ArXiv论文爬虫"""
    
    def __init__(self, max_concurrent: int = 5, delay: float = 1.0):
        self.base_url = "http://export.arxiv.org/api/query"
        self.max_concurrent = max_concurrent
        self.delay = delay  # 请求间隔，避免被限流
        self.session: Optional[aiohttp.ClientSession] = None
        
        # ArXiv分类映射
        self.category_mapping = {
            "cs.AI": "人工智能",
            "cs.CL": "计算语言学",
            "cs.CV": "计算机视觉",
            "cs.LG": "机器学习",
            "cs.RO": "机器人学",
            "math.OC": "优化与控制",
            "stat.ML": "统计机器学习",
            "physics.data-an": "数据分析",
            "q-bio.QM": "定量方法"
        }
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=self.max_concurrent)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    def build_query_string(self, 
                          keywords: List[str],
                          categories: Optional[List[str]] = None,
                          logic: str = "AND") -> str:
        """构建ArXiv查询字符串"""
        # 构建关键词查询
        if logic.upper() == "AND":
            keyword_query = " AND ".join([f'all:"{kw}"' for kw in keywords])
        else:
            keyword_query = " OR ".join([f'all:"{kw}"' for kw in keywords])
        
        # 添加分类过滤
        if categories:
            category_query = " OR ".join([f'cat:{cat}' for cat in categories])
            keyword_query = f"({keyword_query}) AND ({category_query})"
        
        return keyword_query
    
    def build_query_params(self, 
                          query: SearchQuery,
                          start: int = 0) -> Dict[str, Any]:
        """构建查询参数"""
        params = {
            'search_query': query.query_string,
            'start': start,
            'max_results': min(query.max_results, 100),  # ArXiv限制单次最多100个结果
            'sortBy': 'relevance' if query.sort_by == 'relevance' else 'lastUpdatedDate',
            'sortOrder': 'descending'
        }
        
        return params
    
    async def fetch_papers(self, query: SearchQuery) -> CrawlResult:
        """获取论文列表"""
        if not self.session:
            raise CrawlerException("Session not initialized. Use async context manager.")
        
        all_papers = []
        errors = []
        start = 0
        total_found = 0
        
        try:
            while len(all_papers) < query.max_results:
                # 构建请求参数
                params = self.build_query_params(query, start)
                
                # 发送请求
                async with self.session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        error_msg = f"HTTP {response.status}: {await response.text()}"
                        errors.append(error_msg)
                        break
                    
                    xml_content = await response.text()
                
                # 解析XML响应
                papers_batch, batch_total = self.parse_arxiv_response(xml_content)
                
                if not papers_batch:
                    break
                
                # 设置总数（第一次请求时）
                if total_found == 0:
                    total_found = batch_total
                
                # 过滤日期范围
                if query.date_range:
                    papers_batch = self.filter_by_date(papers_batch, query.date_range)
                
                all_papers.extend(papers_batch)
                start += len(papers_batch)
                
                # 如果获取的论文数少于请求数，说明已经到底了
                if len(papers_batch) < params['max_results']:
                    break
                
                # 添加延迟，避免被限流
                await asyncio.sleep(self.delay)
        
        except Exception as e:
            errors.append(f"Crawling error: {str(e)}")
        
        # 限制返回数量
        all_papers = all_papers[:query.max_results]
        
        success_rate = len(all_papers) / query.max_results if query.max_results > 0 else 0
        
        return CrawlResult(
            query=query,
            papers=all_papers,
            total_found=total_found,
            crawled_count=len(all_papers),
            success_rate=success_rate,
            errors=errors
        )
    
    def parse_arxiv_response(self, xml_content: str) -> tuple[List[Paper], int]:
        """解析ArXiv API响应"""
        try:
            # 使用feedparser解析Atom feed
            feed = feedparser.parse(xml_content)
            
            papers = []
            total_results = int(feed.feed.get('opensearch_totalresults', 0))
            
            for entry in feed.entries:
                paper = self.parse_paper_entry(entry)
                if paper:
                    papers.append(paper)
            
            return papers, total_results
            
        except Exception as e:
            raise CrawlerException(f"Failed to parse ArXiv response: {str(e)}")
    
    def parse_paper_entry(self, entry) -> Optional[Paper]:
        """解析单个论文条目"""
        try:
            # 提取基本信息
            title = entry.title.replace('\n', ' ').strip()
            abstract = entry.summary.replace('\n', ' ').strip()
            
            # 提取作者
            authors = []
            if hasattr(entry, 'authors'):
                authors = [author.name for author in entry.authors]
            elif hasattr(entry, 'author'):
                authors = [entry.author]
            
            # 提取ArXiv ID
            arxiv_id = entry.id.split('/')[-1]
            if 'v' in arxiv_id:
                arxiv_id = arxiv_id.split('v')[0]  # 移除版本号
            
            # 构建URL
            url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # 提取发布日期
            published_date = entry.published if hasattr(entry, 'published') else ""
            
            # 提取分类
            categories = []
            if hasattr(entry, 'arxiv_primary_category'):
                categories.append(entry.arxiv_primary_category.get('term', ''))
            if hasattr(entry, 'tags'):
                categories.extend([tag.term for tag in entry.tags if tag.term not in categories])
            
            return Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                arxiv_id=arxiv_id,
                url=url,
                pdf_url=pdf_url,
                published_date=published_date,
                categories=categories
            )
            
        except Exception as e:
            # 记录解析错误但不中断整个过程
            print(f"Error parsing paper entry: {str(e)}")
            return None
    
    def filter_by_date(self, papers: List[Paper], date_range: Dict[str, str]) -> List[Paper]:
        """按日期范围过滤论文"""
        filtered_papers = []
        
        start_date = datetime.fromisoformat(date_range.get('start', '2000-01-01'))
        end_date = datetime.fromisoformat(date_range.get('end', '2030-12-31'))
        
        for paper in papers:
            try:
                # 解析论文发布日期
                paper_date = datetime.fromisoformat(paper.published_date[:10])
                if start_date <= paper_date <= end_date:
                    filtered_papers.append(paper)
            except:
                # 如果日期解析失败，保留论文
                filtered_papers.append(paper)
        
        return filtered_papers
    
    def generate_search_queries(self, keywords: List[str], max_queries: int = 5) -> List[str]:
        """生成多个搜索查询组合"""
        queries = []
        
        # 单个关键词查询
        for keyword in keywords[:3]:  # 限制前3个最重要的关键词
            queries.append(f'all:"{keyword}"')
        
        # 组合查询
        if len(keywords) >= 2:
            # 两两组合
            for i in range(min(3, len(keywords))):
                for j in range(i+1, min(3, len(keywords))):
                    queries.append(f'all:"{keywords[i]}" AND all:"{keywords[j]}"')
        
        # 全部关键词的OR查询
        if len(keywords) > 1:
            or_query = " OR ".join([f'all:"{kw}"' for kw in keywords[:5]])
            queries.append(f"({or_query})")
        
        return queries[:max_queries]
    
    def remove_duplicates(self, papers: List[Paper]) -> List[Paper]:
        """去除重复论文"""
        seen_ids = set()
        unique_papers = []
        
        for paper in papers:
            if paper.arxiv_id not in seen_ids:
                seen_ids.add(paper.arxiv_id)
                unique_papers.append(paper)
        
        return unique_papers
    
    async def search_multiple_queries(self, 
                                    queries: List[str],
                                    max_results_per_query: int = 50) -> List[Paper]:
        """使用多个查询搜索论文"""
        all_papers = []
        
        for query_string in queries:
            search_query = SearchQuery(
                query_string=query_string,
                max_results=max_results_per_query
            )
            
            result = await self.fetch_papers(search_query)
            all_papers.extend(result.papers)
            
            # 添加延迟
            await asyncio.sleep(self.delay)
        
        # 去重并按相关性排序
        unique_papers = self.remove_duplicates(all_papers)
        return unique_papers