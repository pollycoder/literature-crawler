from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class TaskType(Enum):
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    PAPER_CLASSIFICATION = "paper_classification"
    DOMAIN_EXTRACTION = "domain_extraction"
    REVIEW_GENERATION = "review_generation"
    RELEVANCE_SCORING = "relevance_scoring"
    QUERY_GENERATION = "query_generation"

class ResearchType(Enum):
    THEORY = "theory"
    APPLICATION = "application"
    SURVEY = "survey"
    EXPERIMENTAL = "experimental"

class NoveltyLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class Paper:
    """学术论文数据模型"""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    url: str
    pdf_url: str
    published_date: str
    categories: List[str]
    keywords: List[str] = field(default_factory=list)
    classification: Optional[Dict[str, Any]] = None
    relevance_score: float = 0.0
    
    def __post_init__(self):
        if not self.pdf_url and self.arxiv_id:
            self.pdf_url = f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"

@dataclass
class PaperClassification:
    """论文分类结果"""
    paper_id: str
    primary_domain: str
    sub_domains: List[str]
    research_type: ResearchType
    technical_approaches: List[str]
    relevance_score: float
    keywords_extracted: List[str]
    application_areas: List[str]
    novelty_level: NoveltyLevel
    citation_potential: str
    confidence_score: float = 0.0

@dataclass
class ResearchRequirement:
    """用户研究需求"""
    user_input: str
    research_domains: List[str]
    specific_topics: List[str]
    keywords: List[str]
    time_preference: Dict[str, Any]
    paper_count_estimate: int
    research_focus: List[str]
    search_queries: List[str]
    priority_areas: List[str] = field(default_factory=list)

@dataclass
class LiteratureReview:
    """文献综述"""
    title: str
    content: str
    papers_analyzed: List[str]  # paper IDs
    domains_covered: List[str]
    key_findings: List[str]
    research_gaps: List[str]
    future_directions: List[str]
    generated_at: datetime = field(default_factory=datetime.now)
    word_count: int = 0
    
    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.content.split())


@dataclass
class SearchQuery:
    """搜索查询"""
    query_string: str
    category_filter: Optional[List[str]] = None
    date_range: Optional[Dict[str, str]] = None
    max_results: int = 100
    sort_by: str = "relevance"  # relevance, date, citations
    
@dataclass
class CrawlResult:
    """爬取结果"""
    query: SearchQuery
    papers: List[Paper]
    total_found: int
    crawled_count: int
    success_rate: float
    errors: List[str] = field(default_factory=list)
    crawl_time: datetime = field(default_factory=datetime.now)