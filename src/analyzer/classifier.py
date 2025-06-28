import json
import asyncio
from typing import List, Dict, Any, Optional
from collections import defaultdict
from ..core.models import Paper, PaperClassification, TaskType, ResearchType, NoveltyLevel
from ..llm.model_router import ModelRouter
from ..prompts.manager import PromptManager
from ..core.exceptions import APIException

class PaperClassifier:
    """论文分类器"""
    
    def __init__(self, model_router: ModelRouter, prompt_manager: PromptManager):
        self.model_router = model_router
        self.prompt_manager = prompt_manager
        
        # 预定义研究领域映射
        self.domain_keywords = {
            "机器学习": ["machine learning", "deep learning", "neural network", "ML", "DL"],
            "计算机视觉": ["computer vision", "image", "visual", "object detection", "CNN"],
            "自然语言处理": ["natural language", "NLP", "text", "language model", "transformer"],
            "机器人学": ["robotics", "robot", "autonomous", "control", "navigation"],
            "优化算法": ["optimization", "algorithm", "genetic", "evolutionary", "gradient"],
            "强化学习": ["reinforcement learning", "RL", "policy", "reward", "Q-learning"],
            "生物信息学": ["bioinformatics", "genomics", "proteomics", "biomedical"],
            "量子计算": ["quantum", "qubit", "quantum computing", "quantum algorithm"]
        }
    
    async def classify_papers(self, 
                            papers: List[Paper],
                            model_preference: str = None,
                            batch_size: int = 5) -> List[PaperClassification]:
        """批量分类论文"""
        classifications = []
        
        # 分批处理，避免单次请求过大
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            batch_results = await self._classify_batch(batch, model_preference)
            classifications.extend(batch_results)
            
            # 添加延迟避免API限流
            if i + batch_size < len(papers):
                await asyncio.sleep(1)
        
        return classifications
    
    async def _classify_batch(self, 
                            papers: List[Paper],
                            model_preference: str = None) -> List[PaperClassification]:
        """分类一批论文"""
        tasks = []
        
        for paper in papers:
            task = self._classify_single_paper(paper, model_preference)
            tasks.append(task)
        
        # 并发处理
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        classifications = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 如果分类失败，创建基础分类
                classification = self._create_fallback_classification(papers[i])
            else:
                classification = result
            
            classifications.append(classification)
        
        return classifications
    
    async def _classify_single_paper(self, 
                                   paper: Paper,
                                   model_preference: str = None) -> PaperClassification:
        """分类单篇论文"""
        try:
            # 准备提示词变量
            variables = {
                "title": paper.title,
                "authors": ", ".join(paper.authors),
                "abstract": paper.abstract,
                "categories": ", ".join(paper.categories) if paper.categories else ""
            }
            
            # 获取格式化的提示词
            prompt = self.prompt_manager.get_prompt(
                TaskType.PAPER_CLASSIFICATION,
                variables,
                model_name=model_preference
            )
            
            # 调用LLM分类
            response = await self.model_router.generate(
                prompt=prompt,
                model_ref=model_preference,
                task_type=TaskType.PAPER_CLASSIFICATION,
                temperature=0.2  # 低温度保证分类一致性
            )
            
            # 解析分类结果
            classification_data = self.parse_classification_result(response)
            
            return PaperClassification(
                paper_id=paper.arxiv_id,
                primary_domain=classification_data.get("primary_domain", "未知领域"),
                sub_domains=classification_data.get("sub_domains", []),
                research_type=self._parse_research_type(classification_data.get("research_type", "application")),
                technical_approaches=classification_data.get("technical_approaches", []),
                relevance_score=float(classification_data.get("relevance_score", 5.0)),
                keywords_extracted=classification_data.get("keywords_extracted", []),
                application_areas=classification_data.get("application_areas", []),
                novelty_level=self._parse_novelty_level(classification_data.get("novelty_level", "medium")),
                citation_potential=classification_data.get("citation_potential", "medium"),
                confidence_score=0.8  # 基于LLM的置信度
            )
            
        except Exception as e:
            # 创建后备分类
            return self._create_fallback_classification(paper)
    
    def parse_classification_result(self, response: str) -> Dict[str, Any]:
        """解析分类结果"""
        try:
            # 提取JSON内容
            response = response.strip()
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            
            return result
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in classification response: {str(e)}")
    
    def _parse_research_type(self, type_str: str) -> ResearchType:
        """解析研究类型"""
        type_str = type_str.lower()
        if "theory" in type_str:
            return ResearchType.THEORY
        elif "survey" in type_str or "review" in type_str:
            return ResearchType.SURVEY
        elif "experiment" in type_str:
            return ResearchType.EXPERIMENTAL
        else:
            return ResearchType.APPLICATION
    
    def _parse_novelty_level(self, level_str: str) -> NoveltyLevel:
        """解析新颖度级别"""
        level_str = level_str.lower()
        if "high" in level_str:
            return NoveltyLevel.HIGH
        elif "low" in level_str:
            return NoveltyLevel.LOW
        else:
            return NoveltyLevel.MEDIUM
    
    def _create_fallback_classification(self, paper: Paper) -> PaperClassification:
        """创建后备分类（当LLM分类失败时）"""
        # 基于ArXiv分类和关键词的简单分类
        primary_domain = self._infer_domain_from_categories(paper.categories)
        if not primary_domain:
            primary_domain = self._infer_domain_from_text(paper.title + " " + paper.abstract)
        
        keywords = self._extract_simple_keywords(paper.title + " " + paper.abstract)
        
        return PaperClassification(
            paper_id=paper.arxiv_id,
            primary_domain=primary_domain,
            sub_domains=[],
            research_type=ResearchType.APPLICATION,
            technical_approaches=[],
            relevance_score=5.0,
            keywords_extracted=keywords,
            application_areas=[],
            novelty_level=NoveltyLevel.MEDIUM,
            citation_potential="medium",
            confidence_score=0.3  # 低置信度
        )
    
    def _infer_domain_from_categories(self, categories: List[str]) -> str:
        """从ArXiv分类推断研究领域"""
        category_mapping = {
            "cs.AI": "人工智能",
            "cs.LG": "机器学习", 
            "cs.CV": "计算机视觉",
            "cs.CL": "自然语言处理",
            "cs.RO": "机器人学",
            "stat.ML": "统计机器学习",
            "math.OC": "优化与控制",
            "q-bio": "生物信息学",
            "quant-ph": "量子计算"
        }
        
        for category in categories:
            for key, domain in category_mapping.items():
                if category.startswith(key):
                    return domain
        
        return "计算机科学"
    
    def _infer_domain_from_text(self, text: str) -> str:
        """从文本推断研究领域"""
        text_lower = text.lower()
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return domain
        
        return "其他"
    
    def _extract_simple_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """提取简单关键词"""
        import re
        
        # 简单的关键词提取（基于常见技术术语）
        technical_terms = [
            "neural network", "deep learning", "machine learning", "algorithm",
            "optimization", "classification", "regression", "clustering",
            "transformer", "attention", "convolution", "LSTM", "GAN",
            "reinforcement learning", "computer vision", "NLP"
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for term in technical_terms:
            if term in text_lower and term not in found_keywords:
                found_keywords.append(term)
                if len(found_keywords) >= max_keywords:
                    break
        
        return found_keywords
    
    def group_papers_by_domain(self, 
                              papers: List[Paper],
                              classifications: List[PaperClassification]) -> Dict[str, List[Paper]]:
        """按领域分组论文"""
        domain_groups = defaultdict(list)
        
        paper_dict = {paper.arxiv_id: paper for paper in papers}
        
        for classification in classifications:
            paper = paper_dict.get(classification.paper_id)
            if paper:
                domain_groups[classification.primary_domain].append(paper)
        
        return dict(domain_groups)
    
    def get_domain_statistics(self, classifications: List[PaperClassification]) -> Dict[str, Any]:
        """获取领域统计信息"""
        stats = {
            "total_papers": len(classifications),
            "domains": defaultdict(int),
            "research_types": defaultdict(int),
            "novelty_levels": defaultdict(int),
            "avg_relevance_score": 0,
            "high_potential_papers": 0
        }
        
        total_relevance = 0
        
        for classification in classifications:
            stats["domains"][classification.primary_domain] += 1
            stats["research_types"][classification.research_type.value] += 1
            stats["novelty_levels"][classification.novelty_level.value] += 1
            
            total_relevance += classification.relevance_score
            
            if classification.citation_potential == "high":
                stats["high_potential_papers"] += 1
        
        if classifications:
            stats["avg_relevance_score"] = total_relevance / len(classifications)
        
        # 转换为普通dict
        stats["domains"] = dict(stats["domains"])
        stats["research_types"] = dict(stats["research_types"])
        stats["novelty_levels"] = dict(stats["novelty_levels"])
        
        return stats