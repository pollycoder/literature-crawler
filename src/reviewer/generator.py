import asyncio
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
from ..core.models import Paper, PaperClassification, LiteratureReview, TaskType, ResearchRequirement
from ..core.llm_service import LLMService
from ..core.exceptions import APIException

class ReviewGenerator:
    """文献综述生成器"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def generate_literature_review(self,
                                       papers: List[Paper],
                                       classifications: List[PaperClassification],
                                       requirement: ResearchRequirement,
                                       model_preference: str = None) -> LiteratureReview:
        """生成文献综述"""
        try:
            # 准备综述数据
            review_data = self._prepare_review_data(papers, classifications, requirement)
            
            # 生成综述内容
            content = await self._generate_review_content(review_data, model_preference)
            
            # 提取关键信息
            key_findings = self._extract_key_findings(papers, classifications)
            research_gaps = self._identify_research_gaps(papers, classifications)
            future_directions = self._suggest_future_directions(papers, classifications)
            
            # 生成标题
            title = self._generate_title(requirement)
            
            return LiteratureReview(
                title=title,
                content=content,
                papers_analyzed=[paper.arxiv_id for paper in papers],
                domains_covered=list(set(cls.primary_domain for cls in classifications)),
                key_findings=key_findings,
                research_gaps=research_gaps,
                future_directions=future_directions,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            raise APIException(f"Failed to generate literature review: {str(e)}")
    
    def _prepare_review_data(self, 
                           papers: List[Paper],
                           classifications: List[PaperClassification],
                           requirement: ResearchRequirement) -> Dict[str, Any]:
        """准备综述生成所需的数据"""
        # 创建论文-分类映射
        classification_map = {cls.paper_id: cls for cls in classifications}
        
        # 组织论文数据
        organized_papers = []
        for paper in papers:
            classification = classification_map.get(paper.arxiv_id)
            if classification:
                paper_data = {
                    "title": paper.title,
                    "authors": ", ".join(paper.authors),
                    "abstract": paper.abstract[:500] + "..." if len(paper.abstract) > 500 else paper.abstract,
                    "primary_domain": classification.primary_domain,
                    "technical_approaches": ", ".join(classification.technical_approaches),
                    "relevance_score": classification.relevance_score,
                    "novelty_level": classification.novelty_level.value
                }
                organized_papers.append(paper_data)
        
        # 按相关性排序
        organized_papers.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # 确定研究主题
        research_topic = " + ".join(requirement.research_domains[:3])
        if not research_topic:
            research_topic = "相关领域研究"
        
        return {
            "research_topic": research_topic,
            "research_domains": requirement.research_domains,
            "papers": organized_papers[:20],  # 限制论文数量避免提示词过长
            "total_papers": len(papers),
            "time_range": requirement.time_preference
        }
    
    async def _generate_review_content(self, 
                                     review_data: Dict[str, Any],
                                     model_preference: str = None) -> str:
        """生成综述内容"""
        try:
            # 调用LLM生成综述
            content = await self.llm_service.call_with_template(
                task_type=TaskType.REVIEW_GENERATION,
                variables=review_data,
                model_preference=model_preference,
                temperature=0.4,  # 中等创造性
                max_tokens=8000   # 支持长文本生成
            )
            
            return content
            
        except Exception as e:
            raise APIException(f"Failed to generate review content: {str(e)}")
    
    def _extract_key_findings(self, 
                            papers: List[Paper], 
                            classifications: List[PaperClassification]) -> List[str]:
        """提取关键发现"""
        findings = []
        
        # 按领域分组
        domain_groups = defaultdict(list)
        classification_map = {cls.paper_id: cls for cls in classifications}
        
        for paper in papers:
            classification = classification_map.get(paper.arxiv_id)
            if classification:
                domain_groups[classification.primary_domain].append((paper, classification))
        
        # 每个领域提取主要发现
        for domain, paper_cls_pairs in domain_groups.items():
            # 找到该领域最相关的论文
            paper_cls_pairs.sort(key=lambda x: x[1].relevance_score, reverse=True)
            top_papers = paper_cls_pairs[:3]
            
            domain_findings = []
            for paper, classification in top_papers:
                if classification.novelty_level.value == "high":
                    finding = f"{domain}领域：{paper.title[:50]}...提出了创新方法"
                    domain_findings.append(finding)
            
            if domain_findings:
                findings.extend(domain_findings[:2])  # 每个领域最多2个发现
        
        return findings[:10]  # 总共最多10个关键发现
    
    def _identify_research_gaps(self,
                              papers: List[Paper],
                              classifications: List[PaperClassification]) -> List[str]:
        """识别研究空白"""
        gaps = []
        
        # 分析技术方法的覆盖情况
        all_approaches = []
        for cls in classifications:
            all_approaches.extend(cls.technical_approaches)
        
        approach_counts = defaultdict(int)
        for approach in all_approaches:
            approach_counts[approach] += 1
        
        # 识别研究不足的领域
        total_papers = len(papers)
        
        # 基于论文数量识别空白
        if total_papers < 10:
            gaps.append("该研究领域论文数量相对较少，存在进一步研究的空间")
        
        # 基于研究类型分布识别空白
        type_counts = defaultdict(int)
        for cls in classifications:
            type_counts[cls.research_type.value] += 1
        
        if type_counts.get("experimental", 0) < total_papers * 0.3:
            gaps.append("实验验证研究相对不足，需要更多实证研究")
        
        if type_counts.get("survey", 0) < 2:
            gaps.append("缺乏全面的综述性研究")
        
        # 基于应用领域识别空白
        application_areas = []
        for cls in classifications:
            application_areas.extend(cls.application_areas)
        
        if len(set(application_areas)) < 3:
            gaps.append("应用领域相对集中，跨领域应用研究有待拓展")
        
        return gaps[:5]  # 最多5个研究空白
    
    def _suggest_future_directions(self,
                                 papers: List[Paper],
                                 classifications: List[PaperClassification]) -> List[str]:
        """建议未来研究方向"""
        directions = []
        
        # 基于新颖度高的论文建议方向
        high_novelty_papers = [
            cls for cls in classifications 
            if cls.novelty_level.value == "high" and cls.relevance_score > 7.0
        ]
        
        if high_novelty_papers:
            directions.append("继续深入探索高新颖度技术方法的潜力")
        
        # 基于引用潜力建议方向
        high_potential_papers = [
            cls for cls in classifications 
            if cls.citation_potential == "high"
        ]
        
        if len(high_potential_papers) > len(classifications) * 0.3:
            directions.append("重点关注具有高引用潜力的研究方向")
        
        # 基于技术方法建议方向
        all_approaches = []
        for cls in classifications:
            all_approaches.extend(cls.technical_approaches)
        
        approach_counts = defaultdict(int)
        for approach in all_approaches:
            approach_counts[approach] += 1
        
        # 找出热门技术方法
        popular_approaches = [
            approach for approach, count in approach_counts.items() 
            if count >= 2
        ]
        
        if popular_approaches:
            directions.append(f"深入研究{', '.join(popular_approaches[:3])}等热门技术的改进与应用")
        
        # 通用建议
        directions.extend([
            "加强跨学科合作，探索多领域融合的可能性",
            "注重实际应用场景的验证和优化",
            "关注技术方法的可解释性和鲁棒性研究"
        ])
        
        return directions[:8]  # 最多8个未来方向
    
    def _generate_title(self, requirement: ResearchRequirement) -> str:
        """生成综述标题"""
        if requirement.research_domains:
            main_domain = requirement.research_domains[0]
            if len(requirement.research_domains) > 1:
                title = f"{main_domain}等领域研究综述"
            else:
                title = f"{main_domain}研究综述"
        else:
            title = "相关领域研究综述"
        
        return title
    
    async def generate_domain_summary(self,
                                    domain: str,
                                    papers: List[Paper],
                                    classifications: List[PaperClassification],
                                    model_preference: str = None) -> str:
        """生成单个领域的总结"""
        # 过滤该领域的论文
        domain_papers = []
        domain_classifications = []
        
        classification_map = {cls.paper_id: cls for cls in classifications}
        
        for paper in papers:
            classification = classification_map.get(paper.arxiv_id)
            if classification and classification.primary_domain == domain:
                domain_papers.append(paper)
                domain_classifications.append(classification)
        
        if not domain_papers:
            return f"在{domain}领域未找到相关论文。"
        
        try:
            # 准备领域总结变量
            variables = self.llm_service.create_standard_variables(
                domain=domain,
                paper_titles=[paper.title for paper in domain_papers[:10]],
                paper_count=len(domain_papers)
            )
            
            # 使用LLM服务生成总结
            summary = await self.llm_service.call_with_template(
                task_type=TaskType.REVIEW_GENERATION,
                variables=variables,
                model_preference=model_preference,
                temperature=0.3,
                max_tokens=1000
            )
            return summary
        except Exception:
            return f"{domain}领域包含{len(domain_papers)}篇相关论文，涵盖了该领域的主要研究方向。"
    
    def export_to_markdown(self, review: LiteratureReview) -> str:
        """导出为Markdown格式"""
        markdown_content = f"""# {review.title}

**生成时间**: {review.generated_at.strftime("%Y-%m-%d %H:%M:%S")}
**分析论文数量**: {len(review.papers_analyzed)}
**涵盖领域**: {', '.join(review.domains_covered)}
**字数**: {review.word_count}

---

{review.content}

---

## 参考文献

本综述分析了以下 {len(review.papers_analyzed)} 篇论文：

{chr(10).join([f"{i+1}. ArXiv ID: {paper_id}" for i, paper_id in enumerate(review.papers_analyzed)])}

---

*本综述由AI文献爬取助手自动生成*
"""
        return markdown_content