import asyncio
from typing import Dict, Any, List
from ..core.models import ResearchRequirement, TaskType
from ..core.llm_service import LLMService
from ..core.exceptions import APIException

class RequirementAnalyzer:
    """用户需求分析器"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def analyze_requirement(self, 
                                user_input: str,
                                model_preference: str = None) -> ResearchRequirement:
        """分析用户研究需求"""
        try:
            # 准备提示词变量
            variables = self.llm_service.create_standard_variables(
                user_input=user_input
            )
            
            # 调用LLM分析并解析JSON结果
            analysis_result = await self.llm_service.call_with_json_response(
                task_type=TaskType.REQUIREMENT_ANALYSIS,
                variables=variables,
                model_preference=model_preference,
                temperature=0.3  # 较低温度保证结果稳定性
            )
            
            # 后处理和验证
            analysis_result = self._post_process_analysis_result(analysis_result)
            
            # 构建ResearchRequirement对象
            return ResearchRequirement(
                user_input=user_input,
                research_domains=analysis_result.get("research_domains", []),
                specific_topics=analysis_result.get("specific_topics", []),
                keywords=analysis_result.get("keywords", []),
                time_preference=analysis_result.get("time_preference", {}),
                paper_count_estimate=analysis_result.get("paper_count_estimate", 50),
                research_focus=analysis_result.get("research_focus", []),
                search_queries=analysis_result.get("search_queries", []),
                priority_areas=analysis_result.get("priority_areas", [])
            )
            
        except Exception as e:
            raise APIException(f"Failed to analyze requirement: {str(e)}")
    
    def _post_process_analysis_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """后处理分析结果，设置默认值"""
        # 验证必需字段
        required_fields = [
            "research_domains", "specific_topics", "keywords", 
            "search_queries", "research_focus"
        ]
        
        for field in required_fields:
            if field not in result:
                result[field] = []
        
        # 设置默认值
        if "time_preference" not in result:
            result["time_preference"] = {
                "start_year": 2020,
                "end_year": 2024,
                "priority": "recent"
            }
        
        if "paper_count_estimate" not in result:
            result["paper_count_estimate"] = 50
        
        return result
    
    def validate_analysis_result(self, result: Dict[str, Any]) -> bool:
        """验证分析结果的完整性"""
        required_fields = [
            "research_domains", "specific_topics", "keywords", "search_queries"
        ]
        
        for field in required_fields:
            if field not in result or not result[field]:
                return False
        
        return True
    
    def enhance_search_queries(self, 
                              requirement: ResearchRequirement,
                              max_queries: int = 10) -> List[str]:
        """增强搜索查询"""
        enhanced_queries = []
        
        # 基于原始查询
        enhanced_queries.extend(requirement.search_queries[:5])
        
        # 生成基于关键词的查询
        for keyword in requirement.keywords[:3]:
            enhanced_queries.append(f'all:"{keyword}"')
        
        # 生成组合查询
        if len(requirement.keywords) >= 2:
            for i in range(min(2, len(requirement.keywords))):
                for j in range(i+1, min(3, len(requirement.keywords))):
                    query = f'all:"{requirement.keywords[i]}" AND all:"{requirement.keywords[j]}"'
                    enhanced_queries.append(query)
        
        # 基于研究领域的查询
        for domain in requirement.research_domains[:2]:
            enhanced_queries.append(f'all:"{domain}"')
        
        # 去重并限制数量
        unique_queries = list(dict.fromkeys(enhanced_queries))
        return unique_queries[:max_queries]
    
    async def refine_requirement(self, 
                               original_requirement: ResearchRequirement,
                               feedback: str,
                               model_preference: str = None) -> ResearchRequirement:
        """根据用户反馈细化需求"""
        try:
            # 准备细化变量
            variables = self.llm_service.create_standard_variables(
                research_domains=original_requirement.research_domains,
                specific_topics=original_requirement.specific_topics,
                keywords=original_requirement.keywords,
                user_feedback=feedback
            )
            
            # 调用LLM细化需求
            refined_result = await self.llm_service.call_with_json_response(
                task_type=TaskType.REQUIREMENT_ANALYSIS,
                variables=variables,
                model_preference=model_preference,
                temperature=0.3
            )
            
            # 后处理
            refined_result = self._post_process_analysis_result(refined_result)
            
            return ResearchRequirement(
                user_input=f"{original_requirement.user_input}\n\n用户反馈：{feedback}",
                research_domains=refined_result.get("research_domains", []),
                specific_topics=refined_result.get("specific_topics", []),
                keywords=refined_result.get("keywords", []),
                time_preference=refined_result.get("time_preference", {}),
                paper_count_estimate=refined_result.get("paper_count_estimate", 50),
                research_focus=refined_result.get("research_focus", []),
                search_queries=refined_result.get("search_queries", [])
            )
            
        except Exception:
            # 如果细化失败，返回原始需求
            return original_requirement