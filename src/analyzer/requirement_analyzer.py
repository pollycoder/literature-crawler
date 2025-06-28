import json
import asyncio
from typing import Dict, Any, List
from ..core.models import ResearchRequirement, TaskType
from ..llm.model_router import ModelRouter
from ..prompts.manager import PromptManager
from ..core.exceptions import APIException

class RequirementAnalyzer:
    """用户需求分析器"""
    
    def __init__(self, model_router: ModelRouter, prompt_manager: PromptManager):
        self.model_router = model_router
        self.prompt_manager = prompt_manager
    
    async def analyze_requirement(self, 
                                user_input: str,
                                model_preference: str = None) -> ResearchRequirement:
        """分析用户研究需求"""
        try:
            # 准备提示词变量
            variables = {
                "user_input": user_input
            }
            
            # 获取格式化的提示词
            prompt = self.prompt_manager.get_prompt(
                TaskType.REQUIREMENT_ANALYSIS,
                variables,
                model_name=model_preference
            )
            
            # 调用LLM分析
            response = await self.model_router.generate(
                prompt=prompt,
                model_ref=model_preference,
                task_type=TaskType.REQUIREMENT_ANALYSIS,
                temperature=0.3  # 较低温度保证结果稳定性
            )
            
            # 解析JSON响应
            analysis_result = self.parse_analysis_result(response)
            
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
    
    def parse_analysis_result(self, response: str) -> Dict[str, Any]:
        """解析LLM分析结果"""
        try:
            # 尝试提取JSON内容
            response = response.strip()
            
            # 找到JSON部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            
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
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to parse analysis result: {str(e)}")
    
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
        refine_prompt = f"""
        原始需求分析结果：
        研究领域：{original_requirement.research_domains}
        具体主题：{original_requirement.specific_topics}
        关键词：{original_requirement.keywords}
        
        用户反馈：{feedback}
        
        请根据用户反馈调整需求分析结果，以JSON格式返回更新后的分析。
        """
        
        try:
            response = await self.model_router.generate(
                prompt=refine_prompt,
                model_ref=model_preference,
                task_type=TaskType.REQUIREMENT_ANALYSIS,
                temperature=0.3
            )
            
            refined_result = self.parse_analysis_result(response)
            
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
            
        except Exception as e:
            # 如果细化失败，返回原始需求
            return original_requirement