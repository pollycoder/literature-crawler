"""
统一的LLM调用服务
用于消除分析器中重复的LLM调用模式
"""

import json
import asyncio
from typing import Dict, Any, Optional, List, Type, TypeVar, Callable
from enum import Enum

from .models import TaskType
from .exceptions import APIException
from ..llm.model_router import ModelRouter
from ..prompts.manager import PromptManager

T = TypeVar('T')

class ResponseParser:
    """统一的响应解析器"""
    
    @staticmethod
    def parse_json_response(response: str) -> Dict[str, Any]:
        """解析JSON响应的通用方法"""
        try:
            response = response.strip()
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            
            return result
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {str(e)}")
    
    @staticmethod
    def parse_enum_value(value: str, enum_class: Type[Enum], default: Any = None) -> Any:
        """解析枚举值的通用方法"""
        if not value:
            return default
        
        value_lower = value.lower()
        for enum_item in enum_class:
            if enum_item.value.lower() == value_lower or enum_item.name.lower() == value_lower:
                return enum_item
        
        return default

class LLMService:
    """统一的LLM调用服务"""
    
    def __init__(self, model_router: ModelRouter, prompt_manager: PromptManager):
        self.model_router = model_router
        self.prompt_manager = prompt_manager
        self.parser = ResponseParser()
    
    async def call_with_template(self,
                               task_type: TaskType,
                               variables: Dict[str, Any],
                               model_preference: Optional[str] = None,
                               temperature: float = 0.2,
                               max_retries: int = 3,
                               **kwargs) -> str:
        """
        使用模板调用LLM的统一方法
        
        Args:
            task_type: 任务类型
            variables: 模板变量
            model_preference: 模型偏好
            temperature: 温度参数
            max_retries: 最大重试次数
            **kwargs: 其他参数
        
        Returns:
            LLM的响应文本
        """
        # 获取格式化的提示词
        prompt = self.prompt_manager.get_prompt(
            task_type,
            variables,
            model_name=model_preference
        )
        
        # 调用LLM
        for attempt in range(max_retries):
            try:
                response = await self.model_router.generate(
                    prompt=prompt,
                    model_ref=model_preference,
                    task_type=task_type,
                    temperature=temperature,
                    **kwargs
                )
                return response
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise APIException(f"LLM call failed after {max_retries} attempts: {str(e)}")
                
                # 指数退避
                await asyncio.sleep(2 ** attempt)
        
        # 这行不应该被执行到，但为了类型检查
        raise APIException("Unexpected error in LLM call")
    
    async def call_with_json_response(self,
                                    task_type: TaskType,
                                    variables: Dict[str, Any],
                                    model_preference: Optional[str] = None,
                                    temperature: float = 0.2,
                                    **kwargs) -> Dict[str, Any]:
        """
        调用LLM并解析JSON响应
        
        Returns:
            解析后的JSON数据
        """
        response = await self.call_with_template(
            task_type=task_type,
            variables=variables,
            model_preference=model_preference,
            temperature=temperature,
            **kwargs
        )
        
        return self.parser.parse_json_response(response)
    
    async def batch_call_with_template(self,
                                     task_type: TaskType,
                                     variables_list: List[Dict[str, Any]],
                                     model_preference: Optional[str] = None,
                                     temperature: float = 0.2,
                                     batch_size: int = 5,
                                     delay_between_batches: float = 1.0,
                                     **kwargs) -> List[str]:
        """
        批量调用LLM的统一方法
        
        Args:
            task_type: 任务类型
            variables_list: 模板变量列表
            model_preference: 模型偏好
            temperature: 温度参数
            batch_size: 批次大小
            delay_between_batches: 批次间延迟
            **kwargs: 其他参数
        
        Returns:
            LLM响应列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(variables_list), batch_size):
            batch = variables_list[i:i + batch_size]
            batch_results = await self._process_batch(
                task_type=task_type,
                variables_batch=batch,
                model_preference=model_preference,
                temperature=temperature,
                **kwargs
            )
            results.extend(batch_results)
            
            # 添加延迟避免API限流
            if i + batch_size < len(variables_list):
                await asyncio.sleep(delay_between_batches)
        
        return results
    
    async def batch_call_with_json_response(self,
                                          task_type: TaskType,
                                          variables_list: List[Dict[str, Any]],
                                          model_preference: Optional[str] = None,
                                          temperature: float = 0.2,
                                          batch_size: int = 5,
                                          **kwargs) -> List[Dict[str, Any]]:
        """
        批量调用LLM并解析JSON响应
        
        Returns:
            解析后的JSON数据列表
        """
        responses = await self.batch_call_with_template(
            task_type=task_type,
            variables_list=variables_list,
            model_preference=model_preference,
            temperature=temperature,
            batch_size=batch_size,
            **kwargs
        )
        
        results = []
        for response in responses:
            try:
                json_data = self.parser.parse_json_response(response)
                results.append(json_data)
            except ValueError as e:
                # 如果解析失败，添加空字典作为占位符
                results.append({"error": str(e)})
        
        return results
    
    async def _process_batch(self,
                           task_type: TaskType,
                           variables_batch: List[Dict[str, Any]],
                           model_preference: Optional[str] = None,
                           temperature: float = 0.2,
                           **kwargs) -> List[str]:
        """处理一批请求"""
        tasks = []
        
        for variables in variables_batch:
            task = self.call_with_template(
                task_type=task_type,
                variables=variables,
                model_preference=model_preference,
                temperature=temperature,
                **kwargs
            )
            tasks.append(task)
        
        # 并发处理
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                # 如果请求失败，返回错误信息
                processed_results.append(f"Error: {str(result)}")
            else:
                processed_results.append(result)
        
        return processed_results
    
    def create_standard_variables(self, **kwargs) -> Dict[str, Any]:
        """创建标准的模板变量字典"""
        variables = {}
        for key, value in kwargs.items():
            if value is not None:
                if isinstance(value, list):
                    variables[key] = ", ".join(str(v) for v in value)
                else:
                    variables[key] = str(value)
            else:
                variables[key] = ""
        
        return variables
    
    async def call_with_fallback(self,
                               task_type: TaskType,
                               variables: Dict[str, Any],
                               fallback_handler: Callable[[], T],
                               model_preference: Optional[str] = None,
                               **kwargs) -> T:
        """
        调用LLM，失败时使用后备处理器
        
        Args:
            task_type: 任务类型
            variables: 模板变量
            fallback_handler: 失败时的后备处理函数
            model_preference: 模型偏好
            **kwargs: 其他参数
        
        Returns:
            处理结果或后备结果
        """
        try:
            response = await self.call_with_template(
                task_type=task_type,
                variables=variables,
                model_preference=model_preference,
                **kwargs
            )
            # 类型转换，假设response是T类型或可以转换为T类型
            return response  # type: ignore
            
        except Exception:
            # 如果LLM调用失败，使用后备处理器
            return fallback_handler()
    
    def get_parser(self) -> ResponseParser:
        """获取响应解析器"""
        return self.parser