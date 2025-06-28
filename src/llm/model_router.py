import yaml
import asyncio
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
from .providers.base_provider import BaseLLMProvider
from .providers.openai_provider import OpenAIProvider
from .providers.claude_provider import ClaudeProvider
from .providers.gemini_provider import GeminiProvider
from .providers.deepseek_provider import DeepSeekProvider
from ..core.models import TaskType
from ..core.config_models import ProviderConfig, ModelInfo, ModelReference, AppConfig
from ..core.exceptions import ModelProviderException, ConfigException
from ..utils.config_utils import expand_env_vars, load_env_file

class ModelRouter:
    """新的模型路由器 - 按提供商管理模型"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.app_config: AppConfig = AppConfig()
        self.provider_classes: Dict[str, Type[BaseLLMProvider]] = {
            "openai": OpenAIProvider,
            "claude": ClaudeProvider,
            "gemini": GeminiProvider,
            "deepseek": DeepSeekProvider,
        }
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """加载配置文件"""
        try:
            # 加载环境变量
            load_env_file()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            # 展开环境变量
            raw_config = expand_env_vars(raw_config)
            
            # 解析配置
            self.app_config = self._parse_config(raw_config)
            
            # 初始化提供商
            self._initialize_providers()
            
        except Exception as e:
            raise ConfigException(f"Failed to load config: {str(e)}")
    
    def _parse_config(self, raw_config: Dict[str, Any]) -> AppConfig:
        """解析配置文件"""
        app_config = AppConfig()
        
        # 解析提供商配置
        providers_config = raw_config.get('providers', {})
        models_config = raw_config.get('models', {})
        
        for provider_name, provider_data in providers_config.items():
            # 创建提供商配置
            provider_config = ProviderConfig(
                provider_type=provider_data.get('provider_type'),
                api_key=provider_data.get('api_key'),
                base_url=provider_data.get('base_url'),
                timeout=provider_data.get('timeout', 30),
                retry_count=provider_data.get('retry_count', 3),
                requests_per_minute=provider_data.get('requests_per_minute', 60),
                max_concurrent_requests=provider_data.get('max_concurrent_requests', 5)
            )
            
            # 添加支持的模型
            provider_models = models_config.get(provider_name, [])
            for model_data in provider_models:
                model_info = ModelInfo(
                    name=model_data.get('name'),
                    display_name=model_data.get('display_name'),
                    default_temperature=model_data.get('default_temperature', 0.7),
                    default_max_tokens=model_data.get('default_max_tokens', 4000),
                    cost_per_1k_input=model_data.get('cost_per_1k_input', 0.0),
                    cost_per_1k_output=model_data.get('cost_per_1k_output', 0.0)
                )
                provider_config.models.append(model_info)
            
            app_config.providers[provider_name] = provider_config
        
        # 解析任务偏好
        app_config.task_preferences = raw_config.get('task_preferences', {})
        
        # 其他配置
        app_config.crawler_config = raw_config.get('crawler', {})
        app_config.output_config = raw_config.get('output', {})
        app_config.logging_config = raw_config.get('logging', {})
        
        return app_config
    
    def _initialize_providers(self):
        """初始化提供商"""
        for provider_name, provider_config in self.app_config.providers.items():
            # 检查API key
            if not provider_config.api_key:
                print(f"跳过提供商 {provider_name}: API key 未配置")
                continue
            
            try:
                # 获取提供商类
                provider_class = self.provider_classes.get(provider_config.provider_type)
                if not provider_class:
                    print(f"警告: 未知的提供商类型 {provider_config.provider_type}")
                    continue
                
                # 创建提供商实例
                provider = provider_class(provider_config)
                self.providers[provider_name] = provider
                
                print(f"提供商 {provider_name} 已加载 ({len(provider_config.models)} 个模型)")
                
            except Exception as e:
                print(f"警告: 初始化提供商 {provider_name} 失败: {str(e)}")
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """获取所有可用的模型"""
        available_models = {}
        
        for provider_name, provider_config in self.app_config.providers.items():
            if provider_name in self.providers:  # 只返回已成功初始化的提供商
                model_names = [model.name for model in provider_config.models]
                available_models[provider_name] = model_names
        
        return available_models
    
    def get_model_info(self, model_ref: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        try:
            ref = ModelReference.parse(model_ref)
            provider_config = self.app_config.providers.get(ref.provider_name)
            
            if not provider_config:
                return None
            
            for model in provider_config.models:
                if model.name == ref.model_name:
                    return model
            
            return None
            
        except ValueError:
            return None
    
    def get_optimal_model(self, 
                         task_type: TaskType, 
                         criteria: str = "balanced",
                         exclude: Optional[List[str]] = None) -> Optional[str]:
        """获取最优模型"""
        exclude = exclude or []
        
        # 获取任务偏好列表
        task_preferences = self.app_config.task_preferences.get(task_type.value, [])
        
        # 按偏好顺序查找可用模型
        for model_ref in task_preferences:
            if model_ref in exclude:
                continue
            
            try:
                ref = ModelReference.parse(model_ref)
                
                # 检查提供商是否可用
                if ref.provider_name not in self.providers:
                    continue
                
                # 检查模型是否存在
                model_info = self.get_model_info(model_ref)
                if not model_info:
                    continue
                
                # 测试连接
                provider = self.providers[ref.provider_name]
                if provider.test_connection():
                    return model_ref
                
            except ValueError:
                continue
        
        # 如果没有找到偏好模型，返回第一个可用的
        for provider_name in self.providers:
            provider_config = self.app_config.providers[provider_name]
            if provider_config.models:
                first_model = provider_config.models[0]
                candidate = f"{provider_name}/{first_model.name}"
                if candidate not in exclude:
                    return candidate
        
        return None
    
    async def generate(self, 
                      prompt: str,
                      model_ref: Optional[str] = None,
                      task_type: TaskType = TaskType.REQUIREMENT_ANALYSIS,
                      **kwargs) -> str:
        """生成文本"""
        # 选择模型
        if not model_ref:
            model_ref = self.get_optimal_model(task_type)
            
        if not model_ref:
            raise ModelProviderException("No available models found")
        
        try:
            ref = ModelReference.parse(model_ref)
        except ValueError as e:
            raise ModelProviderException(f"Invalid model reference: {e}")
        
        # 获取提供商
        provider = self.providers.get(ref.provider_name)
        if not provider:
            raise ModelProviderException(f"Provider {ref.provider_name} not available")
        
        # 获取模型信息
        model_info = self.get_model_info(model_ref)
        if not model_info:
            raise ModelProviderException(f"Model {ref.model_name} not found")
        
        # 设置默认参数
        generate_kwargs = {
            'model_name': ref.model_name,
            'temperature': kwargs.get('temperature', model_info.default_temperature),
            'max_tokens': kwargs.get('max_tokens', model_info.default_max_tokens)
        }
        
        # o1/o3/o4推理模型不支持temperature
        if self._is_reasoning_model(ref.model_name) and 'temperature' in generate_kwargs:
            del generate_kwargs['temperature']
        
        return await provider.generate_with_retry(prompt, **generate_kwargs)
    
    async def generate_batch(self,
                           prompts: List[str],
                           model_ref: Optional[str] = None,
                           task_type: TaskType = TaskType.REQUIREMENT_ANALYSIS,
                           **kwargs) -> List[str]:
        """批量生成文本"""
        if not model_ref:
            model_ref = self.get_optimal_model(task_type)
        
        if not model_ref:
            raise ModelProviderException("No available models found")
        
        ref = ModelReference.parse(model_ref)
        provider = self.providers.get(ref.provider_name)
        if not provider:
            raise ModelProviderException(f"Provider {ref.provider_name} not available")
        
        model_info = self.get_model_info(model_ref)
        generate_kwargs = {
            'model_name': ref.model_name,
            'temperature': kwargs.get('temperature', model_info.default_temperature),
            'max_tokens': kwargs.get('max_tokens', model_info.default_max_tokens)
        }
        
        if self._is_reasoning_model(ref.model_name) and 'temperature' in generate_kwargs:
            del generate_kwargs['temperature']
        
        return await provider.generate_batch(prompts, **generate_kwargs)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有提供商统计"""
        return {name: provider.get_stats() for name, provider in self.providers.items()}
    
    def test_all_connections(self) -> Dict[str, bool]:
        """测试所有连接"""
        results = {}
        
        for provider_name, provider in self.providers.items():
            provider_config = self.app_config.providers[provider_name]
            
            # 为每个模型测试连接
            for model in provider_config.models:
                model_ref = f"{provider_name}/{model.name}"
                try:
                    # 简单的连接测试
                    is_connected = provider.test_connection()
                    results[model_ref] = is_connected
                except Exception:
                    results[model_ref] = False
        
        return results
    
    def _is_reasoning_model(self, model_name: str) -> bool:
        """判断是否为推理模型（不支持temperature参数）"""
        reasoning_prefixes = ('o')
        return model_name.startswith(reasoning_prefixes)