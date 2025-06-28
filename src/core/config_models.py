from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class ModelInfo:
    """单个模型信息"""
    name: str
    display_name: str
    default_temperature: float = 0.7
    default_max_tokens: int = 4000
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    supports_temperature: bool = True
    supports_streaming: bool = True

@dataclass
class ProviderConfig:
    """提供商配置"""
    provider_type: str  # openai, claude, etc.
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    requests_per_minute: int = 60
    max_concurrent_requests: int = 5
    
    # 支持的模型列表
    models: List[ModelInfo] = field(default_factory=list)

@dataclass
class ModelReference:
    """模型引用 (provider_name/model_name)"""
    provider_name: str
    model_name: str
    
    @classmethod
    def parse(cls, reference: str) -> 'ModelReference':
        """解析 provider_name/model_name 格式的引用"""
        if '/' not in reference:
            raise ValueError(f"Invalid model reference format: {reference}. Expected: provider_name/model_name")
        
        provider_name, model_name = reference.split('/', 1)
        return cls(provider_name=provider_name, model_name=model_name)
    
    def __str__(self) -> str:
        return f"{self.provider_name}/{self.model_name}"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return str(self) == other
        elif isinstance(other, ModelReference):
            return self.provider_name == other.provider_name and self.model_name == other.model_name
        return False

@dataclass 
class AppConfig:
    """应用配置"""
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    task_preferences: Dict[str, List[str]] = field(default_factory=dict)
    crawler_config: Dict[str, Any] = field(default_factory=dict)
    output_config: Dict[str, Any] = field(default_factory=dict)
    logging_config: Dict[str, Any] = field(default_factory=dict)