"""
统一的配置管理器
整合配置加载、验证和管理功能
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path

from .config_models import AppConfig
from ..utils.config_utils import expand_env_vars


class ConfigManager:
    """统一配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为 config/config.yaml
        """
        self.config_path = config_path or self._get_default_config_path()
        self.app_config: Optional[AppConfig] = None
        self._raw_config: Optional[Dict[str, Any]] = None
    
    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        current_dir = Path(__file__).parent.parent.parent
        return str(current_dir / "config" / "config.yaml")
    
    def load_config(self) -> AppConfig:
        """
        加载并解析配置
        
        Returns:
            AppConfig: 解析后的配置对象
        """
        if self.app_config is not None:
            return self.app_config
        
        # 加载原始配置
        self._raw_config = self._load_raw_config()
        
        # 展开环境变量
        expanded_config = expand_env_vars(self._raw_config)
        
        # 验证配置
        validation_errors = self._validate_raw_config(expanded_config)
        if validation_errors:
            raise ValueError(f"配置验证失败: {'; '.join(validation_errors)}")
        
        # 解析为配置对象
        try:
            # 转换为AppConfig期望的格式
            app_config_data = self._convert_to_app_config_format(expanded_config)
            self.app_config = AppConfig(**app_config_data)
            return self.app_config
        except Exception as e:
            raise ValueError(f"配置解析失败: {str(e)}")
    
    def _load_raw_config(self) -> Dict[str, Any]:
        """加载原始配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise ValueError("配置文件内容必须是字典格式")
            
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {str(e)}")
        except Exception as e:
            raise ValueError(f"读取配置文件失败: {str(e)}")
    
    def _validate_raw_config(self, config: Dict[str, Any]) -> List[str]:
        """验证原始配置的完整性"""
        errors = []
        
        # 检查必需的顶级键
        required_keys = ["providers", "crawler"]
        for key in required_keys:
            if key not in config:
                errors.append(f"缺少必需的配置节: {key}")
        
        # 验证providers配置
        if "providers" in config:
            provider_errors = self._validate_providers_config(config["providers"])
            errors.extend(provider_errors)
        
        # 验证爬虫配置
        if "crawler" in config:
            crawler_errors = self._validate_crawler_config(config["crawler"])
            errors.extend(crawler_errors)
        
        return errors
    
    def _validate_providers_config(self, providers_config: Dict[str, Any]) -> List[str]:
        """验证providers配置"""
        errors = []
        
        if not isinstance(providers_config, dict):
            errors.append("providers配置必须是字典格式")
            return errors
        
        # 检查每个provider的配置
        for provider_name, provider_config in providers_config.items():
            if not isinstance(provider_config, dict):
                errors.append(f"Provider {provider_name} 配置必须是字典格式")
                continue
            
            # 检查必需字段
            if "provider_type" not in provider_config:
                errors.append(f"Provider {provider_name} 缺少provider_type配置")
            
            if "api_key" not in provider_config:
                errors.append(f"Provider {provider_name} 缺少api_key配置")
        
        return errors
    
    def _validate_crawler_config(self, crawler_config: Dict[str, Any]) -> List[str]:
        """验证爬虫配置"""
        errors = []
        
        # 检查基本配置
        if "max_papers_per_query" not in crawler_config:
            errors.append("缺少max_papers_per_query配置")
        
        if "request_delay" not in crawler_config:
            errors.append("缺少request_delay配置")
        
        return errors
    
    def _convert_to_app_config_format(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """将原始配置转换为AppConfig期望的格式"""
        app_config_data = {
            "providers": {},
            "task_preferences": config.get("task_preferences", {}),
            "crawler_config": config.get("crawler", {}),
            "output_config": config.get("output", {}),
            "logging_config": config.get("logging", {})
        }
        
        # 转换providers配置
        if "providers" in config:
            for provider_name, provider_config in config["providers"].items():
                # 提取models配置
                models_data = []
                if "models" in config and provider_name in config["models"]:
                    for model_config in config["models"][provider_name]:
                        models_data.append(model_config)
                
                # 创建ProviderConfig数据
                provider_data = {
                    "provider_type": provider_config.get("provider_type", provider_name),
                    "api_key": provider_config.get("api_key"),
                    "base_url": provider_config.get("base_url"),
                    "timeout": provider_config.get("timeout", 30),
                    "retry_count": provider_config.get("retry_count", 3),
                    "requests_per_minute": provider_config.get("requests_per_minute", 60),
                    "max_concurrent_requests": provider_config.get("max_concurrent_requests", 5),
                    "models": models_data
                }
                
                app_config_data["providers"][provider_name] = provider_data
        
        return app_config_data
    
    def get_provider_config(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """
        获取特定提供商配置
        
        Args:
            provider_name: 提供商名称
        
        Returns:
            提供商配置字典，如果不存在则返回None
        """
        if self.app_config is None:
            self.load_config()
        
        return self.app_config.providers.get(provider_name)
    
    def get_enabled_providers(self) -> List[str]:
        """
        获取启用的提供商列表
        
        Returns:
            启用的提供商名称列表
        """
        if self.app_config is None:
            self.load_config()
        
        enabled_providers = []
        
        for provider_name, provider_config in self.app_config.providers.items():
            # 如果配置了api_key，则认为是启用的
            if provider_config and provider_config.get("api_key"):
                enabled_providers.append(provider_name)
        
        return enabled_providers
    
    def validate_config(self) -> List[str]:
        """
        验证配置完整性
        
        Returns:
            验证错误列表，空列表表示验证通过
        """
        try:
            self.load_config()
            return []
        except ValueError as e:
            return [str(e)]
        except Exception as e:
            return [f"配置验证时发生未知错误: {str(e)}"]
    
    def get_raw_config(self) -> Dict[str, Any]:
        """获取原始配置字典（环境变量已展开）"""
        if self._raw_config is None:
            self.load_config()
        
        return self._raw_config.copy()
    
    def reload_config(self) -> AppConfig:
        """重新加载配置"""
        self.app_config = None
        self._raw_config = None
        return self.load_config()
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        更新配置（仅在内存中，不写入文件）
        
        Args:
            updates: 要更新的配置字典
        """
        if self._raw_config is None:
            self.load_config()
        
        # 深度合并配置
        self._deep_update(self._raw_config, updates)
        
        # 重新解析配置
        expanded_config = expand_env_vars(self._raw_config)
        try:
            self.app_config = AppConfig(**expanded_config)
        except Exception as e:
            raise ValueError(f"更新配置后解析失败: {str(e)}")
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """深度更新字典"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        保存当前配置到文件
        
        Args:
            output_path: 输出文件路径，默认为当前配置文件路径
        """
        if self._raw_config is None:
            raise ValueError("没有配置可保存，请先加载配置")
        
        output_path = output_path or self.config_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._raw_config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            raise ValueError(f"保存配置文件失败: {str(e)}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要信息"""
        if self.app_config is None:
            self.load_config()
        
        enabled_providers = self.get_enabled_providers()
        
        return {
            "config_file": self.config_path,
            "enabled_providers": enabled_providers,
            "provider_count": len(enabled_providers),
            "crawler_max_papers": self.app_config.crawler_config.get("max_papers_per_query", "未配置"),
            "crawler_delay": self.app_config.crawler_config.get("request_delay", "未配置"),
            "output_format": self.app_config.output_config.get("default_format", "未配置"),
            "logging_level": self.app_config.logging_config.get("level", "未配置")
        }