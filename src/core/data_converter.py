"""
通用的数据转换工具
消除重复的序列化/反序列化逻辑
"""

import json
from typing import Dict, Any, List, Type, TypeVar, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import asdict, is_dataclass
from pydantic import BaseModel

T = TypeVar('T')


class DataConverter:
    """数据对象转换工具"""
    
    @staticmethod
    def to_dict(obj: Any, exclude_none: bool = True) -> Dict[str, Any]:
        """
        对象转字典的通用方法
        
        Args:
            obj: 要转换的对象
            exclude_none: 是否排除None值
        
        Returns:
            转换后的字典
        """
        if obj is None:
            return {}
        
        # 处理基本类型
        if isinstance(obj, (str, int, float, bool)):
            return {"value": obj}
        
        # 处理字典
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if exclude_none and value is None:
                    continue
                result[key] = DataConverter._serialize_value(value)
            return result
        
        # 处理列表
        if isinstance(obj, (list, tuple)):
            return {
                "items": [DataConverter._serialize_value(item) for item in obj],
                "type": "list"
            }
        
        # 处理Pydantic模型
        if isinstance(obj, BaseModel):
            return obj.model_dump(exclude_none=exclude_none)
        
        # 处理dataclass
        if is_dataclass(obj):
            data = asdict(obj)
            if exclude_none:
                return {k: v for k, v in data.items() if v is not None}
            return data
        
        # 处理枚举
        if isinstance(obj, Enum):
            return {
                "value": obj.value,
                "name": obj.name,
                "type": "enum",
                "enum_class": obj.__class__.__name__
            }
        
        # 处理datetime
        if isinstance(obj, datetime):
            return {
                "value": obj.isoformat(),
                "type": "datetime"
            }
        
        # 处理其他对象，尝试提取__dict__
        if hasattr(obj, '__dict__'):
            data = {}
            for key, value in obj.__dict__.items():
                if key.startswith('_'):
                    continue
                if exclude_none and value is None:
                    continue
                data[key] = DataConverter._serialize_value(value)
            return data
        
        # 最后的后备方案
        return {"value": str(obj), "type": "string"}
    
    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """序列化单个值"""
        if value is None:
            return None
        
        if isinstance(value, (str, int, float, bool)):
            return value
        
        if isinstance(value, datetime):
            return value.isoformat()
        
        if isinstance(value, Enum):
            return value.value
        
        if isinstance(value, (list, tuple)):
            return [DataConverter._serialize_value(item) for item in value]
        
        if isinstance(value, dict):
            return {k: DataConverter._serialize_value(v) for k, v in value.items()}
        
        if isinstance(value, BaseModel):
            return value.model_dump()
        
        if is_dataclass(value):
            return asdict(value)
        
        # 尝试转换为字符串
        return str(value)
    
    @staticmethod
    def from_dict(data: Dict[str, Any], target_class: Type[T]) -> T:
        """
        字典转对象的通用方法
        
        Args:
            data: 源字典
            target_class: 目标类型
        
        Returns:
            转换后的对象
        """
        if not isinstance(data, dict):
            raise ValueError("data必须是字典类型")
        
        # 处理Pydantic模型
        if issubclass(target_class, BaseModel):
            return target_class(**data)
        
        # 处理dataclass
        if is_dataclass(target_class):
            return target_class(**data)
        
        # 处理其他类，尝试直接构造
        try:
            return target_class(**data)
        except Exception as e:
            raise ValueError(f"无法将字典转换为{target_class.__name__}: {str(e)}")
    
    @staticmethod
    def batch_convert_to_dict(objects: List[Any], exclude_none: bool = True) -> List[Dict[str, Any]]:
        """
        批量转换对象为字典
        
        Args:
            objects: 对象列表
            exclude_none: 是否排除None值
        
        Returns:
            字典列表
        """
        return [DataConverter.to_dict(obj, exclude_none) for obj in objects]
    
    @staticmethod
    def batch_convert_from_dict(data_list: List[Dict[str, Any]], target_class: Type[T]) -> List[T]:
        """
        批量从字典转换对象
        
        Args:
            data_list: 字典列表
            target_class: 目标类型
        
        Returns:
            对象列表
        """
        return [DataConverter.from_dict(data, target_class) for data in data_list]
    
    @staticmethod
    def to_json(obj: Any, exclude_none: bool = True, indent: Optional[int] = None) -> str:
        """
        对象转JSON字符串
        
        Args:
            obj: 要转换的对象
            exclude_none: 是否排除None值
            indent: JSON缩进
        
        Returns:
            JSON字符串
        """
        dict_data = DataConverter.to_dict(obj, exclude_none)
        return json.dumps(dict_data, ensure_ascii=False, indent=indent)
    
    @staticmethod
    def from_json(json_str: str, target_class: Type[T]) -> T:
        """
        从JSON字符串转换对象
        
        Args:
            json_str: JSON字符串
            target_class: 目标类型
        
        Returns:
            转换后的对象
        """
        try:
            data = json.loads(json_str)
            return DataConverter.from_dict(data, target_class)
        except json.JSONDecodeError as e:
            raise ValueError(f"无效的JSON字符串: {str(e)}")
    
    @staticmethod
    def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any], 
                   deep_merge: bool = True) -> Dict[str, Any]:
        """
        合并两个字典
        
        Args:
            dict1: 第一个字典
            dict2: 第二个字典  
            deep_merge: 是否深度合并
        
        Returns:
            合并后的字典
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and deep_merge:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = DataConverter.merge_dicts(result[key], value, deep_merge)
                else:
                    result[key] = value
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def flatten_dict(data: Dict[str, Any], parent_key: str = '', 
                    separator: str = '.') -> Dict[str, Any]:
        """
        扁平化嵌套字典
        
        Args:
            data: 嵌套字典
            parent_key: 父键名
            separator: 分隔符
        
        Returns:
            扁平化后的字典
        """
        items = []
        
        for key, value in data.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.extend(
                    DataConverter.flatten_dict(value, new_key, separator).items()
                )
            else:
                items.append((new_key, value))
        
        return dict(items)
    
    @staticmethod
    def unflatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
        """
        反扁平化字典
        
        Args:
            data: 扁平化的字典
            separator: 分隔符
        
        Returns:
            嵌套字典
        """
        result = {}
        
        for key, value in data.items():
            keys = key.split(separator)
            current = result
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
        
        return result
    
    @staticmethod
    def filter_dict(data: Dict[str, Any], include_keys: Optional[List[str]] = None,
                   exclude_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        过滤字典键
        
        Args:
            data: 源字典
            include_keys: 包含的键列表
            exclude_keys: 排除的键列表
        
        Returns:
            过滤后的字典
        """
        result = {}
        
        for key, value in data.items():
            # 检查包含列表
            if include_keys and key not in include_keys:
                continue
            
            # 检查排除列表
            if exclude_keys and key in exclude_keys:
                continue
            
            result[key] = value
        
        return result
    
    @staticmethod
    def deep_copy_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度复制字典
        
        Args:
            data: 源字典
        
        Returns:
            复制后的字典
        """
        import copy
        return copy.deepcopy(data)
    
    @staticmethod
    def validate_dict_structure(data: Dict[str, Any], 
                              required_keys: List[str],
                              optional_keys: Optional[List[str]] = None) -> List[str]:
        """
        验证字典结构
        
        Args:
            data: 要验证的字典
            required_keys: 必需键列表
            optional_keys: 可选键列表
        
        Returns:
            验证错误列表
        """
        errors = []
        
        # 检查必需键
        for key in required_keys:
            if key not in data:
                errors.append(f"缺少必需的键: {key}")
        
        # 检查未知键
        if optional_keys is not None:
            allowed_keys = set(required_keys + optional_keys)
            for key in data.keys():
                if key not in allowed_keys:
                    errors.append(f"未知的键: {key}")
        
        return errors


class JSONSerializationMixin:
    """JSON序列化混入类"""
    
    def to_json(self, exclude_none: bool = True, indent: Optional[int] = None) -> str:
        """转换为JSON字符串"""
        return DataConverter.to_json(self, exclude_none, indent)
    
    @classmethod
    def from_json(cls, json_str: str):
        """从JSON字符串创建对象"""
        return DataConverter.from_json(json_str, cls)
    
    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """转换为字典"""
        return DataConverter.to_dict(self, exclude_none)