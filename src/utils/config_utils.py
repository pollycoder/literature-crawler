import os
import re
from typing import Any, Dict
from pathlib import Path

def expand_env_vars(config: Any) -> Any:
    """递归展开配置中的环境变量"""
    if isinstance(config, dict):
        return {key: expand_env_vars(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        # 提取环境变量名
        env_var = config[2:-1]
        default_value = None
        
        # 支持默认值语法: ${VAR_NAME:default_value}
        if ":" in env_var:
            env_var, default_value = env_var.split(":", 1)
        
        # 获取环境变量值
        env_value = os.getenv(env_var, default_value)
        
        # 如果环境变量不存在且没有默认值，返回None
        if env_value is None:
            return None
            
        return env_value
    else:
        return config


def load_env_file(env_file: str = ".env") -> bool:
    """加载.env文件中的环境变量"""
    env_path = Path(env_file)
    if not env_path.exists():
        return False
    
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    
                    # 只设置尚未设置的环境变量
                    if key not in os.environ:
                        os.environ[key] = value
        
        return True
    except Exception:
        return False



def setup_proxy_config(proxy_api_key: str, proxy_base_url: str) -> str:
    """快速设置代理配置的辅助函数"""
    env_content = f"""# 第三方代理配置
OPENAI_PROXY_API_KEY={proxy_api_key}
OPENAI_PROXY_BASE_URL={proxy_base_url}
"""
    
    try:
        with open(".env", "a", encoding="utf-8") as f:
            f.write(env_content)
        return "代理配置已添加到 .env 文件"
    except Exception as e:
        return f"配置添加失败: {str(e)}"