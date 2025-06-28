import yaml
import jinja2
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
from ..core.models import TaskType
from ..core.exceptions import PromptException

@dataclass
class PromptTemplate:
    name: str
    task_type: TaskType
    template: str
    variables: List[str]
    model_specific: Dict[str, str]
    language: str = "zh"
    version: str = "1.0"
    description: str = ""
    examples: List[Dict[str, Any]] = None

class PromptManager:
    """提示词管理器"""
    
    def __init__(self, prompts_dir: str = "config/prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.templates: Dict[str, PromptTemplate] = {}
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        if self.prompts_dir.exists():
            self.load_all_prompts()
    
    def load_all_prompts(self):
        """加载所有提示词模板"""
        for task_type in TaskType:
            config_file = self.prompts_dir / f"{task_type.value}.yaml"
            if config_file.exists():
                self.load_prompt_for_task(task_type)
    
    def load_prompt_for_task(self, task_type: TaskType):
        """加载特定任务的提示词"""
        config_file = self.prompts_dir / f"{task_type.value}.yaml"
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 确保examples字段存在且为列表
            if 'examples' not in config:
                config['examples'] = []
            elif config['examples'] is None:
                config['examples'] = []
            
            template = PromptTemplate(**config)
            self.templates[task_type.value] = template
            
        except Exception as e:
            raise PromptException(f"Failed to load prompt for {task_type.value}: {str(e)}")
    
    def get_prompt(self, 
                   task_type: TaskType, 
                   variables: Dict[str, Any],
                   model_name: Optional[str] = None,
                   language: str = "zh") -> str:
        """获取格式化的提示词"""
        template = self.templates.get(task_type.value)
        if not template:
            raise PromptException(f"No template found for task: {task_type.value}")
        
        # 验证必需变量
        if not self.validate_variables(task_type, variables):
            missing = set(template.variables) - set(variables.keys())
            raise PromptException(f"Missing variables for {task_type.value}: {missing}")
        
        # 选择模型特定版本或默认版本
        prompt_text = template.template
        if model_name and model_name in template.model_specific:
            prompt_text = template.model_specific[model_name]
        
        try:
            # 使用Jinja2渲染模板
            jinja_template = jinja2.Template(prompt_text)
            return jinja_template.render(**variables)
        except Exception as e:
            raise PromptException(f"Failed to render prompt: {str(e)}")
    
    def validate_variables(self, 
                          task_type: TaskType, 
                          variables: Dict[str, Any]) -> bool:
        """验证变量是否完整"""
        template = self.templates.get(task_type.value)
        if not template:
            return False
        
        required_vars = set(template.variables)
        provided_vars = set(variables.keys())
        
        return required_vars.issubset(provided_vars)
    
    def get_example(self, task_type: TaskType, index: int = 0) -> Dict[str, Any]:
        """获取示例"""
        template = self.templates.get(task_type.value)
        if template and template.examples and len(template.examples) > index:
            return template.examples[index]
        return {}
    
    def get_all_templates(self) -> Dict[str, PromptTemplate]:
        """获取所有模板"""
        return self.templates.copy()
    
    def add_template(self, template: PromptTemplate):
        """添加新模板"""
        self.templates[template.task_type.value] = template
    
    def update_template(self, task_type: TaskType, updates: Dict[str, Any]):
        """更新模板"""
        if task_type.value not in self.templates:
            raise PromptException(f"Template for {task_type.value} not found")
        
        template = self.templates[task_type.value]
        for key, value in updates.items():
            if hasattr(template, key):
                setattr(template, key, value)
    
    def export_template(self, task_type: TaskType, output_path: str):
        """导出模板到文件"""
        template = self.templates.get(task_type.value)
        if not template:
            raise PromptException(f"Template for {task_type.value} not found")
        
        template_dict = {
            'name': template.name,
            'task_type': template.task_type.value,
            'template': template.template,
            'variables': template.variables,
            'model_specific': template.model_specific,
            'language': template.language,
            'version': template.version,
            'description': template.description,
            'examples': template.examples or []
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(template_dict, f, allow_unicode=True, default_flow_style=False)