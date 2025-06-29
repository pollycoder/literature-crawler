"""
服务工厂
用于创建和初始化所有服务组件
"""

from typing import Tuple, Optional

from .config_manager import ConfigManager
from .llm_service import LLMService
from ..llm.model_router import ModelRouter
from ..prompts.manager import PromptManager
from ..analyzer.requirement_analyzer import RequirementAnalyzer
from ..analyzer.classifier import PaperClassifier
from ..reviewer.generator import ReviewGenerator
from ..crawler.arxiv_crawler import ArxivCrawler


class ServiceFactory:
    """服务工厂类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化服务工厂
        
        Args:
            config_path: 配置文件路径
        """
        self.config_manager = ConfigManager(config_path)
        self._model_router: Optional[ModelRouter] = None
        self._prompt_manager: Optional[PromptManager] = None
        self._llm_service: Optional[LLMService] = None
    
    async def get_config_manager(self) -> ConfigManager:
        """获取配置管理器"""
        return self.config_manager
    
    async def get_model_router(self) -> ModelRouter:
        """获取模型路由器"""
        if self._model_router is None:
            # ModelRouter期望配置文件路径，不是AppConfig对象
            config_path = self.config_manager.config_path
            self._model_router = ModelRouter(config_path)
            # ModelRouter在构造时已经初始化，不需要额外调用initialize
        
        return self._model_router
    
    async def get_prompt_manager(self) -> PromptManager:
        """获取提示词管理器"""
        if self._prompt_manager is None:
            self._prompt_manager = PromptManager()
        
        return self._prompt_manager
    
    async def get_llm_service(self) -> LLMService:
        """获取LLM服务"""
        if self._llm_service is None:
            model_router = await self.get_model_router()
            prompt_manager = await self.get_prompt_manager()
            self._llm_service = LLMService(model_router, prompt_manager)
        
        return self._llm_service
    
    async def get_requirement_analyzer(self) -> RequirementAnalyzer:
        """获取需求分析器"""
        llm_service = await self.get_llm_service()
        return RequirementAnalyzer(llm_service)
    
    async def get_paper_classifier(self) -> PaperClassifier:
        """获取论文分类器"""
        llm_service = await self.get_llm_service()
        return PaperClassifier(llm_service)
    
    async def get_review_generator(self) -> ReviewGenerator:
        """获取综述生成器"""
        llm_service = await self.get_llm_service()
        return ReviewGenerator(llm_service)
    
    async def get_arxiv_crawler(self) -> ArxivCrawler:
        """获取ArXiv爬虫"""
        config = self.config_manager.load_config()
        crawler_config = config.crawler_config
        
        # 从配置中提取参数
        max_concurrent = crawler_config.get("max_concurrent_requests", 5)
        delay = crawler_config.get("request_delay", 1.0)
        
        return ArxivCrawler(max_concurrent=max_concurrent, delay=delay)
    
    async def initialize_all_services(self) -> Tuple[ModelRouter, PromptManager, LLMService]:
        """
        初始化所有核心服务
        
        Returns:
            (模型路由器, 提示词管理器, LLM服务)的元组
        """
        model_router = await self.get_model_router()
        prompt_manager = await self.get_prompt_manager()
        llm_service = await self.get_llm_service()
        
        return model_router, prompt_manager, llm_service
    
    async def get_service_status(self) -> dict:
        """获取所有服务的状态"""
        status = {
            "config": "not_loaded",
            "model_router": "not_initialized",
            "prompt_manager": "not_initialized", 
            "llm_service": "not_initialized"
        }
        
        try:
            # 检查配置
            self.config_manager.load_config()
            status["config"] = "loaded"
            
            # 检查模型路由器
            if self._model_router is not None:
                status["model_router"] = "initialized"
            
            # 检查提示词管理器
            if self._prompt_manager is not None:
                status["prompt_manager"] = "initialized"
            
            # 检查LLM服务
            if self._llm_service is not None:
                status["llm_service"] = "initialized"
                
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    async def cleanup(self):
        """清理资源"""
        if self._model_router is not None:
            # 如果模型路由器有清理方法的话
            if hasattr(self._model_router, 'cleanup'):
                await self._model_router.cleanup()
        
        # 重置所有服务引用
        self._model_router = None
        self._prompt_manager = None
        self._llm_service = None


# 全局服务工厂实例
_global_service_factory: Optional[ServiceFactory] = None


def get_service_factory(config_path: Optional[str] = None) -> ServiceFactory:
    """
    获取全局服务工厂实例
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        服务工厂实例
    """
    global _global_service_factory
    
    if _global_service_factory is None:
        _global_service_factory = ServiceFactory(config_path)
    
    return _global_service_factory


async def initialize_global_services(config_path: Optional[str] = None) -> Tuple[ModelRouter, PromptManager, LLMService]:
    """
    初始化全局服务
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        (模型路由器, 提示词管理器, LLM服务)的元组
    """
    factory = get_service_factory(config_path)
    return await factory.initialize_all_services()


async def cleanup_global_services():
    """清理全局服务"""
    global _global_service_factory
    
    if _global_service_factory is not None:
        await _global_service_factory.cleanup()
        _global_service_factory = None