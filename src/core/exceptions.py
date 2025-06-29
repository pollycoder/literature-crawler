class LiteratureCrawlerException(Exception):
    """基础异常类"""
    pass

class ModelProviderException(LiteratureCrawlerException):
    """模型提供商相关异常"""
    pass

class PromptException(LiteratureCrawlerException):
    """提示词相关异常"""
    pass

class CrawlerException(LiteratureCrawlerException):
    """爬虫相关异常"""
    pass

class ConfigException(LiteratureCrawlerException):
    """配置相关异常"""
    pass

class APIException(LiteratureCrawlerException):
    """API调用异常"""
    def __init__(self, message: str, status_code: int = None, provider: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider

class RateLimitException(APIException):
    """API限流异常"""
    pass

class AuthenticationException(APIException):
    """认证异常"""
    pass

class ValidationException(APIException):
    """数据验证异常"""
    pass

class ConfigurationException(LiteratureCrawlerException):
    """配置异常"""
    pass