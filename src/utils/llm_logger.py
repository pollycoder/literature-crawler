"""
LLM调用日志系统
支持4种日志类型：请求、响应、错误、重试
支持按日轮转
支持HTTP状态码标识不同状态
"""

import logging
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler


class StatusCode:
    """LLM调用状态码定义"""
    # 成功状态
    SUCCESS = 200
    
    # 客户端错误
    BAD_REQUEST = 400          # 请求参数错误
    UNAUTHORIZED = 401         # 认证失败
    FORBIDDEN = 403           # 权限不足
    NOT_FOUND = 404           # 模型不存在
    UNPROCESSABLE_ENTITY = 422 # 请求格式正确但语义错误
    RATE_LIMITED = 429        # 速率限制
    
    # 服务端错误
    SERVER_ERROR = 500        # 服务器内部错误
    BAD_GATEWAY = 502         # 网关错误
    SERVICE_UNAVAILABLE = 503 # 服务不可用
    TIMEOUT = 504             # 超时
    
    # 重试状态
    RETRY_RATE_LIMIT = 1001   # 因速率限制重试
    RETRY_SERVER_ERROR = 1002 # 因服务器错误重试
    RETRY_TIMEOUT = 1003      # 因超时重试
    RETRY_UNKNOWN = 1004      # 因未知错误重试


class LLMLogger:
    """LLM专用日志记录器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建4个专用logger
        self.request_logger = self._create_logger("llm_requests", "requests.log")
        self.response_logger = self._create_logger("llm_responses", "responses.log") 
        self.error_logger = self._create_logger("llm_errors", "errors.log")
        self.retry_logger = self._create_logger("llm_retries", "retries.log")
        
        # 添加ArXiv链接记录logger
        self.arxiv_logger = self._create_logger("arxiv_links", "arxiv_links.log")
        
        # 全局session id，用于跟踪整个会话
        self.current_session_id = None
        
    def _create_logger(self, name: str, filename: str) -> logging.Logger:
        """创建单独的日志记录器"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if logger.handlers:
            return logger
            
        # 创建按日轮转的文件处理器
        handler = TimedRotatingFileHandler(
            filename=self.log_dir / filename,
            when='midnight',
            interval=1,
            backupCount=30,  # 保留30天
            encoding='utf-8'
        )
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def generate_session_id(self) -> str:
        """生成新的session ID并设为当前session"""
        self.current_session_id = str(uuid.uuid4())
        return self.current_session_id
    
    def set_session_id(self, session_id: str):
        """设置当前session ID"""
        self.current_session_id = session_id
    
    def get_session_id(self) -> str:
        """获取当前session ID，如果没有则生成一个"""
        if self.current_session_id is None:
            self.generate_session_id()
        return self.current_session_id
    
    def log_request(self, 
                   provider: str,
                   model: str, 
                   prompt: str,
                   params: Dict[str, Any],
                   request_id: str = None,
                   session_id: str = None) -> tuple[str, str]:
        """记录LLM请求
        
        Returns:
            tuple[str, str]: (request_id, session_id)
        """
        if not request_id:
            request_id = f"{provider}_{model}_{int(time.time() * 1000)}"
        
        # 每个请求生成唯一的session_id
        if session_id is None:
            session_id = str(uuid.uuid4())
            
        log_data = {
            "request_id": request_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "status_code": "PENDING",  # 请求状态为待处理
            "provider": provider,
            "model": model,
            "prompt_length": len(prompt),
            "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "parameters": params
        }
        
        self.request_logger.info(json.dumps(log_data, ensure_ascii=False))
        return request_id, session_id
    
    def log_response(self,
                    request_id: str,
                    response: str,
                    tokens_used: int = None,
                    cost: float = None,
                    duration: float = None,
                    status_code: int = StatusCode.SUCCESS,
                    session_id: str = None,
                    model: str = None):
        """记录LLM响应"""
        # session_id是必须的，应该从request中传递过来
        if session_id is None:
            raise ValueError("session_id is required for log_response")
            
        log_data = {
            "request_id": request_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "status_code": status_code,
            "status_message": self._get_status_message(status_code),
            "model": model,
            "response_length": len(response),
            "response_preview": response[:200] + "..." if len(response) > 200 else response,
            "tokens_used": tokens_used,
            "cost": cost,
            "duration_seconds": duration
        }
        
        self.response_logger.info(json.dumps(log_data, ensure_ascii=False))
    
    def log_error(self,
                 request_id: str,
                 provider: str,
                 model: str,
                 error_type: str,
                 error_message: str,
                 duration: float = None,
                 status_code: int = None,
                 session_id: str = None,
                 exception: Exception = None):
        """记录LLM错误"""
        # 根据错误类型自动判断状态码，优先使用异常中的真实状态码
        if status_code is None:
            status_code = self._map_error_to_status_code(error_type, exception)
        
        # session_id是必须的，应该从request中传递过来
        if session_id is None:
            raise ValueError("session_id is required for log_error")
            
        log_data = {
            "request_id": request_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "status_code": status_code,
            "status_message": self._get_status_message(status_code),
            "provider": provider,
            "model": model,
            "error_type": error_type,
            "error_message": error_message,
            "duration_seconds": duration
        }
        
        self.error_logger.error(json.dumps(log_data, ensure_ascii=False))
    
    def log_retry(self,
                 request_id: str,
                 provider: str,
                 model: str,
                 attempt: int,
                 error_type: str,
                 wait_time: float = None,
                 status_code: int = None,
                 session_id: str = None):
        """记录LLM重试"""
        # 根据错误类型自动判断重试状态码
        if status_code is None:
            status_code = self._map_error_to_retry_status_code(error_type)
        
        # session_id是必须的，应该从request中传递过来
        if session_id is None:
            raise ValueError("session_id is required for log_retry")
            
        log_data = {
            "request_id": request_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "status_code": status_code,
            "status_message": self._get_status_message(status_code),
            "provider": provider,
            "model": model,
            "attempt": attempt,
            "error_type": error_type,
            "wait_time_seconds": wait_time
        }
        
        self.retry_logger.warning(json.dumps(log_data, ensure_ascii=False))
    
    def log_arxiv_links(self, papers: list, session_id: str = None):
        """记录爬取到的ArXiv链接"""
        if session_id is None:
            session_id = self.get_session_id()
        
        for paper in papers:
            log_data = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "arxiv_id": getattr(paper, 'arxiv_id', ''),
                "title": getattr(paper, 'title', ''),
                "url": getattr(paper, 'url', ''),
                "pdf_url": getattr(paper, 'pdf_url', ''),
                "authors": getattr(paper, 'authors', []),
                "published_date": str(getattr(paper, 'published_date', '')),
                "categories": getattr(paper, 'categories', [])
            }
            
            self.arxiv_logger.info(json.dumps(log_data, ensure_ascii=False))
    
    def _get_status_message(self, status_code: int) -> str:
        """获取状态码对应的消息"""
        status_messages = {
            StatusCode.SUCCESS: "Success",
            StatusCode.BAD_REQUEST: "Bad Request",
            StatusCode.UNAUTHORIZED: "Unauthorized",
            StatusCode.FORBIDDEN: "Forbidden",
            StatusCode.NOT_FOUND: "Not Found",
            StatusCode.UNPROCESSABLE_ENTITY: "Unprocessable Entity",
            StatusCode.RATE_LIMITED: "Rate Limited",
            StatusCode.SERVER_ERROR: "Server Error",
            StatusCode.BAD_GATEWAY: "Bad Gateway",
            StatusCode.SERVICE_UNAVAILABLE: "Service Unavailable",
            StatusCode.TIMEOUT: "Timeout",
            StatusCode.RETRY_RATE_LIMIT: "Retry - Rate Limited",
            StatusCode.RETRY_SERVER_ERROR: "Retry - Server Error",
            StatusCode.RETRY_TIMEOUT: "Retry - Timeout",
            StatusCode.RETRY_UNKNOWN: "Retry - Unknown Error"
        }
        return status_messages.get(status_code, f"Unknown Status ({status_code})")
    
    def _map_error_to_status_code(self, error_type: str, exception: Exception = None) -> int:
        """将错误类型映射到状态码，优先使用异常中的真实状态码"""
        # 首先尝试从异常实例中获取状态码
        if exception and hasattr(exception, 'status_code') and exception.status_code:
            return exception.status_code
            
        # 如果没有真实状态码，使用错误类型映射
        error_mapping = {
            "AuthenticationException": StatusCode.UNAUTHORIZED,
            "RateLimitException": StatusCode.RATE_LIMITED,
            "APIException": StatusCode.SERVER_ERROR,
            "TimeoutException": StatusCode.TIMEOUT,
            "ValidationException": StatusCode.BAD_REQUEST,
            "ModelNotFoundException": StatusCode.NOT_FOUND,
            "InsufficientQuotaException": StatusCode.FORBIDDEN,
            "ServiceUnavailableException": StatusCode.SERVICE_UNAVAILABLE,
            "BadGatewayException": StatusCode.BAD_GATEWAY,
            "ValueError": StatusCode.BAD_REQUEST,
            "ConnectionError": StatusCode.BAD_GATEWAY,
            # 通用HTTP错误类型
            "BadRequestError": StatusCode.BAD_REQUEST,
            "NotFoundError": StatusCode.NOT_FOUND,
            "PermissionDeniedError": StatusCode.FORBIDDEN,
            "UnprocessableEntityError": StatusCode.UNPROCESSABLE_ENTITY,
            "InternalServerError": StatusCode.SERVER_ERROR,
            "APITimeoutError": StatusCode.TIMEOUT
        }
        return error_mapping.get(error_type, StatusCode.SERVER_ERROR)
    
    def _map_error_to_retry_status_code(self, error_type: str) -> int:
        """将错误类型映射到重试状态码"""
        retry_mapping = {
            "RateLimitException": StatusCode.RETRY_RATE_LIMIT,
            "TimeoutException": StatusCode.RETRY_TIMEOUT,
            "APIException": StatusCode.RETRY_SERVER_ERROR,
            "ServerException": StatusCode.RETRY_SERVER_ERROR,
            "ServiceUnavailableException": StatusCode.RETRY_SERVER_ERROR
        }
        return retry_mapping.get(error_type, StatusCode.RETRY_UNKNOWN)


# 全局日志实例
_llm_logger: Optional[LLMLogger] = None


def get_llm_logger(log_dir: str = "logs") -> LLMLogger:
    """获取LLM日志记录器单例"""
    global _llm_logger
    if _llm_logger is None:
        _llm_logger = LLMLogger(log_dir)
    return _llm_logger


def init_llm_logging(log_dir: str = "logs"):
    """初始化LLM日志系统"""
    global _llm_logger
    _llm_logger = LLMLogger(log_dir)
    return _llm_logger