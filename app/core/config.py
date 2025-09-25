"""
应用程序配置管理
"""
import os
from typing import List
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """应用程序设置"""
    
    # 应用程序信息
    VERSION: str = Field(default="1.0.0", description="应用程序版本")
    
    # API配置
    API_HOST: str = Field(default="0.0.0.0", description="API服务主机地址")
    API_PORT: int = Field(default=8000, description="API服务端口")
    DEBUG: bool = Field(default=False, description="调试模式")
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    
    # Redis配置
    REDIS_HOST: str = Field(default="localhost", description="Redis主机地址")
    REDIS_PORT: int = Field(default=6379, description="Redis端口")
    REDIS_DB: int = Field(default=0, description="Redis数据库编号")
    REDIS_PASSWORD: str = Field(default="", description="Redis密码")
    
    # 存储路径配置
    MODEL_STORAGE_PATH: str = Field(default="./models", description="模型存储路径")
    CONFIG_STORAGE_PATH: str = Field(default="./configs", description="配置存储路径")
    DATA_STORAGE_PATH: str = Field(default="./data", description="数据存储路径")
    LOG_STORAGE_PATH: str = Field(default="./logs", description="日志存储路径")
    
    # DeepAR模型配置
    DEFAULT_PREDICTION_LENGTH: int = Field(default=24, description="默认预测长度")
    DEFAULT_CONTEXT_LENGTH: int = Field(default=168, description="默认上下文长度")
    DEFAULT_FREQ: str = Field(default="H", description="默认频率")
    DEFAULT_TRAINER_EPOCHS: int = Field(default=100, description="默认训练轮数")
    
    # API安全配置
    API_KEY_HEADER: str = Field(default="X-API-Key", description="API密钥头部")
    ALLOWED_HOSTS: List[str] = Field(default=["*"], description="允许的主机列表")
    MAX_REQUEST_SIZE: int = Field(default=10485760, description="最大请求大小(10MB)")
    REQUEST_TIMEOUT: int = Field(default=300, description="请求超时时间(秒)")
    
    # 缓存配置
    CACHE_TTL: int = Field(default=3600, description="缓存生存时间(秒)")
    MODEL_CACHE_SIZE: int = Field(default=5, description="模型缓存数量")
    
    # 系统监控配置
    MONITOR_INTERVAL: int = Field(default=60, description="监控间隔(秒)")
    MAX_CPU_USAGE: float = Field(default=80.0, description="最大CPU使用率(%)")
    MAX_MEMORY_USAGE: float = Field(default=80.0, description="最大内存使用率(%)")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 创建全局设置实例
settings = Settings()


def create_directories():
    """创建必要的目录"""
    directories = [
        settings.MODEL_STORAGE_PATH,
        settings.CONFIG_STORAGE_PATH,
        settings.DATA_STORAGE_PATH,
        settings.LOG_STORAGE_PATH
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# 初始化时创建目录
create_directories()