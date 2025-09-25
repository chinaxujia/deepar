"""
日志配置管理
"""
import logging
import logging.config
import os
from datetime import datetime
from app.core.config import settings


def setup_logging():
    """设置应用程序日志配置"""
    
    # 确保日志目录存在
    log_dir = settings.LOG_STORAGE_PATH
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件路径
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"deepar_{current_date}.log")
    error_log_file = os.path.join(log_dir, f"deepar_error_{current_date}.log")
    
    # 日志配置
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'level': settings.LOG_LEVEL,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'INFO',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'detailed',
                'filename': log_file,
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            },
            'error_file': {
                'level': 'ERROR',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'detailed',
                'filename': error_log_file,
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            }
        },
        'loggers': {
            '': {  # 根日志器
                'level': settings.LOG_LEVEL,
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False
            },
            'uvicorn': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'uvicorn.error': {
                'level': 'INFO',
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False
            },
            'uvicorn.access': {
                'level': 'INFO',
                'handlers': ['file'],
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    # 记录启动日志
    logger = logging.getLogger(__name__)
    logger.info("日志系统初始化完成")
    logger.info(f"日志级别: {settings.LOG_LEVEL}")
    logger.info(f"日志文件: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志器"""
    return logging.getLogger(name)