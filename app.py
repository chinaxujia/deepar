"""
FastAPI主应用程序入口
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1 import training, prediction, monitoring, data, models, health
from app.services.redis_service import RedisService


# 设置日志
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理"""
    # 启动时初始化
    logger.info("正在启动 DeepAR 能耗预测服务...")
    
    # 初始化Redis连接
    redis_service = RedisService()
    await redis_service.connect()
    app.state.redis = redis_service
    
    logger.info("服务启动完成")
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭服务...")
    if hasattr(app.state, 'redis'):
        await app.state.redis.close()
    logger.info("服务已关闭")


# 创建FastAPI应用实例
app = FastAPI(
    title="DeepAR 工厂能耗预测服务",
    description="基于GluonTS的DeepAR算法的工厂能耗时间序列预测服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"全局异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误"}
    )


# 注册路由
app.include_router(health.router, prefix="/api/v1", tags=["健康检查"])
app.include_router(training.router, prefix="/api/v1", tags=["模型训练"])
app.include_router(prediction.router, prefix="/api/v1", tags=["模型预测"])
app.include_router(data.router, prefix="/api/v1", tags=["数据管理"])
app.include_router(models.router, prefix="/api/v1", tags=["模型管理"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["系统监控"])


@app.get("/", summary="根路径", description="API服务根路径")
async def root():
    return {
        "message": "DeepAR 工厂能耗预测服务",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )