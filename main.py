"""
DeepAR工厂能耗预测服务主程序
FastAPI应用入口
"""
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1 import api_router
from app.services.redis_service import RedisService


# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

# Redis服务实例
redis_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global redis_service
    
    logger.info("🚀 启动 DeepAR 工厂能耗预测服务...")
    
    # 初始化Redis连接
    try:
        redis_service = RedisService()
        await redis_service.ping()
        logger.info("✅ Redis连接成功")
    except Exception as e:
        logger.error(f"❌ Redis连接失败: {e}")
        logger.warning("⚠️  将在无缓存模式下运行")
    
    # 确保必要的目录存在
    os.makedirs(settings.DATA_STORAGE_PATH, exist_ok=True)
    os.makedirs(settings.MODEL_STORAGE_PATH, exist_ok=True)
    os.makedirs(settings.LOG_STORAGE_PATH, exist_ok=True)
    logger.info("📁 数据目录初始化完成")
    
    logger.info("🎯 服务启动完成!")
    
    yield
    
    # 清理资源
    logger.info("🔄 正在关闭服务...")
    if redis_service:
        await redis_service.close()
        logger.info("✅ Redis连接已关闭")
    
    logger.info("👋 服务已停止")


# 创建FastAPI应用
app = FastAPI(
    title="DeepAR工厂能耗预测服务",
    description="""
    ## 基于Amazon DeepAR算法的工厂设备能耗预测服务
    
    ### 主要功能
    - 📊 **数据管理**: 能耗数据上传、验证、存储和管理
    - 🧠 **模型训练**: DeepAR时间序列预测模型训练和管理
    - 🔮 **预测服务**: 单次预测、批量预测、实时预测
    - 📈 **分析报告**: 预测结果分析、趋势分析、异常检测
    
    ### 技术架构
    - **算法核心**: Amazon GluonTS DeepAR
    - **Web框架**: FastAPI + Pydantic
    - **数据存储**: Redis缓存 + CSV文件
    - **部署方式**: Docker容器化
    
    ### 快速开始
    1. 上传训练数据 (`POST /api/v1/data/submit`)
    2. 训练预测模型 (`POST /api/v1/training/submit`)
    3. 执行能耗预测 (`POST /api/v1/prediction/submit`)
    4. 查看预测结果 (`GET /api/v1/prediction/result/{prediction_id}`)
    """,
    version=settings.VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan
)

# CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {type(exc).__name__}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "message": "请稍后重试或联系管理员" if not settings.DEBUG else str(exc),
            "type": type(exc).__name__
        }
    )


# 根路径重定向到文档
@app.get("/", include_in_schema=False)
async def root():
    """根路径重定向到API文档"""
    if settings.DEBUG:
        return RedirectResponse(url="/docs")
    else:
        return {
            "service": "DeepAR工厂能耗预测服务",
            "version": settings.VERSION,
            "status": "running",
            "docs": "API文档在生产环境中不可用"
        }


# 健康检查端点
@app.get("/health", tags=["系统"], summary="健康检查")
async def health_check():
    """系统健康检查"""
    health_status = {
        "status": "healthy",
        "service": "DeepAR工厂能耗预测服务",
        "version": settings.VERSION,
        "timestamp": "2024-01-01T00:00:00Z"
    }
    
    # 检查Redis连接
    if redis_service:
        try:
            await redis_service.ping()
            health_status["redis"] = "connected"
        except Exception as e:
            health_status["redis"] = f"disconnected: {str(e)}"
            health_status["status"] = "degraded"
    else:
        health_status["redis"] = "not_configured"
        health_status["status"] = "degraded"
    
    # 检查必要目录
    dirs_status = {}
    for dir_name, dir_path in [
        ("data", settings.DATA_STORAGE_PATH),
        ("models", settings.MODEL_STORAGE_PATH), 
        ("logs", settings.LOG_STORAGE_PATH)
    ]:
        dirs_status[dir_name] = "exists" if os.path.exists(dir_path) else "missing"
    
    health_status["directories"] = dirs_status
    
    return health_status


# 系统信息端点
@app.get("/info", tags=["系统"], summary="系统信息")
async def system_info():
    """获取系统信息"""
    return {
        "service": "DeepAR工厂能耗预测服务",
        "version": settings.VERSION,
        "environment": "development" if settings.DEBUG else "production",
        "features": {
            "data_management": "✅ 数据上传、验证、存储",
            "model_training": "✅ DeepAR模型训练和管理",
            "prediction_service": "✅ 单次、批量、实时预测",
            "analytics": "✅ 预测分析和报告"
        },
        "api": {
            "base_url": f"http://localhost:{settings.API_PORT}",
            "docs_url": "/docs" if settings.DEBUG else "生产环境不可用",
            "version": "v1"
        },
        "configuration": {
            "debug_mode": settings.DEBUG,
            "redis_enabled": redis_service is not None,
            "max_upload_size": f"{settings.MAX_REQUEST_SIZE // 1024 // 1024}MB",
            "supported_formats": ["CSV", "JSON"]
        }
    }


# 包含API路由
app.include_router(api_router, prefix="/api/v1")


# 静态文件服务（如果需要）
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


def main():
    """主函数，用于命令行启动"""
    logger.info(f"🌟 启动DeepAR工厂能耗预测服务 v{settings.VERSION}")
    logger.info(f"🌐 服务地址: http://localhost:{settings.API_PORT}")
    logger.info(f"📚 API文档: http://localhost:{settings.API_PORT}/docs")
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
        access_log=settings.DEBUG
    )


if __name__ == "__main__":
    main()