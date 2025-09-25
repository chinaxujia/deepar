"""
健康检查API路由
"""
import psutil
from fastapi import APIRouter, Depends
from app.services.redis_service import RedisService
from app.schemas.health import HealthResponse, SystemStatus
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_redis_service() -> RedisService:
    """获取Redis服务依赖"""
    from app import app
    return app.state.redis


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check(redis: RedisService = Depends(get_redis_service)):
    """
    系统健康检查接口
    
    检查项目：
    - API服务状态
    - Redis连接状态
    - 系统资源使用情况
    """
    try:
        # 检查Redis连接
        redis_status = "healthy"
        try:
            await redis.redis_client.ping()
        except:
            redis_status = "unhealthy"
        
        # 获取系统资源信息
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_status = SystemStatus(
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            memory_total=memory.total,
            memory_used=memory.used,
            disk_usage=disk.percent,
            disk_total=disk.total,
            disk_used=disk.used
        )
        
        # 判断整体健康状态
        overall_status = "healthy"
        if redis_status == "unhealthy" or cpu_percent > 90 or memory.percent > 90:
            overall_status = "unhealthy"
        elif cpu_percent > 70 or memory.percent > 70:
            overall_status = "warning"
        
        return HealthResponse(
            status=overall_status,
            message="系统运行正常" if overall_status == "healthy" else "系统存在问题",
            redis_status=redis_status,
            system_status=system_status,
            timestamp=psutil.boot_time()
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            message=f"健康检查失败: {str(e)}",
            redis_status="unknown",
            system_status=None,
            timestamp=psutil.boot_time()
        )