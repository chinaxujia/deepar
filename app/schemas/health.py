"""
健康检查相关的数据模型
"""
from typing import Optional
from pydantic import BaseModel, Field


class SystemStatus(BaseModel):
    """系统状态信息"""
    cpu_usage: float = Field(..., description="CPU使用率(%)")
    memory_usage: float = Field(..., description="内存使用率(%)")
    memory_total: int = Field(..., description="总内存(字节)")
    memory_used: int = Field(..., description="已用内存(字节)")
    disk_usage: float = Field(..., description="磁盘使用率(%)")
    disk_total: int = Field(..., description="总磁盘空间(字节)")
    disk_used: int = Field(..., description="已用磁盘空间(字节)")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="整体状态: healthy/warning/unhealthy")
    message: str = Field(..., description="状态描述信息")
    redis_status: str = Field(..., description="Redis连接状态")
    system_status: Optional[SystemStatus] = Field(None, description="系统资源状态")
    timestamp: float = Field(..., description="检查时间戳")