"""
API路由汇总  
"""
from fastapi import APIRouter
from app.api.v1 import data, prediction, training, health

api_router = APIRouter()

# 包含各个模块的路由
api_router.include_router(health.router, tags=["健康检查"])
api_router.include_router(data.router, tags=["数据管理"])
api_router.include_router(training.router, tags=["训练管理"])
api_router.include_router(prediction.router, tags=["预测服务"])