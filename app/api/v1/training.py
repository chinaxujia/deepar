"""
模型训练相关的API路由
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from app.schemas.training import (
    TrainingRequest, TrainingResponse, TrainingStatusResponse,
    TrainingHistoryResponse, TrainingResultResponse
)
from app.services.training_service import DeepARTrainingService
from app.services.redis_service import RedisService
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["训练管理"])

# 依赖注入
def get_redis_service() -> RedisService:
    return RedisService()

def get_training_service(redis_service: RedisService = Depends(get_redis_service)) -> DeepARTrainingService:
    return DeepARTrainingService(redis_service)


@router.post("/submit", response_model=TrainingResponse, summary="提交训练任务")
async def submit_training_task(
    request: TrainingRequest,
    training_service: DeepARTrainingService = Depends(get_training_service)
):
    """
    提交一个新的模型训练任务
    
    - **model_name**: 模型名称，用于标识和管理
    - **data_submission_ids**: 用于训练的数据集ID列表
    - **device_id**: 目标设备ID
    - **config**: 训练配置参数，包括学习率、批次大小等
    - **description**: 可选的训练描述
    
    返回训练任务ID和状态信息
    """
    try:
        response = await training_service.submit_training_task(request)
        logger.info(f"训练任务提交成功: {response.task_id}")
        return response
        
    except ValueError as e:
        logger.error(f"训练任务提交失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"训练服务异常: {str(e)}")
        raise HTTPException(status_code=500, detail="内部服务错误")


@router.get("/status/{task_id}", response_model=TrainingStatusResponse, summary="获取训练状态")
async def get_training_status(
    task_id: str,
    training_service: DeepARTrainingService = Depends(get_training_service)
):
    """
    获取指定训练任务的当前状态
    
    返回详细的训练进度信息，包括：
    - 当前状态（等待中、运行中、已完成、失败等）
    - 训练进度百分比
    - 当前轮次和总轮次
    - 最新的训练指标
    - 错误信息（如果失败）
    - 最近的日志信息
    """
    try:
        status = await training_service.get_training_status(task_id)
        return status
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"获取训练状态异常: {str(e)}")
        raise HTTPException(status_code=500, detail="内部服务错误")


@router.post("/cancel/{task_id}", summary="取消训练任务")
async def cancel_training_task(
    task_id: str,
    training_service: DeepARTrainingService = Depends(get_training_service)
):
    """
    取消指定的训练任务
    
    只能取消状态为"等待中"或"运行中"的任务
    已完成、失败或已取消的任务无法取消
    """
    try:
        success = await training_service.cancel_training(task_id)
        if success:
            return {"message": f"训练任务 {task_id} 已取消"}
        else:
            raise HTTPException(status_code=400, detail="任务无法取消（可能已完成或不存在）")
            
    except Exception as e:
        logger.error(f"取消训练任务异常: {str(e)}")
        raise HTTPException(status_code=500, detail="内部服务错误")


@router.get("/history", response_model=TrainingHistoryResponse, summary="获取训练历史")
async def get_training_history(
    device_id: Optional[str] = Query(None, description="设备ID过滤"),
    limit: int = Query(50, description="返回数量限制", ge=1, le=200),
    training_service: DeepARTrainingService = Depends(get_training_service)
):
    """
    获取训练历史记录
    
    - **device_id**: 可选，按设备ID过滤
    - **limit**: 返回的记录数量限制（1-200）
    
    返回训练历史统计和详细列表，包括：
    - 总任务数、完成数、失败数统计
    - 每个任务的基本信息和最终状态
    - 训练时长和性能指标
    """
    try:
        history = await training_service.get_training_history(device_id, limit)
        return history
        
    except Exception as e:
        logger.error(f"获取训练历史异常: {str(e)}")
        raise HTTPException(status_code=500, detail="内部服务错误")


@router.get("/result/{task_id}", response_model=TrainingResultResponse, summary="获取训练结果")
async def get_training_result(
    task_id: str,
    training_service: DeepARTrainingService = Depends(get_training_service)
):
    """
    获取已完成训练任务的详细结果
    
    返回完整的训练结果，包括：
    - 模型性能指标（MAPE、RMSE、MAE、R²等）
    - 训练过程历史数据
    - 模型文件信息
    - 评估结果和预测准确度
    
    只有状态为"已完成"的任务才能获取结果
    """
    try:
        result = await training_service.get_training_result(task_id)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"获取训练结果异常: {str(e)}")
        raise HTTPException(status_code=500, detail="内部服务错误")


@router.get("/models", summary="获取可用模型列表")
async def list_trained_models(
    device_id: Optional[str] = Query(None, description="设备ID过滤"),
    training_service: DeepARTrainingService = Depends(get_training_service)
):
    """
    获取所有已训练完成的模型列表
    
    - **device_id**: 可选，按设备ID过滤
    
    返回模型基本信息列表，用于预测服务选择模型
    """
    try:
        # 获取已完成的训练历史
        history = await training_service.get_training_history(device_id, limit=200)
        
        # 筛选已完成的任务
        completed_models = []
        for item in history.items:
            if item.status.value == "completed":
                completed_models.append({
                    "task_id": item.task_id,
                    "model_name": item.model_name,
                    "device_id": item.device_id,
                    "created_time": item.created_time,
                    "duration": item.duration,
                    "final_loss": item.final_loss,
                    "data_points_used": item.data_points_used
                })
        
        return {
            "total": len(completed_models),
            "models": completed_models
        }
        
    except Exception as e:
        logger.error(f"获取模型列表异常: {str(e)}")
        raise HTTPException(status_code=500, detail="内部服务错误")


@router.delete("/model/{task_id}", summary="删除训练模型")
async def delete_trained_model(
    task_id: str,
    training_service: DeepARTrainingService = Depends(get_training_service)
):
    """
    删除指定的训练模型和相关文件
    
    将删除：
    - 模型文件
    - 训练结果文件
    - Redis中的任务记录
    
    注意：此操作不可恢复
    """
    try:
        # 获取任务信息
        task_data = await training_service.redis_service.get(f"training_task:{task_id}")
        if not task_data:
            raise HTTPException(status_code=404, detail="训练任务不存在")
        
        import json
        import os
        from pathlib import Path
        
        task = json.loads(task_data)
        
        # 删除模型文件
        if task.get("model_path"):
            model_path = Path(task["model_path"])
            if model_path.exists():
                os.remove(model_path)
        
        # 删除结果文件
        result_path = Path(settings.MODEL_DIR) / f"{task_id}_result.json"
        if result_path.exists():
            os.remove(result_path)
        
        # 删除Redis记录
        await training_service.redis_service.delete(f"training_task:{task_id}")
        
        logger.info(f"训练模型已删除: {task_id}")
        return {"message": f"训练模型 {task_id} 已删除"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除训练模型异常: {str(e)}")
        raise HTTPException(status_code=500, detail="内部服务错误")


@router.get("/config/default", summary="获取默认训练配置")
async def get_default_training_config():
    """
    获取默认的训练配置参数
    
    返回推荐的训练配置，用户可以基于此配置进行调整
    """
    from app.schemas.training import TrainingConfig
    
    default_config = TrainingConfig()
    return {
        "config": default_config.dict(),
        "description": {
            "prediction_length": "预测长度（小时），通常设置为24小时（1天）",
            "context_length": "上下文长度（小时），用于学习历史模式，推荐7天",
            "freq": "时间频率，H表示小时级别",
            "epochs": "训练轮数，更多轮数可能提高精度但增加训练时间",
            "learning_rate": "学习率，控制训练速度和稳定性",
            "batch_size": "批次大小，影响训练速度和内存使用",
            "num_layers": "LSTM层数，更多层可能提高模型复杂度",
            "hidden_size": "隐藏层大小，影响模型容量",
            "dropout_rate": "Dropout率，防止过拟合",
            "early_stopping_patience": "早停耐心值，防止过度训练"
        }
    }