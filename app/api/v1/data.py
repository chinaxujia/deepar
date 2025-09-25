"""
数据管理API路由
"""
import os
import logging
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse

from app.schemas.data import (
    TrainingDataSubmission, DataSubmissionResponse, DataListResponse,
    DataStatistics, DataExportRequest, DataListItem
)
from app.services.data_service import DataProcessingService
from app.services.redis_service import RedisService
from app.utils.helpers import generate_id

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_redis_service() -> RedisService:
    """获取Redis服务依赖"""
    from app import app
    return app.state.redis


async def get_data_service() -> DataProcessingService:
    """获取数据处理服务依赖"""
    from app.services.data_service import data_processing_service
    return data_processing_service


@router.post("/data/submit", response_model=DataSubmissionResponse, summary="提交训练数据")
async def submit_training_data(
    data_submission: TrainingDataSubmission,
    data_service: DataProcessingService = Depends(get_data_service),
    redis: RedisService = Depends(get_redis_service)
):
    """
    提交训练数据
    
    功能：
    - 接收JSON格式的时间序列数据
    - 进行数据验证和质量检查
    - 转换为CSV格式存储
    - 保存元数据信息
    - 缓存提交状态
    """
    try:
        # 生成提交ID
        submission_id = generate_id("data_", 12)
        
        # 验证数据
        validation_result = data_service.validate_data(data_submission.data)
        
        if not validation_result.is_valid:
            return DataSubmissionResponse(
                submission_id=submission_id,
                status="failed",
                message="数据验证失败",
                device_id=data_submission.device_id,
                data_name=data_submission.data_name,
                validation_result=validation_result,
                timestamp=datetime.now()
            )
        
        # 保存为CSV格式
        csv_file_path, save_success = data_service.save_data_as_csv(
            submission_id, data_submission.data
        )
        
        if not save_success:
            raise HTTPException(status_code=500, detail="数据保存失败")
        
        # 保存元数据
        metadata = {
            "submission_id": submission_id,
            "device_id": data_submission.device_id,
            "data_name": data_submission.data_name,
            "description": data_submission.description,
            "total_points": len(data_submission.data),
            "submission_time": datetime.now().isoformat(),
            "status": "completed",
            "csv_file_path": csv_file_path,
            "validation_result": validation_result.dict()
        }
        
        metadata_saved = data_service.save_metadata(submission_id, metadata)
        if not metadata_saved:
            logger.warning(f"元数据保存失败: {submission_id}")
        
        # 缓存提交状态到Redis
        await redis.set(f"data_submission:{submission_id}", {
            "status": "completed",
            "device_id": data_submission.device_id,
            "data_name": data_submission.data_name,
            "timestamp": datetime.now().isoformat()
        }, ttl=3600)
        
        logger.info(f"数据提交成功: {submission_id}, 设备: {data_submission.device_id}")
        
        return DataSubmissionResponse(
            submission_id=submission_id,
            status="completed",
            message="数据提交成功",
            device_id=data_submission.device_id,
            data_name=data_submission.data_name,
            file_path=csv_file_path,
            validation_result=validation_result,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"提交训练数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"数据提交失败: {str(e)}")


@router.get("/data/list", response_model=DataListResponse, summary="获取数据集列表")
async def list_datasets(
    data_service: DataProcessingService = Depends(get_data_service)
):
    """
    获取所有数据集列表
    
    返回所有已提交的训练数据集信息，包括：
    - 提交ID和基本信息
    - 数据统计信息
    - 文件大小和状态
    """
    try:
        datasets = data_service.list_datasets()
        
        # 转换为响应模型
        items = []
        for dataset in datasets:
            item = DataListItem(
                submission_id=dataset['submission_id'],
                device_id=dataset['device_id'],
                data_name=dataset['data_name'],
                description=dataset['description'],
                total_points=dataset['total_points'],
                file_size=dataset['file_size'],
                submission_time=datetime.fromisoformat(dataset['submission_time']) if dataset['submission_time'] else datetime.now(),
                status=dataset['status']
            )
            items.append(item)
        
        return DataListResponse(
            total=len(items),
            items=items
        )
        
    except Exception as e:
        logger.error(f"获取数据集列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取数据集列表失败: {str(e)}")


@router.get("/data/statistics", response_model=DataStatistics, summary="获取数据统计信息")
async def get_data_statistics(
    data_service: DataProcessingService = Depends(get_data_service)
):
    """
    获取数据统计信息
    
    提供系统中所有数据的统计信息：
    - 总数据集数量和数据点数量
    - 存储使用情况
    - 设备数量统计
    - 最近提交的数据集
    """
    try:
        stats = data_service.get_statistics()
        
        # 转换最近提交数据格式
        recent_items = []
        for item in stats['recent_submissions']:
            recent_item = DataListItem(
                submission_id=item['submission_id'],
                device_id=item['device_id'],
                data_name=item['data_name'],
                description=item['description'],
                total_points=item['total_points'],
                file_size=item['file_size'],
                submission_time=datetime.fromisoformat(item['submission_time']) if item['submission_time'] else datetime.now(),
                status=item['status']
            )
            recent_items.append(recent_item)
        
        return DataStatistics(
            total_datasets=stats['total_datasets'],
            total_data_points=stats['total_data_points'],
            total_storage_size=stats['total_storage_size'],
            device_count=stats['device_count'],
            date_range=stats['date_range'],
            recent_submissions=recent_items
        )
        
    except Exception as e:
        logger.error(f"获取数据统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取数据统计失败: {str(e)}")


@router.get("/data/{submission_id}", summary="获取数据集详情")
async def get_dataset_details(
    submission_id: str,
    data_service: DataProcessingService = Depends(get_data_service)
):
    """
    获取特定数据集的详细信息
    
    包括元数据、验证结果、统计信息等
    """
    try:
        metadata = data_service.get_metadata(submission_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="数据集不存在")
        
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据集详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取数据集详情失败: {str(e)}")


@router.delete("/data/{submission_id}", summary="删除数据集")
async def delete_dataset(
    submission_id: str,
    data_service: DataProcessingService = Depends(get_data_service),
    redis: RedisService = Depends(get_redis_service)
):
    """
    删除指定的数据集
    
    将删除：
    - CSV数据文件
    - 元数据文件
    - Redis缓存
    """
    try:
        # 检查数据集是否存在
        metadata = data_service.get_metadata(submission_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="数据集不存在")
        
        # 删除数据集
        delete_success = data_service.delete_dataset(submission_id)
        if not delete_success:
            raise HTTPException(status_code=500, detail="删除数据集失败")
        
        # 删除Redis缓存
        await redis.delete(f"data_submission:{submission_id}")
        
        logger.info(f"数据集已删除: {submission_id}")
        
        return {
            "message": "数据集删除成功",
            "submission_id": submission_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除数据集失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除数据集失败: {str(e)}")


@router.post("/data/export", summary="导出数据集")
async def export_datasets(
    export_request: DataExportRequest,
    data_service: DataProcessingService = Depends(get_data_service)
):
    """
    导出多个数据集为单个文件
    
    支持CSV和JSON格式导出
    """
    try:
        export_path = data_service.export_datasets(
            export_request.submission_ids,
            export_request.format
        )
        
        if not export_path or not os.path.exists(export_path):
            raise HTTPException(status_code=500, detail="数据导出失败")
        
        # 返回文件下载
        filename = os.path.basename(export_path)
        return FileResponse(
            path=export_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出数据集失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"导出数据集失败: {str(e)}")


@router.get("/data/submission/{submission_id}/status", summary="获取提交状态")
async def get_submission_status(
    submission_id: str,
    redis: RedisService = Depends(get_redis_service)
):
    """
    从Redis缓存中获取数据提交状态
    """
    try:
        status = await redis.get(f"data_submission:{submission_id}")
        if not status:
            raise HTTPException(status_code=404, detail="提交记录不存在")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取提交状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取提交状态失败: {str(e)}")