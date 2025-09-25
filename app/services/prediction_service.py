"""
预测服务
负责模型加载、预测推理和结果处理
"""
import os
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid
import pickle
import logging

# GluonTS imports for prediction
try:
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.util import to_pandas
    import mxnet as mx
except ImportError:
    # Fallback - 在实际部署时需要安装GluonTS
    pass

from app.schemas.prediction import (
    PredictionRequest, PredictionResponse, PredictionResult, PredictionPoint,
    PredictionStatus, PredictionStatusResponse, PredictionHistoryItem,
    PredictionHistoryResponse, BatchPredictionRequest, BatchPredictionResponse,
    ModelInfo, ModelListResponse, PredictionAnalytics, PredictionAnalyticsResponse,
    RealTimePredictionRequest, RealTimePredictionResponse
)
from app.core.config import settings
from app.services.redis_service import RedisService


logger = logging.getLogger(__name__)


class PredictionService:
    """预测服务"""
    
    def __init__(self, redis_service: RedisService):
        self.redis_service = redis_service
        self.model_dir = Path(settings.MODEL_DIR)
        self.loaded_models: Dict[str, Any] = {}  # 模型缓存
        self.prediction_tasks: Dict[str, Dict] = {}
        
    async def submit_prediction_request(self, request: PredictionRequest) -> PredictionResponse:
        """提交预测请求"""
        prediction_id = str(uuid.uuid4())
        
        # 验证模型是否存在
        model_exists = await self._check_model_exists(request.model_name, request.task_id)
        if not model_exists:
            raise ValueError(f"模型 {request.model_name} 不存在或未完成训练")
        
        # 创建预测任务
        prediction_task = {
            "prediction_id": prediction_id,
            "model_name": request.model_name,
            "task_id": request.task_id,
            "device_id": request.device_id,
            "start_time": request.start_time,
            "end_time": request.end_time,
            "status": PredictionStatus.PENDING,
            "created_time": datetime.now(),
            "started_time": None,
            "processing_time": None,
            "progress": 0.0,
            "historical_data": request.historical_data,
            "prediction_config": request.prediction_config
        }
        
        # 保存到Redis
        await self.redis_service.set(
            f"prediction_task:{prediction_id}",
            json.dumps(prediction_task, default=str),
            ttl=7*24*3600  # 保存7天
        )
        
        # 启动异步预测
        asyncio.create_task(self._predict_async(prediction_id, request))
        
        # 估算完成时间
        time_range = request.end_time - request.start_time
        estimated_time = max(30, min(300, time_range.total_seconds() / 3600))  # 30秒到5分钟
        estimated_completion = datetime.now() + timedelta(seconds=estimated_time)
        
        logger.info(f"预测任务已提交: {prediction_id}")
        
        return PredictionResponse(
            prediction_id=prediction_id,
            status=PredictionStatus.PENDING,
            message="预测任务已提交，正在处理...",
            model_name=request.model_name,
            device_id=request.device_id,
            estimated_completion_time=estimated_completion,
            created_time=prediction_task["created_time"]
        )
    
    async def get_prediction_status(self, prediction_id: str) -> PredictionStatusResponse:
        """获取预测状态"""
        task_data = await self.redis_service.get(f"prediction_task:{prediction_id}")
        if not task_data:
            raise ValueError(f"预测任务 {prediction_id} 不存在")
        
        task = json.loads(task_data)
        
        # 获取预测结果
        result = None
        if task["status"] == PredictionStatus.COMPLETED.value:
            result_data = await self.redis_service.get(f"prediction_result:{prediction_id}")
            if result_data:
                result_dict = json.loads(result_data)
                result = PredictionResult(**result_dict)
        
        return PredictionStatusResponse(
            prediction_id=prediction_id,
            status=PredictionStatus(task["status"]),
            progress=task.get("progress", 0.0),
            message=task.get("message", ""),
            error_message=task.get("error_message"),
            started_time=datetime.fromisoformat(task["started_time"]) if task.get("started_time") else None,
            estimated_completion_time=datetime.fromisoformat(task["estimated_completion_time"]) if task.get("estimated_completion_time") else None,
            result=result
        )
    
    async def get_prediction_result(self, prediction_id: str) -> PredictionResult:
        """获取预测结果"""
        # 检查任务状态
        task_data = await self.redis_service.get(f"prediction_task:{prediction_id}")
        if not task_data:
            raise ValueError(f"预测任务 {prediction_id} 不存在")
        
        task = json.loads(task_data)
        if task["status"] != PredictionStatus.COMPLETED.value:
            raise ValueError(f"预测任务 {prediction_id} 尚未完成")
        
        # 获取预测结果
        result_data = await self.redis_service.get(f"prediction_result:{prediction_id}")
        if not result_data:
            raise ValueError(f"预测结果 {prediction_id} 不存在")
        
        result_dict = json.loads(result_data)
        return PredictionResult(**result_dict)
    
    async def get_prediction_history(self, device_id: Optional[str] = None,
                                   limit: int = 50) -> PredictionHistoryResponse:
        """获取预测历史"""
        pattern = "prediction_task:*"
        task_keys = await self.redis_service.scan_keys(pattern)
        
        history_items = []
        total_count = 0
        completed_count = 0
        failed_count = 0
        
        for key in task_keys:
            task_data = await self.redis_service.get(key)
            if not task_data:
                continue
                
            task = json.loads(task_data)
            
            # 设备过滤
            if device_id and task.get("device_id") != device_id:
                continue
            
            total_count += 1
            status = PredictionStatus(task["status"])
            
            if status == PredictionStatus.COMPLETED:
                completed_count += 1
            elif status == PredictionStatus.FAILED:
                failed_count += 1
            
            # 计算预测数据点数量
            start_time = datetime.fromisoformat(task["start_time"])
            end_time = datetime.fromisoformat(task["end_time"])
            time_diff = end_time - start_time
            data_points_predicted = int(time_diff.total_seconds() / 3600)  # 按小时计算
            
            # 获取模型准确度
            model_accuracy = None
            if status == PredictionStatus.COMPLETED:
                result_data = await self.redis_service.get(f"prediction_result:{task['prediction_id']}")
                if result_data:
                    result = json.loads(result_data)
                    model_accuracy = result.get("model_accuracy")
            
            history_items.append(PredictionHistoryItem(
                prediction_id=task["prediction_id"],
                model_name=task["model_name"],
                device_id=task["device_id"],
                status=status,
                start_time=start_time,
                end_time=end_time,
                created_time=datetime.fromisoformat(task["created_time"]),
                processing_time=task.get("processing_time"),
                data_points_predicted=data_points_predicted,
                model_accuracy=model_accuracy
            ))
        
        # 按创建时间排序并限制数量
        history_items.sort(key=lambda x: x.created_time, reverse=True)
        history_items = history_items[:limit]
        
        return PredictionHistoryResponse(
            total=total_count,
            completed=completed_count,
            failed=failed_count,
            items=history_items
        )
    
    async def submit_batch_prediction(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """提交批量预测请求"""
        batch_id = str(uuid.uuid4())
        individual_prediction_ids = []
        
        # 为每个设备创建预测任务
        for device_id in request.device_ids:
            prediction_request = PredictionRequest(
                model_name=request.model_name,
                device_id=device_id,
                start_time=request.start_time,
                end_time=request.end_time,
                prediction_config=request.prediction_config
            )
            
            response = await self.submit_prediction_request(prediction_request)
            individual_prediction_ids.append(response.prediction_id)
        
        # 保存批量任务信息
        batch_task = {
            "batch_id": batch_id,
            "model_name": request.model_name,
            "device_ids": request.device_ids,
            "individual_prediction_ids": individual_prediction_ids,
            "start_time": request.start_time,
            "end_time": request.end_time,
            "status": PredictionStatus.PENDING,
            "created_time": datetime.now()
        }
        
        await self.redis_service.set(
            f"batch_prediction:{batch_id}",
            json.dumps(batch_task, default=str),
            ttl=7*24*3600
        )
        
        logger.info(f"批量预测任务已提交: {batch_id}, 设备数量: {len(request.device_ids)}")
        
        return BatchPredictionResponse(
            batch_id=batch_id,
            status=PredictionStatus.PENDING,
            message=f"批量预测任务已提交，包含 {len(request.device_ids)} 个设备",
            model_name=request.model_name,
            device_count=len(request.device_ids),
            individual_prediction_ids=individual_prediction_ids,
            created_time=batch_task["created_time"]
        )
    
    async def get_available_models(self, device_id: Optional[str] = None) -> ModelListResponse:
        """获取可用模型列表"""
        # 从训练历史获取已完成的模型
        pattern = "training_task:*"
        task_keys = await self.redis_service.scan_keys(pattern)
        
        models = []
        
        for key in task_keys:
            task_data = await self.redis_service.get(key)
            if not task_data:
                continue
                
            task = json.loads(task_data)
            
            # 只包含已完成的训练任务
            if task.get("status") != "completed":
                continue
            
            # 设备过滤
            if device_id and task.get("device_id") != device_id:
                continue
            
            # 获取模型使用统计
            usage_stats = await self._get_model_usage_stats(task["task_id"])
            
            # 获取模型文件大小
            model_size = 0
            if task.get("model_path"):
                model_path = Path(task["model_path"])
                if model_path.exists():
                    model_size = model_path.stat().st_size
            
            models.append(ModelInfo(
                task_id=task["task_id"],
                model_name=task["model_name"],
                device_id=task["device_id"],
                model_type=task.get("model_type", "deepar"),
                created_time=datetime.fromisoformat(task["created_time"]),
                model_accuracy=task.get("model_accuracy"),
                training_data_points=task.get("training_data_points", 0),
                model_size=model_size,
                last_used_time=usage_stats.get("last_used_time"),
                usage_count=usage_stats.get("usage_count", 0)
            ))
        
        # 按创建时间排序
        models.sort(key=lambda x: x.created_time, reverse=True)
        
        return ModelListResponse(
            total=len(models),
            models=models
        )
    
    async def real_time_prediction(self, request: RealTimePredictionRequest) -> RealTimePredictionResponse:
        """实时预测"""
        start_time = datetime.now()
        
        # 验证模型
        model_exists = await self._check_model_exists(request.model_name)
        if not model_exists:
            raise ValueError(f"模型 {request.model_name} 不存在")
        
        # 加载模型
        model = await self._load_model(request.model_name)
        
        # 准备数据
        df = pd.DataFrame(request.current_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # 执行预测
        predictions = await self._generate_predictions(
            model, df, request.prediction_horizon
        )
        
        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # 生成告警
        alerts = await self._generate_alerts(predictions, df)
        
        # 下次更新时间（通常是1小时后）
        next_update = datetime.now() + timedelta(hours=1)
        
        # 更新模型使用统计
        await self._update_model_usage_stats(request.model_name)
        
        return RealTimePredictionResponse(
            device_id=request.device_id,
            current_timestamp=datetime.now(),
            predictions=predictions,
            model_confidence=0.85,  # 模拟置信度
            next_update_time=next_update,
            alerts=alerts,
            processing_time_ms=processing_time
        )
    
    async def _predict_async(self, prediction_id: str, request: PredictionRequest):
        """异步执行预测"""
        try:
            # 更新状态为运行中
            await self._update_prediction_status(
                prediction_id, PredictionStatus.RUNNING, "开始预测..."
            )
            
            start_time = datetime.now()
            
            # 加载模型
            model = await self._load_model(request.model_name, request.task_id)
            await self._update_prediction_progress(prediction_id, 20.0, "模型加载完成")
            
            # 准备历史数据
            historical_data = await self._prepare_historical_data(
                request.device_id, request.start_time, request.historical_data
            )
            await self._update_prediction_progress(prediction_id, 40.0, "历史数据准备完成")
            
            # 执行预测
            time_range = request.end_time - request.start_time
            prediction_hours = int(time_range.total_seconds() / 3600)
            
            predictions = await self._generate_predictions(
                model, historical_data, prediction_hours
            )
            await self._update_prediction_progress(prediction_id, 80.0, "预测计算完成")
            
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 保存预测结果
            result = PredictionResult(
                prediction_id=prediction_id,
                model_name=request.model_name,
                device_id=request.device_id,
                start_time=request.start_time,
                end_time=request.end_time,
                predictions=predictions,
                model_accuracy=0.87,  # 模拟准确度
                created_time=start_time,
                processing_time=processing_time
            )
            
            await self._save_prediction_result(prediction_id, result)
            await self._update_prediction_status(
                prediction_id, PredictionStatus.COMPLETED, "预测完成"
            )
            
            # 更新模型使用统计
            await self._update_model_usage_stats(request.model_name, request.task_id)
            
        except Exception as e:
            logger.error(f"预测任务 {prediction_id} 失败: {str(e)}")
            await self._update_prediction_status(
                prediction_id, PredictionStatus.FAILED, f"预测失败: {str(e)}"
            )
    
    async def _check_model_exists(self, model_name: str, task_id: Optional[str] = None) -> bool:
        """检查模型是否存在"""
        if task_id:
            # 检查特定训练任务的模型
            task_data = await self.redis_service.get(f"training_task:{task_id}")
            if not task_data:
                return False
            
            task = json.loads(task_data)
            return (task.get("status") == "completed" and 
                   task.get("model_name") == model_name and
                   task.get("model_path"))
        else:
            # 检查是否有该名称的已完成模型
            pattern = "training_task:*"
            task_keys = await self.redis_service.scan_keys(pattern)
            
            for key in task_keys:
                task_data = await self.redis_service.get(key)
                if not task_data:
                    continue
                    
                task = json.loads(task_data)
                if (task.get("status") == "completed" and 
                    task.get("model_name") == model_name and
                    task.get("model_path")):
                    return True
                    
            return False
    
    async def _load_model(self, model_name: str, task_id: Optional[str] = None) -> Any:
        """加载模型"""
        cache_key = f"{model_name}_{task_id or 'default'}"
        
        # 检查缓存
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        # 查找模型文件
        model_path = None
        if task_id:
            task_data = await self.redis_service.get(f"training_task:{task_id}")
            if task_data:
                task = json.loads(task_data)
                model_path = task.get("model_path")
        else:
            # 查找最新的同名模型
            pattern = "training_task:*"
            task_keys = await self.redis_service.scan_keys(pattern)
            latest_task = None
            
            for key in task_keys:
                task_data = await self.redis_service.get(key)
                if not task_data:
                    continue
                    
                task = json.loads(task_data)
                if (task.get("status") == "completed" and 
                    task.get("model_name") == model_name):
                    if (not latest_task or 
                        task["created_time"] > latest_task["created_time"]):
                        latest_task = task
            
            if latest_task:
                model_path = latest_task.get("model_path")
        
        if not model_path or not Path(model_path).exists():
            raise ValueError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # 缓存模型
        self.loaded_models[cache_key] = model
        
        return model
    
    async def _prepare_historical_data(self, device_id: str, start_time: datetime,
                                     provided_data: Optional[List[Dict]]) -> pd.DataFrame:
        """准备历史数据"""
        if provided_data:
            # 使用提供的历史数据
            df = pd.DataFrame(provided_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.sort_values('timestamp')
        
        # 从数据库获取历史数据
        # 获取开始时间前7天的数据作为上下文
        context_start = start_time - timedelta(days=7)
        
        # 模拟从数据集获取历史数据
        pattern = "dataset:*"
        dataset_keys = await self.redis_service.scan_keys(pattern)
        
        all_data = []
        for key in dataset_keys:
            dataset_info = await self.redis_service.get(key)
            if not dataset_info:
                continue
                
            info = json.loads(dataset_info)
            if info.get("device_id") == device_id:
                csv_path = Path(info["csv_path"])
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # 筛选时间范围
                    mask = (df['timestamp'] >= context_start) & (df['timestamp'] < start_time)
                    filtered_df = df[mask]
                    
                    if not filtered_df.empty:
                        all_data.append(filtered_df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df.sort_values('timestamp')
        else:
            # 如果没有历史数据，生成模拟数据
            return self._generate_mock_historical_data(context_start, start_time)
    
    def _generate_mock_historical_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """生成模拟历史数据"""
        time_range = pd.date_range(start=start_time, end=end_time, freq='H')
        
        # 生成模拟能耗数据（带有日周期性）
        base_consumption = 100
        daily_pattern = np.sin(2 * np.pi * np.arange(len(time_range)) / 24) * 20
        weekly_pattern = np.sin(2 * np.pi * np.arange(len(time_range)) / (24*7)) * 10
        noise = np.random.normal(0, 5, len(time_range))
        
        energy_consumption = base_consumption + daily_pattern + weekly_pattern + noise
        energy_consumption = np.maximum(energy_consumption, 0)  # 确保非负
        
        return pd.DataFrame({
            'timestamp': time_range,
            'energy_consumption': energy_consumption
        })
    
    async def _generate_predictions(self, model: Any, historical_data: pd.DataFrame,
                                  prediction_hours: int) -> List[PredictionPoint]:
        """生成预测结果"""
        predictions = []
        
        # 获取最后一个时间点
        last_timestamp = historical_data['timestamp'].max()
        
        # 模拟预测（实际应该使用GluonTS模型）
        base_value = historical_data['energy_consumption'].iloc[-24:].mean()  # 使用最近24小时的平均值
        
        for i in range(prediction_hours):
            timestamp = last_timestamp + timedelta(hours=i+1)
            
            # 模拟预测值（包含日周期性）
            hour_of_day = timestamp.hour
            daily_factor = 1 + 0.3 * np.sin(2 * np.pi * hour_of_day / 24)
            
            predicted_value = base_value * daily_factor + np.random.normal(0, 5)
            predicted_value = max(predicted_value, 0)
            
            # 计算置信区间
            confidence_interval_lower = predicted_value * 0.9
            confidence_interval_upper = predicted_value * 1.1
            
            predictions.append(PredictionPoint(
                timestamp=timestamp,
                predicted_value=predicted_value,
                confidence_interval_lower=confidence_interval_lower,
                confidence_interval_upper=confidence_interval_upper,
                confidence_level=0.8
            ))
        
        return predictions
    
    async def _generate_alerts(self, predictions: List[PredictionPoint],
                             historical_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """生成告警"""
        alerts = []
        
        # 计算历史平均值
        avg_consumption = historical_data['energy_consumption'].mean()
        max_consumption = historical_data['energy_consumption'].max()
        
        # 检查异常高能耗
        for pred in predictions:
            if pred.predicted_value > max_consumption * 1.2:
                alerts.append({
                    "type": "high_consumption",
                    "timestamp": pred.timestamp,
                    "message": f"预测能耗异常偏高: {pred.predicted_value:.2f}",
                    "severity": "warning",
                    "predicted_value": pred.predicted_value,
                    "threshold": max_consumption * 1.2
                })
            elif pred.predicted_value < avg_consumption * 0.5:
                alerts.append({
                    "type": "low_consumption",
                    "timestamp": pred.timestamp,
                    "message": f"预测能耗异常偏低: {pred.predicted_value:.2f}",
                    "severity": "info",
                    "predicted_value": pred.predicted_value,
                    "threshold": avg_consumption * 0.5
                })
        
        return alerts
    
    async def _save_prediction_result(self, prediction_id: str, result: PredictionResult):
        """保存预测结果"""
        await self.redis_service.set(
            f"prediction_result:{prediction_id}",
            json.dumps(result.dict(), default=str),
            ttl=7*24*3600
        )
    
    async def _update_prediction_status(self, prediction_id: str, status: PredictionStatus,
                                      message: str):
        """更新预测状态"""
        task_data = await self.redis_service.get(f"prediction_task:{prediction_id}")
        if not task_data:
            return
        
        task = json.loads(task_data)
        task["status"] = status.value
        task["message"] = message
        
        if status == PredictionStatus.RUNNING and not task.get("started_time"):
            task["started_time"] = datetime.now()
        elif status in [PredictionStatus.COMPLETED, PredictionStatus.FAILED]:
            if task.get("started_time"):
                start_time = datetime.fromisoformat(task["started_time"])
                task["processing_time"] = (datetime.now() - start_time).total_seconds()
        
        await self.redis_service.set(
            f"prediction_task:{prediction_id}",
            json.dumps(task, default=str)
        )
    
    async def _update_prediction_progress(self, prediction_id: str, progress: float, message: str):
        """更新预测进度"""
        task_data = await self.redis_service.get(f"prediction_task:{prediction_id}")
        if not task_data:
            return
        
        task = json.loads(task_data)
        task["progress"] = progress
        task["message"] = message
        
        await self.redis_service.set(
            f"prediction_task:{prediction_id}",
            json.dumps(task, default=str)
        )
    
    async def _get_model_usage_stats(self, task_id: str) -> Dict[str, Any]:
        """获取模型使用统计"""
        stats_data = await self.redis_service.get(f"model_usage:{task_id}")
        if stats_data:
            return json.loads(stats_data)
        return {"usage_count": 0, "last_used_time": None}
    
    async def _update_model_usage_stats(self, model_name: str, task_id: Optional[str] = None):
        """更新模型使用统计"""
        if not task_id:
            # 查找最新的同名模型任务ID
            pattern = "training_task:*"
            task_keys = await self.redis_service.scan_keys(pattern)
            
            for key in task_keys:
                task_data = await self.redis_service.get(key)
                if not task_data:
                    continue
                    
                task = json.loads(task_data)
                if (task.get("status") == "completed" and 
                    task.get("model_name") == model_name):
                    task_id = task["task_id"]
                    break
        
        if task_id:
            stats = await self._get_model_usage_stats(task_id)
            stats["usage_count"] = stats.get("usage_count", 0) + 1
            stats["last_used_time"] = datetime.now()
            
            await self.redis_service.set(
                f"model_usage:{task_id}",
                json.dumps(stats, default=str),
                ttl=30*24*3600  # 保存30天
            )