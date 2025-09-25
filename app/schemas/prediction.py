"""
预测服务相关的数据模型
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class PredictionStatus(str, Enum):
    """预测状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PredictionRequest(BaseModel):
    """预测请求"""
    model_name: str = Field(..., description="模型名称", min_length=1, max_length=100)
    task_id: Optional[str] = Field(None, description="训练任务ID（可选，用于直接使用训练任务的模型）")
    device_id: str = Field(..., description="设备ID")
    start_time: datetime = Field(..., description="预测开始时间")
    end_time: datetime = Field(..., description="预测结束时间")
    historical_data: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="历史数据（可选，如果不提供将从数据库获取）",
        max_items=10000
    )
    prediction_config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="预测配置参数"
    )
    
    @validator('end_time')
    def validate_time_range(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('结束时间必须晚于开始时间')
        
        # 限制预测时间范围不超过30天
        if 'start_time' in values:
            time_diff = v - values['start_time']
            if time_diff.days > 30:
                raise ValueError('预测时间范围不能超过30天')
        
        return v
    
    @validator('historical_data')
    def validate_historical_data(cls, v):
        if v is not None:
            for item in v:
                if 'timestamp' not in item or 'energy_consumption' not in item:
                    raise ValueError('历史数据必须包含timestamp和energy_consumption字段')
        return v


class PredictionPoint(BaseModel):
    """预测数据点"""
    timestamp: datetime = Field(..., description="时间戳")
    predicted_value: float = Field(..., description="预测值")
    confidence_interval_lower: float = Field(..., description="置信区间下限")
    confidence_interval_upper: float = Field(..., description="置信区间上限")
    confidence_level: float = Field(default=0.8, description="置信水平", ge=0.5, le=0.99)


class PredictionResult(BaseModel):
    """预测结果"""
    prediction_id: str = Field(..., description="预测任务ID")
    model_name: str = Field(..., description="使用的模型名称")
    device_id: str = Field(..., description="设备ID")
    start_time: datetime = Field(..., description="预测开始时间")
    end_time: datetime = Field(..., description="预测结束时间")
    predictions: List[PredictionPoint] = Field(..., description="预测结果数据点")
    model_accuracy: Optional[float] = Field(None, description="模型准确度", ge=0, le=1)
    created_time: datetime = Field(default_factory=datetime.now, description="预测创建时间")
    processing_time: float = Field(..., description="处理时间（秒）")


class PredictionResponse(BaseModel):
    """预测响应"""
    prediction_id: str = Field(..., description="预测任务ID")
    status: PredictionStatus = Field(..., description="预测状态")
    message: str = Field(..., description="响应消息")
    model_name: str = Field(..., description="使用的模型名称")
    device_id: str = Field(..., description="设备ID")
    estimated_completion_time: Optional[datetime] = Field(None, description="预计完成时间")
    created_time: datetime = Field(default_factory=datetime.now, description="创建时间")


class PredictionStatusResponse(BaseModel):
    """预测状态响应"""
    prediction_id: str = Field(..., description="预测任务ID")
    status: PredictionStatus = Field(..., description="预测状态")
    progress: float = Field(..., description="预测进度(0-100)", ge=0, le=100)
    message: str = Field(..., description="状态信息")
    error_message: Optional[str] = Field(None, description="错误信息")
    started_time: Optional[datetime] = Field(None, description="开始时间")
    estimated_completion_time: Optional[datetime] = Field(None, description="预计完成时间")
    result: Optional[PredictionResult] = Field(None, description="预测结果（完成时才有）")


class PredictionHistoryItem(BaseModel):
    """预测历史项"""
    prediction_id: str = Field(..., description="预测任务ID")
    model_name: str = Field(..., description="模型名称")
    device_id: str = Field(..., description="设备ID")
    status: PredictionStatus = Field(..., description="预测状态")
    start_time: datetime = Field(..., description="预测开始时间")
    end_time: datetime = Field(..., description="预测结束时间")
    created_time: datetime = Field(..., description="创建时间")
    processing_time: Optional[float] = Field(None, description="处理时间（秒）")
    data_points_predicted: int = Field(..., description="预测数据点数量")
    model_accuracy: Optional[float] = Field(None, description="模型准确度")


class PredictionHistoryResponse(BaseModel):
    """预测历史响应"""
    total: int = Field(..., description="总预测任务数量")
    completed: int = Field(..., description="已完成任务数量")
    failed: int = Field(..., description="失败任务数量")
    items: List[PredictionHistoryItem] = Field(..., description="预测历史列表")


class BatchPredictionRequest(BaseModel):
    """批量预测请求"""
    model_name: str = Field(..., description="模型名称")
    device_ids: List[str] = Field(..., description="设备ID列表", min_items=1, max_items=50)
    start_time: datetime = Field(..., description="预测开始时间")
    end_time: datetime = Field(..., description="预测结束时间")
    prediction_config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="预测配置参数"
    )
    
    @validator('device_ids')
    def validate_device_ids(cls, v):
        if len(set(v)) != len(v):
            raise ValueError('设备ID列表中不能有重复项')
        return v


class BatchPredictionResponse(BaseModel):
    """批量预测响应"""
    batch_id: str = Field(..., description="批量预测任务ID")
    status: PredictionStatus = Field(..., description="批量预测状态")
    message: str = Field(..., description="响应消息")
    model_name: str = Field(..., description="使用的模型名称")
    device_count: int = Field(..., description="设备数量")
    individual_prediction_ids: List[str] = Field(..., description="各设备预测任务ID列表")
    created_time: datetime = Field(default_factory=datetime.now, description="创建时间")


class ModelInfo(BaseModel):
    """模型信息"""
    task_id: str = Field(..., description="训练任务ID")
    model_name: str = Field(..., description="模型名称")
    device_id: str = Field(..., description="目标设备ID")
    model_type: str = Field(..., description="模型类型")
    created_time: datetime = Field(..., description="创建时间")
    model_accuracy: Optional[float] = Field(None, description="模型准确度")
    training_data_points: int = Field(..., description="训练数据点数量")
    model_size: int = Field(..., description="模型文件大小（字节）")
    last_used_time: Optional[datetime] = Field(None, description="最后使用时间")
    usage_count: int = Field(default=0, description="使用次数")


class ModelListResponse(BaseModel):
    """模型列表响应"""
    total: int = Field(..., description="总模型数量")
    models: List[ModelInfo] = Field(..., description="模型列表")


class PredictionAnalytics(BaseModel):
    """预测分析"""
    device_id: str = Field(..., description="设备ID")
    time_period: str = Field(..., description="时间段")
    total_predictions: int = Field(..., description="总预测次数")
    avg_energy_consumption: float = Field(..., description="平均能耗预测")
    peak_consumption: float = Field(..., description="峰值能耗预测")
    energy_trend: str = Field(..., description="能耗趋势（上升/下降/稳定）")
    anomaly_alerts: List[Dict[str, Any]] = Field(default=[], description="异常告警列表")
    efficiency_score: float = Field(..., description="效率分数（0-100）", ge=0, le=100)


class PredictionAnalyticsResponse(BaseModel):
    """预测分析响应"""
    total_devices: int = Field(..., description="总设备数量")
    analytics: List[PredictionAnalytics] = Field(..., description="分析结果列表")
    summary: Dict[str, Any] = Field(..., description="总体摘要")
    generated_time: datetime = Field(default_factory=datetime.now, description="生成时间")


class RealTimePredictionRequest(BaseModel):
    """实时预测请求"""
    model_name: str = Field(..., description="模型名称")
    device_id: str = Field(..., description="设备ID")
    current_data: List[Dict[str, Any]] = Field(
        ..., 
        description="当前实时数据", 
        min_items=1, 
        max_items=100
    )
    prediction_horizon: int = Field(
        default=24, 
        description="预测时间范围（小时）", 
        ge=1, 
        le=168
    )
    
    @validator('current_data')
    def validate_current_data(cls, v):
        for item in v:
            if 'timestamp' not in item or 'energy_consumption' not in item:
                raise ValueError('实时数据必须包含timestamp和energy_consumption字段')
        return v


class RealTimePredictionResponse(BaseModel):
    """实时预测响应"""
    device_id: str = Field(..., description="设备ID")
    current_timestamp: datetime = Field(default_factory=datetime.now, description="当前时间")
    predictions: List[PredictionPoint] = Field(..., description="预测结果")
    model_confidence: float = Field(..., description="模型置信度", ge=0, le=1)
    next_update_time: datetime = Field(..., description="下次更新时间")
    alerts: List[Dict[str, Any]] = Field(default=[], description="告警信息")
    processing_time_ms: float = Field(..., description="处理时间（毫秒）")