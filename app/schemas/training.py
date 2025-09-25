"""
模型训练相关的数据模型
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class TrainingStatus(str, Enum):
    """训练状态枚举"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelType(str, Enum):
    """模型类型枚举"""
    DEEPAR = "deepar"


class TrainingConfig(BaseModel):
    """训练配置"""
    prediction_length: int = Field(default=24, description="预测长度（小时）", ge=1, le=168)
    context_length: int = Field(default=168, description="上下文长度（小时）", ge=24, le=720)
    freq: str = Field(default="H", description="时间频率: H(小时), D(天), W(周)")
    epochs: int = Field(default=100, description="训练轮数", ge=10, le=1000)
    learning_rate: float = Field(default=0.001, description="学习率", gt=0, le=1)
    batch_size: int = Field(default=32, description="批次大小", ge=8, le=256)
    num_layers: int = Field(default=2, description="LSTM层数", ge=1, le=5)
    hidden_size: int = Field(default=40, description="隐藏层大小", ge=10, le=200)
    dropout_rate: float = Field(default=0.1, description="Dropout率", ge=0, le=0.5)
    early_stopping_patience: int = Field(default=10, description="早停耐心值", ge=5, le=50)
    
    @validator('freq')
    def validate_freq(cls, v):
        allowed_freqs = ['H', 'D', 'W', 'M']
        if v not in allowed_freqs:
            raise ValueError(f'频率必须是以下之一: {allowed_freqs}')
        return v


class TrainingRequest(BaseModel):
    """训练请求"""
    model_name: str = Field(..., description="模型名称", min_length=1, max_length=100)
    model_type: ModelType = Field(default=ModelType.DEEPAR, description="模型类型")
    data_submission_ids: List[str] = Field(..., description="训练数据提交ID列表", min_items=1)
    device_id: str = Field(..., description="目标设备ID")
    description: Optional[str] = Field(None, description="训练描述", max_length=500)
    config: TrainingConfig = Field(default_factory=TrainingConfig, description="训练配置")
    
    @validator('data_submission_ids')
    def validate_data_ids(cls, v):
        if len(v) > 10:  # 限制最大数据集数量
            raise ValueError('单次训练最多支持10个数据集')
        return v


class TrainingMetrics(BaseModel):
    """训练指标"""
    epoch: int = Field(..., description="当前轮数")
    train_loss: float = Field(..., description="训练损失")
    val_loss: Optional[float] = Field(None, description="验证损失")
    learning_rate: float = Field(..., description="当前学习率")
    time_elapsed: float = Field(..., description="已用时间(秒)")
    estimated_time_remaining: Optional[float] = Field(None, description="预计剩余时间(秒)")


class TrainingResponse(BaseModel):
    """训练响应"""
    task_id: str = Field(..., description="训练任务ID")
    model_name: str = Field(..., description="模型名称")
    status: TrainingStatus = Field(..., description="训练状态")
    message: str = Field(..., description="响应消息")
    device_id: str = Field(..., description="设备ID")
    data_submission_ids: List[str] = Field(..., description="使用的数据集ID")
    config: TrainingConfig = Field(..., description="训练配置")
    created_time: datetime = Field(default_factory=datetime.now, description="创建时间")
    started_time: Optional[datetime] = Field(None, description="开始时间")
    completed_time: Optional[datetime] = Field(None, description="完成时间")
    current_metrics: Optional[TrainingMetrics] = Field(None, description="当前训练指标")
    model_path: Optional[str] = Field(None, description="模型保存路径")


class TrainingStatusResponse(BaseModel):
    """训练状态响应"""
    task_id: str = Field(..., description="训练任务ID")
    status: TrainingStatus = Field(..., description="训练状态")
    progress: float = Field(..., description="训练进度(0-100)", ge=0, le=100)
    current_epoch: int = Field(..., description="当前轮数")
    total_epochs: int = Field(..., description="总轮数")
    metrics: Optional[TrainingMetrics] = Field(None, description="训练指标")
    error_message: Optional[str] = Field(None, description="错误信息")
    logs: List[str] = Field(default=[], description="最近的日志信息")


class TrainingHistoryItem(BaseModel):
    """训练历史项"""
    task_id: str = Field(..., description="训练任务ID")
    model_name: str = Field(..., description="模型名称")
    device_id: str = Field(..., description="设备ID")
    status: TrainingStatus = Field(..., description="最终状态")
    created_time: datetime = Field(..., description="创建时间")
    duration: Optional[float] = Field(None, description="训练时长(秒)")
    final_loss: Optional[float] = Field(None, description="最终损失值")
    data_points_used: int = Field(..., description="使用的数据点数量")


class TrainingHistoryResponse(BaseModel):
    """训练历史响应"""
    total: int = Field(..., description="总任务数量")
    completed: int = Field(..., description="已完成任务数量")
    failed: int = Field(..., description="失败任务数量")
    items: List[TrainingHistoryItem] = Field(..., description="训练历史列表")


class ModelEvaluationMetrics(BaseModel):
    """模型评估指标"""
    mape: float = Field(..., description="平均绝对百分比误差")
    rmse: float = Field(..., description="均方根误差")
    mae: float = Field(..., description="平均绝对误差")
    r2_score: float = Field(..., description="R²决定系数")
    training_data_points: int = Field(..., description="训练数据点数量")
    validation_data_points: int = Field(..., description="验证数据点数量")
    prediction_accuracy: Dict[str, float] = Field(..., description="不同时间段的预测准确度")


class TrainingResultResponse(BaseModel):
    """训练结果响应"""
    task_id: str = Field(..., description="训练任务ID")
    model_name: str = Field(..., description="模型名称")
    status: TrainingStatus = Field(..., description="训练状态")
    model_path: str = Field(..., description="模型保存路径")
    model_size: int = Field(..., description="模型文件大小(字节)")
    training_duration: float = Field(..., description="训练时长(秒)")
    total_epochs: int = Field(..., description="实际训练轮数")
    best_epoch: int = Field(..., description="最佳轮数")
    evaluation_metrics: ModelEvaluationMetrics = Field(..., description="评估指标")
    training_history: List[TrainingMetrics] = Field(..., description="训练历史指标")
    created_time: datetime = Field(..., description="创建时间")
    completed_time: datetime = Field(..., description="完成时间")