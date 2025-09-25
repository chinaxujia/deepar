"""
数据管理相关的数据模型
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class TimeSeriesDataPoint(BaseModel):
    """时间序列数据点"""
    timestamp: str = Field(..., description="时间戳，格式：YYYY-MM-DD HH:MM:SS")
    energy_consumption: float = Field(..., description="能耗值")
    temperature: Optional[float] = Field(None, description="温度")
    humidity: Optional[float] = Field(None, description="湿度")
    production_load: Optional[float] = Field(None, description="生产负荷")
    
    @validator('energy_consumption')
    def validate_energy_consumption(cls, v):
        if v < 0:
            raise ValueError('能耗值不能为负数')
        if v > 50000:  # 假设最大能耗值
            raise ValueError('能耗值超出合理范围')
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None:
            if v < -50 or v > 100:
                raise ValueError('温度值超出合理范围(-50°C 到 100°C)')
        return v
    
    @validator('humidity')
    def validate_humidity(cls, v):
        if v is not None:
            if v < 0 or v > 100:
                raise ValueError('湿度值必须在0-100之间')
        return v
    
    @validator('production_load')
    def validate_production_load(cls, v):
        if v is not None:
            if v < 0 or v > 100:
                raise ValueError('生产负荷必须在0-100之间')
        return v


class TrainingDataSubmission(BaseModel):
    """训练数据提交请求"""
    device_id: str = Field(..., description="设备ID")
    data_name: str = Field(..., description="数据集名称")
    description: Optional[str] = Field(None, description="数据描述")
    data: List[TimeSeriesDataPoint] = Field(..., description="时间序列数据点列表")
    
    @validator('data')
    def validate_data_length(cls, v):
        if len(v) < 24:  # 至少需要24个小时的数据
            raise ValueError('训练数据至少需要24个数据点')
        if len(v) > 10000:  # 限制最大数据量
            raise ValueError('单次提交数据点不能超过10000个')
        return v


class DataValidationResult(BaseModel):
    """数据验证结果"""
    is_valid: bool = Field(..., description="数据是否有效")
    total_points: int = Field(..., description="总数据点数")
    valid_points: int = Field(..., description="有效数据点数")
    invalid_points: int = Field(..., description="无效数据点数")
    missing_values: int = Field(..., description="缺失值数量")
    errors: List[str] = Field(default=[], description="错误信息列表")
    warnings: List[str] = Field(default=[], description="警告信息列表")
    statistics: Optional[Dict[str, Any]] = Field(None, description="数据统计信息")


class DataSubmissionResponse(BaseModel):
    """数据提交响应"""
    submission_id: str = Field(..., description="提交ID")
    status: str = Field(..., description="提交状态")
    message: str = Field(..., description="响应消息")
    device_id: str = Field(..., description="设备ID")
    data_name: str = Field(..., description="数据集名称")
    file_path: Optional[str] = Field(None, description="保存的文件路径")
    validation_result: DataValidationResult = Field(..., description="数据验证结果")
    timestamp: datetime = Field(default_factory=datetime.now, description="提交时间")


class DataListItem(BaseModel):
    """数据列表项"""
    submission_id: str = Field(..., description="提交ID")
    device_id: str = Field(..., description="设备ID")
    data_name: str = Field(..., description="数据集名称")
    description: Optional[str] = Field(None, description="数据描述")
    total_points: int = Field(..., description="数据点总数")
    file_size: int = Field(..., description="文件大小(字节)")
    submission_time: datetime = Field(..., description="提交时间")
    status: str = Field(..., description="数据状态")


class DataListResponse(BaseModel):
    """数据列表响应"""
    total: int = Field(..., description="总数据集数量")
    items: List[DataListItem] = Field(..., description="数据集列表")


class DataStatistics(BaseModel):
    """数据统计信息"""
    total_datasets: int = Field(..., description="总数据集数量")
    total_data_points: int = Field(..., description="总数据点数量")
    total_storage_size: int = Field(..., description="总存储大小(字节)")
    device_count: int = Field(..., description="设备数量")
    date_range: Dict[str, str] = Field(..., description="数据时间范围")
    recent_submissions: List[DataListItem] = Field(..., description="最近提交的数据集")


class DataExportRequest(BaseModel):
    """数据导出请求"""
    submission_ids: List[str] = Field(..., description="要导出的提交ID列表")
    format: str = Field(default="csv", description="导出格式: csv, json")
    include_metadata: bool = Field(default=True, description="是否包含元数据")