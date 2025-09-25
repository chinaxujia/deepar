"""
数据处理服务
"""
import os
import json
import csv
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from app.core.config import settings
from app.utils.helpers import generate_id, save_json, load_json, format_bytes
from app.schemas.data import TimeSeriesDataPoint, DataValidationResult
import logging

logger = logging.getLogger(__name__)


class DataProcessingService:
    """数据处理服务类"""
    
    def __init__(self):
        self.raw_data_path = os.path.join(settings.DATA_STORAGE_PATH, "raw")
        self.processed_data_path = os.path.join(settings.DATA_STORAGE_PATH, "processed")
        self.metadata_path = os.path.join(settings.DATA_STORAGE_PATH, "metadata")
        
        # 确保目录存在
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)
    
    def validate_data(self, data: List[TimeSeriesDataPoint]) -> DataValidationResult:
        """验证时间序列数据"""
        try:
            total_points = len(data)
            valid_points = 0
            invalid_points = 0
            missing_values = 0
            errors = []
            warnings = []
            
            # 检查数据完整性
            timestamps = []
            energy_values = []
            
            for i, point in enumerate(data):
                try:
                    # 验证时间戳格式
                    timestamp = datetime.strptime(point.timestamp, "%Y-%m-%d %H:%M:%S")
                    timestamps.append(timestamp)
                    
                    # 验证能耗值
                    if point.energy_consumption is None:
                        missing_values += 1
                        errors.append(f"数据点{i+1}: 能耗值缺失")
                        invalid_points += 1
                        continue
                    
                    energy_values.append(point.energy_consumption)
                    valid_points += 1
                    
                except ValueError as e:
                    errors.append(f"数据点{i+1}: {str(e)}")
                    invalid_points += 1
            
            # 检查时间序列连续性
            if len(timestamps) > 1:
                timestamps.sort()
                time_gaps = []
                for i in range(1, len(timestamps)):
                    gap = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # 小时
                    time_gaps.append(gap)
                
                # 检查是否有异常的时间间隔
                avg_gap = sum(time_gaps) / len(time_gaps)
                for i, gap in enumerate(time_gaps):
                    if abs(gap - avg_gap) > avg_gap * 0.5:  # 超过平均间隔50%
                        warnings.append(f"时间间隔异常: 第{i+1}到{i+2}个数据点间隔{gap:.2f}小时")
            
            # 计算统计信息
            statistics = None
            if valid_points > 0:
                df = pd.DataFrame([{
                    'timestamp': point.timestamp,
                    'energy_consumption': point.energy_consumption,
                    'temperature': point.temperature,
                    'humidity': point.humidity,
                    'production_load': point.production_load
                } for point in data if point.energy_consumption is not None])
                
                statistics = {
                    'energy_consumption': {
                        'mean': float(df['energy_consumption'].mean()),
                        'std': float(df['energy_consumption'].std()),
                        'min': float(df['energy_consumption'].min()),
                        'max': float(df['energy_consumption'].max()),
                        'median': float(df['energy_consumption'].median())
                    },
                    'date_range': {
                        'start': df['timestamp'].min(),
                        'end': df['timestamp'].max()
                    },
                    'data_quality': {
                        'completeness': valid_points / total_points * 100,
                        'has_temperature': df['temperature'].notna().sum() > 0,
                        'has_humidity': df['humidity'].notna().sum() > 0,
                        'has_production_load': df['production_load'].notna().sum() > 0
                    }
                }
            
            # 判断数据是否有效
            is_valid = valid_points >= 24 and len(errors) == 0
            
            return DataValidationResult(
                is_valid=is_valid,
                total_points=total_points,
                valid_points=valid_points,
                invalid_points=invalid_points,
                missing_values=missing_values,
                errors=errors,
                warnings=warnings,
                statistics=statistics
            )
            
        except Exception as e:
            logger.error(f"数据验证失败: {str(e)}")
            return DataValidationResult(
                is_valid=False,
                total_points=len(data),
                valid_points=0,
                invalid_points=len(data),
                missing_values=0,
                errors=[f"数据验证异常: {str(e)}"],
                warnings=[],
                statistics=None
            )
    
    def save_data_as_csv(self, submission_id: str, data: List[TimeSeriesDataPoint]) -> Tuple[str, bool]:
        """将数据保存为CSV格式"""
        try:
            csv_file_path = os.path.join(self.processed_data_path, f"{submission_id}.csv")
            
            # 准备CSV数据
            csv_data = []
            for point in data:
                row = {
                    'timestamp': point.timestamp,
                    'energy_consumption': point.energy_consumption,
                    'temperature': point.temperature if point.temperature is not None else '',
                    'humidity': point.humidity if point.humidity is not None else '',
                    'production_load': point.production_load if point.production_load is not None else ''
                }
                csv_data.append(row)
            
            # 写入CSV文件
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_file_path, index=False, encoding='utf-8')
            
            logger.info(f"数据已保存到CSV文件: {csv_file_path}")
            return csv_file_path, True
            
        except Exception as e:
            logger.error(f"保存CSV文件失败: {str(e)}")
            return "", False
    
    def save_metadata(self, submission_id: str, metadata: Dict[str, Any]) -> bool:
        """保存数据元信息"""
        try:
            metadata_file = os.path.join(self.metadata_path, f"{submission_id}.json")
            return save_json(metadata, metadata_file)
        except Exception as e:
            logger.error(f"保存元数据失败: {str(e)}")
            return False
    
    def get_metadata(self, submission_id: str) -> Optional[Dict[str, Any]]:
        """获取数据元信息"""
        try:
            metadata_file = os.path.join(self.metadata_path, f"{submission_id}.json")
            return load_json(metadata_file)
        except Exception as e:
            logger.error(f"获取元数据失败: {str(e)}")
            return None
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """列出所有数据集"""
        try:
            datasets = []
            
            # 遍历元数据文件
            for filename in os.listdir(self.metadata_path):
                if filename.endswith('.json'):
                    submission_id = filename[:-5]  # 移除.json后缀
                    metadata = self.get_metadata(submission_id)
                    
                    if metadata:
                        # 获取文件大小
                        csv_file = os.path.join(self.processed_data_path, f"{submission_id}.csv")
                        file_size = os.path.getsize(csv_file) if os.path.exists(csv_file) else 0
                        
                        dataset_info = {
                            'submission_id': submission_id,
                            'device_id': metadata.get('device_id', ''),
                            'data_name': metadata.get('data_name', ''),
                            'description': metadata.get('description', ''),
                            'total_points': metadata.get('total_points', 0),
                            'file_size': file_size,
                            'submission_time': metadata.get('submission_time', ''),
                            'status': metadata.get('status', 'unknown')
                        }
                        datasets.append(dataset_info)
            
            # 按提交时间排序
            datasets.sort(key=lambda x: x['submission_time'], reverse=True)
            return datasets
            
        except Exception as e:
            logger.error(f"列出数据集失败: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        try:
            datasets = self.list_datasets()
            
            total_datasets = len(datasets)
            total_data_points = sum(d['total_points'] for d in datasets)
            total_storage_size = sum(d['file_size'] for d in datasets)
            device_count = len(set(d['device_id'] for d in datasets))
            
            # 计算时间范围
            date_range = {'start': '', 'end': ''}
            if datasets:
                submission_times = [d['submission_time'] for d in datasets if d['submission_time']]
                if submission_times:
                    date_range['start'] = min(submission_times)
                    date_range['end'] = max(submission_times)
            
            # 最近的提交
            recent_submissions = datasets[:5]  # 最近5个
            
            return {
                'total_datasets': total_datasets,
                'total_data_points': total_data_points,
                'total_storage_size': total_storage_size,
                'device_count': device_count,
                'date_range': date_range,
                'recent_submissions': recent_submissions
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {
                'total_datasets': 0,
                'total_data_points': 0,
                'total_storage_size': 0,
                'device_count': 0,
                'date_range': {'start': '', 'end': ''},
                'recent_submissions': []
            }
    
    def delete_dataset(self, submission_id: str) -> bool:
        """删除数据集"""
        try:
            # 删除CSV文件
            csv_file = os.path.join(self.processed_data_path, f"{submission_id}.csv")
            if os.path.exists(csv_file):
                os.remove(csv_file)
            
            # 删除元数据文件
            metadata_file = os.path.join(self.metadata_path, f"{submission_id}.json")
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            
            logger.info(f"数据集已删除: {submission_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除数据集失败: {str(e)}")
            return False
    
    def export_datasets(self, submission_ids: List[str], format: str = "csv") -> Optional[str]:
        """导出数据集"""
        try:
            if format == "csv":
                # 合并多个CSV文件
                combined_data = []
                
                for submission_id in submission_ids:
                    csv_file = os.path.join(self.processed_data_path, f"{submission_id}.csv")
                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file)
                        df['submission_id'] = submission_id
                        combined_data.append(df)
                
                if combined_data:
                    combined_df = pd.concat(combined_data, ignore_index=True)
                    export_filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    export_path = os.path.join(self.processed_data_path, export_filename)
                    combined_df.to_csv(export_path, index=False, encoding='utf-8')
                    return export_path
            
            return None
            
        except Exception as e:
            logger.error(f"导出数据集失败: {str(e)}")
            return None


# 全局数据处理服务实例
data_processing_service = DataProcessingService()