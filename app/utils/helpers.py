"""
常用工具函数
"""
import os
import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd


def generate_id(prefix: str = "", length: int = 8) -> str:
    """生成唯一ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    hash_obj = hashlib.md5(f"{timestamp}{prefix}".encode())
    return f"{prefix}{hash_obj.hexdigest()[:length]}"


def save_json(data: Any, file_path: str) -> bool:
    """保存数据到JSON文件"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return True
    except Exception as e:
        print(f"保存JSON文件失败: {str(e)}")
        return False


def load_json(file_path: str) -> Optional[Any]:
    """从JSON文件加载数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON文件失败: {str(e)}")
        return None


def validate_time_series_data(data: List[Dict]) -> Dict[str, Any]:
    """验证时间序列数据格式"""
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {}
    }
    
    if not data:
        validation_result["valid"] = False
        validation_result["errors"].append("数据为空")
        return validation_result
    
    # 检查必需字段
    required_fields = ["timestamp", "value"]
    for i, record in enumerate(data):
        for field in required_fields:
            if field not in record:
                validation_result["valid"] = False
                validation_result["errors"].append(f"记录{i}缺少必需字段: {field}")
    
    # 生成统计信息
    if validation_result["valid"]:
        df = pd.DataFrame(data)
        validation_result["statistics"] = {
            "total_records": len(data),
            "date_range": {
                "start": df['timestamp'].min(),
                "end": df['timestamp'].max()
            },
            "value_stats": {
                "mean": float(df['value'].mean()),
                "std": float(df['value'].std()),
                "min": float(df['value'].min()),
                "max": float(df['value'].max())
            }
        }
    
    return validation_result


def format_bytes(bytes_value: int) -> str:
    """格式化字节数为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"


def get_file_size(file_path: str) -> int:
    """获取文件大小"""
    try:
        return os.path.getsize(file_path)
    except:
        return 0