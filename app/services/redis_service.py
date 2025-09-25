"""
Redis缓存服务
"""
import json
import logging
from typing import Any, Optional, Dict, List
import aioredis
from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisService:
    """Redis缓存服务类"""
    
    def __init__(self):
        self.redis_client: Optional[aioredis.Redis] = None
        self.connection_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
        if settings.REDIS_PASSWORD:
            self.connection_url = f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
    
    async def connect(self):
        """连接到Redis"""
        try:
            self.redis_client = await aioredis.from_url(
                self.connection_url,
                encoding="utf-8",
                decode_responses=True,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            # 测试连接
            await self.redis_client.ping()
            logger.info("Redis连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {str(e)}")
            raise e
    
    async def close(self):
        """关闭Redis连接"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis连接已关闭")
    
    async def ping(self) -> bool:
        """检查Redis连接状态"""
        try:
            if not self.redis_client:
                await self.connect()
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping失败: {str(e)}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            value = await self.redis_client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logger.error(f"Redis获取数据失败 key={key}: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        try:
            json_value = json.dumps(value, ensure_ascii=False, default=str)
            if ttl is None:
                ttl = settings.CACHE_TTL
            await self.redis_client.setex(key, ttl, json_value)
            return True
        except Exception as e:
            logger.error(f"Redis设置数据失败 key={key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis删除数据失败 key={key}: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            result = await self.redis_client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis检查键存在失败 key={key}: {str(e)}")
            return False
    
    async def get_all_keys(self, pattern: str = "*") -> List[str]:
        """获取所有匹配的键"""
        try:
            keys = await self.redis_client.keys(pattern)
            return keys if keys else []
        except Exception as e:
            logger.error(f"Redis获取键列表失败 pattern={pattern}: {str(e)}")
            return []
    
    async def set_hash(self, key: str, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """设置哈希表"""
        try:
            # 将值转换为JSON字符串
            json_mapping = {k: json.dumps(v, ensure_ascii=False, default=str) for k, v in mapping.items()}
            await self.redis_client.hset(key, mapping=json_mapping)
            if ttl:
                await self.redis_client.expire(key, ttl)
            return True
        except Exception as e:
            logger.error(f"Redis设置哈希表失败 key={key}: {str(e)}")
            return False
    
    async def get_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """获取哈希表"""
        try:
            hash_data = await self.redis_client.hgetall(key)
            if not hash_data:
                return None
            # 将JSON字符串转换回原始值
            return {k: json.loads(v) for k, v in hash_data.items()}
        except Exception as e:
            logger.error(f"Redis获取哈希表失败 key={key}: {str(e)}")
            return None
    
    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """递增计数器"""
        try:
            result = await self.redis_client.incr(key, amount)
            return result
        except Exception as e:
            logger.error(f"Redis递增失败 key={key}: {str(e)}")
            return None
    
    async def get_info(self) -> Dict[str, Any]:
        """获取Redis服务器信息"""
        try:
            info = await self.redis_client.info()
            return {
                "version": info.get("redis_version", "unknown"),
                "memory_used": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace": info.get("db0", {})
            }
        except Exception as e:
            logger.error(f"获取Redis信息失败: {str(e)}")
            return {}


# 全局Redis服务实例
redis_service = RedisService()