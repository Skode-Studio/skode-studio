

import redis
import json
import hashlib
from typing import Any, Optional
import pickle
import logging


logger = logging.getLogger(__name__)


class RedisCache:
  """Redis-based caching for query results"""
  
  def __init__(
    self,
    redis_url: str = "redis://localhost:6379",
    ttl: int = 3600
  ):
    self.redis_client = redis.from_url(redis_url)
    self.ttl = ttl # Time to live in seconds
    
  
  def _generate_key(self, query: str, **kwrags) -> str:
    """Generate cache key from query and parameters"""
    
    key_data = f"{query}_{json.dumps(kwrags, sort_keys=True)}"
    return hashlib.sha256(key_data.encode()).hexdigest()
  
  
  async def get(self, query: str, **kwrags) -> Optional[Any]:
    """Get cached result"""
    
    try:
      key = self._generate_key(query, **kwrags)
      cached_data = self.redis_client.get(key)
      
      if cached_data:
        result = pickle.loads(cached_data)
        logger.debug(f"Cache hit for query: {query[:50]}...")
        return result
      
      logger.debug(f"Cache miss for query: {query[:50]}...")
      return None
    
    except Exception as e:
      logger.error(f"Cache get error: {e}")
      return None
    
    
  async def set(self, query: str, result: Any, **kwrags):
    """Cache result"""
    try:
      key = self._generate_key(query, **kwrags)
      serialized_data = pickle.dumps(result)
      
      self.redis_client.setex(key, self.ttl, serialized_data)
      logger.debug(f"Cached result for query: {query[:50]}...")
      
    except Exception as e:
      logger.error(f"Cache set error: {e}")
      
  
  def clear_cache(self):
    """Clear all cached data"""
    self.redis_client.flushall()
    logger.info("Cache Cleared")
  















