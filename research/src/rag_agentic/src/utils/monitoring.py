
import time
import logging
from typing import Dict, Any, Callable
from functools import wraps
import asyncio


logger = logging.getLogger(__name__)



class PerformanceMonitor:
  """Monitor performance and colelct metrics"""
  
  def __init__(self):
    self.metrics = {
      'query_count': 0,
      'avg_response_time': 0.0,
      'total_documents': 0,
      'total_chunks': 0,
      'cache_hits': 0,
      'cache_misses': 0
    }
    self.response_times = []
    
    
  def time_function(self, func_name: str):
    """Decorator to time function execution"""
    
    def decorator(func: Callable):
      if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwrags):
          start_time = time.time()
          try:
            result = await func(*args, **kwrags)
            return result
          finally:
            execution_time = time.time() - start_time
            self._record_timing(func_name, execution_time)
        return async_wrapper
      
      else:
        @wraps(func)
        def sync_wrapper(*args, **kwrags):
          start_time = time.time()
          try:
            result = func(*args, **kwrags)
            return result
          finally:
            execution_time = time.time() - start_time
            self._record_timing(func_name, execution_time)
        return async_wrapper
      
    return decorator
              
    
  
  def _record_timing(self, func_name: str, execution_time: float):
    """Record timing information"""
    logger.info(f"{func_name} executed in {execution_time:.3f}s")
    
    if func_name == 'process_query':
      self.metrics['query_count'] += 1
      self.response_times.append(execution_time)
      self.metrics['avg_response_time'] = sum(self.response_times) / len(self.response_times)
    
  def get_metrics(self) -> Dict[str, Any]:
    """Get current metrics"""
    return self.metrics.copy()

  
  
  






