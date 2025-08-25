
from typing import List, Dict
from models.schemas import ContextItem, ContextPriority
from collections import defaultdict
import logging
import hashlib
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class ContextWindow:
  """Advanced context window manager"""
  
  def __init__(
    self,
    max_tokens: int = 8000,
    compression_ratio: float = 0.3
  ):
    self.max_tokens = max_tokens
    self.compression_ratio = compression_ratio
    self.items: List[ContextItem] = []
    self.compressed_items: List[ContextItem] = []
    self.semantic_index: Dict[str, List[int]] = defaultdict(list)
    self.current_tokens = 0
    
    
  def add_item(self, item: ContextItem) -> None:
    """Add item to context window with intelligent management"""
    
    self.items.append(item)
    self.current_tokens += item.token_count
    self._index_item(item, len(self.items) - 1)
    
    if self.current_tokens > self.max_tokens:
      self._manage_overflow()
    
    
    
  def _index_item(self, item: ContextItem, index: int) -> None:
    """Index item for semantic search"""
    
    for tag in item.tags:
      self.semantic_index[tag].append(index)
    self.semantic_index[item.semantic_hash].append(index)
    
    
  
  def _manage_overflow(self) -> None:
    """Manage context overflow using multiple strategies"""
    
    logger.info(f"Context overflow detected. Current tokens: {self.current_tokens}")
    
    # Strategy 1: Remove low priority items
    self._remove_low_priority_items()
    
    # Strategy 2: Compress medium priority items
    if self.current_tokens > self.max_tokens:
      self._compress_medium_priority_items()
      
    # Strategy 3: Sliding window for old items
    if self.current_tokens > self.max_tokens:
      self._apply_sliding_window()
    
    
    
  def _remove_low_priority_items(self) -> None:
    """Remove items with low priority and low access count"""
    
    to_remove = []
    for i, item in enumerate(self.items):
      if(item.priority in [ContextPriority.LOW, ContextPriority.ARCHIVE] and item.access_count < 2):
        to_remove.append(i)
    
    for i in reversed(to_remove):
      removed_item = self.items.pop(i)
      self.current_tokens -= removed_item.token_count
      logger.info(f"Removed low priority item: {removed_item.source}")
    
    
  
  def _compress_medium_priority_items(self) -> None:
    """Compress medium priority items"""    
    
    for item in self.items:
      if item.priority == ContextPriority.MEDIUM and len(item.content) > 500:
        compressed_content = self._compress_text(item.content)
        token_savings = item.token_count - len(compressed_content.split())
        item.content = compressed_content
        item.token_count = len(compressed_content.split())
        self.current_tokens -= token_savings
        logger.info(f"Compressed item, saved {token_savings} tokens")
    
    
    
    
  def _compress_text(self, text: str) -> str:
    """Simple text compression - extract key sentences"""
    
    sentences = text.split('.')
    # Keep first and last sentences, and every 3rd sentence
    key_sentences = []
    if sentences:
      key_sentences.append(sentences[0])
      for i in range(2, len(sentences) - 1, 3):
        key_sentences.append(sentences[i])
      if len(sentences) > 1:
        key_sentences.append(sentences[-1])
    return '. '.join(key_sentences)
    
    
  
  def _apply_sliding_window(self) -> None:
    """Apply sliding window to remove oldest items"""
    
    while self.current_tokens > self.max_tokens and self.items:
      # Find oldest non-critical item
      oldest_idx = -1
      oldest_time = float('inf')
      for i, item in enumerate(self.items):
        if item.priority != ContextPriority.CRITICAL and item.timestamp < oldest_time:
          oldest_time = item.timestamp
          oldest_idx = i
    
      if oldest_idx >= 0:
        removed_item = self.items.pop(oldest_idx)
        self.current_tokens -= removed_item.token_count
        logger.info(f"Applied sliding window, removed: {removed_item.source}")
      else:
        break
      
    
  def search_relevant(self, query: str, top_k: int = 5) -> List[ContextItem]:
    """Search for relevant context items"""
    
    query_hast = hashlib.md5(query.encode()).hexdigest()
    relevant_items = []
    
    # Semantic search by tags and hash
    query_words = query.lower().split()
    for word in query_words:
      if word in self.semantic_index:
        for idx in self.semantic_index[word]:
          if idx < len(self.items):
            item = self.items[idx]
            item.access_count += 1
            item.last_accessed = time.time()
            relevant_items.append(item)
    
    # Remove duplicates and sort by relevance
    unique_items = list({item.semantic_hash: item for item in relevant_items}.values())
    return sorted(unique_items, key=lambda x: (x.priority.value, -x.access_count))[:top_k]
    
    

