

from typing import List, Dict, Any
from core.context_window import ContextWindow, ContextItem
from models.schemas import ContextPriority
import time




class MemoryHierarchy:
  """Multi-level memory hierarchy for agents"""
  
  def __init__(self):
    self.working_memory = ContextWindow(max_tokens=2000) # Immediate context
    self.short_term_memory = ContextWindow(max_tokens=8000) # Recent important info
    self.long_term_memory = ContextWindow(max_tokens=32000) # Historical context
    self.episodic_memory: List[Dict] = [] # Conversation episodes
    
  
  def store_interaction(self, interaction: Dict[str, Any]) -> None:
    """Store interaction across memory levels"""
    
    timestamp = time.time()
    
    # Create context items for different parts
    if 'user_input' in interaction:
      user_item = ContextItem(
        content=interaction['user_input'],
        priority=ContextPriority.HIGH,
        timestamp=timestamp,
        token_count=len(interaction['user_input'].split()),
        source="user_input",
        tags=['user', 'input']
      )
      self.working_memory.add_item(user_item)
      
    if 'agent_response' in interaction:
      agent_item = ContextItem(
        content=interaction['agent_response'],
        priority=ContextPriority.MEDIUM,
        timestamp=timestamp,
        token_count=len(interaction['agent_response'].split()),
        source="agent_response",
        tags=['agent', 'response']
      )
      self.working_memory.add_item(agent_item)
      
      
    # Store full interaction in episodic memory
    self.episodic_memory.append({
      **interaction,
      'timestamp': timestamp,
      'episode_id': len(self.episodic_memory)
    })
  
  
  def retrieve_context(self, query: str, max_tokens: int = 4000) -> str:
    """Retrieve relevant context from memory hierarchy"""
    context_parts = []
    remaining_tokens = max_tokens
    
    # 1. Working memory (most recent/relevant)
    working_items = self.working_memory.search_relevant(query, top_k=3)
    for item in working_items:
      if remaining_tokens > item.token_count:
        context_parts.append(f"[RECENT] {item.content}")
        remaining_tokens -= item.token_count
        
        
    # 2. Short-term memory
    if remaining_tokens > 500:
      short_term_items = self.short_term_memory.search_relevant(query, top_k=2)
      for item in short_term_items:
        if remaining_tokens > item.token_count:
          context_parts.append(f"[SHORT-TERM] {item.content}")
          remaining_tokens -= item.token_count
          
          
    # 3. Long-term memory (if space available)
    if remaining_tokens > 500:
      long_term_items = self.long_term_memory.search_relevant(query, top_k=1)
      for item in long_term_items:
        if remaining_tokens > item.token_count:
          context_parts.append(f"[HISTORICAL] {item.content}")
          remaining_tokens -= item.token_count
          
          
    return "\n\n".join(context_parts)
    
    
    
    
  



