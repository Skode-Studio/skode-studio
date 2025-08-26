

from abc import ABC, abstractmethod
from core.memory_hierarchy import MemoryHierarchy
from collections import deque
from models.schemas import AgentState




class ContextAwareAgent(ABC):
  """Base class for context-aware agents"""
  
  def __init__(self, name: str, memory_hierarchy: MemoryHierarchy):
    self.name = name
    self.memory = memory_hierarchy
    self.conversation_history = deque(maxlen=20)
    self.performance_metrics = {
      'interactions': 0,
      'successful_tasks': 0,
      'context_retrievals': 0
    }

  @abstractmethod
  async def process(self, state: AgentState) -> AgentState:
    """Process the current state and return updated state"""
    
    pass
  
  
  def update_context(self, state: AgentState) -> None:
    """Update context based on current state"""
    
    relevant_context = self.memory.retrieve_context(
      state.current_task,
      max_tokens=2000
    )
    state.context = relevant_context
    self.performance_metrics['context_retrievals'] += 1


