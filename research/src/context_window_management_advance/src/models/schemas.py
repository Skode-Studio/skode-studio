
from typing import List, Dict, Any
from enum import Enum
import logging
import hashlib
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)






class ContextPriority(Enum):
  """Priority levels for context items"""
  CRITICAL = 1
  HIGH = 2
  MEDIUM = 3
  LOW = 4
  ARCHIVE = 5


@dataclass
class ContextItem:
  """Represents a single context item with metadata"""
  content: str
  priority: ContextPriority
  timestamp: float
  token_count: int
  access_count: int = 0
  last_accessed: float = 0.0
  importance_score: float = 0.0
  semantic_hash: str = ""
  tags: List[str] = field(default_factory=list)
  source: str = ""
  
  def __post_init__(self):
    if not self.semantic_hash:
      self.semantic_hash = hashlib.md5(self.content.encode()).hexdigest()
    if not self.last_accessed:
      self.last_accessed = self.timestamp



@dataclass
class AgentState:
  """State management for the agent system"""
  messages: List[BaseMessage] = field(default_factory=list)
  current_task: str = ""
  context: str = ""
  memory_summary: str = ""
  active_agent: List[str] = field(default_factory=list)
  code_context: Dict[str, Any] = field(default_factory=dict)
  execution_history: List[Dict] = field(default_factory=list)






