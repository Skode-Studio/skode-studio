

from typing import Dict, Any
from core.memory_hierarchy import MemoryHierarchy
from agents.code_analyzer_agent import CodeAnalyzerAgent
from agents.root_coordinator_agent import RootCoordinatorAgent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.schemas import AgentState, ContextPriority
import time
import logging
import inspect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StateGraph:
  def __init__(self, state_type=None):
    self.state_type = state_type
    self.nodes = {}
    self.edges = {}
    self.entry_point = None
    self.checkpointer = None

  def add_node(self, name, func):
    self.nodes[name] = func

  def add_edge(self, from_node, to_node):
    self.edges.setdefault(from_node, []).append(to_node)

  def set_entry_point(self, node):
    self.entry_point = node

  def compile(self, **kwargs):
      # capture checkpointer if provided
      self.checkpointer = kwargs.get("checkpointer", None)
      return self

  async def invoke(self, state, config=None):
    """Async execution flow"""
    current = self.entry_point
    while current and current != END:
      node_func = self.nodes[current]

      if inspect.iscoroutinefunction(node_func):
        state = await node_func(state)
      else:
        state = node_func(state)

      next_nodes = self.edges.get(current, [])
      current = next_nodes[0] if next_nodes else None
    return state
  
class MemorySaver: pass
class BaseMessage:
  def __init__(self, content: str = "", role: str = "base"):
    self.content = content
    self.role = role


class HumanMessage(BaseMessage):
  def __init__(self, content: str = ""):
    super().__init__(content, role="human")


class AIMessage(BaseMessage):
  def __init__(self, content: str = ""):
    super().__init__(content, role="ai")


class SystemMessage(BaseMessage):
  def __init__(self, content: str = ""):
    super().__init__(content, role="system")
    
START, END = "start", "end"


class LangGraphContextSystem:
  """Main system orchestrator using LangGraph"""
  
  def __init__(self):
    self.memory_hierarchy = MemoryHierarchy()
    self.root_agent = RootCoordinatorAgent(self.memory_hierarchy)
    self.code_analyzer = CodeAnalyzerAgent("CodeAnalyzer", self.memory_hierarchy)
    self.checkpointer = MemorySaver()
    self.graph = self._create_graph()
  
  def _create_graph(self) -> StateGraph:
    """Create the LangGraph workflow"""
    workflow = StateGraph(StateGraph)
    
    # Add nodes
    workflow.add_node("root_coordinator", self._root_coordinator_node)
    workflow.add_node("code_analyzer", self._code_analyzer_node)
    workflow.add_node("context_manager", self._context_manager_node)
    workflow.add_node("memory_consolidator", self._memory_consolidator_node)
    
    # Add edges
    workflow.set_entry_point("context_manager")
    workflow.add_edge("context_manager", "root_coordinator")
    workflow.add_edge("root_coordinator", "code_analyzer")
    workflow.add_edge("code_analyzer", "memory_consolidator")
    workflow.add_edge("memory_consolidator", END)
    
    return workflow.compile(checkpointer=self.checkpointer)

  async def _root_coordinator_node(self, state: AgentState) -> AgentState:
    """Root coordinator node"""
    return await self.root_agent.process(state)

  async def _code_analyzer_node(self, state: AgentState) -> AgentState:
    """Code analyzer node"""
    return await self.code_analyzer.process(state)
  

  async def _context_manager_node(self, state: AgentState) -> AgentState:
    """Context management node"""
    
    # Advance context preparation
    if state.current_task:
      # Retrieve and prepare context
      relevant_context = self.memory_hierarchy.retrieve_context(
        state.current_task,
        max_tokens=3000
      )
      state.context = relevant_context
    
      # Add system context about available capabilities
      system_context = """
      Available capabilities:
      - Advanced code analysis with context awareness
      - Multi-level memory hierarchy (working, short-term, long-term)
      - Intelligent context compression and prioritization
      - Task complexity analysis and adaptive planning
      - Performance monitoring and optimization
      """
      state.context = system_context + "\n\n" + state.context
    
    return state
  
  
  async def _memory_consolidator_node(self, state: AgentState) -> AgentState:
    """Memory consolidation node"""
    
    # Promote important information from working to short-term memory
    working_items = self.memory_hierarchy.working_memory.items
    
    for item in working_items:
      if item.access_count > 2 or item.priority == ContextPriority.CRITICAL:
        # Promote to short-term memory
        self.memory_hierarchy.short_term_memory.add_item(item)
        
        
    # Update memory summary
    state.memory_summary = f"Consolidated {len(working_items)} working memory items"
    
    return state
        
    
  async def process_request(self, user_request: str) -> Dict[str, Any]:
    """Process a user request through the system"""
    
    initial_state = AgentState(
      messages=[
        HumanMessage(content=user_request)
      ],
      current_task=user_request
    )
    
    # Create unique thread ID for this conversation
    thread_id = f"thread_{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}}
  
    try:
      # Execute the graph
      final_state = await self.graph.invoke(initial_state, config)
    
      return {
        'success': True,
        'response': f"Task processed: {user_request}",
        'execution_plan': getattr(final_state, 'execution_history', []),
        'memory_summary': getattr(final_state, 'memory_summary', ''),
        'context_used': len(getattr(final_state, 'context', '').split()),
        'performance_metrics': self.root_agent.performance_metrics
      }
    except Exception as e:
      logger.error(f"Error processing request: {e}")
      return {
        'success': False,
        'error': str(e),
        'context': 'System error occurred during processing'
      }
  
  