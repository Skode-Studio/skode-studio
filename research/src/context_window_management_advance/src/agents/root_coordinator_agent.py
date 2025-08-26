

from collections import deque
from agents.context_aware_agent import ContextAwareAgent
from core.memory_hierarchy import MemoryHierarchy
from models.schemas import AgentState
import logging
from typing import Dict, Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class RootCoordinatorAgent(ContextAwareAgent):
  """Root agent that coordinates the entire system"""
  
  def __init__(self, memory_hierarchy: MemoryHierarchy):
    super().__init__("RootCoordinate", memory_hierarchy)
    self.sub_agents = {}
    self.task_queue = deque()
    self.execution_plan = []
    
    
  async def process(self, state: AgentState) -> AgentState:
    """Main coordinator logic with advanced context management"""
    
    logging.info(f"Root agent processing task: {state.current_task}")
    
    # Update the context with memmory hierarchy
    self.update_context(state)
    
    
    # Analyze task complexity and decide coordination strategy
    task_analysis = await self._analyze_task(state)
    
    # Create execution plan
    execution_plan = await self._create_execution_plan(task_analysis, state)
    
    # Store interaction in memory
    self.memory.store_interaction({
      'user_input': state.current_task,
      'task_analysis': task_analysis,
      'execution_plan': execution_plan,
      'context_used': state.context[:500] + "..." if len(state.context) > 500 else state.context
    })
    
    return state
    
    
    
  async def _analyze_task(self, state: AgentState) -> Dict[str, Any]:
    """Analyze task complexity and requirements"""
    
    task = state.current_task.lower()
    
    analysis = {
      'complexity': 'medium',
      'required_agents': ['code_analyzer'],
      'estimated_steps': 3,
      'context_requirements': 'medium'
    }
    
    # Task complexity analysis
    complexity_indicators = {
      'high': ['refactor', 'architecture', 'design pattern', 'multiple files'],
      'medium': ['function', 'class', 'module', 'debug'],
      'low': ['variable', 'comment', 'simple fix']
    }
    
    for complexity, indicators in complexity_indicators.items():
      if any(indicator in task for indicator in indicators):
        analysis['complexity'] = complexity
        break
      
      
    # Determine required agents based on task
    if 'test' in task:
      analysis['required_agents'].append('test_agent')
    if 'document' in task:
      analysis['required_agents'].append('documentation_agent')
    if 'performance' in task or 'optimize' in task:
      analysis['required_agents'].append('performance_agent')
    
    return analysis
    
  
  async def _create_execution_plan(self, analysis: Dict[str, Any], state: AgentState) -> list[str]:
    """Created detailed execution plan"""
    
    plan = []
    
    # Base plan structure
    plan.append(f"1. Analyze code context for: {state.current_task}")
    plan.append("2. Retrieve relevant historical context")
    plan.append(f"3. Execute task with {analysis['complexity']} complexity approach")
    
    # Add specific steps based on required agents
    for i, agent in enumerate(analysis['required_agents'], start=4):
      plan.append(f"{i}. Delegate subtask to {agent}")
      
    plan.append(f"{len(plan)+1}. Integrate results and provide final response")
    plan.append(f"{len(plan)+1}. Update memory hierarchy with new knowledge")
    
    return plan
    
  
  def _create_memory_summary(self) -> str:
    """Create a summmary of current memory state"""
    
    working_items = len(self.memory.working_memory.items())
    short_term_items = len(self.memory.short_term_memory.items())
    long_term_items = len(self.memory.long_term_memory.items())
    episodes = len(self.memory.episodic_memory)
    
    return f"Memory State - Working: {working_items}, Short-term: {short_term_items}, Long-term: {long_term_items}, Episodes: {episodes}"
    
    
    
    
    
    
    
    
    