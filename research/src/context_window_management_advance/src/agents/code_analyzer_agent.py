

from agents.context_aware_agent import ContextAwareAgent
from models.schemas import AgentState
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeAnalyzerAgent(ContextAwareAgent):
  """Specialized agent for code analysis"""
  
  async def process(self, state: AgentState) -> AgentState:
    logger.info("Code analyzer processing...")
    
    # Update context for code analysis
    self.update_context(state)
    
    
    # Simulate code analysis
    analysis_result = {
      'code_quality': 'good',
      'suggestions': ['Add hype hints', 'Improve error handling'],
      'complextity_score': 7.5
    }
  

    state.code_context['analysis'] = analysis_result
    
    # Store specialized code knowledge
    self.memory.store_interaction({
      'agent_response': f"Code analysis completed: {analysis_result}",
      'code_analysis': analysis_result,
      'context_tokens_used': len(state.context.split())
    })
    
    
    return state
    
    
    
    

