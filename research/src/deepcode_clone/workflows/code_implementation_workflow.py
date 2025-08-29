"""
Paper Code Implementation Workflow - MCP-compliant Iterative Development

Features:
1. File Tree Creation
2. Code Implementation - Based on aisi-basic-agent iterative development

MCP Architecture:
- MCP Server: tools/code_implementation_server.py
- MCP Client: Called through mcp_agent framework
- Configuration: mcp_agent.config.yaml
"""



from agents.memory_agent_concise import ConciseMemoryAgent
from agents.code_implementation_agent import CodeImplementationAgent
from typing import Dict, Any, List, Optional
import logging
import os



class CodeImplementationWorkflow:
  """
  Paper Code Implementation Workflow Manager
  
  Uses standard MCP architecture:
  1. Connect to code-implementation server via MCP client
  2. Use MCP protocol for tool calls
  3. Support workspace management and operation history tracking
  """


  # ==================== 1. Class Initialization and Configuration (Infrastructure Layer) ====================

  def __init__(self, config_path: str = "mcp_agent.secrets.yaml"):
    """Initialize workflow with configuration"""


  def _load_api_config(self) -> Dict[str, Any]:
    """Load API configuration from YAML file"""
    
    
  def _create_logger(self) -> logging.Logger:
    """Create and configure logger"""


  def _read_plan_file(self, plan_file_path: str) -> str:
    """Read implementation plan file"""


  def _check_file_tree_exists(self, target_directory: str) -> bool:
    """Check if file tree structure already exists"""



  # ==================== 2. Public Interface Methods (External API Layer) ====================
  
  async def run_workflow(
    self,
    plan_file_path: str,
    target_directory: Optional[str] = None,
    pure_code_mode: bool = False,
    enable_read_tools: bool = True
  ):
    """Run complete workflow - Main public interface"""


  async def create_file_structure(
    self, plan_content: str, target_directory: str
  ) -> str:
    """Create file tree structure based on implementation plan"""


  async def implement_code_pure(
    self, plan_content: str, target_directory: str, code_directory: str = None
  ) -> str:
    """Pure code implementation - focus on code writing without testing"""


  # ==================== 3. Core Business Logic (Implementation Layer) ====================
  
  async def _pure_code_implementation_loop(
    self,
    client,
    client_type,
    system_message,
    messages,
    tools,
    plan_content,
    target_directory
  ):
    """Pure code implementation loop with memory optimization and phase consistency"""


  # ==================== 4. MCP Agent and LLM Communication Management (Communication Layer) ====================
  
  async def _initialize_mcp_agent(self, code_directory: str):
    """Initialize MCP agent and conenct to code-implementation server"""

    
    
  async def _cleanup_mcp_agent(self):
    """Clean up MCP agent resources"""
    
    
  async def _initalize_llm_client(self):
    """Initalize LLM client (Anthropic or OpenAI) based on API key availability"""  
    
    
  async def _call_llm_with_tools(
    self, client, client_type, system_message, messages, tools, max_tokens=8192
  ):
    """Call LLM with tools"""
    
    
  async def _call_anthropic_with_tools(
    self, client, system_message, messages, tools, max_tokens
  ):
    """Call Anthropic API"""
  
  
  async def _call_openai_with_tools(
    self, client, system_message, messages, tools, max_tokens
  ):
    """Call OpenAI API"""
  
  
  # ==================== 5. Tools and Utility Methods (Utility Layer) ====================
  
  def _validate_messages(self, messages: List[Dict]) -> List[Dict]:
    """Validate and clean message list"""
    
    
  def _prepare_mcp_tool_definitions(self) -> List[Dict[str, Any]]:
    """Prepare tool definitions in Anthropic API standard format"""
  
  
  def _check_tool_results_for_errors(self, tool_results: List[Dict]) -> bool:
    """Check tool results for errors"""
    
  
  # ==================== 6. User Interaction and Feedback (Interaction Layer) ====================
  
  def _generate_success_guidance(self, files_count: int) -> str:
    """Generate concise success guidance for continuing implementation"""
    
    
  def _generate_error_guidance(self) -> str:
    """Generate error guidance for handling issues"""
    
    
  def _generate_no_tools_guidance(self, files_count: int) -> str:
    """Generate concise guidance when no tools are called"""


  def _compile_user_response(self, tool_results: List[Dict], guidance: str) -> str:
    """Compile tool results and guidance into a single user response"""


  # ==================== 7. Reporting and Output (Output Layer) ====================
  
  async def _generate_pure_code_final_report_with_concise_agents(
    self,
    iterations: int,
    elapsed_time: float,
    code_agent: CodeImplementationAgent,
    memory_agent: ConciseMemoryAgent
  ):
    """Generate final report using concise agent statistics"""


