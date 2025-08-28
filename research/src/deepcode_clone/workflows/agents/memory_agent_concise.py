"""
Consise Memory Agent for Code Implementation Workflow

This memory agent implements a focused approach:
1. Before first file: Normal conversation flow
2. After first file: Keep only system_prompt + initial_plan + current round tool results
3. Clean slate for each new code file generation

Key Features:
- Preserves system prompt and initial plan always
- After first file generation, discards previous conversation history
- Keeps only current round tool results from essential tools:
  * read_code_mem, read_file, write_file
  * execute_python, execute_bash
  * search_code, search_reference_code, get_file_structure
- Provides clean, focused input for next write_file operation
"""


import logging
from typing import Optional, Dict, Any, List



class ConciseMemoryAgent:
  """
  Concise Memory Agent - Focused Information Retention
  
  Core Philosophy:
  - Preserve essential context (system prompt + initial plan)
  - After first file generation, use clean slate approach
  - Keep only current round tool results from all essential MCP tools
  - Remove conversational clutter and previous tool calls
  
  Essential Tools Tracked:
  - File Operation: read_code_mem, read_file, write_file
  - Code Analysis: search_code, search_reference_code, get_file_structure
  - Execution: execute_python, execute_bash
  """

  def __init__(
    self,
    initial_plan_content: str,
    logger: Optional[logging.Logger] = None,
    target_directory: Optional[str] = None,
    default_models: Optional[Dict[str, str]] = None
  ):
    """
    Initialize Concise Memory Agent
    
    Args:
      initial_plan_content: Content of initial_plan.txt
      logger: Logger instance
      target_directory: Target directory for saving summaries
      default_models: Default models configuration from workflow
    """
    
    
  def _create_default_logger(self) -> logging.Logger:
    """Create default logger"""
    
    
  def _parse_phase_structure(self) -> Dict[str, List[str]]:
    """Parse implementation phases from initial plan"""


  def _extract_all_files_from_plan(self) -> List[str]:
    """
    Extract all file paths from the file_structure section in initial plan
    Handles multiple formats: tree structure, YAML, and simple lists
    
    Returns:
      List of all file paths that should be implemented
    """
    
    
  def _extract_from_tree_structure(self, lines: List[str]) -> List[str]:
    """Extract files from tree structure format - only from file_structure section"""


  def _extract_from_simple_list(self, lines: List[str]) -> List[str]:
    """Extract files from simple list format (- filename)"""
    
    
  def _extract_from_plan_content(self, lines: List[str]) -> List[str]:
    """Extract files from anywhere in the plan content"""


  def _clean_and_validate_files(self, files: List[str]) -> List[str]:
    """Clean and validate extracted file paths - only keep code files"""
    
    
  def record_file_implementation(
    self, file_path: str, implentation_content: str = ""
  ):
    """
    Record a newly implemented file (simplified version)
    NEW LOGIC: File implementation is tracked via write_file tool detection
    
    Args:
      file_path: Path of the implemented file
      implementation_content: Content of the implemented file
    """


  async def create_code_implementation_summary(
    self,
    client,
    client_type: str,
    file_path: str,
    implementation_content: str,
    files_implemented: int
  ) -> str:
    """
    Create LLM-based code implementation summary after writing a file
    Uses LLM to analyze and summarize the implemented code
    
    Args:
      client: LLM client instance
      client_type: Type of LLM client ("anthropic" or "openai")
      file_path: Path of the implemented file
      implementation_content: Content of the implemented file
      files_implemented: Number of files implemented so far
      
    Returns:
      LLM-generated formatted code implementation summary
    """


  def _create_code_summary_prompt(
    self, file_path: str, implementation_content: str, files_implemented: int
  ) -> str:
    """
    Create prompt for LLM to generate code implementation summary
    
    Args:
      file_path: Path of the implemented file
      implementation_content: Content of the implemented file
      files_implemented: Number of files implemented so far
    
    Returns:
      Prompt for LLM summarization
    """
    
    
  def _extract_summary_sections(self, llm_summary: str) -> Dict[str, str]:
    """
    Extract different sections from LLM-generated summary
    
    Args:
      llm_summary: Raw LLM-generated summary text
      
    Returns:
      Dictionary with extracted sections: implementation_progress, dependencies, next_steps
    """  
  
  
  def _format_code_implementation_summary(
    self, file_path: str, llm_summary: str, files_implemented: int
  ) -> str:
    """
    Format the LLM-generated summary into the final structure
    
    Args:
      file_path: Path of the implemented file
      llm_summary: LLM-generated summary content
      files_implemented: Number of files implemented so far
      
    Returns:
      Formatted summary
    """
  
  
  def _create_fallback_code_summary(
    self, file_path: str, implementation_content: str, files_implemented: int
  ) -> str:
    """
    Create fallback summary when LLM is unavailable
    
    Args:
      file_path: Path of the implemented file
      implementation_content: Content of the implemented file
      files_implemented: Number of files implemented so far
      
    Returns:
      Fallback summary
    """
    
    
  async def _save_code_summary_to_file(self, new_summary: str, file_path: str):
    """
    Append code implementation summary to implement_code_summary.md
    Accumulates all implementations with clear separators
    
    Args:
      new_summary: New summary content to append
      file_path: Path of the file for which the summary was generated
    """  
    
  
  async def _call_llm_for_summary(
    self, client, client_type: str, summary_messages: List[Dict]
  ) -> Dict[str, Any]:
    """
    Call LLM for code implementation summary generation ONLY
    
    This method is used only for creating code implementation summaries,
    NOT for conversation summarization which has been removed.
    """
    
    
  def start_new_round(self, iteration: Optional[int] = None):
    """
    Start a new dialogue round and reset tool results
    
    Args:
      iteration: Optional iteration number from workflow to sync with current_round 
    """  
  
  
  def record_tool_result(
    self, tool_name: str, tool_input: Dict[str, Any], tool_result: Any
  ):
    """
    Record tool result for current round and detect write_file calls
    
    Args:
      tool_name: Name of the tool called
      tool_input: Input parameters for the tool
      tool_result: Result returned by the tool
    """


  def should_use_concise_mode(self) -> bool:
    """
    Check if concise memory mode should be used
    
    Returns:
      True if first file has been generated and concise mode should be active
    """
    
    
  def create_concise_messages(
    self,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    files_implemented: int
  ) -> List[Dict[str, Any]]:
    """
    Create concise message list for LLM input
    NEW LOGIC: Always clear after write_file, keep system_prompt + initial_plan + current round tools
    
    Args:
      system_prompt: Current system prompt
      messages: Original message list
      files_implemented: Number of files implemented so far
      
    Returns:
      Concise message list containing only essential information
    """
    
    
  def _read_code_knowledge_base(self) -> Optional[str]:
    """
    Read the implement_code_summary.md file as code knowledge base
    Returns only the final/latest implementaion entry, not all historical entries
    
    Returns:
      Content of the latest implementation entry if it exists, None otherwise
    """


  def _extract_latest_implementation_entry(self, content: str) -> Optional[str]:
    """
    Extract the latest/final implementation entry from the implement_code_summary.md content
    Uses a simpler approach to find the last imlementation section
    
    Args:
      content: Full content of implement_code_summary.md
      
    Returns:
      Latest implementation entry content, or None if not found
    """


  def _format_tool_results(self) -> str:
    """
    Format current round tool resulst for LLM input
    
    Returns:
      Formatted string of tool results
    """


  def _format_tool_result_content(self, tool_result: Any) -> str:
    """
    Format tool result content for display
    
    Args:
      tool_result: Tool result to format
      
    Returns:
      Formatted string representation
    """
    
    
  def get_memory_statistics(self, files_implemented: int = 0) -> Dict[str, Any]:
    """Get memory agent statistics"""
    
    
  def get_implemented_files(self) -> List[str]:
    """Get list of all implemented files"""  
    
    
  def get_all_files_list(self) -> List[str]:
    """Get list of all files that should be implemented according to the plan"""
    
    
  def get_unimplemented_files(self) -> List[str]:
    """
    Get list of files that haven't been implemented yet
    
    Returns:
      List of file paths that still need to be implemented
    """


  def get_formatted_files_lists(self) -> Dict[str, str]:
    """
    Get formatted strings for implemented and unimplemented files
    
    Returns:
      Dictionary with 'implemented' and 'unimplemented' formatted lists
    """
    
  
  def get_current_next_steps(self) -> str:
    """Get the current Next Steps information"""


  def clear_next_steps(self):
    """Clear the stored Next Steps information"""


  def set_next_steps(self, next_steps: str):
    """Manually set Next Steps information"""
    
    
  def should_trigger_memory_optimization(
    self, messages: List[Dict[str, Any]], files_implemented: int = 0
  ) -> bool:
    """
    Check if memory optimization should be triggered
    NEW LOGIC: Trigger after write_file has been detected
    
    Args:
      messages: Current message list
      files_implemented: Number of files implemented so far
      
    Returns:
      True if concise mode should be applied
    """


  def apply_memory_optimization(
    self, system_prompt: str, messages: List[Dict[str, Any]], files_implemented: int
  ) -> List[Dict[str, Any]]:
    """
    Apply memory optimization using concise approach
    NEW LOGIC: Clear all history after write_file, keep only system_prompt + initial_plan + current tools
    
    Args:
      system_prompt: Current system prompt
      messages: Original message list
      files_implemented: Number of files implemented so far
      
    Returns:
      Optimized message list
    """


  def clear_current_round_tool_resulst(self):
    """Clear current round tool results (called when starting new round)"""


  def debug_concise_state(self, files_implemented: int = 0):
    """Debug method to show current concise memory state"""






