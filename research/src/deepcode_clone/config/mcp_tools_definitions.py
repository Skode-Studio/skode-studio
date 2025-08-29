"""
MCP Tool Definitions Configuration Module

Separate tool definitions from main program logic, providing standardized tool definition format

- File Operations
- Code Execution
- Search Tools
- Project Structure Tools
"""


from typing import Dict, List, Any



class MCPToolDefinitions:
  
  
  @staticmethod
  def get_code_implementation_tools() -> List[Dict[str, Any]]:
    """Get tool definitions for code implementation"""


  @staticmethod
  def _get_read_file_tool() -> Dict[str, Any]:
    """Read file tool"""


  @staticmethod
  def _get_read_code_mem_tool() -> Dict[str, Any]:
    """Read code memory tool definition - reads from implement_code_summary.md"""


  @staticmethod
  def _get_write_file_tool() -> Dict[str, Any]:
    """Write file tool"""


  @staticmethod
  def _get_execute_python_tool() -> Dict[str, Any]:
    """Python execute tool"""
  
  
  @staticmethod
  def _get_execute_bash_tool() -> Dict[str, Any]:
    """Bash command execute tool"""
  
  
  @staticmethod
  def _get_file_structure_tool() -> Dict[str, Any]:
    """File structure tool"""


  @staticmethod
  def _get_search_code_references_tool() -> Dict[str, Any]:
    """Search code references tool"""
    
    
  @staticmethod
  def _get_get_indexes_overview_tool() -> Dict[str, Any]:
    """Get indexes overview tool"""  
    
    
  @staticmethod
  def _get_set_workspace_tool() -> Dict[str, Any]:
    """Set workspace directory tool definition"""  
    
    
  @staticmethod
  def get_available_tool_sets() -> Dict[str, str]:
    """Get available tool sets"""
    
    
  @staticmethod
  def get_tool_set(tool_set_name: str) -> List[Dict[str, Any]]:
    """Get specific tool set by name"""
  
  
  @staticmethod
  def get_all_tools() -> List[Dict[str, Any]]:
    """Get all available tools"""
    
    
    
# Convenient Access Functions
def get_mcp_tools(tool_set: str = "code_implementation") -> List[Dict[str, Any]]:
  """
  Convenience function: Get MCP tool definitions
  
  Args:
    tool_set: tool set name
    
  Returns:
    tools definition list
  """
