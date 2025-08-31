"""
Code Implementation MCP Server

This MCP server provides core functions needed for paper code reproduction
1. File read/write operations
2. Code execution and testing
3. Code search and analysis
4. Iterative improvement support

Usage:
python tools/code_implementation_server.py
"""


from pathlib import Path
from typing import Dict, Any, List
import logging



# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import MCP related modules
from mcp.server.fastmcp import FastMCP


# Create FastMCP Server instance
mcp = FastMCP("code-implementation-server")



def initialize_workspace(workspace_dir: str = None):
  """
  Initialize workspace
  
  By default, the workspace will be set by the workflow via the set_workspace tool to: {plan_file_parent}/generate_code
  
  Args:
    workspace_dir: Optional workspace directory path
  """
  
  

def ensure_workspace_exists():
  """Ensure workspace directory exists"""
  
  

def validate_path(path: str) -> Path:
  """Validate if path is within workspace"""



def log_operation(action: str, details: Dict[str, Any]):
  """Log operation history"""



# ==================== File Operation Tools ====================

@mcp.tool()
async def read_file(
  file_path: str,
  start_line: int = None,
  end_line: int = None
):
  """
  Read file content, supports specifying line number range
  
  Args:
    file_path: File path, relative to workspace
    start_line: Starting line number (1-based, optional)
    end_line: Ending line number (1-based, optional)
    
  Returns:
    JSON string of file content or error message
  """


@mcp.tool()
async def write_file(
  file_path: str,
  content: str,
  create_dirs: bool = True,
  create_backup: bool = False
) -> str:
  """
  Write content to file
  
  Args:
    file_path: File path, relative to workspace
    content: Content to write to file
    create_dirs: Whether to create directories if they don't exist
    create_backup: Whether to create backup file if file already exists
    
  Returns:
    JSON string of operation result
  """
  

# ==================== Code Execution Tools ====================

@mcp.tool()
async def execute_python(code: str, timeout: int = 30) -> str:
  """
  Execute Python code and return output
  
  Args:
    code: Python code to execute
    timeout: Timeout in seconds
    
  Returns:
    JSON string of execution result
  """
  
  

@mcp.tool()
async def execute_bash(command: str, timeout: int = 30) -> str:
  """
  Execute bash command
  
  Args:
    command: Bash command to execute
    timeout: Timeout in seconds
    
  Returns:
    JSON string of execution result
  """


@mcp.tool()
async def read_code_mem(file_paths: List[str]) -> str:
  """
  Check if file summaries exist in implement_code_summary.md for multiple files
  
  Args:
    file_paths: List of file paths to check for summary information in implement_code_summary.md
    
  Returns:
    Summary information for all requested files if available
  """

def _extract_file_section_from_summary(
  summary_content: str,
  target_file_path: str
) -> str:
  """
  Extract the specific section for a file from the summary content
  
  Args:
    summary_content: Full summary content
    target_file_path: Path of the target file
    
  Returns:
    File-specific section or None if not found
  """

def _normalize_file_path(file_path: str) -> str:
  """Normalize file path for comparison"""

def _paths_match(
  normalized_target: str,
  normalized_summary: str,
  original_target: str,
  original_summary: str
) -> bool:
  """Check if two file paths match using multiple strategies"""

def _remove_common_prefixes(file_path: str) -> str:
  """Remove common prefixes from file path"""
  
def _extract_file_section_alternative(
  summary_content: str, target_file_path: str
) -> str:
  """Alternative method to extract file section using simpler pattern matching"""
  



# ==================== Code Search Tools ====================

@mcp.tool()
async def search_code(
  pattern: str,
  file_pattern: str = "*.json",
  use_regex: bool = False,
  search_directory: str = None
) -> str:
  """
  Search patterns in code files
  
  Args:
    pattern: Search pattern
    file_pattern: File pattern (e.g., '*.py)
    use_regex: Whether to use regular expressions
    search_directory: Specify search directory (optional, uses WORKSPACE_DIR if not specified)
    
  Returns:
    JSON string of search results
  """
  
  
# ==================== File Structure Tools ====================

@mcp.tool()
async def get_file_structure(directory: str = ".", max_depth: int = 5) -> str:
  """
  Get directory file structure
  
  Args:
    directory: Directory path, relative to workspace
    max_depth: Maximum traversal depth
  
  Returns:
    JSON string of file structure
  """
  
  
# ==================== Workspace Management Tools ====================

@mcp.tool()
async def set_workspace(workspace_path: str) -> str:
  """
  Set workspace directory
  
  Called by workflow to set workspace to: {plan_file_parent}/generate_code
  This ensures all file operations are executed relative to the correct project directory
  
  Args:
    workspace_path: Workspace path (Usually {plan_file_parent}/generate_code)
    
  Returns:
    JSON string of operation result
  """
  
  
@mcp.tool()
async def get_operation_history(last_n: int = 10) -> str:
  """
  Get operation history
  
  Args:
    last_n: Return the last N operations
    
  Returns:
    JSON string of operation history
  """
  
  
# ==================== Server Initialization ====================

def main():
  
  # Start server
  mcp.run()
  

if __name__ == "__main__":
  main()