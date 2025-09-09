"""
Code Reference Indexer MCP Tool - Unified Version

Specialized MCP Tool for searching relevant index content in indexes folder
and formatting it for LLM Code implementation reference

Core Features:
1. **UNIFIED TOOL**L Combined search_code_references that hanldes directory setup, loading, and searching in one call
2. Match relevant reference code based on target file path and functionality requirements
3. Format output of relevant code examples, funcitons and concepts
4. Provide structured reference information for LLM use

Key Inprovement:
- Single tool call that handles all steps internally
- Agent only needs to provide indexes_path and target_file
- No dependency on calling order or global state management
"""


from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import json


# Import MCP modules
from mcp.server.fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create FastMCP server instance
mcp = FastMCP("code-reference-indexer")



@dataclass
class CodeReference:
  """Code Reference information structure"""

  file_path: str
  file_type: str
  main_functions: List[str]
  key_concepts: List[str]
  dependencies: List[str]
  summary: str
  lines_of_code: int
  repo_name: str
  confidence_score: float = 0.0


@dataclass
class RelationshipInfo:
  """Relationship information structure"""
  
  repo_file_path: str
  target_file_path: str
  relationship_type: str
  confidence_score: float
  helpful_aspects: List[str]
  potential_contributions: List[str]
  usage_suggestions: str
  


def load_index_files_from_directory(indexes_directory: str) -> Dict[str, Dict]:
  """Load all index files from specified directory"""
  
  indexes_path = Path(indexes_directory).resolve()
  
  if not indexes_path.exists():
    logger.warning(f"Indexes directory does not exist: {indexes_path}")
    return {}
  
  index_cache = {}
  
  for index_file in indexes_path.glob("*.json"):
    try:
      with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)
        index_cache[index_file.stem] = index_data
        logger.info(f"Loaded index file: {index_file.name}")
    except Exception as e:
      logger.error(f"Failed to load index file {index_file.name}: {e}")
      
  logger.info(f"Loaded {len(index_cache)} index files from {indexes_path}")
  return index_cache
  
  

def extract_code_references(index_data: Dict) -> List[CodeReference]:
  """Extract code reference information from index data"""
  
  

def extract_relationships(index_data: Dict) -> List[RelationshipInfo]:
  """Extract relationship information from index data"""



def calculate_relevance_score(
  target_file: str, reference: CodeReference, keywords: List[str] = None
) -> float:
  """Calculate relevance score between reference code and target file"""



def find_relevant_references_in_cache(
  target_file: str,
  index_cache: Dict[str, Dict],
  keywords: List[str] = None,
  max_results: int = 10
) -> List[Tuple[CodeReference, float]]:
  """Find reference code relevant to target file from provided cache"""



def find_direct_relationships_in_cache(
  target_file: str, index_cache: Dict[str, Dict]
) -> List[RelationshipInfo]:
  """Find direct relationships with target file form provided cache"""



def format_reference_output(
  target_file: str,
  relevant_refs: List[Tuple[CodeReference, float]],
  relationships: List[RelationshipInfo]
) -> str:
  """Format reference information output"""


# ==================== MCP Tool Definitions ====================

@mcp.tool()
async def search_code_references(
  indexes_path: str, 
  target_file: str, 
  keywords: str = "", 
  max_results: int = 10
) -> str:
  """
  **UNIFIED TOOL**: Search relevant reference code from index files for target file implementation.
  This tool combines directory setup, index loading, and searching in a single call.
  
  Args:
    indexes_path: Path to the indexes directory containing JSON index files
    target_file: Target file path (file to be implemented)
    keywords: Search keywords, comma-separated
    max_results: Maximum number of results to return
    
  Returns:
    Formatted reference code information JSON string
  """
  
  try:
    """"""
    # Step 1: Load index files from specified directory
    
    # Step 2: Parse keywords
    
    # Step 3: Find relevant reference code
    
    # Step 4: Find direct relationships
    
    # Step 5: Format output
    
  except Exception as e:
    """"""



@mcp.tool()
async def get_indexes_overview(indexes_path: str) -> str:
  """
  Get overview of all available reference code index information from specified directory
  
  Args:
    indexes_path: Path to the indexes directory containing JSON index files
    
  Returns:
    Overview information of all available reference code JSON string
  """
  
  try:
    # Load index files from specified directory
    index_cache = load_index_files_from_directory(indexes_path)
    
    if not index_cache:
      result = {
        "status": "error",
        "message": f"No index files found in: {indexes_path}",
        "indexes_path": indexes_path
      }
      return json.dumps(result, ensure_ascii=False, indent=2)
    
    overview = {
      "total_repos": len(index_cache),
      "repositories": {}
    }
    
    for repo_name, index_data in index_cache.items():
      repo_info = {
        "repo_name": index_data.get("repo_name", repo_name),
        "total_files": index_data.get("total_files", 0),
        "file_types": [],
        "main_concepts": [],
        "total_relationships": len(index_data.get("relationships", []))
      }
      
      # Collect file types and concepts
      file_summaries = index_data.get("file_summaries", [])
      file_types = set()
      concepts = set()
      
      for file_summary in file_summaries:
        file_types.add(file_summary.get("file_type", "Unknown"))
        concepts.update(file_summary.get("key_concepts", []))
        
      repo_info["file_types"] = sorted(list(file_types))
      repo_info["main_concepts"] = sorted(list(concepts))[:20] # Limit concept count
      
      overview["repositories"][repo_name] = repo_info
      
    
    result = {
      "status": "success",
      "overview": overview,
      "indexes_directory": str(Path(indexes_path).resolve()),
      "total_indexes_loaded": len(index_cache)
    }
    
    return json.dumps(result, ensure_ascii=False, indent=2)
  
  except Exception as e:
    result = {
      "status": "error",
      "message": f"Failed to get indexes overview: {str(e)}",
      "indexes_path": indexes_path
    }
    return json.dumps(result, ensure_ascii=False, indent=2)
    
    
    



def main():
  """Main function"""
  logger.info("Starting unified Code Reference Indexer MCP server")
  logger.info("Available tools:")
  logger.info(
    "1. search_code_references(indexes_path, target_file, keywords, max_results) - UNIFIED TOOL"
  )
  logger.info(
    "2. get_indexes_overview(indexes_path) - Get overview of available indexes"
  )
  
  # Run MCP Server  
  mcp.run()
  

if __name__ == "__main__":
  main()