"""
Code Indexer for Repository Analysis

Analyzes code repositories to build comprehensive indexes for each subdirectory,
identifying file relationships and reusable components for implementation.

Features:
- Recursive file traversal
- LLM-powered code similarity analysis using augmented LLM classes
- JSON-based relationship storage
- Configurable matching strategies
- Progress tracking and error handling
- Automatic LLM provider selection based on API key availability
"""


from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import asyncio
import yaml
import os
from pathlib import Path




def get_default_models(config_path: str = "mcp_agent.config.yaml"):
  """
  Get default models from configuration file.
  
  Args:
    config_path: Path to the configuration file
    
  Returns:
    dict: Dictionary with 'anthropic' and 'openai' default models
  """
  try:
    if os.path.exists(config_path):
      with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
      anthropic_model = config.get("anthropic", {}).get(
        "default_model", "claude-sonnet-4-20250514"
      )

      openai_model = config.get("openai", {}).get(
        "default_model", "o3-mini"
      )
      
      return {
        "anthropic": anthropic_model,
        "openai": openai_model
      }
      
    else:
      print(f"Config file {config_path} not found, using default models")
      return {"anthropic": "claude-sonnet-4-20250514", "openai": "o3-mini"}
    
  except Exception as e:
    print(f"Error reading config file {config_path}: {e}")
    return {"anthropic": "claude-sonnet-4-20250514", "openai": "o3-mini"}
    
  
  
  
@dataclass
class FileRelationship:
  """Represents a relationship between a repo file and target structure file"""
  
  repo_file_path: str
  target_file_path: str
  relationship_type: str # 'direct_match', 'partial_match', 'reference', 'utility'
  confidence_score: float # 0.0 to 1.0
  helpful_aspects: List[str]
  potential_contribution: List[str]
  usage_suggestions: str
  
  

@dataclass
class FileSummary:
  """Summary information for a repository file"""
  
  file_path: str
  file_type: str
  main_functions: List[str]
  key_concepts: List[str]
  dependencies: List[str]
  summary: str
  lines_of_code: int
  last_modified: str
  
  

@dataclass
class RepoIndex:
  """Complete index for a repository"""
  
  repo_name: str
  total_files: int
  file_summaries: List[FileSummary]
  relationships: List[FileRelationship]
  analysis_metadata: Dict[str, Any]



class CodeIndexer:
  """Main class for building code repository indexes"""
  
  def __init__(
    self,
    code_base_path: str = None,
    target_structure: str = None,
    output_dir: str = None,
    config_path: str = "mcp_agent.secrets.yaml",
    indexer_config_path: str = None,
    enable_pre_filtering: bool = True
  ):
    """Initialization"""
    
    # Load configurations first
    self.config_path = config_path
    self.indexer_config_path = indexer_config_path
    self.api_config = self._load_api_config()
    self.indexer_config = self._load_indexer_config()
    self.default_models = get_default_models("mcp_agent.config.yaml")
    
    
    # Use config paths if not provided as parameters
    paths_config = self.indexer_config.get("paths", {})
    self.code_base_path = Path(
      code_base_path or paths_config.get("code_base_path", "code_base")
    )
    self.output_dir = Path(output_dir or paths_config.get("output_dir", "indexes"))
    self.target_structure = (
      target_structure # This must be provided as it's project specific
    )
    self.enable_pre_filtering = enable_pre_filtering
    
    
    # LLM clients
    self.llm_client = None
    self.llm_client_type = None
    
    
    # Initialize logger early
    self.logger = self._setup_logger()
    
    
    # Create output directory if it doesn't exist
    self.output_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Load file analysis configuration
    file_analysis_config = self.indexer_config.get("file_analysis", {})
    self.supported_extensions = set(
      file_analysis_config.get(
        "supported_extensions",
        [
          ".py",
          ".js",
          ".ts",
          ".java",
          ".cpp",
          ".c",
          ".h",
          ".hpp",
          ".cs",
          ".php",
          ".rb",
          ".go",
          ".rs",
          ".scala",
          ".kt",
          ".swift",
          ".m",
          ".mm",
          ".r",
          ".matlab",
          ".sql",
          ".sh",
          ".bat",
          ".ps1",
          ".yaml",
          ".yml",
          ".json",
          ".xml",
          ".toml",
        ]
      )
    )
    self.skip_directories = set(
      file_analysis_config.get(
        "skip_directories",
        [
          "__pycache__",
          "node_modules",
          "target",
          "build",
          "dist",
          "venv",
          "env",
        ]
      )
    )
    self.max_file_size = file_analysis_config.get("max_file_size", 1048576)
    self.max_content_length = file_analysis_config.get("max_content_length", 3000)
    
    
    # Load LLM configuration
    llm_config = self.indexer_config.get("llm", {})
    self.model_provider = llm_config.get("model_provider", "anthropic")
    self.llm_max_tokens = llm_config.get("max_tokens", 4000)
    self.llm_temperature = llm_config.get("temperature", 0.3)
    self.llm_system_prompt = llm_config.get(
      "system_prompt",
      "You are a code analysis expert. Provide precise, structured analysis of code relationships and similarities."
    )
    self.request_delay = llm_config.get("request_delay", 0.1)
    self.max_retries = llm_config.get("max_retries", 3)
    self.retry_delay = llm_config.get("retry_delay", 1.0)
    
    
    # Load relationship configuration
    relationship_config = self.indexer_config.get("relationships", {})
    self.min_confidence_score = relationship_config.get("min_confidence_score", 0.3)
    self.high_confidence_threshold = relationship_config.get(
      "high_confidence_threshold", 0.7
    )
    self.relationship_types = relationship_config.get(
      "relationship_types",
      {
        "direct_match": 1.0,
        "partial_match": 0.8,
        "reference": 0.6,
        "utility": 0.4,
      },
    )
    
    
    # Load performance configuration
    performace_config = self.indexer_config.get("performance", {})
    self.enable_concurrent_analysis = performace_config.get(
      "enable_concurrent_analysis", False
    )
    self.max_concurrent_files = performace_config.get("max_concurrent_files", 5)
    self.enable_content_caching = performace_config.get("enable_content_caching", False)
    self.max_cache_size = performace_config.get("max_cache_size", 100)
    
    
    # Load debug configuration
    debug_config = self.indexer_config.get("debug", {})
    self.save_raw_responses = debug_config.get("save_raw_responses", False)
    self.raw_responses_dir = debug_config.get(
      "raw_responses_dir", "debug_responses"
    )
    self.verbose_output = debug_config.get("verbose_output", False)
    self.mock_llm_responses = debug_config.get("mock_llm_responses", False)
    
    
    # Load output configuration
    output_config = self.indexer_config.get("output", {})
    self.generate_summary = output_config.get("generate_summary", True)
    self.generate_statistics = output_config.get("generate_statistics", True)
    self.include_metadata = output_config.get("include_metadata", True)
    self.index_filename_pattern = output_config.get(
      "index_filename_pattern", "{repo_name}_index.json"
    )
    self.summary_filename = output_config.get(
      "summary_filename", "indexing_summary.json"
    )
    self.stats_filename = output_config.get(
      "stats_filename", "indexing_statistics.json"
    )
    
    
    # Initialize caching if enabled
    self.content_cache = {} if self.enable_content_caching else None
    
    
    # Create debug directory if needed
    if self.save_raw_responses:
      Path(self.raw_responses_dir).mkdir(parents=True, exist_ok=True)
    
    
    # Debug logging
    if self.verbose_output:
      self.logger.info(
        f"Initialized CodeIndexer with config: {self.indexer_config_path}"
      )
      self.logger.info(f"Code base path: {self.code_base_path}")
      self.logger.info(f"Output directory: {self.output_dir}")
      self.logger.info(f"Model provider: {self.model_provider}")
      self.logger.info(f"Concurrent analysis: {self.enable_concurrent_analysis}")
      self.logger.info(f"Content caching: {self.enable_content_caching}")
      self.logger.info(f"Mock LLM responses: {self.mock_llm_responses}")
    
    
    

  def _setup_logger(self) -> logging.Logger:
    """Setup logging configuration from config file"""
    logger = logging.getLogger("CodeIndexer")
    
    # Get logging config
    logging_config = self.indexer_config.get("logging", {})
    log_level = logging_config.get("level", "INFO")
    log_format = logging_config.get(
      "log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # File handler if enabled
    if logging_config.get("log_to_file", False):
      log_file = logging_config.get("log_file", "indexer.log")
      file_handler = logging.FileHandler(log_file, encoding="utf-8")
      file_handler.setFormatter(formatter)
      logger.addHandler(file_handler)
      
    return logger
    
    
    
  
  def _load_api_config(self) -> Dict[str, Any]:
    """Load API configuration from YAML file"""
    
    try:
      with open(self.config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
    except Exception as e:
      # Create a basic logger for this error since self.logger doesn't exist yet
      print(f"Warning: Failed to load API config from {self.config_path}: {e}")
      return {}
    
    
  
  def _load_indexer_config(self) -> Dict[str, Any]:
    """Load indexer configuration from YAML file"""
    
    try:
      with open(self.indexer_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        if config is None:
          config = {}
          
        return config
    
    except Exception as e:
      print(
        f"Warning: Failed to load indexer config from {self.indexer_config_path}: {e}"
      )
      print("Using default configuration values")
      return {}
    
    
  
  async def _initialize_llm_client(self):
    """Initialize LLM client (Anthropic or OpenAI) based on API key availability"""
    
    
    
  async def _call_llm(
    self,
    prompt: str,
    system_prompt: str = None,
    max_tokens: int = None
  ) -> str:
    """Call LLM for code analysis with retry mechanism and debugging support"""



  def _generate_mock_response(self, prompt: str) -> str:
    """Generate mock LLM response for testing"""
    
    
  
  def _save_debug_response(
    self,
    provider: str,
    prompt: str,
    response: str
  ):
    """Save LLM response for debugging"""
  
  
  
  def get_all_repo_files(self, repo_path: Path) -> List[Path]:
    """Recursively get all supported files in a repository"""
  
  
  
  def generate_file_tree(self, repo_path: Path, max_depth: int = 5) -> str:
    """Generate file tree structure string for the repository"""
  
  
  
    def add_to_tree(current_path: Path, prefix: str = "", depth: int = 0):
      """Add to Tree"""
    
  

  async def pre_filter_files(self, repo_path: Path, file_tree: str) -> List[str]:
    """Use LLM to pre-filter relevant files based on target structure"""
  
  
  
  def filter_files_by_paths(
    self,
    all_files: List[Path],
    selected_paths: List[str],
    repo_path: Path
  ) -> List[Path]:
    """Filter file list based on LLM-selected paths"""
    
    
  
  def _get_cache_key(self, file_path: Path) -> str:
    """Generate cache key for file content"""
    
  
  
  def _manage_cache_size(self):
    """Manage cache size to stay within limits"""
    
    
  
  async def analyze_file_content(self, file_path: Path) -> FileSummary:
    """Analyze a single fiel and create summary with caching support"""

    try:
      """"""
      # Check file size before reading
      
      # Check cache if enabled
      
      # Get file stats
      
      # Truncate content based on config
      
      # Create analysis prompt
      
      # Get LLM analysis with configured parameters
      
      # Cache the result if caching is enabled
      
    except Exception as e:
      """"""
      
  
  
  async def find_relationships(self, file_summary: FileSummary) -> List[FileRelationship]:
    """Find relationships between a repo file and target structure"""    
  
  
  
  async def _analyze_single_file_with_relationships(
    self,
    file_path: Path,
    index: int,
    total: int
  ) -> tuple:
    """Analyze a single file and its relationships (for concurrent processing)"""
  
  
  
  async def process_repository(self, repo_path: Path) -> RepoIndex:
    """Process a single repository and create complete index with optional concurrent processing"""
  
    # Step 1: Generate file tree
  
    # Step 2: Get all files
  
    # Step 3: LLM pre-filtering of relevant files
    
    # Step 4: Filter file list based on filtering results

    # Step 5: Analyze filtered files (concurrent or sequential)
    
    # Step 6: Create repository index
    
    
    
  async def _process_files_sequentially(self, files_to_analyze: list) -> tuple:
    """Process files sequentially (original method)"""
    
    
    
  async def _process_files_concurrently(self, files_to_analysis: list) -> tuple:
    """Process files concurrently with semaphore limiting"""
    
    
    
  async def build_all_indexes(self) -> Dict[str, str]:
    """Build indexes for all repositories in code_base"""
    
    
    
  def _extract_repository_statistics(self, repo_index: RepoIndex) -> Dict[str, Any]:
    """Extract statistical information from a repository index"""
    
    
    
  def generate_statistics_report(self, statistics_data: List[Dict[str, Any]]) -> str:
    """Generate a detailed statistics report"""
    
    
    
  def generate_summary_report(self, output_files: Dict[str, str]) -> str:
    """Generate a summary report of all indexes created"""
    



async def main():
  """Main function to run the code indexer with full configuration support"""
  
  
  
def print_usage_example():
    """Print usage examples for different scenarios"""
    print("""
    📖 Code Indexer Usage Examples:

    1. Basic usage with config file:
       - Update paths in indexer_config.yaml
       - Run: python code_indexer.py

    2. Enable debugging:
       - Set debug.verbose_output: true in config
       - Set debug.save_raw_responses: true to save LLM responses

    3. Enable concurrent processing:
       - Set performance.enable_concurrent_analysis: true
       - Adjust performance.max_concurrent_files as needed

    4. Enable caching:
       - Set performance.enable_content_caching: true
       - Adjust performance.max_cache_size as needed

    5. Mock mode for testing:
       - Set debug.mock_llm_responses: true
       - No API calls will be made

    6. Custom output:
       - Modify output.index_filename_pattern
       - Set output.generate_statistics: true for detailed reports

    📋 Configuration file location: tools/indexer_config.yaml
    """)
    
    
    
if __name__ == "__main__":
  import sys

  if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
    print_usage_example()
  else:
    asyncio.run(main())
