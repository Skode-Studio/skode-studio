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
from utils.llm_utils import get_preferred_llm_class




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
    if self.llm_client is not None:
      return self.llm_client, self.llm_client_type
    
    # Check if mock responses are enabled
    if self.mock_llm_responses:
      self.logger.info("Using mock LLM responses for testing")
      self.llm_client = "mock"
      self.llm_client_type = "mock"
      return "mock", "mock"
    
    # Check which API has available key and try that first
    anthropic_key = self.api_config.get("anthropic", {}).get("api_key", "")
    openai_key = self.api_config.get("openai", {}).get("api_key", "")
    
    # Try Anthropic API first if key is available
    if anthropic_key and anthropic_key.strip():
      try:
        from anthropic import AsyncAnthropic
        
        client = AsyncAnthropic(
          api_key=anthropic_key
        )
        
        # Test connection with default model from config
        await client.messages.create(
          model=self.default_models["anthropic"],
          max_tokens=10,
          messages=[
            {
              "role": "user",
              "content": "test"
            }
          ]
        )
        self.logger.info(f"Using Anthropic API with model: {self.default_models["anthropic"]}")
        
        self.llm_client = client
        self.llm_client_type = "anthropic"
        
        return client, "anthropic"
      
      except Exception as e:
        self.logger.warning(f"Anthropic API unavailable: {e}")
        
        
    # Try OpenAI API if Anthropic failed or key not available
    if openai_key and openai_key.strip():
      try:
        from openai import AsyncOpenAI
        
        # Hanlde custom base_url if specified
        openai_config = self.api_config.get("openai", {})
        base_url = openai_config.get("base_url")
        
        if base_url:
          client = AsyncOpenAI(
            api_key=openai_key,
            base_url=base_url
          )
        else:
          client = AsyncOpenAI(
            api_key=openai_key
          )
          
        # Test connection with default model from config
        await client.chat.completions.create(
          model=self.default_models["openai"],
          max_tokens=10,
          messages=[
            {
              "role": "user",
              "content": "test"
            }
          ]
        )
        self.logger.warning(f"OpenAI API unavailable: {e}")
        
        if base_url:
          self.logger.info(f"Using custom base URL: {base_url}")
        self.llm_client = client
        self.llm_client_type = "openai"
        
        return client, "openai"
        
      except Exception as e:
        self.logger.warning(f"OpenAI API unavailable: {e}")
    
    raise ValueError(
      "No available LLM API - please check your API keys in configuration"
    )
      
    
    
    
    
  async def _call_llm(
    self,
    prompt: str,
    system_prompt: str = None,
    max_tokens: int = None
  ) -> str:
    """Call LLM for code analysis with retry mechanism and debugging support"""

    if system_prompt is None:
      system_prompt = self.llm_system_prompt
    if max_tokens is None:
      max_tokens = self.llm_max_tokens
      
    # Mock response for testing
    if self.mock_llm_responses:
      mock_response = self._generate_mock_response(prompt)
      if self.save_raw_responses:
        self._save_debug_response("mock", prompt, mock_response)
      return mock_response
    
    last_error = None
    
    # Retry mechanism
    for attempt in range(self.max_retries):
      try:
        if self.verbose_output and attempt > 0:
          self.logger.info(
            f"LLM call attempt {attempt + 1}/{self.max_retries}"
          )
          
        client, client_type = await self._initialize_llm_client()
        
        # === ANTHROPIC ===
        if client_type == "anthropic":
          response = await client.messages.create(
            model=self.default_models["anthropic"],
            system=system_prompt,
            messages=[
              {
                "role": "user",
                "content": prompt
              }
            ],
            max_tokens=max_tokens,
            temperature=self.llm_temperature
          )
          
          content = ""
          for block in response.content:
            if block.type == "text":
              content += block.text
              
          # Save debug response if enabled
          if self.save_raw_responses:
            self._save_debug_response("anthropic", prompt, content)
            
          return content
        
        # === OPENAI ===
        elif client_type == "openai":
          messages = [
            {
              "role": "system",
              "content": system_prompt
            },
            {
              "role": "user",
              "content": prompt
            }
          ]
          
          response = await client.chat.completions.create(
            model=self.default_models["openai"],
            messages=messages,
            max_tokens=max_tokens
          )
          
          content = response.choices[0].message.content or ""
          
          # Save debug response if enabled
          if self.save_raw_responses:
            self._save_debug_response("openai", prompt, content)
            
          return content
        
        # === OTHERS ===
        else:
          raise ValueError(f"Unsupported client type: {client_type}")
        
      except Exception as e:
        last_error = e
        self.logger.warning(f"LLM call attempt {attempt + 1} failed {e}")
        
        if attempt < self.max_retries - 1:
          await asyncio.sleep(
            self.retry_delay * (attempt + 1)
          ) # Exponential backoff

    
    # All retries failed
    error_msg = f"LLM call failed after {self.max_retries} attempts. Last error: {str(last_error)}"
    self.logger.error(error_msg)
    return f"Error in LLM analysis: {error_msg}"
  
  


  def _generate_mock_response(
    self, 
    prompt: str
  ) -> str:
    """Generate mock LLM response for testing"""
    
    # CASE 1: File analysis mock
    if "JSON format" in prompt and "file_type" in prompt:
      return """
      {
        "file_type": "Python module",
        "main_functions": ["main_function", "helper_function"],
        "key_concepts": ["data_processing", "algorithm"],
        "dependencies": ["numpy", "pandas"],
        "summary": "Mock analysis of code file functionality."
      }
      """
    
    # CASE 2: Relationship analysis mock
    elif "relationships" in prompt:
      return """
      {
        "relationships": [
          {
            "target_file_path": "src/core/mock.py",
            "relationship_type": "partial_match",
            "confidence_score": 0.8,
            "helpful_aspects": ["algorithm implementation", "data structures"],
            "potential_contributions": ["core functionality", "utility methods"],
            "usage_suggestions": "Mock relationship suggestion for testing"
          }
        ]
      }
      """
      
    # CASE 3: File filtering mock
    elif "relevant_files" in prompt:
      return """
      {
        "relevant_files": [
          {
            "file_path": "mock_file.py",
            "relevance_reason": "Mock relevance reason",
            "confidence": 0.9,
            "expected_contribution": "Mock contribution"
          }
        ],
        "summary": {
          "total_files_analyzed": "10",
          "relevant_files_count": "1",
          "filtering_strategy": "Mock filtering strategy"
        }
      }
      """
    else:
      return "Mock LLM response for testing purposes."

    
      
    
    
  
  def _save_debug_response(
    self,
    provider: str,
    prompt: str,
    response: str
  ):
    """Save LLM response for debugging"""
  
  
  
  def get_all_repo_files(self, repo_path: Path) -> List[Path]:
    """Recursively get all supported files in a repository"""
    
    files = []
    
    try:
      
      for root, dirs, filenames in os.walk(repo_path):
        # Skip common non-code directories
        
        dirs[:] = [
          d
          for d in dirs
          if not d.startswith(".") and d not in self.skip_directories
        ]
        
        for filename in filenames:
          file_path = Path(root) / filename
          if file_path.suffix.lower() in self.supported_extensions:
            files.append(file_path)
      
    except Exception as e:
      self.logger.error(f"Error traversing {repo_path}: {e}")
      
    return files
    
  
  
  
  def generate_file_tree(self, repo_path: Path, max_depth: int = 5) -> str:
    """Generate file tree structure string for the repository"""
    
    tree_lines = []
    
    # Recursive to crawl file structure
    def add_to_tree(current_path: Path, prefix: str = "", depth: int = 0):
      """Add to Tree"""
      
      if depth > max_depth:
        return
      
      try:
        items = sorted(
          current_path.iterdir(),
          # Use lambdas for short, simple functions 
          key=lambda x: (x.is_file(), x.name.lower())
        )
        
        # Filter out irrelevant directories and files
        items = [
          item
          for item in items
          if not item.name.startswith(".") and item.name not in self.skip_directories
        ]
        
        # Compose structure
        for i, item in enumerate(items):
          is_last = i == len(items) - 1
          current_prefix = "└── " if is_last else "├── "
          tree_lines.append(f"{prefix}{current_prefix}{item.name}")
          
          if item.is_dir():
            extension_prefix = "    " if is_last else "│   "
            add_to_tree(item, prefix + extension_prefix, depth + 1)
          elif item.suffix.lower() in self.supported_extensions:
            # Add file size information
            try:
              size = item.stat().st_size
              
              if size > 1024:
                size_str = f" ({size // 1024}KB)"
              else:
                size_str = f" ({size}B)"
                
              tree_lines[-1] += size_str
            except (OSError, PermissionError):
              pass
      
      except PermissionError:
        tree_lines.append(f"{prefix}├── [Permission Denied]")
      except Exception as e:
        tree_lines.append(f"{prefix}├── [Error: {str(e)}]")      
    
    
    tree_lines.append(f"{repo_path.name}/")
    add_to_tree(repo_path)
    return "\n".join(tree_lines)
    
  

  async def pre_filter_files(self, repo_path: Path, file_tree: str) -> List[str]:
    """Use LLM to pre-filter relevant files based on target structure"""
    
    filter_prompt = f"""
    You are a code analysis expert, Please analyze the following code repository file tree based on the targer structure and filter out files that may be relevant to the target project
    
    Target Project Structure:
    {self.target_structure}
    
    Code Repository Tree:
    {file_tree}
    
    Please analyze which files might be helpful for implementing the target project structur, including:
    - Core algorithm implementation files (such as GCN, recommendation system, graph neural networks, etc.)
    - Data processing and preprocessing files
    - Lost functions and evaluation metric files
    - Test files
    - Documentation files
    
    Please return the filtering resulst in JSON format:
    {{
      "relevant_files": [
        {{
          "file_path": "file path relative to repository root",
          "relevance_reason": "why this file relevant",
          "confidence": 0.0-1.0,
          "expected_contribution": "expected contribution to the target project"
        }}
      ],
      "summary": {{
        "total_files_analyzed": "total number of files analyzed",
        "relevant_files_count": "number of relevant files",
        "filtering_strategy": "explanation of filtering strategy"
      }}
    }}
    
    Only return files with confidence > {self.min_confidence_score}, Focus on files related to recommendation systems, graph neural networks, and diffusion models.
    """
    
    
      
  
  
  
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
  
    repo_name = repo_path.name
    self.logger.info(f"Processing repository: {repo_name}")
  
    # ========== Step 1: Generate file tree ==========
    self.logger.info("Generating file tree structure")
    file_tree = self.generate_file_tree(repo_path, max_depth=5)
    
    # ========== Step 2: Get all files ==========
    all_files = self.get_all_repo_files(repo_path)
    self.logger.info(f"Found {len(all_files)} files in {repo_name}")
    
    # ========== Step 3: LLM pre-filtering of relevant files ==========
    
    
    # ========== Step 4: Filter file list based on filtering results ==========

    # ========== Step 5: Analyze filtered files (concurrent or sequential) ==========
    
    # ========== Step 6: Create repository index ==========
    
    
    
  async def _process_files_sequentially(self, files_to_analyze: list) -> tuple:
    """Process files sequentially (original method)"""
    
    
    
  async def _process_files_concurrently(self, files_to_analysis: list) -> tuple:
    """Process files concurrently with semaphore limiting"""
    
    
    
  async def build_all_indexes(self) -> Dict[str, str]:
    """Build indexes for all repositories in code_base"""
    
    if not self.code_base_path.exists():
      raise FileNotFoundError(
        f"Code base path does not exist: {self.code_base_path}"
      )
      
    # Get all repository directories
    # /my_project_example/
    # ├── src/           ← included
    # ├── tests/         ← included  
    # ├── docs/          ← included
    # ├── .git/          ← excluded (hidden)
    # ├── .vscode/       ← excluded (hidden)
    # └── README.md      ← excluded (not a directory)
    repo_dirs = [
      d
      for d in self.code_base_path.iterdir()
      if d.is_dir() and not d.name.startswith(".")
    ]
    
    if not repo_dirs:
      raise ValueError(f"No repositories found in {self.code_base_path}")
    
    self.logger.info(f"Found {len(repo_dirs)} repositories to process")

    
    
    
    
    
  def _extract_repository_statistics(self, repo_index: RepoIndex) -> Dict[str, Any]:
    """Extract statistical information from a repository index"""
    
    
    
  def generate_statistics_report(self, statistics_data: List[Dict[str, Any]]) -> str:
    """Generate a detailed statistics report"""
    
    
    
  def generate_summary_report(self, output_files: Dict[str, str]) -> str:
    """Generate a summary report of all indexes created"""
    



async def main():
  """Main function to run the code indexer with full configuration support"""
  
  # Configuration - can be overridden by config file
  config_file = "DeepCode/tools/indexer_config.yaml"
  api_config_file = "DeepCode/mcp_agent.secrets.yaml"

  # You can override these parameters or let them be read from config
  code_base_path = "DeepCode/deepcode_lab/papers/1/code_base" # Will use config file value if None
  output_dir = (
    "DeepCode/deepcode_lab/papers/1/indexes/" # Will use config file if None
  )
  
  # Taret structure - this shoulde be customized for your specific project
  target_structure = """
                    project/
                    ├── src/
                    │   ├── core/
                    │   │   ├── gcn.py        # GCN encoder
                    │   │   ├── diffusion.py  # forward/reverse processes
                    │   │   ├── denoiser.py   # denoising MLP
                    │   │   └── fusion.py     # fusion combiner
                    │   ├── models/           # model wrapper classes
                    │   │   └── recdiff.py
                    │   ├── utils/
                    │   │   ├── data.py       # loading & preprocessing
                    │   │   ├── predictor.py  # scoring functions
                    │   │   ├── loss.py       # loss functions
                    │   │   ├── metrics.py    # NDCG, Recall etc.
                    │   │   └── sched.py      # beta/alpha schedule utils
                    │   └── configs/
                    │       └── default.yaml  # hyperparameters, paths
                    ├── tests/
                    │   ├── test_gcn.py
                    │   ├── test_diffusion.py
                    │   ├── test_denoiser.py
                    │   ├── test_loss.py
                    │   └── test_pipeline.py
                    ├── docs/
                    │   ├── architecture.md
                    │   ├── api_reference.md
                    │   └── README.md
                    ├── experiments/
                    │   ├── run_experiment.py
                    │   └── notebooks/
                    │       └── analysis.ipynb
                    ├── requirements.txt
                    └── setup.py
                    """
  
  print("Starting Code Indexer with Enhanced COnfiguration Support")
  print("Configuration file: {config_file}")
  print("API Configuration file: {api_config_file}")
  
  # Create indexer with fill configuration support
  try:
    
    # Initialize Code Indexer
    indexer = CodeIndexer(
      code_base_path=code_base_path, # None = read from config
      target_structure=target_structure,
      output_dir=output_dir,
      config_path=api_config_file,
      indexer_config_path=config_file,
      enable_pre_filtering=True # Can be overridden in config
    )
    
    # Display configuration information
    print(f"📁 Code base path: {indexer.code_base_path}")
    print(f"📂 Output directory: {indexer.output_dir}")
    print(
        f"🤖 Default models: Anthropic={indexer.default_models['anthropic']}, OpenAI={indexer.default_models['openai']}"
    )
    print(f"🔧 Preferred LLM: {get_preferred_llm_class(api_config_file).__name__}")
    print(
        f"⚡ Concurrent analysis: {'enabled' if indexer.enable_concurrent_analysis else 'disabled'}"
    )
    print(
        f"🗄️  Content caching: {'enabled' if indexer.enable_content_caching else 'disabled'}"
    )
    print(
        f"🔍 Pre-filtering: {'enabled' if indexer.enable_pre_filtering else 'disabled'}"
    )
    print(f"🐛 Debug mode: {'enabled' if indexer.verbose_output else 'disabled'}")
    print(
        f"🎭 Mock responses: {'enabled' if indexer.mock_llm_responses else 'disabled'}"
    )
    
    
    # Validate configuration
    if not indexer.code_base_path.exists():
      raise FileNotFoundError(
        f"Code base path doesnot exist: {indexer.code_base_path}"
      )
      
    if not target_structure:
      raise ValueError("Target structure is required for analysis")
    
    
    print("STARTING INDEXING PROCESS...")
    
    
    
    
    
  except FileNotFoundError as e:
    print(f"❌ File not found error: {e}")
    print("💡 Please check your configuration file paths")
  except ValueError as e:
    int(f"❌ Configuration error: {e}")
    print("💡 Please check your configuration file settings")
  except Exception as e:
    print(f"❌ Indexing failed: {e}")
    print("💡 Check the logs for more details")
    
    
    
    
  
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
