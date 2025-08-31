"""
Document Segmentation MCP Server

This MCP Server provides intelligent document segmentation adn retrieval functions for handling
large research papers and technical documents that exceed LLM token limits.

==== CORE FUNCTIONALITY ====
1. Analyze document structure and type using semantic content analysis
2. Create intelligent segments based on content semantics, not just structure
3. Provide query-aware segment retrieval with relevance scoring
4. Support both structured (papers with headers) and unstructured documents
5. Configurable segmentation strategies based on document complexity

==== MCP TOOLS PROVIDED ====

📄  analyze_and_segment_document(
      paper_dir: str, 
      force_refresh: bool = False
    )
    > Purpose: Analyzes document structure and creates intelligent segments
    - Detects document type (research paper, technical doc, algorithm-focused, etc.)
    - Selects optimal segmentation strategy based on content analysis
    - Creates semantic segments preserving algorithm and concept integrity
    - Stores segmentation index for efficient retrieval
    - Returns: JSON with segmentation status, strategy used, and segment count
    
    
    
📖  read_document_segments(
      paper_dir: str,
      query_type: str,
      keywords: List[str] = None,
      max_segments: int = 3,
      max_total_chars: int = None
    )
    > Purpose: Intelligently retrieves relevant document segments based on query context
    - query_type: "concept_analysis", "algorithm_extraction", or "code_planning"
    - Uses semantic relevance scoring to rank segments
    - Applies query-specific filtering and keyword matching
    - Dynamically calculates optimal character limits based on content complexity
    - Returns: JSON with selected segments optimized for the specific query type
    
    
    
📋  get_document_overview(
      paper_dir: str
    )
    > Purpose: Provides high-level overview of document structure and available segments
    - Shows document type and segmentation strategy used
    - Lists all segments with titles, content types, and relevance scores
    - Displays segment statistics (character counts, keyword summaries)
    - Returns: JSON with complete document analysis metadata
    
    
    
========== SEGMENTATION STRATEGIES ==========
> semantic_research_focused : For academic papers with complex algorithmic content
> algorithm_preserve_integrity : Maintains algorithm blocks and formula chains intact
> concept_implementaion_hybrid : Merges related concepts with implementation details
> semantic_chunking_enhanced : Advanced boundary detection for long documents
> content_aware_segmentation : Adaptive chunking based on content density
=============================================


========== INTELLIGENT FEATURES ==========
> Semantic boundary detection (not just structural)
> Algorithm block identification and preservation
> Formula chain recognition and grouping
> Concept-implementation relationship mapping
> Multi-level relevance scoring (content type, importance, keyword matching)
> Backward compatibility with existing document indexes
> Configurable via mcp_agent.config.yaml (enabled/disabled, size thresholds)
==========================================
"""


import logging
import hashlib
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import MCP Related Modules
from mcp.server.fastmcp import FastMCP

# Create FastMCP server instance
mcp = FastMCP("document-segmentation-server")




@dataclass
class DocumentSegment:
  """Represents a document segment with metadata"""
  
  id: str
  title: str
  content: str
  content_type: str # "introduction", "methodology", "algorithm", "results", etc.
  keywords: List[str]
  char_start: int
  char_end: int
  char_count: int
  relevance_scores: Dict[str, float] # Scores for different query types
  section_path: str # e.g., "3.2.1" for nested sections


@dataclass
class DocumentIndex:
  """Document index containing all segments and metadata"""

  document_path: str
  document_path: str # "academic_paper", "technical_doc", "code_doc", "general"
  segmentation_strategy: str
  total_segments: int
  total_chars: int
  segments: List[DocumentSegment]
  created_at: str
  
