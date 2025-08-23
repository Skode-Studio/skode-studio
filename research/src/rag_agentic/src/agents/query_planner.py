
from typing import List, Dict, Any
from models.schemas import QueryPlan, QueryType
import re
import logging


logger = logging.getLogger(__name__)


class QueryPlanner:
  """Intelligent query planning and decomposition"""
  
  def __init__(self):
    self.query_patterns = {
      QueryType.FACTUAL: [
        r'\bwhat is\b', r'\bwho is\b', r'\bwhen did\b', r'\bwhere is\b',
        r'\bdefine\b', r'\bexplain\b'
      ],
      QueryType.COMPARATIVE: [
        r'\bcompare\b', r'\bdifference\b', r'\bvs\b', r'\bbetter\b',
        r'\bsimilar\b', r'\bunlike\b'
      ],
      QueryType.ANALYTICAL: [
        r'\bwhy\b', r'\bhow\b', r'\banalyze\b', r'\bevaluate\b',
        r'\breason\b', r'\bcause\b'
      ]
    }
    
    
  def create_plan(self, query: str) -> QueryPlan:
    """Create an execution plan for the query"""

    query_type = self._classify_query(query)
    sub_queries = self._decompose_query(query, query_type)
    strategy = self._select_strategy(query_type)
    reasoning = self._generate_reasoning(query, query_type, strategy)
    
    return QueryPlan(
      original_query=query,
      query_type=query_type,
      sub_queries=sub_queries,
      retrieval_strategy=strategy,
      reasoning=reasoning
    )
    


  def _classify_query(self, query: str) -> QueryType:
    """Classify the type of query"""
    
    query_lower = query.lower()
    
    scores = {}
    for query_type, patterns in self.query_patterns.items():
      score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
      scores[query_type] = score
      
    if max(scores.values()) == 0:
      return QueryType.SEMANTIC
    
    return max(scores.items(), key=lambda x: x[1])[0]


  def _decompose_query(self, query: str, query_type: QueryType) -> List[str]:
    """Decompose complex queries into sub-queires"""
    
    # Simple decomposition based on query type
    if query_type == QueryType.COMPARATIVE:
      # Extract entities to compare
      words = query.split()
      sub_queries = [
        f"What is {word}?" for word in words
        if word[0].isupper() and len(word) > 3
      ][:3]
      
      if not sub_queries:
        sub_queries = [query]

    elif query_type == QueryType.ANALYTICAL:
      # Break down analytical questions
      sub_queries = [
        query,
        f"Background information about {' '.join(query.split()[1:4])}"
      ]
      
    else:
      sub_queries = [query]

    return sub_queries


  def _select_strategy(self, query_type: QueryType) -> str:
    """Select retrieval strategy based on query type"""
    
    strategy_map = {
      QueryType.SEMANTIC: "vector_similarity",
      QueryType.FACTUAL: "hybrid_vector_graph",
      QueryType.COMPARATIVE: "multi_vector_rerank",
      QueryType.ANALYTICAL: "graph_enhanced_vector"
    }
    
    return strategy_map.get(query_type, "vector_similarity")
    
    
  def _generate_reasoning(self, query: str, query_type: QueryType, strategy: str) -> str:
    """Generate reasoning for the chosen approach"""
    
    return f"""
    Query analysis:
    - Type: {query_type.value}
    - Strategy: {strategy}
    - Rationale: {self._get_strategy_rationale(strategy)}
    """
  
  
  def _get_strategy_rationale(self, strategy: str) -> str:
    """Get rationale for strategy choice"""
    
    rationales = {
      "vector_similarity": "Using semantic similarity for conceptual matching",
      "hybrid_vector_graph": "Combining sematic search with relationship analysis",
      "multi_vector_rerank": "Multiple retrievals with cross-encoder reranking",
      "graph_enhanced_vector": "Graph relationships to enhance semantic understanding"
    }

    return rationales.get(strategy, "Standard retrieval approach")



