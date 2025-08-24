
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
    """Decompose complex queries into sub-queries"""
    
    query_lower = query.lower()
    
    # COMPARATIVE
    if query_type == QueryType.COMPARATIVE:
      # Extract entities from comparative queries
      
      # Look for "X and Y", "X vs Y", "X compared to Y" patterns
      and_pattern = r'(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)'
      vs_pattern = r'(\w+(?:\s+\w+)*)\s+vs\s+(\w+(?:\s+\w+)*)'
      compare_pattern = r'compare\s+(\w+(?:\s+\w+)*)\s+(?:and|to|with)\s+(\w+(?:\s+\w+)*)'
      
      
      entities = []
      for pattern in [and_pattern, vs_pattern, compare_pattern]:
        match = re.search(pattern, query_lower)
        if match:
          entities = [match.group(1).strip(), match.group(2).strip()]
          break
        
      if entities:
        sub_queries = [f"What is {entity}?" for entity in entities]
        sub_queries.append(f"How do {entities[0]} and {entities[1]} differ?")
      else:
        # Fallback: extract capitalized terms
        words = query.split()
        potential_entities = [word for word in words
                              if len(word) > 2 and (
                                  word[0].isupper() or word.lower() in
                                  ['machine learning', 'neural networks', 'ai', 'artificial intelligence']
                                )
                              ]
        if len(potential_entities) >= 2:
          sub_queries = [f"What is {entity}?" for entity in potential_entities[:2]]
        else:
          sub_queries = [query]

    # ANALYTICAL 
    elif query_type == QueryType.ANALYTICAL:
      # Break down analytical questions better
      if query_lower.startswith('why'):
        # "Why is X importan for Y?" -> ["What is X?", "What is Y?", original question]
        words = query.split()
        if 'important' in query_lower and 'for' in query_lower:
          for_idx = words.index(next(word for word in words if word.lower() == 'for'))
          if for_idx > 2:
            subject = ' '.join(words[2:for_idx]).replace('important', '').strip()
            object_part = ' '.join(words[for_idx+1:]).rstrip('?')
            sub_queries = [
              f"What is {subject}?",
              f"What is {object_part}?",
              query
            ]
          else:
            sub_queries = [query]
        else:
          sub_queries = [query]
        
      elif query_lower.startswith('how'):
        # "How are X related to Y?" -> ["What is X?", "What is Y?", original question]
        if 'related to' in query_lower:
          parts = query_lower.split('related to')
          if len(parts) == 2:
            subject = parts[0].replace('how are', '').strip()
            object_part = parts[1].rstrip('?').strip()
            sub_queries = [
              f"What is {subject}?",
              f"What is {object_part}?",
              query
            ]
          else:
            sub_queries = [query]
        else:
          sub_queries = [query]
      
      else:
        sub_queries = [query]
        
    # OTHERS 
    else:
      sub_queries = [query]
      
      
    return sub_queries
  

  # def _decompose_query(self, query: str, query_type: QueryType) -> List[str]:
  #   """Decompose complex queries into sub-queires"""
    
  #   # Simple decomposition based on query type
  #   if query_type == QueryType.COMPARATIVE:
  #     # Extract entities to compare
  #     words = query.split()
  #     sub_queries = [
  #       f"What is {word}?" for word in words
  #       if word[0].isupper() and len(word) > 3
  #     ][:3]
      
  #     if not sub_queries:
  #       sub_queries = [query]

  #   elif query_type == QueryType.ANALYTICAL:
  #     # Break down analytical questions
  #     sub_queries = [
  #       query,
  #       f"Background information about {' '.join(query.split()[1:4])}"
  #     ]
      
  #   else:
  #     sub_queries = [query]

  #   return sub_queries


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



