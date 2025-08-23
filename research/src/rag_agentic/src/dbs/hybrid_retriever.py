

from typing import List, Dict, Any, Optional
from core.embeddings import EmbeddingService
from core.reranks import RerankingService
from dbs.vector_store import ChromaVectorStore
from dbs.graph_store import Neo4jGraphStore
from models.schemas import RetrievalResult, QueryType
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
  """Advanced hybrid retrieval combining multiple strategies"""
  
  def __init__(
    self,
    embedding_service: EmbeddingService,
    vector_store: ChromaVectorStore,
    graph_store: Neo4jGraphStore,
    reranker: RerankingService,
  ):
    self.embedding_service = embedding_service
    self.vector_store = vector_store
    self.graph_store = graph_store
    self.reranker = reranker
    
    
  def retrieve(
    self,
    query: str,
    query_type: QueryType,
    top_k: int = 20,
    rerank_top_k: int = 5
  ) -> List[RetrievalResult]:
    """Hybrid retrieval with multiple strategies"""
    
    all_results = []
    
    # 1. Vector similarity search
    query_embedding = self.embedding_service.encode([query])[0]
    vector_results = self.vector_store.search(
      query_embedding,
      top_k=top_k//2
    )
    all_results.extend(vector_results)
    
    # 2. Graph-based retrieval for relationship queries
    if query_type in [QueryType.ANALYTICAL, QueryType.COMPARATIVE]:
      graph_results = self.graph_store.find_related_content(
        query,
        max_depth=2
      )
      all_results.extend(graph_results)
    
    # 3. Remove duplicates based on chunk ID
    unique_results = {}
    for result in all_results:
      if result.chunk.id not in unique_results:
        unique_results[result.chunk.id] = result
      else:
        # Keep the one with higher score
        if result.score > unique_results[result.chunk.id].score:
          unique_results[result.chunk.id] = result
    
    # 4. Rerank using cross-encoder
    final_results = list(unique_results.values())
    if len(final_results) > rerank_top_k:
      passages = [result.chunk.content for result in final_results]
      reranked = self.reranker.rerank(query, passages)
      
      # Map back to RetrievalResult objects
      passage_to_result = {result.chunk.content: result for result in final_results}
      final_results = []

      for passage, score in reranked[:rerank_top_k]:
        result = passage_to_result[passage]
        result.score = float(score) # Update with reranking score
        final_results.append(result)
        

    logger.info(f"Retrieved {len(final_results)} results for query: {query[:50]}...")
    return final_results[:rerank_top_k]




