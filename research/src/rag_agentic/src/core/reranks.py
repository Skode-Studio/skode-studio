

import numpy as np
from typing import List, Union
from sentence_transformers import CrossEncoder
import torch
import logging

logger = logging.getLogger(__name__)

class RerankingService:
  """Cross-encoder reranking service"""
  
  def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    self.model = CrossEncoder(model_name)
    
    if torch.cuda.is_available():
      self.model.model = self.model.model.to('cuda')
    
    logger.info(f"Initialized reranking model: {model_name}")
    
    
  def rerank(self, query: str, passages: List[str]) -> List[tuple]:
    """Rerank passages based on query relevance"""
    
    if not passages:
      return []
    
    # Create query-passage pairs
    pairs = [(query, passage) for passage in passages]
    
    # Get relevance scores
    scores = self.model.predict(pairs)
    
    # Sort by relevance
    ranked_pairs = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    
    return ranked_pairs
    
  
    
      
    