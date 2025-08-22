

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import torch
import logging


logger = logging.getLogger(__name__)


class EmbeddingService:
  """Advanced embedding service with multiple models"""
  
  def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    self.model_name = model_name
    self.model = SentenceTransformer(model_name)
    self.dimension = self.model.get_sentence_embedding_dimension()
    
    
    #  GPU optimization
    if torch.cuda.is_available():
      self.model = self.model.to('cuda')
    
    logger.info(f"Initialized embedding model: {model_name} (dim: {self.dimension})")
    
  
  def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
    """Generate embeddings for texts"""
    
    if isinstance(texts, str):
      texts = [texts]
    
    embeddings = self.model.encode(
      texts,
      convert_to_numpy=True,
      normalize_embeddings=normalize,
      show_progress_bar=len(texts) > 100
    )
    
    return embeddings
  
  
  def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Normally, you pass one sentence at a time into a SentenceTransformer model:
      But this is slow if you have thousands (or millions) of sentences.
      Instead, we process them in batches → multiple sentences together → fewer GPU/CPU calls.
    """
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
      batch = texts[i:i + batch_size]
      batch_embeddings = self.encode(batch)
      all_embeddings.extend(batch_embeddings)
      
      
    return np.array(all_embeddings)
  
  


    
    
    