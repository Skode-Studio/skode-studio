
import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
from typing import List, Optional, Dict, Any
from models.schemas import Chunk, RetrievalResult
import logging


logger = logging.getLogger(__name__)

class ChromaVectorStore:
  """ChromaDB-based vector store with advanced features"""
  
  def __init__(self, persist_directory: str = "./db/chroma_db"):
    self.client = chromadb.PersistentClient(
      path=persist_directory,
      settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    self.collection = self.client.get_or_create_collection(
      name="rag_documents",
      metadata={
        "hnsw:space": "cosine" # Use cosine similarity
      }
    )
    
    logger.info(f"Initialized ChromaDB vector store at {persist_directory}")
    
  
  def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
    """Add chunks with embeddings to vector store"""
    
    ids = [chunk.id for chunk in chunks]
    documents = [chunk.content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    embeddings_list = embeddings.tolist()
    
    
    self.collection.add(
      ids=ids,
      documents=documents,
      metadatas=metadatas,
      embeddings=embeddings_list
    )
    
    logger.info(f"Added {len(chunks)} chunks to vector store")


  def search(
    self, 
    query_embeddings: np.ndarray, 
    top_k: int = 10,
    where: Optional[Dict] = None
  ) -> List[RetrievalResult]:
    
    """Search for similary chunks"""

    results = self.collection.query(
      query_embeddings=query_embeddings.tolist(),
      n_results=top_k,
      where=where,
      include=["documents", "metadatas", "distances"]
    )
    
    retrieval_results = []
    for i in range(len(results['id'][0])):
      chunk = Chunk(
        id=results['ids'][0][i],
        content=results['documents'][0][i],
        document_id=results['metadatas'][0][i].get('document_id', ''),
        chunk_index=results['metadatas'][0][i].get('chunk_index', 0),
        metadata=results['metadatas'][0][i]
      )
      
      # Convert distance to similarity score
      distance = results['distances'][0][i]
      score = 1 / (1 + distance) # Convert distance to similarity
      
      retrieval_results.append(RetrievalResult(
        chunk=chunk,
        score=score,
        source="vector"
      ))
      
    return retrieval_results


