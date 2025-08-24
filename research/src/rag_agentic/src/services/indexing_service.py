
import asyncio
from typing import List, Dict, Any, Optional
from models.schemas import Document, Chunk
from core.embeddings import EmbeddingService
from core.chunking import AdvancedChunker
from dbs.vector_store import ChromaVectorStore
from dbs.graph_store import Neo4jGraphStore
import logging


logger = logging.getLogger(__name__)


class IndexingService:
  """Advanced indexing service for multiple databases"""
  
  def __init__(
    self,
    embedding_service: EmbeddingService,
    chunker: AdvancedChunker,
    vector_store: ChromaVectorStore,
    graph_store: Neo4jGraphStore
  ):
    self.embedding_service = embedding_service
    self.chunker = chunker
    self.vector_store = vector_store
    self.graph_store = graph_store
    
    
  async def index_documents(
    self,
    documents: List[Document],
    batch_size: int = 10
  ) -> Dict[str, int]:
    """Index documents across all databases"""
    
    stats = {
      'processed': 0,
      'chunks_created': 0,
      'embeddings_generated': 0
    }
    
    for i in range(0, len(documents), batch_size):
      batch = documents[i:i + batch_size]
      await self._process_batch(batch, stats)
      
      logger.info(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
      
    return stats
      
      
  async def _process_batch(
    self,
    documents: List[Document],
    stats: Dict[str, int]
  ):
    """Process a batch of documents"""
    
    for document in documents:
      # 1. Chunk the document
      chunks = self.chunker.chunk_document(document)
      stats['chunks_created'] += len(chunks)
      
      # 2. Generate embeddings for chunks
      chunk_texts = [chunk.content for chunk in chunks]
      embeddings = self.embedding_service.encode_batch(chunk_texts, batch_size=32)
      
      # Set embeddings on chunks
      for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding.toList()
        
      stats['embeddings_generated'] += len(embeddings)
      
      # 3. Store in vector database
      self.vector_store.add_chunks(chunks, embeddings)
      
      # 4. Store in graph database
      self.graph_store.add_document(document, chunks)
      
      stats['processed'] += 1
      
      logger.debug(f"Indexed document {document.id} with {len(chunks)} chunks")  
        

  async def update_document(self, document: Document):
    """Update an existing document in all databases"""
    
    # In a production system, you'd need to:
    # 1. Remove old chunks from vector store
    # 2. Remove old relationships from graph store
    # 3. Re-index the updated document
    
    # For now, we'll just re-add (which might create duplicates)
    await self._process_batch(
      [document],
      {'processed': 0, 'chunks_created': 0, 'embeddings_generated': 0}
    )
    
  async def delete_document(self, document_id: str):
    """Delete document from all databases"""

    # This would require implementing deletion in both stores
    # For now, we'll log the operation
    logger.info(f"Delete request for document {document_id} (not implemented)")
    
    
    
    
    
      
