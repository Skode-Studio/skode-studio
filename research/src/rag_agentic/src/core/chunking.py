


from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models.schemas import Document, Chunk
import uuid


class AdvancedChunker:
  """Advanced text chunking with semantic awareness"""
  
  def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
    
    # Different splitters for different content types
    self.text_splitter = RecursiveCharacterTextSplitter(
      chunk_siz=chunk_size,
      chunk_overlap=chunk_overlap,
      separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
    )
    
    self.code_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      separators=["\n\nclass ", "\n\ndef ", "\n\n", "\n", " "]
    )
    
  def chunk_document(self, document: Document) -> List[Chunk]:
    """Chunk document based on its type"""
    
    if document.doc_type == "code":
      splitter = self.code_splitter
    else:
      splitter = self.text_splitter
      
    # Split the content
    texts = splitter.split_text(document.content)
    
    chunks = []
    for i, text in enumerate(texts):
      chunk = Chunk(
        id=str(uuid.uuid4()),
        content=text,
        document_id=document.id,
        chunk_index=i,
        metadata={
          **document.metadata,
          "chunk_type": "semantic",
          "original_doc_type": document.doc_type
        }
      )
      chunks.append(chunk)
      
    return chunks
    
    
    
    
    
    
    