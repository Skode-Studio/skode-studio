
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import aiofiles
from models.schemas import Document, DocumentType, Chunk
from core.chunking import AdvancedChunker
from core.embeddings import EmbeddingService
import logging
import hashlib
import PyPDF2


logger = logging.getLogger(__name__)


class DocumentProcessor:
  """Advanced document processing with multiple format support"""

  def __init__(
    self,
    embedding_service: EmbeddingService,
    chunker: AdvancedChunker
  ):
    self.embedding_service = embedding_service
    self.chunker = chunker
    
    
  async def process_file(self, file_path: Path) -> Document:
    """Process a file into a Document object"""
    
    file_ext = file_path.suffix.lower()
    
    if file_ext == ".txt":
      return await self._process_text_file(file_path)
    elif file_ext == ".pdf":
      return await self._process_pdf_file(file_path)
    elif file_ext in ['.py', '.js', '.java', '.cpp', '.c']:
      return await self._process_code_file(file_path)
    else:
      raise ValueError(f"Unsupported file type: {file_ext}")
    
    
    
  # TEXT
  async def _process_text_file(self, file_path: Path) -> Document:
    """Process text file"""
    
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
      content = await f.read()
      
    doc_id = self._generate_doc_id(content)
    
    return Document(
      id=doc_id,
      content=content,
      metadata={
        'filename': file_path.name,
        'filepath': str(file_path),
        'size': len(content),
        'format': 'text'
      },
      doc_type=DocumentType.TEXT
    )
    
  # PDF
  async def _process_pdf_file(self, file_path: Path) -> Document:
    """Process PDF file - requires PyPDF2 or similar"""
    
    with open(file_path, 'rb') as f:
      pdf_reader = PyPDF2.PdfReader(f)
      content = ""
      for page in pdf_reader.pages:
        content += page.extract_text() + "\n"
        
    doc_id = self._generate_doc_id(content)
    
    return Document(
      id=doc_id,
      content=content,
      metadata={
        'filename': file_path.name,
        'filepath': str(file_path),
        'pages': len(pdf_reader.pages),
        'format': 'pdf'
      },
      doc_type=Document.PDF
    )
    
    
  # Code
  async def _process_code_file(self, file_path: Path) -> Document:
    """Process code file"""
    
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
      content = await f.read()
      
    doc_id = self._generate_doc_id(content)
    
    # Extract language from extension
    lang_map = {
      '.py': 'python', '.js': 'javascript', '.java': 'java', '.cpp': 'cpp', '.c': 'c'
    }
    language = lang_map.get(file_path.suffix.lower(), 'unknown')
    
    return Document(
      id=doc_id,
      content=content,
      metadata={
        'filename': file_path.name,
        'filepath': str(file_path),
        'language': language,
        'lines': len(content.split('\n')),
        'format': 'code'
      },
      doc_type=DocumentType.CODE
    )
    
  
  def _generate_doc_id(self, content: str) -> str:
    """Generate unique document ID from content hash"""
    return hashlib.sha256(content.encode()).hexdigest()[:16]
  
  
  async def process_batch(
    self,
    file_paths: List[Path],
    max_concurrent: int = 5
  ) -> List[Document]:
    """Process multiple files concurrently"""
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single(file_path):
      async with semaphore:
        try:
          return await self.process_file(file_path)
        except Exception as e:
          logger.error(f"Error processing {file_path}: {e}")
          return None
        
    tasks = [process_single(fp) for fp in file_paths]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results (failed processing)
    return [doc for doc in results if doc is not None]    
    
  
  
  
  
  
  
  
  
