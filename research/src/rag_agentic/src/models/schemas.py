
from pydantic import BaseModel, Field
from typing import List,Dict,Any,Optional,Union
from datetime import datetime
from enum import Enum
import numpy as np



class DocumentType(str, Enum):
  TEXT = "text"
  PDF = "pdf"
  WEB = "web"
  CODE = "code"
  
  
class QueryType(str, Enum):
  SEMANTIC = "semantic"
  FACTUAL = "factual"
  ANALYTICAL = "analytical"
  COMPARATIVE = "comparative"


class Document(BaseModel):
  id: str
  content: str
  metadata: Dict[str, Any] = Field(default_factory=dict)
  doc_type: DocumentType
  created_at: datetime = Field(default_factory=datetime.now)
  embedding: Optional[List[float]] = None
  
  class Config:
    arbitrary_types_allowed = True
    

class Chunk(BaseModel):
  id: str
  content: str
  document_id: str
  chunk_index: int
  metadata: Dict[str, Any] = Field(default_factory=dict)
  embedding: Optional[List[float]] = None    


class RetrievalResult(BaseModel):
  chunk: Chunk
  score: float
  source: str # vector, graph, hybrid
  
  
class QueryPlan(BaseModel):
  original_query: str
  query_type: QueryType
  sub_queries: List[str]
  retrieval_strategy: str
  reasoning: str
  
  
class AgentResponse(BaseModel):
  answer: str
  sources: List[RetrievalResult]
  confidence: float
  reasoning_steps: List[str]
  query_plan: QueryPlan
  
  

