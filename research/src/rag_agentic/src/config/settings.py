

import os
from typing import Dict, Any
from pydantic_settings import BaseSettings




class Settings(BaseSettings):
  # Database Configurations
  POSTGRES_URL: str = "postgresql://user:password@localhost:5432/rag_db"
  NEO4J_URL: str = "bolt://localhost:7687"
  NEO4J_USER: str = "neo4j"
  NEO4J_PASSWORD: str = "password"
  REDIS_URL: str = "redis://localhost:6379"
  CHROMA_PERSIST_DIR: str = "./data/chroma_db"
  
  # ML Model Configurations
  EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
  RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
  LLM_MODEL: str = "ollama/llama2"
  
  # RAG Parameters
  CHUNK_SIZE: int = 512
  CHUNK_OVERLAP: int = 50
  TOP_K_RETRIEVAL: int = 20
  TOP_K_RERANK: int = 5
  SIMILARITY_THRESHOLD: float = 0.7
  
  # Agent Parameters
  MAX_ITERATIONS: int = 5
  AGENT_TEMPERATURE: float = 0.1
  
 
    
settings = Settings()