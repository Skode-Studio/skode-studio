
import asyncio
from typing import List, Dict, Any, Optional
from core.embeddings import EmbeddingService
from core.reranks import RerankingService
from dbs.hybrid_retriever import HybridRetriever
from agents.query_planner import QueryPlanner, QueryPlan
from models.schemas import AgentResponse, QueryType
import logging
from google import genai
from dotenv import load_dotenv
import os

# Find the root .env relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)


logger = logging.getLogger(__name__)


class AdvancedRAGAgent:
  """Advanced RAG Agent with query planning and multi-step reasoning"""
  
  def __init__(
    self,
    hybrid_retriever: HybridRetriever,
    query_planner: QueryPlanner,
    llm_service: Any # LLM Service (Ollama, OpenAI, etc.)
  ):
    self.retriever = hybrid_retriever
    self.planner = query_planner
    self.llm_service = llm_service
    
  async def process_query(self, query: str, max_iterations: int = 3) -> AgentResponse:
    """Process query with multi-step reasoning"""
    
    # Step 1: Plan the query
    query_plan = self.planner.create_plan(query)
    logger.info(f"Created query plan: {query_plan.retrieval_strategy}")
    
    # Step 2: Execute retrieval
    all_results = []
    reasoning_steps = []
    
    for sub_query in query_plan.sub_queries:
      reasoning_steps.append(f"Retrieving for: {sub_query}")
      
      results = self.retriever.retrieve(
        sub_query,
        query_plan.query_type,
        top_k=20,
        rerank_top_k=5
      )
      all_results.extend(results)
    
    # Step 3: Remove duplicates and select best results
    unique_results = {}
    for result in all_results:
      if result.chunk.id not in unique_results:
        unique_results[result.chunk.id] = result
      else:
        if result.score > unique_results[result.chunk.id].score:
          unique_results[result.chunk.id] = result

    final_results = sorted(unique_results.values(), key=lambda x: x.score, reverse=True)[:5]
    
    # Step 4: Generate answer using LLM
    context = "\n\n".join([f"Source {i+1}: {result.chunk.content}" for i, result in enumerate(final_results) ])
    
    answer = await self._generate_answer(query, context, query_plan)
    confidence = self._calculate_confidence(final_results)
    
    
    reasoning_steps.append(f"Generated answer with {len(final_results)} sources")
    reasoning_steps.append(f"Confidence: {confidence:.2f}")
    
    return AgentResponse(
      answer=answer,
      sources=final_results,
      confidence=confidence,
      reasoning_steps=reasoning_steps,
      query_plan=query_plan
    )
    


  async def _generate_answer(
    self,
    query: str,
    context: str,
    query_plan: QueryPlan
  ) -> str:
    """Generate answer using LLM with context"""
    
    prompt = f"""Based on the following context, answer the question accurately and comprehensively.

              Query: {query}
              Query Type: {query_plan.query_type.value}
              Strategy: {query_plan.retrieval_strategy}

              Context:
              {context}

              Instructions:
              - Provide a direct and accurate answer
              - Cite relevant sources when making claims
              - If the context doesn't contain enough information, say so
              - For comparative questions, provide balanced analysis
              - For analytical questions, explain reasoning

              Answer:
            """

    
    response = await self._call_llm(prompt)
    return response
    


  async def _call_llm(self, prompt: str) -> str:
    """Call LLM Service - implement based on your chosen LLM"""
   
    client = genai.Client(api_key="")
    
    response = client.models.generate_content(
      model="gemini-2.5-flash-preview-05-20",
      contents=prompt
    )
   
    return response.text



  def _calculate_confidence(self, results: List) -> float:
    """Calculate confidence score based on retrieval results"""

    if not results:
      return 0.0
    
    # Calculate confidence based on:
    # 1. Average retrieval scores
    # 2. Number of sources
    # 3. Score distribution

    avg_score = sum(result.score for result in results) / len(results)
    source_bonus = min(len(results) / 5.0, 1.0) # More resources = higher confidence
    
    # Penalize if all scores are very low
    min_score = min(result.score for result in results)
    score_pentalty = 1.0 if min_score > 0.3 else 0.8
    
    confidence = (avg_score * 0.6 + source_bonus * 0.3) * score_pentalty
    return min(confidence, 1.0)




