
from config.settings import settings
from core.embeddings import EmbeddingService
from core.reranks import RerankingService
from dbs.vector_store import ChromaVectorStore
from dbs.graph_store import Neo4jGraphStore
from dbs.hybrid_retriever import HybridRetriever
from agents.query_planner import QueryPlanner
from agents.rag_agent import AdvancedRAGAgent
from core.chunking import AdvancedChunker
from models.schemas import Document, DocumentType
import traceback
import asyncio


async def main():
  """Example usage of the Advanced RAG Agent System"""
  
  # Initialize services
  embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
  reranker = RerankingService(settings.RERANKER_MODEL)
  
  # Initialize databases
  vector_store = ChromaVectorStore(settings.CHROMA_PERSIST_DIR)
  graph_store = Neo4jGraphStore(
    settings.NEO4J_URL,
    settings.NEO4J_USER,
    settings.NEO4J_PASSWORD
  )
  
  # Initialize retriever and agent
  hybrid_retriever = HybridRetriever(
    embedding_service,
    vector_store,
    graph_store,
    reranker
  )
  
  query_planner = QueryPlanner()
  rag_agent = AdvancedRAGAgent(hybrid_retriever, query_planner, None)
  
  # Example: Add documents to the system
  sample_docs = [
    Document(
      id="doc1",
      content=(
        "Machine learning is a subset of artificial intelligence that enables computer systems "
        "to automatically learn and improve from data without being explicitly programmed. "
        "By analyzing large amounts of examples, machine learning algorithms can recognize "
        "patterns, make predictions, and support decision-making in real-world applications. "
        "Common techniques in ML include supervised learning, unsupervised learning, and "
        "reinforcement learning, each suited for different types of problems such as "
        "classification, clustering, or sequential decision-making."  
      ),
      metadata={
        "source": "textbook",
        "topic": "ML"
      },
      doc_type=DocumentType.TEXT
    ),
    Document(
      id="doc2",
      content=(
        "Neural networks are a foundational concept in AI and represent a class of computing "
        "systems inspired by the structure of the human brain. These networks are composed of "
        "interconnected layers of artificial neurons that process data through weighted "
        "connections. By adjusting the weights during training, neural networks can learn "
        "complex relationships in data. In modern artificial intelligence research, deep "
        "neural networks power breakthroughs in computer vision, natural language processing, "
        "and speech recognition. Their success demonstrates how AI systems can achieve "
        "human-level performance in tasks once thought impossible for machines."
      ),
      metadata={
        "source": "research",
        "topic": "AI"
      },
      doc_type=DocumentType.TEXT
    )
  ]
  
  # Process documents
  chunker = AdvancedChunker()
  
  
  for doc in sample_docs:
    # Chunk the document
    chunks = chunker.chunk_document(doc)
    
    # Generate embeddings
    chunk_texts = [chunk.content for chunk in chunks]
    embeddings = embedding_service.encode(chunk_texts)
    
    # Store in vector db
    vector_store.add_chunks(chunks, embeddings)
    
    # Store in graph db
    graph_store.add_document(doc, chunks)
    
    print(f"Processed document: {doc.id} into {len(chunks)} chunks")
  
  
  # Example queries
  test_queries = [
    "What is machine learning?",
    "How are neural networks related to AI?",
    "machine learning vs neural networks",
    "Why is machine learning important for AI?"
  ]
  
  print("\n" + "="*60)
  print("ADVANCED RAG AGENT SYSTEM - TEST QUERIES")
  print("="*60)
  
  for query in test_queries:
    print(f"\n🔍 QUERY: {query}")
    print("-" * 50)
    
    try:
      response = await rag_agent.process_query(query)
      
      print(f"📋 Query Plan:")
      print(f"   Type: {response.query_plan.query_type.value}")
      print(f"   Strategy: {response.query_plan.retrieval_strategy}")
      print(f"   Sub-queries: {len(response.query_plan.sub_queries)}")
      
      print(f"\n💭 Reasoning Steps:")
      for i, step in enumerate(response.reasoning_steps, 1):
        print(f"   {i}. {step}")
        
      print(f"\n📚 Retrieved Sources ({len(response.sources)}):")
      for i, source in enumerate(response.sources, 1):
        print(f"   {i}. Score: {source.score:.3f} | Source: {source.source}")
        print(f"      Content: {source.chunk.content[:100]}...")

      print(f"\n✅ Answer (Confidence: {response.confidence:.2f}):")
      print(f"   {response.answer}")

    except Exception as e:
      print(f"❌ Error processing query: {e}")
      traceback.print_exc()
  
  # Cleanup
  graph_store.close()
    
  
if __name__ == "__main__":
  asyncio.run(main())
  
  
  
  
  
  