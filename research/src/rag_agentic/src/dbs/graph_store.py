

from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
from models.schemas import Chunk, RetrievalResult, Document
import logging
import re


logger = logging.getLogger(__name__)


class Neo4jGraphStore:
  """Neo4j-based graph store for relationship queries"""
  
  def __init__(self, uri: str, user: str, password: str):
    self.driver = GraphDatabase.driver(uri, auth=(user, password))
    self._create_constraints()
    
    logger.info(f"Initialized Neo4j graph store at {uri}")
    
    
    
  def _create_constraints(self):
    """Create necessary constraints and indexes"""
    with self.driver.session() as session:
      # Create constraints
      try:
        session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
        session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
      except Exception as e:
        logger.warning(f"Constraint creation warning: {e}")
        
  
  def add_document(self, document: Document, chunks: List[Chunk]):
    """Add document and chunks to graph"""
    
    with self.driver.session() as session:
      # Create document node
      session.run("""
                  MERGE (d: Document {id: $doc_id})
                  SET d.content = $content,
                      d.doc_type = $doc_type,
                      d.created_at = $created_at,
                      d.metadata = $metadata
                  """,
                  {
                    "doc_id": document.id,
                    "content": document.content[:1000], # Truncate for graph
                    "doc_type": document.doc_type,
                    "created_at": document.created_at.isoformat(),
                    "metadata": str(document.metadata)
                  }
                )

      # Create chunk nodes and relationships
      for chunk in chunks:
        session.run("""
                    MATCH (d:Document {id: $doc_id})
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.content = $content,
                        c.chunk_index = $chunk_index,
                        c.metadata = $metadata
                    MERGE (d)-[:CONTAINS]->(c)
                    """,
                    {
                      "doc_id": document.id,
                      "chunk_id": chunk.id,
                      "content": chunk.content,
                      "chunk_index": chunk.chunk_index,
                      "metadata": str(chunk.metadata)
                    }
                  )
      
      # Extract and link entities (simplified NER)
      

  def _extract_entities(self, doc_id: str, content: str):
    """Simple entity extraction and linking"""
    
    # **This is a simplified example - in production, use spaCy or similar**
    
    # Extract potential entities (capitalized words)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
    entities = list(set(entities))  # Unique entities
    
    with self.driver.session() as session:
      for entity in entities[:20]: # Limit to avoid too many entities
        session.run("""
                    MERGE (e:Entity {name: $entity})
                    WITH e
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:MENTIONS]->(e)
                    """,
                    {
                      'entity': entity,
                      'doc_id': doc_id
                    }
                  )
        

  def find_related_content(self, query: str, max_depth: int = 2) -> List[RetrievalResult]:
    """Find content related through graph relationships"""
    
    with self.driver.session() as session:
      # Search for entities matching query terms
      query_words = query.lower().split()
      
      results = session.run("""
                            MATCH (e:Entiy)
                            WHERE ANY(word IN $query_words WHERE toLower(e.name) CONTAINS word)
                            MATCH (e)<-[:MENTIONS]-(d:Document)-[:CONTAINS]->(e:Chunk)
                            RETURN c.id as chunk_id, c.content as content, c.chunk_index as chunk_index,
                                  d.id as document_id, d.metadata as metadata,
                                  count(*) as relevance_store
                            ORDER BY relevance_score DESC
                            LIMIT 10
                            """,
                            {
                              'query_words': query_words
                            }
                          )

      retrieval_results = []
      for record in results:
        chunk = Chunk(
          id=record['chunk_id'],
          content=record['content'],
          document_id=record['doc_id'],
          chunk_index=record['chunk_index'],
          metadata=eval(record['metadata']) if record['metadata'] else {}
        )
        
        retrieval_results.append(RetrievalResult(
          chunk=chunk,
          score=float(record['relevance_score']) / 10.0, # Normalize score to [0, 1] range
          source="graph"
        ))
        
        
  def close(self):
    """Close the database connection"""
    self.driver.close()
        

    