"""
Simple Vector Store implementation for marketing knowledge base
Provides basic vector storage capabilities without complex TiDB dependencies
"""
from typing import Dict, List, Any, Optional, Tuple
import json
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SimpleVectorStore:
    """
    Simple in-memory vector store for basic vector operations
    This is a lightweight implementation for development and testing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.documents = {}
        self.embeddings = {}
        self.config = config or {}
        logger.info("SimpleVectorStore initialized")
        
    async def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add a document to the store"""
        self.documents[doc_id] = {
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }
        logger.info(f"Added document {doc_id}")
        
    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Simple text-based search (would be vector-based in production)"""
        results = []
        query_lower = query.lower()
        
        for doc_id, doc_data in self.documents.items():
            content = doc_data["content"].lower()
            # Simple relevance scoring based on keyword matches
            score = sum(1 for word in query_lower.split() if word in content)
            
            if score > 0:
                results.append({
                    "id": doc_id,
                    "content": doc_data["content"],
                    "metadata": doc_data["metadata"],
                    "score": score
                })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
    
    async def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Alias for search method to maintain compatibility"""
        return await self.search(query, k)
        
    async def store_knowledge(self, content: str, metadata: Dict[str, Any] = None):
        """Store knowledge for later retrieval"""
        doc_id = hashlib.md5(content.encode()).hexdigest()[:16]
        await self.add_document(doc_id, content, metadata)
        return doc_id
        
    async def retrieve_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge based on query"""
        return await self.search(query, limit)
    
    async def get_document_count(self) -> int:
        """Get total number of documents"""
        return len(self.documents)


# For backward compatibility
VectorStore = SimpleVectorStore