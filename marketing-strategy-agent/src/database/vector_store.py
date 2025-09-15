"""
TiDB Vector Store implementation for semantic search and retrieval.

This module provides vector storage and similarity search capabilities
using TiDB's native vector support for the marketing knowledge base.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import hashlib

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from ..config.settings import load_config
# Simplified imports for now - these modules would need to be implemented
import logging


class TiDBVectorStore:
    """
    TiDB-based vector store for semantic search and document retrieval.
    
    This class provides functionality for storing, indexing, and searching
    marketing knowledge using vector embeddings and TiDB's vector capabilities.
    """
    
    def __init__(self, embeddings: Optional[Embeddings] = None, table_name: str = "marketing_knowledge"):
        """
        Initialize TiDB vector store.
        
        Args:
            embeddings: Embeddings model for generating vectors
            table_name: Name of the table to store vectors
        """
        self.logger = get_component_logger("vector_store", __name__)
        
        # Initialize embeddings model
        if embeddings is None:
            self.embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key
            )
        else:
            self.embeddings = embeddings
        
        self.table_name = table_name
        self.dimension = settings.vector_dimension
        self.distance_metric = settings.vector_distance_metric
        
        # Cache for embeddings to avoid duplicate API calls
        self._embedding_cache = {}
        self._cache_max_size = 1000
        
        self.logger.info("TiDB vector store initialized", extra={
            "table_name": table_name,
            "dimension": self.dimension,
            "distance_metric": self.distance_metric
        })
    
    async def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 10,
        upsert: bool = False
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            batch_size: Number of documents to process in each batch
            upsert: Whether to update existing documents
            
        Returns:
            List of document IDs
        """
        try:
            with timing_context("vector_store_add_documents"):
                self.logger.info("Adding documents to vector store", extra={
                    "document_count": len(documents),
                    "batch_size": batch_size,
                    "upsert": upsert
                })
                
                if not documents:
                    return []
                
                document_ids = []
                
                # Process documents in batches
                for batch in batch_process(documents, batch_size):
                    batch_ids = await self._add_document_batch(batch, upsert)
                    document_ids.extend(batch_ids)
                    
                    # Small delay between batches to avoid overwhelming the API
                    await asyncio.sleep(0.1)
                
                self.logger.info("Documents added successfully", extra={
                    "document_ids_count": len(document_ids)
                })
                
                return document_ids
                
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}", exc_info=True)
            raise handle_exception(self.logger, e, "adding documents to vector store")
    
    async def _add_document_batch(self, documents: List[Document], upsert: bool) -> List[str]:
        """Add a batch of documents to the vector store."""
        try:
            # Generate embeddings for all documents in batch
            texts = [doc.page_content for doc in documents]
            embeddings = await self._generate_embeddings(texts)
            
            document_ids = []
            
            async with (await get_database_connection()) as connection:
                with connection.cursor() as cursor:
                    for doc, embedding in zip(documents, embeddings):
                        # Generate document ID
                        doc_id = self._generate_document_id(doc)
                        document_ids.append(doc_id)
                        
                        # Prepare document data
                        content_type = doc.metadata.get("content_type", "general")
                        title = doc.metadata.get("title", f"Document {doc_id}")
                        summary = doc.metadata.get("summary", "")
                        keywords = doc.metadata.get("keywords", [])
                        
                        # Convert embedding to string format for TiDB
                        embedding_str = json.dumps(embedding.tolist())
                        
                        if upsert:
                            # Check if document already exists
                            check_query = f"""
                            SELECT id FROM {self.table_name} 
                            WHERE content_type = %s AND title = %s
                            """
                            cursor.execute(check_query, (content_type, title))
                            existing = cursor.fetchone()
                            
                            if existing:
                                # Update existing document
                                update_query = f"""
                                UPDATE {self.table_name} 
                                SET content = %s, summary = %s, keywords = %s, 
                                    metadata = %s, embedding = VEC_FROM_TEXT(%s),
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE id = %s
                                """
                                cursor.execute(update_query, (
                                    doc.page_content,
                                    summary,
                                    json.dumps(keywords),
                                    json.dumps(doc.metadata),
                                    embedding_str,
                                    existing['id']
                                ))
                            else:
                                # Insert new document
                                await self._insert_document(cursor, doc, embedding_str, content_type, title, summary, keywords)
                        else:
                            # Always insert new document
                            await self._insert_document(cursor, doc, embedding_str, content_type, title, summary, keywords)
            
            return document_ids
            
        except Exception as e:
            self.logger.error(f"Error adding document batch: {e}")
            raise VectorStoreException(f"Failed to add document batch: {str(e)}")
    
    async def _insert_document(self, cursor, doc: Document, embedding_str: str, content_type: str, 
                             title: str, summary: str, keywords: List[str]) -> None:
        """Insert a single document into the database."""
        insert_query = f"""
        INSERT INTO {self.table_name} 
        (content_type, title, content, summary, keywords, metadata, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, VEC_FROM_TEXT(%s))
        """
        
        cursor.execute(insert_query, (
            content_type,
            title,
            doc.page_content,
            summary,
            json.dumps(keywords),
            json.dumps(doc.metadata),
            embedding_str
        ))
    
    def _generate_document_id(self, document: Document) -> str:
        """Generate a unique ID for a document."""
        # Create hash based on content and metadata
        content_hash = hashlib.md5(
            (document.page_content + str(document.metadata)).encode()
        ).hexdigest()
        return content_hash[:16]
    
    async def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts with caching."""
        embeddings_list = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            if text_hash in self._embedding_cache:
                embeddings_list.append(self._embedding_cache[text_hash])
            else:
                embeddings_list.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                with timing_context("embedding_generation"):
                    new_embeddings = await self.embeddings.aembed_documents(uncached_texts)
                
                # Cache new embeddings
                for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    embedding_array = np.array(embedding, dtype=np.float32)
                    
                    # Store in cache
                    if len(self._embedding_cache) < self._cache_max_size:
                        self._embedding_cache[text_hash] = embedding_array
                    
                    # Update embeddings list
                    original_index = uncached_indices[i]
                    embeddings_list[original_index] = embedding_array
                
            except Exception as e:
                self.logger.error(f"Failed to generate embeddings: {e}")
                raise EmbeddingException(f"Embedding generation failed: {str(e)}")
        
        # Convert to numpy array
        return np.array(embeddings_list)
    
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        distance_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Perform similarity search for documents.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_metadata: Optional metadata filters
            distance_threshold: Optional distance threshold for filtering results
            
        Returns:
            List of similar documents
        """
        try:
            with timing_context("vector_similarity_search"):
                self.logger.info("Performing similarity search", extra={
                    "query_length": len(query),
                    "k": k,
                    "has_filters": filter_metadata is not None
                })
                
                # Generate query embedding
                query_embedding = await self._generate_embeddings([query])
                query_vector = query_embedding[0]
                
                # Perform vector search
                results = await self._vector_search(query_vector, k, filter_metadata, distance_threshold)
                
                # Convert results to Document objects
                documents = []
                for result in results:
                    try:
                        metadata = json.loads(result['metadata']) if result['metadata'] else {}
                        metadata['id'] = result['id']
                        metadata['distance'] = result['distance']
                        metadata['content_type'] = result['content_type']
                        metadata['title'] = result['title']
                        
                        document = Document(
                            page_content=result['content'],
                            metadata=metadata
                        )
                        documents.append(document)
                        
                    except Exception as e:
                        self.logger.warning(f"Error parsing search result: {e}")
                        continue
                
                self.logger.info("Similarity search completed", extra={
                    "results_found": len(documents),
                    "requested": k
                })
                
                return documents
                
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}", exc_info=True)
            raise handle_exception(self.logger, e, "similarity search")
    
    async def _vector_search(
        self,
        query_vector: np.ndarray,
        k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
        distance_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector search in TiDB."""
        try:
            # Convert query vector to string format
            query_vector_str = json.dumps(query_vector.tolist())
            
            # Build search query
            base_query = f"""
            SELECT id, content_type, title, content, summary, keywords, metadata,
                   VEC_{self.distance_metric.upper()}_DISTANCE(embedding, VEC_FROM_TEXT(%s)) as distance
            FROM {self.table_name}
            """
            
            # Add filters
            where_conditions = []
            query_params = [query_vector_str]
            
            if filter_metadata:
                for key, value in filter_metadata.items():
                    if key == "content_type":
                        where_conditions.append("content_type = %s")
                        query_params.append(value)
                    else:
                        # For JSON metadata filtering
                        where_conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') = %s")
                        query_params.append(value)
            
            if distance_threshold:
                where_conditions.append(f"VEC_{self.distance_metric.upper()}_DISTANCE(embedding, VEC_FROM_TEXT(%s)) <= %s")
                query_params.append(distance_threshold)
            
            if where_conditions:
                base_query += " WHERE " + " AND ".join(where_conditions)
            
            # Add ordering and limit
            base_query += " ORDER BY distance ASC LIMIT %s"
            query_params.append(k)
            
            # Execute search
            async with (await get_database_connection()) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(base_query, tuple(query_params))
                    results = cursor.fetchall()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            raise VectorStoreException(f"Vector search failed: {str(e)}")
    
    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return documents with similarity scores.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (document, score) tuples
        """
        documents = await self.similarity_search(query, k, filter_metadata)
        
        # Extract scores from metadata
        results = []
        for doc in documents:
            score = doc.metadata.get('distance', 1.0)
            # Convert distance to similarity score (lower distance = higher similarity)
            similarity_score = max(0.0, 1.0 - score)
            results.append((doc, similarity_score))
        
        return results
    
    async def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform Maximum Marginal Relevance (MMR) search for diverse results.
        
        Args:
            query: Search query text
            k: Number of results to return
            fetch_k: Number of initial results to fetch
            lambda_mult: Lambda parameter for MMR (trade-off between relevance and diversity)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of diverse documents
        """
        try:
            # Fetch more results than needed
            initial_results = await self.similarity_search(query, fetch_k, filter_metadata)
            
            if len(initial_results) <= k:
                return initial_results
            
            # Generate embeddings for all results
            result_texts = [doc.page_content for doc in initial_results]
            result_embeddings = await self._generate_embeddings(result_texts)
            
            # Generate query embedding
            query_embedding = await self._generate_embeddings([query])
            query_vector = query_embedding[0]
            
            # Apply MMR algorithm
            selected_indices = []
            remaining_indices = list(range(len(initial_results)))
            
            # Select first document (most similar to query)
            similarities = np.dot(result_embeddings, query_vector)
            first_idx = np.argmax(similarities)
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)
            
            # Select remaining documents using MMR
            while len(selected_indices) < k and remaining_indices:
                mmr_scores = []
                
                for idx in remaining_indices:
                    # Relevance score (similarity to query)
                    relevance = similarities[idx]
                    
                    # Diversity score (maximum similarity to already selected documents)
                    if selected_indices:
                        selected_embeddings = result_embeddings[selected_indices]
                        diversities = np.dot(selected_embeddings, result_embeddings[idx])
                        max_diversity = np.max(diversities)
                    else:
                        max_diversity = 0
                    
                    # MMR score
                    mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_diversity
                    mmr_scores.append((idx, mmr_score))
                
                # Select document with highest MMR score
                best_idx, _ = max(mmr_scores, key=lambda x: x[1])
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            
            # Return selected documents
            return [initial_results[idx] for idx in selected_indices]
            
        except Exception as e:
            self.logger.error(f"MMR search failed: {e}", exc_info=True)
            return await self.similarity_search(query, k, filter_metadata)  # Fallback to regular search
    
    async def delete_documents(self, document_ids: List[str]) -> int:
        """
        Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        try:
            if not document_ids:
                return 0
            
            self.logger.info("Deleting documents", extra={
                "document_count": len(document_ids)
            })
            
            # Build delete query
            placeholders = ",".join(["%s"] * len(document_ids))
            delete_query = f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})"
            
            async with (await get_database_connection()) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(delete_query, document_ids)
                    deleted_count = cursor.rowcount
            
            self.logger.info("Documents deleted", extra={
                "deleted_count": deleted_count
            })
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
            raise VectorStoreException(f"Failed to delete documents: {str(e)}")
    
    async def get_document_count(self, filter_metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Get the number of documents in the vector store.
        
        Args:
            filter_metadata: Optional metadata filters
            
        Returns:
            Number of documents
        """
        try:
            base_query = f"SELECT COUNT(*) as count FROM {self.table_name}"
            query_params = []
            
            if filter_metadata:
                where_conditions = []
                for key, value in filter_metadata.items():
                    if key == "content_type":
                        where_conditions.append("content_type = %s")
                        query_params.append(value)
                    else:
                        where_conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') = %s")
                        query_params.append(value)
                
                if where_conditions:
                    base_query += " WHERE " + " AND ".join(where_conditions)
            
            async with (await get_database_connection()) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(base_query, tuple(query_params))
                    result = cursor.fetchone()
            
            return result['count'] if result else 0
            
        except Exception as e:
            self.logger.error(f"Error getting document count: {e}")
            raise VectorStoreException(f"Failed to get document count: {str(e)}")
    
    async def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a specific document.
        
        Args:
            document_id: ID of the document to update
            metadata: New metadata to set
            
        Returns:
            True if document was updated, False otherwise
        """
        try:
            update_query = f"""
            UPDATE {self.table_name} 
            SET metadata = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """
            
            async with (await get_database_connection()) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(update_query, (json.dumps(metadata), document_id))
                    updated = cursor.rowcount > 0
            
            if updated:
                self.logger.info("Document metadata updated", extra={
                    "document_id": document_id
                })
            
            return updated
            
        except Exception as e:
            self.logger.error(f"Error updating document metadata: {e}")
            raise VectorStoreException(f"Failed to update document metadata: {str(e)}")
    
    async def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self.logger.info("Embedding cache cleared")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._embedding_cache),
            "cache_max_size": self._cache_max_size,
            "cache_hit_rate": getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
        }