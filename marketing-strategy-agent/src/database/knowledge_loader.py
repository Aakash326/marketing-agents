"""
Knowledge Base Loader for populating the marketing vector store.

This module handles loading, processing, and indexing marketing knowledge
from various sources into the TiDB vector store for semantic retrieval.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import csv
import xml.etree.ElementTree as ET
from datetime import datetime

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, JSONLoader, CSVLoader

from config import settings
from src.database.vector_store import TiDBVectorStore
from src.utils.logging import get_component_logger
from src.utils.exceptions import (
    DatabaseException,
    ValidationException,
    handle_exception
)
from src.utils.helpers import timing_context, sanitize_text, extract_keywords


class KnowledgeBaseLoader:
    """
    Loader for marketing knowledge base content.
    
    This class handles loading various types of marketing content
    and preparing it for vector storage and semantic search.
    """
    
    def __init__(self, vector_store: TiDBVectorStore):
        """
        Initialize knowledge base loader.
        
        Args:
            vector_store: TiDB vector store instance
        """
        self.logger = get_component_logger("knowledge_loader", __name__)
        self.vector_store = vector_store
        
        # Text splitter for chunking large documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Knowledge categories and their processing rules
        self.knowledge_categories = {
            "marketing_strategies": {
                "chunk_size": 1500,
                "overlap": 300,
                "keywords_extract": True
            },
            "brand_examples": {
                "chunk_size": 1000,
                "overlap": 200,
                "keywords_extract": True
            },
            "content_templates": {
                "chunk_size": 800,
                "overlap": 150,
                "keywords_extract": False
            },
            "industry_insights": {
                "chunk_size": 1200,
                "overlap": 250,
                "keywords_extract": True
            },
            "campaign_examples": {
                "chunk_size": 1500,
                "overlap": 300,
                "keywords_extract": True
            }
        }
        
        self.logger.info("Knowledge base loader initialized")
    
    async def load_knowledge_base(
        self,
        data_directory: Union[str, Path],
        batch_size: int = 20,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Load complete knowledge base from directory.
        
        Args:
            data_directory: Path to knowledge base directory
            batch_size: Number of documents to process in each batch
            overwrite: Whether to overwrite existing knowledge
            
        Returns:
            Dictionary containing load statistics
        """
        try:
            with timing_context("knowledge_base_loading"):
                self.logger.info("Starting knowledge base loading", extra={
                    "data_directory": str(data_directory),
                    "batch_size": batch_size,
                    "overwrite": overwrite
                })
                
                data_path = Path(data_directory)
                if not data_path.exists():
                    raise ValidationException(f"Data directory not found: {data_directory}")
                
                # Clear existing knowledge if overwrite is requested
                if overwrite:
                    await self._clear_existing_knowledge()
                
                # Load knowledge by category
                load_stats = {
                    "total_files_processed": 0,
                    "total_documents_created": 0,
                    "total_chunks_stored": 0,
                    "categories_loaded": {},
                    "errors": []
                }
                
                # Process each knowledge category
                for category in self.knowledge_categories.keys():
                    category_path = data_path / category
                    if category_path.exists():
                        try:
                            category_stats = await self._load_category(
                                category, category_path, batch_size
                            )
                            load_stats["categories_loaded"][category] = category_stats
                            load_stats["total_files_processed"] += category_stats["files_processed"]
                            load_stats["total_documents_created"] += category_stats["documents_created"]
                            load_stats["total_chunks_stored"] += category_stats["chunks_stored"]
                            
                        except Exception as e:
                            error_msg = f"Error loading category {category}: {str(e)}"
                            self.logger.error(error_msg)
                            load_stats["errors"].append(error_msg)
                
                # Load general knowledge files
                general_stats = await self._load_general_files(data_path, batch_size)
                load_stats["categories_loaded"]["general"] = general_stats
                load_stats["total_files_processed"] += general_stats["files_processed"]
                load_stats["total_documents_created"] += general_stats["documents_created"]
                load_stats["total_chunks_stored"] += general_stats["chunks_stored"]
                
                self.logger.info("Knowledge base loading completed", extra={
                    "total_files": load_stats["total_files_processed"],
                    "total_documents": load_stats["total_documents_created"],
                    "total_chunks": load_stats["total_chunks_stored"],
                    "errors": len(load_stats["errors"])
                })
                
                return load_stats
                
        except Exception as e:
            self.logger.error(f"Knowledge base loading failed: {e}", exc_info=True)
            raise handle_exception(self.logger, e, "knowledge base loading")
    
    async def _clear_existing_knowledge(self) -> None:
        """Clear existing knowledge from vector store."""
        self.logger.info("Clearing existing knowledge base")
        
        try:
            # Get count of existing documents
            existing_count = await self.vector_store.get_document_count()
            
            if existing_count > 0:
                # For simplicity, we'll truncate the table
                # In production, you might want more selective deletion
                from src.database.tidb_setup import get_database_connection
                
                async with get_database_connection() as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(f"DELETE FROM {self.vector_store.table_name}")
                
                self.logger.info(f"Cleared {existing_count} existing documents")
            
        except Exception as e:
            self.logger.error(f"Error clearing existing knowledge: {e}")
            raise DatabaseException(f"Failed to clear existing knowledge: {str(e)}")
    
    async def _load_category(
        self, 
        category: str, 
        category_path: Path, 
        batch_size: int
    ) -> Dict[str, Any]:
        """Load knowledge for a specific category."""
        self.logger.info(f"Loading category: {category}", extra={
            "category_path": str(category_path)
        })
        
        category_config = self.knowledge_categories[category]
        
        # Configure text splitter for this category
        category_splitter = RecursiveCharacterTextSplitter(
            chunk_size=category_config["chunk_size"],
            chunk_overlap=category_config["overlap"],
            length_function=len
        )
        
        stats = {
            "files_processed": 0,
            "documents_created": 0,
            "chunks_stored": 0,
            "errors": []
        }
        
        # Find all supported files in category
        supported_extensions = [".txt", ".json", ".csv", ".md"]
        files_to_process = []
        
        for ext in supported_extensions:
            files_to_process.extend(category_path.glob(f"*{ext}"))
        
        # Process files
        for file_path in files_to_process:
            try:
                documents = await self._load_file(file_path, category, category_splitter)
                
                if documents:
                    # Add documents to vector store
                    doc_ids = await self.vector_store.add_documents(
                        documents, batch_size=batch_size, upsert=True
                    )
                    
                    stats["files_processed"] += 1
                    stats["documents_created"] += len(documents)
                    stats["chunks_stored"] += len(doc_ids)
                
            except Exception as e:
                error_msg = f"Error processing file {file_path}: {str(e)}"
                self.logger.warning(error_msg)
                stats["errors"].append(error_msg)
        
        return stats
    
    async def _load_general_files(self, data_path: Path, batch_size: int) -> Dict[str, Any]:
        """Load general knowledge files not in specific categories."""
        self.logger.info("Loading general knowledge files")
        
        stats = {
            "files_processed": 0,
            "documents_created": 0,
            "chunks_stored": 0,
            "errors": []
        }
        
        # Look for files in root directory
        supported_extensions = [".txt", ".json", ".csv", ".md"]
        
        for ext in supported_extensions:
            for file_path in data_path.glob(f"*{ext}"):
                # Skip if file is in a category subdirectory
                if any(cat in str(file_path) for cat in self.knowledge_categories.keys()):
                    continue
                
                try:
                    documents = await self._load_file(file_path, "general", self.text_splitter)
                    
                    if documents:
                        doc_ids = await self.vector_store.add_documents(
                            documents, batch_size=batch_size, upsert=True
                        )
                        
                        stats["files_processed"] += 1
                        stats["documents_created"] += len(documents)
                        stats["chunks_stored"] += len(doc_ids)
                
                except Exception as e:
                    error_msg = f"Error processing general file {file_path}: {str(e)}"
                    self.logger.warning(error_msg)
                    stats["errors"].append(error_msg)
        
        return stats
    
    async def _load_file(
        self, 
        file_path: Path, 
        category: str, 
        text_splitter: RecursiveCharacterTextSplitter
    ) -> List[Document]:
        """Load and process a single file."""
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == ".json":
                documents = await self._load_json_file(file_path, category)
            elif file_extension == ".csv":
                documents = await self._load_csv_file(file_path, category)
            elif file_extension in [".txt", ".md"]:
                documents = await self._load_text_file(file_path, category)
            else:
                self.logger.warning(f"Unsupported file type: {file_path}")
                return []
            
            # Split documents into chunks
            chunked_documents = []
            for doc in documents:
                chunks = text_splitter.split_documents([doc])
                chunked_documents.extend(chunks)
            
            # Process and enhance document metadata
            processed_documents = []
            for doc in chunked_documents:
                processed_doc = await self._process_document(doc, category, file_path)
                processed_documents.append(processed_doc)
            
            return processed_documents
            
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {e}")
            raise
    
    async def _load_json_file(self, file_path: Path, category: str) -> List[Document]:
        """Load documents from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        if isinstance(data, list):
            # Array of objects
            for i, item in enumerate(data):
                content = self._extract_content_from_dict(item)
                metadata = {
                    "source": str(file_path),
                    "category": category,
                    "index": i,
                    "original_data": item
                }
                documents.append(Document(page_content=content, metadata=metadata))
        
        elif isinstance(data, dict):
            # Single object or object with nested structure
            if "documents" in data:
                # Structured format with documents array
                for i, doc_data in enumerate(data["documents"]):
                    content = self._extract_content_from_dict(doc_data)
                    metadata = {
                        "source": str(file_path),
                        "category": category,
                        "index": i,
                        **doc_data.get("metadata", {})
                    }
                    documents.append(Document(page_content=content, metadata=metadata))
            else:
                # Single document
                content = self._extract_content_from_dict(data)
                metadata = {
                    "source": str(file_path),
                    "category": category,
                    "original_data": data
                }
                documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    async def _load_csv_file(self, file_path: Path, category: str) -> List[Document]:
        """Load documents from CSV file."""
        documents = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                # Determine content column
                content_column = self._identify_content_column(row.keys())
                
                if content_column:
                    content = str(row[content_column])
                    
                    # Use other columns as metadata
                    metadata = {
                        "source": str(file_path),
                        "category": category,
                        "row_index": i
                    }
                    
                    for key, value in row.items():
                        if key != content_column:
                            metadata[key] = value
                    
                    documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    async def _load_text_file(self, file_path: Path, category: str) -> List[Document]:
        """Load document from text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            "source": str(file_path),
            "category": category,
            "filename": file_path.name
        }
        
        return [Document(page_content=content, metadata=metadata)]
    
    def _extract_content_from_dict(self, data: Dict[str, Any]) -> str:
        """Extract readable content from dictionary data."""
        # Priority order for content fields
        content_fields = [
            "content", "text", "description", "summary", 
            "body", "message", "details", "overview"
        ]
        
        # Try to find content in priority order
        for field in content_fields:
            if field in data and data[field]:
                content = str(data[field])
                if len(content.strip()) > 10:  # Minimum content length
                    return content
        
        # If no content field found, combine title and other text fields
        title = data.get("title", data.get("name", ""))
        other_text = []
        
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 5 and key not in ["id", "source", "category"]:
                other_text.append(f"{key}: {value}")
        
        combined_content = title
        if other_text:
            combined_content += "\n" + "\n".join(other_text)
        
        return combined_content if combined_content.strip() else str(data)
    
    def _identify_content_column(self, columns: List[str]) -> Optional[str]:
        """Identify the main content column in CSV data."""
        # Priority order for content columns
        content_columns = [
            "content", "text", "description", "summary", 
            "body", "message", "details", "overview"
        ]
        
        # Find the first matching column
        for col in content_columns:
            if col in columns:
                return col
        
        # If no exact match, look for partial matches
        for col in columns:
            col_lower = col.lower()
            if any(content_col in col_lower for content_col in content_columns):
                return col
        
        # Return first non-id column as fallback
        for col in columns:
            if "id" not in col.lower():
                return col
        
        return None
    
    async def _process_document(
        self, 
        document: Document, 
        category: str, 
        file_path: Path
    ) -> Document:
        """Process and enhance document with additional metadata."""
        # Sanitize content
        document.page_content = sanitize_text(document.page_content)
        
        # Enhance metadata
        document.metadata.update({
            "content_type": category,
            "processed_at": datetime.utcnow().isoformat(),
            "char_count": len(document.page_content),
            "word_count": len(document.page_content.split())
        })
        
        # Extract keywords if configured for this category
        category_config = self.knowledge_categories.get(category, {})
        if category_config.get("keywords_extract", False):
            keywords = extract_keywords(document.page_content, min_length=3, max_keywords=10)
            document.metadata["keywords"] = keywords
        
        # Add title if not present
        if "title" not in document.metadata:
            title = self._generate_title(document.page_content, file_path.stem)
            document.metadata["title"] = title
        
        # Add summary for long documents
        if len(document.page_content) > 500:
            summary = self._generate_summary(document.page_content)
            document.metadata["summary"] = summary
        
        return document
    
    def _generate_title(self, content: str, filename: str) -> str:
        """Generate a title for the document."""
        # Try to extract title from first line
        first_line = content.split('\n')[0].strip()
        
        # If first line looks like a title (not too long, has meaningful words)
        if len(first_line) < 100 and len(first_line.split()) >= 2:
            return first_line
        
        # Extract first meaningful sentence
        sentences = content.split('.')[:3]  # First 3 sentences
        for sentence in sentences:
            sentence = sentence.strip()
            if 10 <= len(sentence) <= 80:  # Reasonable title length
                return sentence
        
        # Fallback to filename-based title
        return filename.replace('_', ' ').replace('-', ' ').title()
    
    def _generate_summary(self, content: str, max_length: int = 200) -> str:
        """Generate a summary of the document content."""
        # Simple extractive summary - take first few sentences
        sentences = content.split('.')
        
        summary = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(summary + sentence) < max_length:
                summary += sentence + ". "
            else:
                break
        
        return summary.strip()
    
    async def add_marketing_knowledge(
        self,
        content: str,
        title: str,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a single piece of marketing knowledge.
        
        Args:
            content: The knowledge content
            title: Title for the knowledge
            category: Knowledge category
            metadata: Additional metadata
            
        Returns:
            Document ID of the added knowledge
        """
        try:
            # Prepare document
            doc_metadata = {
                "title": title,
                "content_type": category,
                "source": "manual_input",
                "added_at": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            document = Document(
                page_content=sanitize_text(content),
                metadata=doc_metadata
            )
            
            # Process document
            processed_doc = await self._process_document(document, category, Path(title))
            
            # Add to vector store
            doc_ids = await self.vector_store.add_documents([processed_doc], upsert=True)
            
            self.logger.info("Marketing knowledge added", extra={
                "title": title,
                "category": category,
                "doc_id": doc_ids[0] if doc_ids else None
            })
            
            return doc_ids[0] if doc_ids else ""
            
        except Exception as e:
            self.logger.error(f"Error adding marketing knowledge: {e}")
            raise handle_exception(self.logger, e, "adding marketing knowledge")
    
    async def update_knowledge_metadata(
        self,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for existing knowledge.
        
        Args:
            document_id: ID of the document to update
            metadata: New metadata to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add update timestamp
            metadata["updated_at"] = datetime.utcnow().isoformat()
            
            success = await self.vector_store.update_document_metadata(document_id, metadata)
            
            if success:
                self.logger.info("Knowledge metadata updated", extra={
                    "document_id": document_id
                })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating knowledge metadata: {e}")
            return False
    
    async def search_knowledge(
        self,
        query: str,
        category: Optional[str] = None,
        k: int = 5
    ) -> List[Document]:
        """
        Search knowledge base.
        
        Args:
            query: Search query
            category: Optional category filter
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        filter_metadata = {"content_type": category} if category else None
        return await self.vector_store.similarity_search(query, k, filter_metadata)
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            # Get total count
            total_count = await self.vector_store.get_document_count()
            
            # Get counts by category
            category_counts = {}
            for category in self.knowledge_categories.keys():
                count = await self.vector_store.get_document_count({"content_type": category})
                category_counts[category] = count
            
            # Get general count
            general_count = await self.vector_store.get_document_count({"content_type": "general"})
            category_counts["general"] = general_count
            
            return {
                "total_documents": total_count,
                "category_counts": category_counts,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting knowledge stats: {e}")
            return {
                "total_documents": 0,
                "category_counts": {},
                "error": str(e)
            }