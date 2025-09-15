"""
Database package for the Marketing Strategy Agent.

This package contains database setup, vector store integration,
and knowledge base management functionality.
"""

from .company_database import CompanyDatabase, CompanySelector
from .simple_vector_store import VectorStore

__all__ = [
    "CompanyDatabase",
    "CompanySelector", 
    "VectorStore"
]