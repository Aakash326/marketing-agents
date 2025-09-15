#!/usr/bin/env python3
"""
Knowledge Base Initialization Script.

This script loads all sample knowledge base data into the TiDB vector store
and ensures all agents can access the marketing knowledge.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings
from src.database.tidb_setup import initialize_database, get_database_connection
from src.database.vector_store import TiDBVectorStore
from src.database.knowledge_loader import KnowledgeBaseLoader
from src.utils.logging import get_component_logger

logger = get_component_logger("knowledge_init", __name__)


async def main():
    """Initialize the knowledge base with sample data."""
    try:
        logger.info("Starting knowledge base initialization")
        
        # Step 1: Initialize database
        logger.info("Initializing database...")
        await initialize_database()
        
        # Step 2: Create vector store
        logger.info("Creating vector store...")
        vector_store = TiDBVectorStore()
        
        # Step 3: Create knowledge loader
        logger.info("Creating knowledge loader...")
        knowledge_loader = KnowledgeBaseLoader(vector_store)
        
        # Step 4: Load knowledge base
        knowledge_base_path = project_root / "data" / "knowledge_base"
        
        if not knowledge_base_path.exists():
            logger.error(f"Knowledge base directory not found: {knowledge_base_path}")
            return False
            
        logger.info(f"Loading knowledge base from: {knowledge_base_path}")
        
        # Load with overwrite to ensure fresh data
        load_stats = await knowledge_loader.load_knowledge_base(
            data_directory=knowledge_base_path,
            batch_size=20,
            overwrite=True
        )
        
        # Step 5: Display results
        logger.info("Knowledge base loading completed!")
        logger.info(f"Files processed: {load_stats['total_files_processed']}")
        logger.info(f"Documents created: {load_stats['total_documents_created']}")
        logger.info(f"Chunks stored: {load_stats['total_chunks_stored']}")
        
        # Display category breakdown
        logger.info("Category breakdown:")
        for category, stats in load_stats['categories_loaded'].items():
            logger.info(f"  {category}: {stats['files_processed']} files, {stats['chunks_stored']} chunks")
        
        # Display any errors
        if load_stats['errors']:
            logger.warning(f"Errors encountered: {len(load_stats['errors'])}")
            for error in load_stats['errors']:
                logger.warning(f"  - {error}")
        
        # Step 6: Test search functionality
        logger.info("Testing search functionality...")
        test_queries = [
            "marketing strategy frameworks",
            "Tesla brand positioning",
            "social media templates",
            "successful marketing campaigns",
            "market trends 2025"
        ]
        
        for query in test_queries:
            results = await knowledge_loader.search_knowledge(query, k=3)
            logger.info(f"Query '{query}': {len(results)} results found")
            
            for i, doc in enumerate(results[:2]):  # Show first 2 results
                title = doc.metadata.get('title', 'Unknown')
                category = doc.metadata.get('content_type', 'Unknown')
                logger.info(f"  {i+1}. {title} ({category})")
        
        # Step 7: Get final statistics
        stats = await knowledge_loader.get_knowledge_stats()
        logger.info("Final knowledge base statistics:")
        logger.info(f"Total documents: {stats['total_documents']}")
        logger.info("Documents by category:")
        for category, count in stats['category_counts'].items():
            logger.info(f"  {category}: {count}")
        
        logger.info("Knowledge base initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Knowledge base initialization failed: {e}", exc_info=True)
        return False


def verify_knowledge_files():
    """Verify that all expected knowledge files exist."""
    project_root = Path(__file__).parent.parent
    knowledge_base_path = project_root / "data" / "knowledge_base"
    
    expected_files = [
        "marketing_strategies/frameworks_guide.json",
        "brand_examples/tesla_case_study.json",
        "content_templates/social_media_templates.json",
        "content_templates/email_marketing_templates.json",
        "campaign_examples/successful_campaigns.json",
        "industry_insights/market_trends_2025.json"
    ]
    
    missing_files = []
    for file_path in expected_files:
        full_path = knowledge_base_path / file_path
        if not full_path.exists():
            missing_files.append(str(full_path))
    
    if missing_files:
        logger.error("Missing knowledge base files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    logger.info(f"All {len(expected_files)} expected knowledge files found")
    return True


async def test_agent_integration():
    """Test that agents can access the knowledge base."""
    try:
        logger.info("Testing agent integration with knowledge base...")
        
        # Import agents after the knowledge base is loaded
        from src.agents.marketing_agent import MarketingAgent
        from src.agents.brand_analyzer import BrandAnalyzer
        from src.agents.content_creator import ContentCreator
        from src.agents.trend_researcher import TrendResearcher
        
        # Initialize vector store
        vector_store = TiDBVectorStore()
        
        # Create agents
        marketing_agent = MarketingAgent(vector_store)
        brand_analyzer = BrandAnalyzer(vector_store)
        content_creator = ContentCreator(vector_store)
        trend_researcher = TrendResearcher(vector_store)
        
        # Test each agent's knowledge access
        agents = [
            ("Marketing Agent", marketing_agent),
            ("Brand Analyzer", brand_analyzer),
            ("Content Creator", content_creator),
            ("Trend Researcher", trend_researcher)
        ]
        
        for agent_name, agent in agents:
            try:
                # Test knowledge retrieval
                knowledge = await agent.get_relevant_knowledge("marketing strategy")
                logger.info(f"{agent_name}: Retrieved {len(knowledge)} knowledge documents")
                
                if knowledge:
                    first_doc = knowledge[0]
                    title = first_doc.metadata.get('title', 'Unknown')
                    logger.info(f"  Sample knowledge: {title}")
                
            except Exception as e:
                logger.error(f"Error testing {agent_name}: {e}")
        
        logger.info("Agent integration testing completed")
        return True
        
    except Exception as e:
        logger.error(f"Agent integration testing failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    print("🚀 Marketing Strategy Agent - Knowledge Base Initialization")
    print("=" * 60)
    
    # Verify files exist
    print("📋 Verifying knowledge base files...")
    if not verify_knowledge_files():
        print("❌ Knowledge base files verification failed")
        sys.exit(1)
    print("✅ All knowledge base files found")
    
    # Initialize knowledge base
    print("\n📚 Initializing knowledge base...")
    success = asyncio.run(main())
    
    if success:
        print("\n🧪 Testing agent integration...")
        integration_success = asyncio.run(test_agent_integration())
        
        if integration_success:
            print("\n✅ Knowledge base initialization and agent integration completed successfully!")
            print("\n🎯 Your marketing strategy agent is ready to use!")
            print("\nNext steps:")
            print("1. Start the FastAPI server: python app.py")
            print("2. Access the API at: http://localhost:8000")
            print("3. View API docs at: http://localhost:8000/docs")
        else:
            print("\n⚠️  Knowledge base loaded but agent integration had issues")
    else:
        print("\n❌ Knowledge base initialization failed")
        sys.exit(1)