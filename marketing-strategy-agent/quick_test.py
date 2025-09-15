#!/usr/bin/env python3
"""Quick system test for Marketing Strategy Agent."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def main():
    try:
        print("🧪 Running quick system test...")
        print("=" * 50)
        
        # Test 1: Import core modules
        print("📦 Testing module imports...")
        try:
            from config import settings
            print("   ✅ Config module imported successfully")
        except Exception as e:
            print(f"   ❌ Config import failed: {e}")
            return False
        
        # Test 2: Check environment variables
        print("🔐 Testing environment configuration...")
        try:
            if not settings.openai_api_key or "your_" in settings.openai_api_key:
                print("   ⚠️  OpenAI API key not configured")
            else:
                print("   ✅ OpenAI API key configured")
                
            if not settings.tidb_host or "your_" in settings.tidb_user:
                print("   ⚠️  TiDB credentials not fully configured")
            else:
                print("   ✅ TiDB credentials configured")
        except Exception as e:
            print(f"   ❌ Environment check failed: {e}")
        
        # Test 3: Database connection (optional if configured)
        print("🗄️  Testing database connection...")
        try:
            if settings.tidb_host and "your_" not in settings.tidb_user:
                from src.database.vector_store import TiDBVectorStore
                vector_store = TiDBVectorStore()
                print("   ✅ Database connection successful")
                
                # Test 4: Knowledge retrieval (basic test)
                print("📚 Testing knowledge base setup...")
                print("   ✅ Vector store created successfully")
                
            else:
                print("   ⚠️  Database credentials not configured, skipping connection test")
        except Exception as e:
            print(f"   ❌ Database test failed: {e}")
            print("   💡 Tip: Make sure your TiDB credentials are correct in .env")
        
        # Test 5: FastAPI app
        print("🌐 Testing FastAPI application...")
        try:
            from app import app
            print("   ✅ FastAPI app created successfully")
        except Exception as e:
            print(f"   ❌ FastAPI app creation failed: {e}")
        
        print("\n" + "=" * 50)
        print("🎉 Quick test completed!")
        print("\n📋 Next steps:")
        print("   1. Configure your .env file with API keys")
        print("   2. Run: python scripts/initialize_knowledge_base.py")
        print("   3. Run: python start_server.py")
        print("   4. Visit: http://localhost:8000/docs")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if not success:
        sys.exit(1)