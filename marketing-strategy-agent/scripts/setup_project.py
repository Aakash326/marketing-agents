#!/usr/bin/env python3
"""
Complete Project Setup Script for Marketing Strategy Agent.

This script handles the complete setup process including:
- Environment verification
- Dependencies installation
- Database initialization  
- Knowledge base loading
- Configuration validation
- System testing
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import get_component_logger

logger = get_component_logger("setup", __name__)


class ProjectSetup:
    """Handles complete project setup and initialization."""
    
    def __init__(self):
        self.project_root = project_root
        self.setup_steps = []
        
    def run_command(self, command: str, description: str = "") -> bool:
        """Run a shell command and return success status."""
        try:
            if description:
                print(f"📋 {description}")
            
            print(f"   Running: {command}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("   ✅ Success")
                if result.stdout.strip():
                    print(f"   Output: {result.stdout.strip()}")
                return True
            else:
                print("   ❌ Failed")
                if result.stderr.strip():
                    print(f"   Error: {result.stderr.strip()}")
                return False
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            return False
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        print("🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major >= 3 and version.minor >= 9:
            print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
            return True
        else:
            print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
            print("   📝 Python 3.9+ is required")
            return False
    
    def check_environment_file(self) -> bool:
        """Check if .env file exists and contains required variables."""
        print("🔐 Checking environment configuration...")
        
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"
        
        if not env_file.exists():
            if env_example.exists():
                print("   📋 .env file not found, copying from .env.example")
                try:
                    import shutil
                    shutil.copy(env_example, env_file)
                    print("   ✅ .env file created from template")
                    print("   ⚠️  Please update .env with your actual credentials")
                    return True
                except Exception as e:
                    print(f"   ❌ Failed to copy .env.example: {e}")
                    return False
            else:
                print("   ❌ Neither .env nor .env.example found")
                return False
        
        # Check required environment variables
        required_vars = [
            "OPENAI_API_KEY",
            "TIDB_HOST",
            "TIDB_PORT", 
            "TIDB_USER",
            "TIDB_PASSWORD",
            "TIDB_DATABASE"
        ]
        
        try:
            with open(env_file, 'r') as f:
                env_content = f.read()
            
            missing_vars = []
            for var in required_vars:
                if f"{var}=" not in env_content or f"{var}=your_" in env_content:
                    missing_vars.append(var)
            
            if missing_vars:
                print(f"   ⚠️  Missing or placeholder values for: {', '.join(missing_vars)}")
                print("   📝 Please update .env with actual values before continuing")
                return False
            
            print("   ✅ Environment configuration looks complete")
            return True
            
        except Exception as e:
            print(f"   ❌ Error reading .env file: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies."""
        print("📦 Installing dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print("   ❌ requirements.txt not found")
            return False
        
        # Upgrade pip first
        success = self.run_command(
            f"{sys.executable} -m pip install --upgrade pip",
            "Upgrading pip"
        )
        
        if not success:
            print("   ⚠️  Pip upgrade failed, continuing anyway")
        
        # Install requirements
        return self.run_command(
            f"{sys.executable} -m pip install -r {requirements_file}",
            "Installing project dependencies"
        )
    
    def create_directories(self) -> bool:
        """Create necessary directories."""
        print("📁 Creating directories...")
        
        directories = [
            "logs",
            "data/uploads",
            "data/exports",
            "data/cache",
            "tmp"
        ]
        
        try:
            for dir_path in directories:
                full_path = self.project_root / dir_path
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Created/verified: {dir_path}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating directories: {e}")
            return False
    
    async def initialize_database(self) -> bool:
        """Initialize the database and vector store."""
        print("🗄️  Initializing database...")
        
        try:
            from src.database.tidb_setup import initialize_database
            await initialize_database()
            print("   ✅ Database initialized successfully")
            return True
            
        except Exception as e:
            print(f"   ❌ Database initialization failed: {e}")
            print("   📝 Please check your TiDB connection settings in .env")
            return False
    
    async def load_knowledge_base(self) -> bool:
        """Load the sample knowledge base."""
        print("📚 Loading knowledge base...")
        
        try:
            # Import and run the knowledge base initialization
            from scripts.initialize_knowledge_base import main as init_kb
            success = await init_kb()
            
            if success:
                print("   ✅ Knowledge base loaded successfully")
                return True
            else:
                print("   ❌ Knowledge base loading failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Error loading knowledge base: {e}")
            return False
    
    async def test_system(self) -> bool:
        """Run basic system tests."""
        print("🧪 Running system tests...")
        
        try:
            # Test 1: Vector store connection
            print("   Testing vector store connection...")
            from src.database.vector_store import TiDBVectorStore
            vector_store = TiDBVectorStore()
            await vector_store.initialize()
            print("   ✅ Vector store connection successful")
            
            # Test 2: Agent initialization
            print("   Testing agent initialization...")
            from src.agents.marketing_agent import MarketingAgent
            marketing_agent = MarketingAgent(vector_store)
            print("   ✅ Agent initialization successful")
            
            # Test 3: Knowledge retrieval
            print("   Testing knowledge retrieval...")
            knowledge = await marketing_agent.get_relevant_knowledge("marketing strategy")
            if knowledge:
                print(f"   ✅ Knowledge retrieval successful ({len(knowledge)} documents)")
            else:
                print("   ⚠️  No knowledge documents found")
            
            # Test 4: Workflow initialization
            print("   Testing workflow initialization...")
            from src.workflows.marketing_workflow import MarketingWorkflow
            workflow = MarketingWorkflow(vector_store)
            print("   ✅ Workflow initialization successful")
            
            return True
            
        except Exception as e:
            print(f"   ❌ System test failed: {e}")
            return False
    
    def create_startup_scripts(self) -> bool:
        """Create convenient startup scripts."""
        print("📜 Creating startup scripts...")
        
        try:
            # Create start server script
            start_script = self.project_root / "start_server.py"
            with open(start_script, 'w') as f:
                f.write("""#!/usr/bin/env python3
\"\"\"Start the Marketing Strategy Agent server.\"\"\"

import uvicorn
from app import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
""")
            
            # Make executable
            os.chmod(start_script, 0o755)
            print("   ✅ Created start_server.py")
            
            # Create quick test script
            test_script = self.project_root / "quick_test.py"
            with open(test_script, 'w') as f:
                f.write("""#!/usr/bin/env python3
\"\"\"Quick system test for Marketing Strategy Agent.\"\"\"

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def main():
    try:
        from src.database.vector_store import TiDBVectorStore
        from src.agents.marketing_agent import MarketingAgent
        
        print("🧪 Running quick system test...")
        
        # Test vector store
        vector_store = TiDBVectorStore()
        await vector_store.initialize()
        print("✅ Vector store connection: OK")
        
        # Test agent
        agent = MarketingAgent(vector_store)
        knowledge = await agent.get_relevant_knowledge("marketing")
        print(f"✅ Knowledge retrieval: OK ({len(knowledge)} docs)")
        
        print("🎉 All systems operational!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(main())
""")
            
            os.chmod(test_script, 0o755)
            print("   ✅ Created quick_test.py")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating startup scripts: {e}")
            return False
    
    def display_completion_info(self):
        """Display setup completion information."""
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        print()
        print("📋 Your Marketing Strategy Agent is ready to use!")
        print()
        print("🚀 Quick Start:")
        print("   1. Start the server: python start_server.py")
        print("   2. Open browser: http://localhost:8000")
        print("   3. View API docs: http://localhost:8000/docs")
        print()
        print("🧪 Testing:")
        print("   - Quick test: python quick_test.py")
        print("   - Full test: python scripts/initialize_knowledge_base.py")
        print()
        print("📁 Important Files:")
        print("   - Configuration: .env")
        print("   - Main app: app.py")
        print("   - Logs: logs/")
        print("   - Knowledge: data/knowledge_base/")
        print()
        print("📖 API Endpoints:")
        print("   - Generate strategy: POST /api/v1/marketing/strategy")
        print("   - Brand analysis: POST /api/v1/marketing/brand-analysis")
        print("   - Content creation: POST /api/v1/content/generate")
        print("   - Trend research: POST /api/v1/marketing/trends")
        print()
        print("💡 Next Steps:")
        print("   1. Customize knowledge base in data/knowledge_base/")
        print("   2. Adjust configuration in config.py")
        print("   3. Explore the API documentation")
        print("   4. Build your marketing strategies!")
        print()
    
    async def run_setup(self) -> bool:
        """Run the complete setup process."""
        print("🚀 Marketing Strategy Agent - Complete Setup")
        print("="*60)
        print()
        
        steps = [
            ("check_python_version", "Python Version Check", self.check_python_version),
            ("check_environment", "Environment Configuration", self.check_environment_file),
            ("install_dependencies", "Dependency Installation", self.install_dependencies),
            ("create_directories", "Directory Creation", self.create_directories),
            ("initialize_database", "Database Initialization", self.initialize_database),
            ("load_knowledge_base", "Knowledge Base Loading", self.load_knowledge_base),
            ("test_system", "System Testing", self.test_system),
            ("create_startup_scripts", "Startup Scripts", self.create_startup_scripts),
        ]
        
        failed_steps = []
        
        for step_id, step_name, step_func in steps:
            print(f"\n{'='*20} {step_name} {'='*20}")
            
            try:
                if asyncio.iscoroutinefunction(step_func):
                    success = await step_func()
                else:
                    success = step_func()
                
                if success:
                    print(f"✅ {step_name} completed successfully")
                else:
                    print(f"❌ {step_name} failed")
                    failed_steps.append(step_name)
                    
            except Exception as e:
                print(f"❌ {step_name} failed with exception: {e}")
                failed_steps.append(step_name)
        
        print(f"\n{'='*60}")
        
        if failed_steps:
            print("⚠️  Setup completed with issues:")
            for step in failed_steps:
                print(f"   ❌ {step}")
            print()
            print("📝 Please address the failed steps before using the system.")
            return False
        else:
            self.display_completion_info()
            return True


async def main():
    """Main setup function."""
    setup = ProjectSetup()
    success = await setup.run_setup()
    
    if success:
        print("✨ Setup completed successfully!")
        return 0
    else:
        print("💥 Setup failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)