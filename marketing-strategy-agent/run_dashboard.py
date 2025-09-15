#!/usr/bin/env python3
"""
Marketing Strategy Agent Dashboard Runner
Launch the interactive web dashboard for the marketing workflow system
"""
import os
import sys
import asyncio
import uvicorn
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.config.settings import load_config, validate_config

def main():
    """Main entry point for the dashboard application"""
    print("🚀 Starting Marketing Strategy Agent Dashboard...")
    
    try:
        # Load and validate configuration
        config = load_config()
        validate_config(config)
        
        print("✅ Configuration loaded successfully")
        
        # Get server configuration
        host = config.get("HOST", "0.0.0.0")
        port = config.get("PORT", 8000)
        debug = config.get("DEBUG", False)
        
        print(f"🌐 Starting server on http://{host}:{port}")
        print(f"📊 Dashboard will be available at: http://localhost:{port}")
        print(f"🔧 Debug mode: {'ON' if debug else 'OFF'}")
        
        # Check for required API keys
        required_apis = []
        if not config.get("OPENAI_API_KEY"):
            required_apis.append("OPENAI_API_KEY")
        
        optional_apis = []
        if not config.get("GEMINI_API_KEY"):
            optional_apis.append("GEMINI_API_KEY (for visual generation)")
        if not config.get("TAVILY_API_KEY"):
            optional_apis.append("TAVILY_API_KEY (for web research)")
        
        if required_apis:
            print(f"⚠️  Missing required API keys: {', '.join(required_apis)}")
            print("Please add these to your .env file before starting the application.")
            return
        
        if optional_apis:
            print(f"ℹ️  Optional API keys not configured: {', '.join(optional_apis)}")
            print("Some features may not be available.")
        
        print("\n" + "="*60)
        print("Marketing Strategy Agent Dashboard")
        print("="*60)
        print("Features:")
        print("• 🏢 Company selection and management")
        print("• 🤖 Multi-agent marketing workflow execution")
        print("• 📊 Real-time progress tracking")
        print("• 📈 Brand analysis and competitive intelligence")
        print("• 🌐 Market trend research")
        print("• 📝 Content strategy generation")
        print("• 🎨 AI-powered visual content creation")
        print("• 📋 Comprehensive marketing reports")
        print("="*60)
        
        # Start the server
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=debug,
            log_level="info" if debug else "warning",
            access_log=debug
        )
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down Marketing Strategy Agent Dashboard...")
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())