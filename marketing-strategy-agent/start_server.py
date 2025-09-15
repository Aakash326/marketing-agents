#!/usr/bin/env python3
"""Start the Marketing Strategy Agent server."""

import uvicorn
from app import app

if __name__ == "__main__":
    print("🚀 Starting Marketing Strategy Agent Server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔧 Interactive API: http://localhost:8000/redoc")
    print("\n💡 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )