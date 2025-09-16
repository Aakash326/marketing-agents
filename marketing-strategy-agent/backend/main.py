"""
Advanced FastAPI Backend for Marketing Analyzer
Modern API with real-time WebSocket support, file handling, and comprehensive endpoints
"""
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.data_models import CompanyInfo, Industry, CompanySize
from src.database.company_database import CompanyDatabase

# Import the automated marketing analyzer
sys.path.insert(0, str(Path(__file__).parent.parent))
from run_analyzer_auto import AutoMarketingAnalyzer

# Initialize FastAPI app with advanced configuration
app = FastAPI(
    title="🎯 Marketing Strategy Analyzer API",
    description="Advanced AI-powered marketing analysis platform with real-time processing",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend (optional)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Global state management
class AppState:
    def __init__(self):
        self.active_analyses: Dict[str, Dict] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.company_database = CompanyDatabase()
        self.reports_dir = Path("data/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def add_connection(self, analysis_id: str, websocket: WebSocket):
        self.websocket_connections[analysis_id] = websocket
    
    def remove_connection(self, analysis_id: str):
        if analysis_id in self.websocket_connections:
            del self.websocket_connections[analysis_id]
    
    async def broadcast_update(self, analysis_id: str, data: Dict):
        if analysis_id in self.websocket_connections:
            try:
                websocket = self.websocket_connections[analysis_id]
                await websocket.send_json(data)
            except Exception as e:
                print(f"Failed to send WebSocket update: {e}")
                self.remove_connection(analysis_id)

# Initialize global state
state = AppState()

# Pydantic models
class CompanyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    industry: str = Field(..., description="Company industry")
    size: str = Field(..., description="Company size")
    location: str = Field(default="Global", max_length=100)
    description: str = Field(..., min_length=10, max_length=1000)
    website: Optional[str] = Field(None, max_length=200)

class AnalysisRequest(BaseModel):
    company: CompanyRequest
    analysis_type: str = Field(default="complete", description="Type of analysis to run")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)

class AnalysisStatus(BaseModel):
    analysis_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: int = Field(ge=0, le=100)
    current_step: Optional[str] = None
    message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AnalysisResult(BaseModel):
    analysis_id: str
    company_name: str
    status: str
    success_rate: str
    agents_executed: int
    successful_agents: int
    execution_time: float
    pdf_report_path: Optional[str] = None
    text_report_path: Optional[str] = None
    created_at: datetime

# API Endpoints
@app.get("/")
async def root():
    """API root endpoint with system information"""
    return {
        "name": "Marketing Strategy Analyzer API",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Complete marketing analysis with 6 AI agents",
            "Real-time progress tracking via WebSocket",
            "PDF and text report generation",
            "Company database management",
            "File download capabilities"
        ],
        "endpoints": {
            "companies": "/api/companies",
            "analysis": "/api/analysis",
            "reports": "/api/reports",
            "websocket": "/ws/{analysis_id}",
            "docs": "/api/docs"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_analyses": len(state.active_analyses),
        "websocket_connections": len(state.websocket_connections)
    }

# Company Management Endpoints
@app.get("/api/companies")
async def list_companies():
    """Get all companies from database"""
    try:
        companies = state.company_database.list_companies()
        return {
            "companies": companies,
            "total": len(companies),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch companies: {str(e)}")

@app.post("/api/companies")
async def create_company(company: CompanyRequest):
    """Add a new company to database"""
    try:
        company_info = CompanyInfo(
            name=company.name,
            industry=company.industry,
            size=company.size,
            location=company.location,
            description=company.description,
            website=company.website or ""
        )
        
        success = state.company_database.add_company(company_info)
        if success:
            return {
                "message": "Company created successfully",
                "company": company_info.dict(),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=409, detail="Company already exists")
            
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create company: {str(e)}")

@app.get("/api/companies/{company_name}")
async def get_company(company_name: str):
    """Get specific company by name"""
    try:
        company = state.company_database.get_company(company_name)
        if not company:
            raise HTTPException(status_code=404, detail="Company not found")
        
        return {
            "company": company,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch company: {str(e)}")

# Analysis Endpoints
@app.post("/api/analysis/start")
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start a new marketing analysis"""
    try:
        # Generate unique analysis ID
        analysis_id = f"analysis_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert to CompanyInfo
        company_info = CompanyInfo(
            name=request.company.name,
            industry=request.company.industry,
            size=request.company.size,
            location=request.company.location,
            description=request.company.description,
            website=request.company.website or ""
        )
        
        # Initialize analysis tracking
        state.active_analyses[analysis_id] = {
            "status": "pending",
            "progress": 0,
            "current_step": "Initializing",
            "company": company_info.dict(),
            "started_at": datetime.now(),
            "analysis_type": request.analysis_type
        }
        
        # Start analysis in background
        background_tasks.add_task(
            run_marketing_analysis,
            analysis_id,
            company_info,
            request.options
        )
        
        return {
            "analysis_id": analysis_id,
            "status": "started",
            "message": "Marketing analysis started successfully",
            "estimated_duration": "5-15 minutes",
            "websocket_url": f"/ws/{analysis_id}",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

@app.get("/api/analysis/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    """Get status of a specific analysis"""
    if analysis_id not in state.active_analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = state.active_analyses[analysis_id]
    
    return {
        "analysis_id": analysis_id,
        "status": analysis["status"],
        "progress": analysis["progress"],
        "current_step": analysis.get("current_step"),
        "message": analysis.get("message"),
        "started_at": analysis["started_at"].isoformat(),
        "completed_at": analysis.get("completed_at").isoformat() if analysis.get("completed_at") else None,
        "results": analysis.get("results"),
        "error": analysis.get("error"),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/analysis")
async def list_analyses():
    """List all analyses (active and completed)"""
    analyses = []
    for analysis_id, data in state.active_analyses.items():
        analyses.append({
            "analysis_id": analysis_id,
            "company_name": data["company"]["name"],
            "status": data["status"],
            "progress": data["progress"],
            "started_at": data["started_at"].isoformat(),
            "completed_at": data.get("completed_at").isoformat() if data.get("completed_at") else None
        })
    
    return {
        "analyses": analyses,
        "total": len(analyses),
        "active": len([a for a in analyses if a["status"] in ["pending", "running"]]),
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/api/analysis/{analysis_id}")
async def cancel_analysis(analysis_id: str):
    """Cancel a running analysis"""
    if analysis_id not in state.active_analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = state.active_analyses[analysis_id]
    if analysis["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed analysis")
    
    # Update status
    analysis["status"] = "cancelled"
    analysis["completed_at"] = datetime.now()
    analysis["message"] = "Analysis cancelled by user"
    
    # Notify via WebSocket
    await state.broadcast_update(analysis_id, {
        "type": "status_update",
        "status": "cancelled",
        "message": "Analysis cancelled by user"
    })
    
    return {
        "message": "Analysis cancelled successfully",
        "analysis_id": analysis_id,
        "timestamp": datetime.now().isoformat()
    }

# Report Management Endpoints
@app.get("/api/reports")
async def list_reports():
    """List all generated reports"""
    reports = []
    reports_dir = Path("data/reports")
    
    if reports_dir.exists():
        for file_path in reports_dir.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                reports.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "type": file_path.suffix.lower(),
                    "download_url": f"/api/reports/download/{file_path.name}"
                })
    
    # Sort by creation time, newest first
    reports.sort(key=lambda x: x["created"], reverse=True)
    
    return {
        "reports": reports,
        "total": len(reports),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/reports/download/{filename}")
async def download_report(filename: str):
    """Download a specific report file"""
    file_path = Path("data/reports") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/octet-stream'
    )

@app.delete("/api/reports/{filename}")
async def delete_report(filename: str):
    """Delete a specific report file"""
    file_path = Path("data/reports") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    
    try:
        file_path.unlink()
        return {
            "message": "Report deleted successfully",
            "filename": filename,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete report: {str(e)}")

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: str):
    """WebSocket endpoint for real-time analysis updates"""
    await websocket.accept()
    state.add_connection(analysis_id, websocket)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_json()
                
                # Handle different message types
                if data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        state.remove_connection(analysis_id)

# Helper function for running analysis with progress updates
async def run_analysis_with_progress(analyzer, company_info, progress_callback):
    """Run automated analysis with real-time progress updates"""
    agent_order = [
        ("BrandAnalyzer", "🔍 Analyzing brand positioning & competitive landscape...", 15),
        ("TrendResearcher", "📈 Researching market trends & opportunities...", 30),
        ("ContentCreator", "📝 Developing content strategy & social media plans...", 45),
        ("MarketingAgent", "🎯 Creating comprehensive marketing strategy...", 60),
        ("GeminiVisualGenerator", "🎨 Generating visual content recommendations...", 75)
    ]
    
    results = {}
    
    # Run each agent with progress updates
    for i, (agent_name, description, progress) in enumerate(agent_order, 1):
        await progress_callback(agent_name, progress, description)
        
        try:
            agent = analyzer.agents[agent_name]
            
            # Run agent with timeout
            result = await asyncio.wait_for(
                agent._execute_with_tracking(company_info, previous_results=results),
                timeout=180  # 3 minutes per agent
            )
            
            results[agent_name] = {
                "success": result.success,
                "execution_time": result.execution_time,
                "result": result.result,
                "error_message": result.error_message
            }
            
        except asyncio.TimeoutError:
            results[agent_name] = {
                "success": False,
                "execution_time": 180,
                "result": None,
                "error_message": "Agent execution timed out"
            }
        except Exception as e:
            results[agent_name] = {
                "success": False,
                "execution_time": 0,
                "result": None,
                "error_message": str(e)
            }
    
    # Generate reports if any agents succeeded
    successful_agents = [name for name, result in results.items() if result["success"]]
    if successful_agents:
        await progress_callback("PDFGeneratorAgent", 90, "📄 Generating professional PDF and text reports...")
        
        try:
            pdf_agent = analyzer.agents["PDFGeneratorAgent"]
            result = await asyncio.wait_for(
                pdf_agent._execute_with_tracking(company_info, agent_results=results),
                timeout=120  # 2 minutes for report generation
            )
            
            if result.success:
                results["PDFGeneratorAgent"] = {
                    "success": True,
                    "execution_time": result.execution_time,
                    "result": result.result,
                    "error_message": None
                }
                # Only set 100% when PDF is successfully generated
                await progress_callback("Complete", 100, "🎉 Complete analysis finished!")
            else:
                results["PDFGeneratorAgent"] = {
                    "success": False,
                    "execution_time": result.execution_time,
                    "result": None,
                    "error_message": result.error_message
                }
                await progress_callback("Complete", 95, "⚠️ Analysis complete (report generation failed)")
        except Exception as e:
            results["PDFGeneratorAgent"] = {
                "success": False,
                "execution_time": 0,
                "result": None,
                "error_message": str(e)
            }
            await progress_callback("Complete", 95, "⚠️ Analysis complete (report generation failed)")
    else:
        await progress_callback("Complete", 75, "⚠️ Analysis complete (no successful agents)")
    
    return results

# Helper function for quick analysis (fewer agents, faster)
async def run_quick_analysis_with_progress(analyzer, company_info, progress_callback):
    """Run quick analysis with subset of agents"""
    # Quick analysis uses only essential agents  
    agent_order = [
        ("BrandAnalyzer", "🔍 Quick brand analysis...", 25),
        ("TrendResearcher", "📈 Quick trend research...", 50),
        ("MarketingAgent", "🎯 Quick marketing strategy...", 75)
    ]
    
    results = {}
    
    # Run each agent with progress updates
    for i, (agent_name, description, progress) in enumerate(agent_order, 1):
        await progress_callback(agent_name, progress, description)
        
        try:
            agent = analyzer.agents[agent_name]
            
            # Run agent with shorter timeout for quick analysis
            result = await asyncio.wait_for(
                agent._execute_with_tracking(company_info, previous_results=results),
                timeout=120  # 2 minutes per agent for quick analysis
            )
            
            results[agent_name] = {
                "success": result.success,
                "execution_time": result.execution_time,
                "result": result.result,
                "error_message": result.error_message
            }
            
        except asyncio.TimeoutError:
            results[agent_name] = {
                "success": False,
                "execution_time": 120,
                "result": None,
                "error_message": "Agent execution timed out (quick mode)"
            }
        except Exception as e:
            results[agent_name] = {
                "success": False,
                "execution_time": 0,
                "result": None,
                "error_message": str(e)
            }
    
    # Generate reports for quick analysis too (but with simpler report)
    successful_agents = [name for name, result in results.items() if result["success"]]
    if successful_agents:
        await progress_callback("PDFGeneratorAgent", 90, "📄 Generating quick analysis report...")
        
        try:
            pdf_agent = analyzer.agents["PDFGeneratorAgent"]
            result = await asyncio.wait_for(
                pdf_agent._execute_with_tracking(company_info, agent_results=results),
                timeout=60  # 1 minute for quick report generation
            )
            
            if result.success:
                results["PDFGeneratorAgent"] = {
                    "success": True,
                    "execution_time": result.execution_time,
                    "result": result.result,
                    "error_message": None
                }
                await progress_callback("Complete", 100, "📊 Quick analysis complete!")
            else:
                results["PDFGeneratorAgent"] = {
                    "success": False,
                    "execution_time": result.execution_time,
                    "result": None,
                    "error_message": result.error_message
                }
                await progress_callback("Complete", 95, "⚠️ Quick analysis complete (report generation failed)")
        except Exception as e:
            results["PDFGeneratorAgent"] = {
                "success": False,
                "execution_time": 0,
                "result": None,
                "error_message": str(e)
            }
            await progress_callback("Complete", 95, "⚠️ Quick analysis complete (report generation failed)")
    else:
        await progress_callback("Complete", 75, "⚠️ Quick analysis complete (no successful agents)")
    
    return results

# Background task for running marketing analysis
async def run_marketing_analysis(analysis_id: str, company_info: CompanyInfo, options: Dict[str, Any]):
    """Background task to run complete marketing analysis"""
    try:
        # Update status to running
        state.active_analyses[analysis_id]["status"] = "running"
        state.active_analyses[analysis_id]["current_step"] = "Initializing AI agents"
        state.active_analyses[analysis_id]["progress"] = 5
        
        await state.broadcast_update(analysis_id, {
            "type": "status_update",
            "status": "running",
            "progress": 5,
            "current_step": "Initializing AI agents",
            "message": "Setting up marketing analysis environment"
        })
        
        # Initialize the automated marketing analyzer
        analyzer = AutoMarketingAnalyzer()
        
        # Update progress
        state.active_analyses[analysis_id]["progress"] = 10
        state.active_analyses[analysis_id]["current_step"] = "Running AI agents"
        
        await state.broadcast_update(analysis_id, {
            "type": "status_update",
            "progress": 10,
            "current_step": "Running AI agents",
            "message": "Starting comprehensive marketing analysis with 6 AI agents"
        })
        
        # Custom progress callback for real-time updates
        async def progress_callback(agent_name: str, progress: int, message: str):
            if state.active_analyses[analysis_id]["status"] == "cancelled":
                return
            
            state.active_analyses[analysis_id]["progress"] = progress
            state.active_analyses[analysis_id]["current_step"] = agent_name
            
            await state.broadcast_update(analysis_id, {
                "type": "progress_update",
                "progress": progress,
                "current_step": agent_name,
                "message": message
            })
        
        # Determine analysis mode based on request
        analysis_type = state.active_analyses[analysis_id]["analysis_type"]
        
        # Run the analysis with integrated progress updates
        if analysis_type == "quick":
            agent_results = await run_quick_analysis_with_progress(analyzer, company_info, progress_callback)
        else:
            agent_results = await run_analysis_with_progress(analyzer, company_info, progress_callback)
        
        # Calculate results
        successful_agents = sum(1 for result in agent_results.values() if result["success"])
        total_agents = len(agent_results)
        total_time = sum(result["execution_time"] for result in agent_results.values())
        
        # Extract report information from PDFGeneratorAgent if available
        report_info = None
        if "PDFGeneratorAgent" in agent_results and agent_results["PDFGeneratorAgent"]["success"]:
            report_info = agent_results["PDFGeneratorAgent"]["result"]
        
        # Complete the analysis - but don't override progress if already set by agent workflow
        current_progress = state.active_analyses[analysis_id]["progress"]
        
        state.active_analyses[analysis_id]["status"] = "completed"
        # Only set progress to 100 if it hasn't been set by the workflow (fallback)
        if current_progress < 100:
            # Check if PDF generation was successful
            pdf_success = "PDFGeneratorAgent" in agent_results and agent_results["PDFGeneratorAgent"]["success"]
            state.active_analyses[analysis_id]["progress"] = 100 if pdf_success else 95
        
        state.active_analyses[analysis_id]["current_step"] = "Complete"
        state.active_analyses[analysis_id]["completed_at"] = datetime.now()
        state.active_analyses[analysis_id]["results"] = {
            "successful_agents": successful_agents,
            "total_agents": total_agents,
            "success_rate": f"{(successful_agents/total_agents*100):.0f}%",
            "execution_time": round(total_time, 1),
            "pdf_report": report_info.get("pdf_report_path") if report_info else None,
            "text_report": report_info.get("text_report_path") if report_info else None,
            "agent_results": {
                name: {
                    "success": result["success"],
                    "execution_time": result["execution_time"],
                    "error": result["error_message"] if not result["success"] else None
                }
                for name, result in agent_results.items()
            }
        }
        
        # Send completion notification
        await state.broadcast_update(analysis_id, {
            "type": "analysis_completed",
            "progress": 100,
            "message": "Marketing analysis completed successfully!",
            "results": state.active_analyses[analysis_id]["results"]
        })
        
    except Exception as e:
        # Handle analysis failure
        state.active_analyses[analysis_id]["status"] = "failed"
        state.active_analyses[analysis_id]["completed_at"] = datetime.now()
        state.active_analyses[analysis_id]["error"] = str(e)
        
        await state.broadcast_update(analysis_id, {
            "type": "analysis_failed",
            "message": f"Analysis failed: {str(e)}",
            "error": str(e)
        })

# Utility endpoints
@app.get("/api/system/info")
async def system_info():
    """Get system information"""
    return {
        "system": {
            "python_version": sys.version,
            "fastapi_version": "Latest",
            "reports_directory": str(Path("data/reports").absolute()),
            "active_analyses": len(state.active_analyses),
            "websocket_connections": len(state.websocket_connections)
        },
        "capabilities": {
            "agents": [
                "BrandAnalyzer - Brand positioning & competitive analysis",
                "TrendResearcher - Market trends & industry insights", 
                "ContentCreator - Content strategy & social media planning",
                "MarketingAgent - Comprehensive marketing strategy",
                "GeminiVisualGenerator - AI-powered visual content creation",
                "PDFGeneratorAgent - Professional report generation"
            ],
            "output_formats": ["PDF", "Text"],
            "supported_industries": [industry.value for industry in Industry],
            "supported_sizes": [size.value for size in CompanySize]
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting Advanced Marketing Analyzer API...")
    print("=" * 60)
    print("🔧 Backend Features:")
    print("   • FastAPI with modern async architecture")
    print("   • Real-time WebSocket progress tracking")
    print("   • Complete marketing analysis with 6 AI agents")
    print("   • PDF and text report generation")
    print("   • Company database management")
    print("   • File download capabilities")
    print("   • CORS enabled for frontend integration")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )