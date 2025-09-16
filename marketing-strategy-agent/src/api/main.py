"""
FastAPI Application for Marketing Strategy Agent
Interactive dashboard for managing marketing workflows
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, ValidationError
import asyncio
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import logging

from ..models.data_models import (
    CompanyInfo,
    ComprehensiveMarketingPackage,
    WorkflowStatus,
    WorkflowProgress
)
from workflows.enhanced_marketing_workflow import WorkflowManager
from workflows.quick_marketing_workflow import QuickMarketingWorkflow
from ..database.company_database import CompanyDatabase, CompanySelector
from ..reports.report_generator import ReportGenerator
from ..config.settings import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Marketing Strategy Agent",
    description="AI-powered marketing workflow automation system",
    version="1.0.0"
)

# Global instances
workflow_manager = None
company_database = None
company_selector = None
report_generator = None
active_connections: Dict[str, WebSocket] = {}
config = load_config()
completed_workflows: Dict[str, ComprehensiveMarketingPackage] = {}

# Static files and templates
app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")


class CompanyRequest(BaseModel):
    name: str
    industry: str
    size: Optional[str] = "medium"
    location: Optional[str] = "Global"
    description: Optional[str] = ""
    website: Optional[str] = ""


class WorkflowRequest(BaseModel):
    company_info: CompanyRequest
    execution_mode: Optional[str] = "hybrid"


@app.on_event("startup")
async def startup_event():
    """Initialize application components on startup"""
    global workflow_manager, company_database, company_selector, report_generator
    
    try:
        # Initialize workflow manager
        workflow_manager = WorkflowManager(config)
        
        # Initialize company database
        company_database = CompanyDatabase()
        company_selector = CompanySelector(company_database)
        
        # Initialize report generator
        report_generator = ReportGenerator()
        
        logger.info("Marketing Strategy Agent API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Marketing Strategy Agent Dashboard"
    })


@app.get("/api/companies")
async def get_companies():
    """Get all available companies"""
    try:
        companies = company_database.list_companies()
        return {"companies": companies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/companies/{company_name}")
async def get_company(company_name: str):
    """Get specific company information"""
    try:
        company = company_database.get_company(company_name)
        if not company:
            raise HTTPException(status_code=404, detail="Company not found")
        return {"company": company}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/companies")
async def add_company(company_request: CompanyRequest):
    """Add a new company to the database"""
    try:
        company_info = CompanyInfo(
            name=company_request.name,
            industry=company_request.industry,
            size=company_request.size,
            location=company_request.location,
            description=company_request.description,
            website=company_request.website
        )
        
        success = company_database.add_company(company_info)
        if success:
            return {"message": "Company added successfully", "company": company_info.dict()}
        else:
            raise HTTPException(status_code=400, detail="Company already exists")
            
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflows")
async def create_workflow(workflow_request: WorkflowRequest, background_tasks: BackgroundTasks):
    """Create and start a new marketing workflow"""
    try:
        # Convert request to CompanyInfo
        company_info = CompanyInfo(
            name=workflow_request.company_info.name,
            industry=workflow_request.company_info.industry,
            size=workflow_request.company_info.size,
            location=workflow_request.company_info.location,
            description=workflow_request.company_info.description,
            website=workflow_request.company_info.website
        )
        
        # Create workflow
        workflow_id = await workflow_manager.create_workflow(company_info)
        
        # Start workflow execution in background
        background_tasks.add_task(
            execute_workflow_background,
            workflow_id,
            company_info,
            workflow_request.execution_mode
        )
        
        return {
            "workflow_id": workflow_id,
            "status": "created",
            "message": "Workflow created and started"
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def execute_workflow_background(workflow_id: str, company_info: CompanyInfo, execution_mode: str):
    """Execute workflow in background and broadcast updates"""
    try:
        # Add progress callback for websocket updates
        workflow = workflow_manager.active_workflows.get(workflow_id)
        if workflow:
            workflow.add_progress_callback(
                lambda status: broadcast_workflow_update(workflow_id, status)
            )
        
        # Execute workflow
        result = await workflow_manager.execute_workflow(
            workflow_id, company_info, execution_mode
        )
        
        # Store completed workflow for report generation
        completed_workflows[workflow_id] = result
        
        # Broadcast completion
        await broadcast_workflow_update(workflow_id, {
            "status": "completed",
            "message": "Workflow completed successfully",
            "result_summary": {
                "company": result.company_info.name,
                "agents_executed": len(result.workflow_metadata.get("agents_executed", [])),
                "execution_time": result.workflow_metadata.get("execution_time_seconds", 0)
            }
        })
        
    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {e}")
        await broadcast_workflow_update(workflow_id, {
            "status": "failed",
            "message": f"Workflow failed: {str(e)}"
        })


@app.get("/api/workflows")
async def get_workflows():
    """Get all workflows (active and history)"""
    try:
        active_workflows = workflow_manager.get_active_workflows()
        workflow_history = workflow_manager.get_workflow_history()
        
        return {
            "active_workflows": active_workflows,
            "workflow_history": workflow_history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workflows/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get status of a specific workflow"""
    try:
        status = await workflow_manager.get_workflow_status(workflow_id)
        return {"status": status.dict()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/workflows/{workflow_id}")
async def cancel_workflow(workflow_id: str):
    """Cancel a specific workflow"""
    try:
        await workflow_manager.cancel_workflow(workflow_id)
        return {"message": "Workflow cancelled successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{workflow_id}")
async def websocket_endpoint(websocket: WebSocket, workflow_id: str):
    """WebSocket endpoint for real-time workflow updates"""
    await websocket.accept()
    active_connections[workflow_id] = websocket
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            
            # Handle ping/pong for connection health
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        if workflow_id in active_connections:
            del active_connections[workflow_id]
    except Exception as e:
        logger.error(f"WebSocket error for workflow {workflow_id}: {e}")
        if workflow_id in active_connections:
            del active_connections[workflow_id]


async def broadcast_workflow_update(workflow_id: str, update_data):
    """Broadcast workflow updates to connected websockets"""
    if workflow_id in active_connections:
        try:
            websocket = active_connections[workflow_id]
            await websocket.send_text(json.dumps({
                "type": "workflow_update",
                "workflow_id": workflow_id,
                "data": update_data,
                "timestamp": datetime.utcnow().isoformat()
            }))
        except Exception as e:
            logger.error(f"Failed to broadcast update for workflow {workflow_id}: {e}")
            # Remove failed connection
            if workflow_id in active_connections:
                del active_connections[workflow_id]


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "active_workflows": len(workflow_manager.get_active_workflows()) if workflow_manager else 0,
        "active_connections": len(active_connections)
    }


@app.get("/api/agent-capabilities")
async def get_agent_capabilities():
    """Get capabilities of all available agents"""
    try:
        if not workflow_manager:
            raise HTTPException(status_code=503, detail="Workflow manager not initialized")
        
        # Create a temporary workflow to get capabilities
        temp_workflow = workflow_manager.active_workflows.values()
        if temp_workflow:
            capabilities = await list(temp_workflow)[0].get_agent_capabilities()
            return {"capabilities": capabilities}
        else:
            return {"capabilities": {
                "BrandAnalyzer": {"description": "Brand positioning and competitive analysis"},
                "TrendResearcher": {"description": "Market trends and industry insights"},
                "ContentCreator": {"description": "Content strategy and social media planning"},
                "MarketingAgent": {"description": "Comprehensive marketing strategy synthesis"},
                "GeminiVisualGenerator": {"description": "AI-powered visual content generation"}
            }}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ReportRequest(BaseModel):
    workflow_id: str
    format: Optional[str] = "pdf"
    sections: Optional[List[str]] = None


@app.post("/api/reports/generate")
async def generate_report(report_request: ReportRequest):
    """Generate a report for a completed workflow"""
    try:
        # Check if workflow is completed and result is available
        if report_request.workflow_id not in completed_workflows:
            raise HTTPException(status_code=404, detail="Workflow not found or not completed")
        
        # Get the marketing package
        marketing_package = completed_workflows[report_request.workflow_id]
        
        # Generate report
        report_info = await report_generator.generate_comprehensive_report(
            marketing_package,
            report_request.format,
            report_request.sections
        )
        
        return {
            "message": "Report generated successfully",
            "report": report_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reports")
async def list_reports():
    """List all generated reports"""
    try:
        reports = await report_generator.list_reports()
        return {"reports": reports}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reports/download/{file_name}")
async def download_report(file_name: str):
    """Download a specific report file"""
    try:
        # Get reports list to verify file exists
        reports = await report_generator.list_reports()
        report_file = next((r for r in reports if r["file_name"] == file_name), None)
        
        if not report_file:
            raise HTTPException(status_code=404, detail="Report file not found")
        
        file_path = report_file["file_path"]
        
        # Determine media type based on file extension
        media_types = {
            ".pdf": "application/pdf",
            ".html": "text/html",
            ".json": "application/json",
            ".md": "text/markdown"
        }
        
        extension = os.path.splitext(file_name)[1].lower()
        media_type = media_types.get(extension, "application/octet-stream")
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=file_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/reports/{file_name}")
async def delete_report(file_name: str):
    """Delete a specific report file"""
    try:
        success = await report_generator.delete_report(file_name)
        
        if success:
            return {"message": "Report deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Report file not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflows/complete")
async def create_complete_workflow(workflow_request: WorkflowRequest, background_tasks: BackgroundTasks):
    """Create and execute complete marketing analysis workflow with all agents and PDF generation"""
    try:
        # Import complete analyzer
        from marketing_analyzer import CompleteMarketingAnalyzer
        
        # Convert request to CompanyInfo
        company_info = CompanyInfo(
            name=workflow_request.company_info.name,
            industry=workflow_request.company_info.industry,
            size=workflow_request.company_info.size,
            location=workflow_request.company_info.location or "Global",
            description=workflow_request.company_info.description or f"A {workflow_request.company_info.industry} company",
            website=workflow_request.company_info.website or ""
        )
        
        # Create unique workflow ID
        workflow_id = f"complete_{company_info.name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Execute complete analysis in background
        background_tasks.add_task(
            execute_complete_analysis_background,
            workflow_id,
            company_info
        )
        
        return {
            "workflow_id": workflow_id,
            "status": "created",
            "message": "Complete marketing analysis started with all agents + PDF generation",
            "estimated_time": "5-15 minutes",
            "agents": ["BrandAnalyzer", "TrendResearcher", "ContentCreator", "MarketingAgent", "GeminiVisualGenerator", "PDFGeneratorAgent"]
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating complete workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def execute_complete_analysis_background(workflow_id: str, company_info: CompanyInfo):
    """Execute complete marketing analysis in background"""
    try:
        # Broadcast start
        await broadcast_workflow_update(workflow_id, {
            "status": "running",
            "message": "Starting comprehensive marketing analysis",
            "progress": {"progress_percentage": 0, "current_agent": "Initializing"}
        })
        
        # Import and initialize complete analyzer  
        from marketing_analyzer import CompleteMarketingAnalyzer
        analyzer = CompleteMarketingAnalyzer()
        
        # Run all agents
        agent_results = await analyzer.run_all_agents(company_info)
        
        # Broadcast progress
        await broadcast_workflow_update(workflow_id, {
            "status": "running", 
            "message": "Generating PDF and text reports",
            "progress": {"progress_percentage": 90, "current_agent": "PDFGeneratorAgent"}
        })
        
        # Generate reports
        report_info = await analyzer.generate_reports(company_info, agent_results)
        
        # Store results
        successful_agents = sum(1 for result in agent_results.values() if result["success"])
        total_agents = len(agent_results)
        
        # Broadcast completion
        await broadcast_workflow_update(workflow_id, {
            "status": "completed",
            "message": "Complete marketing analysis finished successfully",
            "progress": {"progress_percentage": 100, "current_agent": "Complete"},
            "result_summary": {
                "company": company_info.name,
                "agents_executed": total_agents,
                "successful_agents": successful_agents,
                "success_rate": f"{(successful_agents/total_agents*100):.0f}%",
                "pdf_report": report_info.get("pdf_report_path") if report_info else None,
                "text_report": report_info.get("text_report_path") if report_info else None
            }
        })
        
    except Exception as e:
        logger.error(f"Complete workflow {workflow_id} failed: {e}")
        await broadcast_workflow_update(workflow_id, {
            "status": "failed",
            "message": f"Complete analysis failed: {str(e)}",
            "error": str(e)
        })


@app.post("/api/workflows/quick")
async def create_quick_workflow(workflow_request: WorkflowRequest, background_tasks: BackgroundTasks):
    """Create and execute quick marketing analysis workflow (2-4 minutes)"""
    try:
        # Convert request to CompanyInfo
        company_info = CompanyInfo(
            name=workflow_request.company_info.name,
            industry=workflow_request.company_info.industry,
            size=workflow_request.company_info.size,
            location=workflow_request.company_info.location or "Global",
            description=workflow_request.company_info.description or f"A {workflow_request.company_info.industry} company",
            website=workflow_request.company_info.website or ""
        )
        
        # Create unique workflow ID
        workflow_id = f"quick_{company_info.name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Execute quick analysis in background
        background_tasks.add_task(
            execute_quick_analysis_background,
            workflow_id,
            company_info
        )
        
        return {
            "analysis_id": workflow_id,
            "workflow_id": workflow_id,
            "status": "created",
            "message": "Quick marketing analysis started - core insights only",
            "estimated_time": "2-4 minutes",
            "agents": ["BrandAnalyzer", "ContentCreator", "MarketingAgent"],
            "features": ["Core brand analysis", "Essential content strategy", "Key marketing recommendations"]
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating quick workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflows/complete")
async def create_complete_workflow(workflow_request: WorkflowRequest, background_tasks: BackgroundTasks):
    """Create and execute complete marketing analysis workflow (5-15 minutes)"""
    try:
        # Convert request to CompanyInfo
        company_info = CompanyInfo(
            name=workflow_request.company_info.name,
            industry=workflow_request.company_info.industry,
            size=workflow_request.company_info.size,
            location=workflow_request.company_info.location or "Global",
            description=workflow_request.company_info.description or f"A {workflow_request.company_info.industry} company",
            website=workflow_request.company_info.website or ""
        )
        
        # Create unique workflow ID  
        workflow_id = f"complete_{company_info.name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Execute complete analysis in background
        background_tasks.add_task(
            execute_complete_analysis_background,
            workflow_id,
            company_info
        )
        
        return {
            "analysis_id": workflow_id,
            "workflow_id": workflow_id,
            "status": "created",
            "message": "Complete marketing analysis started - comprehensive insights",
            "estimated_time": "5-15 minutes",
            "agents": ["BrandAnalyzer", "TrendResearcher", "ContentCreator", "MarketingAgent", "GeminiVisualGenerator"],
            "features": ["Complete brand analysis", "Trend research", "Comprehensive content strategy", "Visual content generation", "Complete marketing strategy"]
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating complete workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def execute_quick_analysis_background(workflow_id: str, company_info: CompanyInfo):
    """Execute quick marketing analysis in background"""
    try:
        # Broadcast start
        await broadcast_workflow_update(workflow_id, {
            "status": "running",
            "message": "Starting quick marketing analysis",
            "progress": {"progress_percentage": 0, "current_agent": "Initializing"}
        })
        
        # Initialize quick workflow
        quick_workflow = QuickMarketingWorkflow(config)
        
        # Add progress callback for websocket updates
        quick_workflow.add_progress_callback(
            lambda progress: broadcast_workflow_update(workflow_id, {
                "status": "running",
                "message": progress.current_activity,
                "progress": {
                    "progress_percentage": progress.progress_percentage,
                    "current_agent": progress.stage
                }
            })
        )
        
        # Run quick analysis
        result = await quick_workflow.execute(company_info)
        
        # Store results for quick access
        completed_workflows[workflow_id] = result
        
        # Broadcast completion
        await broadcast_workflow_update(workflow_id, {
            "status": "completed",
            "message": "Quick marketing analysis completed successfully",
            "progress": {"progress_percentage": 100, "current_agent": "Complete"},
            "result_summary": {
                "company": company_info.name,
                "workflow_type": "quick_analysis",
                "agents_executed": len(result.workflow_metadata.get("agents_executed", [])),
                "execution_time": f"{result.workflow_metadata.get('execution_time_seconds', 0):.1f}s",
                "success_rate": result.workflow_metadata.get("success_rate", "100%")
            }
        })
        
    except Exception as e:
        logger.error(f"Quick workflow {workflow_id} failed: {e}")
        await broadcast_workflow_update(workflow_id, {
            "status": "failed",
            "message": f"Quick analysis failed: {str(e)}",
            "error": str(e)
        })


@app.get("/api/workflows/{workflow_id}/report")
async def get_workflow_report_status(workflow_id: str):
    """Check if a workflow has completed and is ready for report generation"""
    try:
        # Check if workflow is in completed workflows
        has_result = workflow_id in completed_workflows
        
        # Check workflow status
        workflow_status = None
        if workflow_id in workflow_manager.active_workflows:
            status = await workflow_manager.get_workflow_status(workflow_id)
            workflow_status = status.status
        
        return {
            "workflow_id": workflow_id,
            "has_result": has_result,
            "status": workflow_status,
            "ready_for_report": has_result and workflow_status == "completed"
        }
        
    except Exception as e:
        logger.error(f"Error checking report status for workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Load configuration
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )