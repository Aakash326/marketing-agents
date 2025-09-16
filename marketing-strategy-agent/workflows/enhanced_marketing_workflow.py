"""
Enhanced Marketing Automation Workflow System
Main orchestrator for the multi-agent marketing strategy system
"""
import asyncio
import uuid
from typing import Dict, Any, List, Callable
from datetime import datetime, timezone
import logging

from src.agents.base_agent import AgentOrchestrator
from src.agents.brand_analyzer import BrandAnalyzer
from src.agents.trend_researcher import TrendResearcher
from src.agents.content_creator import ContentCreator
from src.agents.marketing_agent import MarketingAgent
from src.agents.visual_generator import GeminiVisualGenerator
from src.models.data_models import (
    CompanyInfo,
    WorkflowStatus,
    WorkflowProgress,
    ComprehensiveMarketingPackage,
    AgentResponse
)


class EnhancedMarketingWorkflow:
    """
    Enhanced Marketing Automation Workflow System
    
    Orchestrates the execution of multiple AI agents to create comprehensive
    marketing strategies, content, and visual assets for companies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflow_id = str(uuid.uuid4())
        self.logger = self._setup_logging()
        
        # Initialize agent orchestrator
        self.orchestrator = AgentOrchestrator()
        
        # Initialize agents
        self._initialize_agents()
        
        # Progress tracking
        self.progress_callbacks: List[Callable] = []
        self.current_status = WorkflowStatus(
            workflow_id=self.workflow_id,
            status="initialized",
            progress=WorkflowProgress(
                stage="initialization",
                progress_percentage=0,
                current_activity="Workflow initialized"
            )
        )
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the workflow"""
        logger = logging.getLogger(f"workflow_{self.workflow_id}")
        logger.setLevel(logging.INFO)
        
        # Create console handler if not exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_agents(self):
        """Initialize all marketing agents"""
        try:
            # Agent configurations
            agent_configs = {
                "openai_model": self.config.get("OPENAI_MODEL", "gpt-4o-mini"),
                "temperature": self.config.get("OPENAI_TEMPERATURE", 0.4),
                "max_tokens": self.config.get("OPENAI_MAX_TOKENS", 2000),
                "gemini_api_key": self.config.get("GEMINI_API_KEY"),
                "gemini_model": self.config.get("GEMINI_MODEL", "gemini-1.5-pro-latest"),
                "tavily_api_key": self.config.get("TAVILY_API_KEY"),
                "tavily_max_results": self.config.get("TAVILY_MAX_RESULTS", 10),
                "vector_config": {
                    "host": self.config.get("TIDB_HOST"),
                    "port": self.config.get("TIDB_PORT"),
                    "user": self.config.get("TIDB_USER"),
                    "password": self.config.get("TIDB_PASSWORD"),
                    "database": self.config.get("TIDB_DATABASE")
                }
            }
            
            # Create and register agents
            brand_analyzer = BrandAnalyzer(agent_configs)
            trend_researcher = TrendResearcher(agent_configs)
            content_creator = ContentCreator(agent_configs)
            marketing_agent = MarketingAgent(agent_configs)
            visual_generator = GeminiVisualGenerator(agent_configs)
            
            # Register agents with orchestrator
            self.orchestrator.register_agent(brand_analyzer, 0)
            self.orchestrator.register_agent(trend_researcher, 1)
            self.orchestrator.register_agent(content_creator, 2)
            self.orchestrator.register_agent(marketing_agent, 3)
            self.orchestrator.register_agent(visual_generator, 4)
            
            # Add progress callback
            self.orchestrator.add_progress_callback(self._update_progress)
            
            self.logger.info("All agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing agents: {e}")
            raise
    
    async def _update_progress(self, progress: WorkflowProgress):
        """Update workflow progress and notify callbacks"""
        self.current_status.progress = progress
        self.current_status.updated_at = datetime.now(timezone.utc)
        
        # Notify all progress callbacks
        for callback in self.progress_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.current_status)
                else:
                    callback(self.current_status)
            except Exception as e:
                self.logger.error(f"Progress callback error: {e}")
    
    def add_progress_callback(self, callback: Callable):
        """Add a progress callback function"""
        self.progress_callbacks.append(callback)
    
    async def execute_workflow(self, company_info: CompanyInfo, 
                              execution_mode: str = "hybrid") -> ComprehensiveMarketingPackage:
        """
        Execute the complete marketing workflow
        
        Args:
            company_info: Company information and context
            execution_mode: "parallel", "sequential", or "hybrid"
            
        Returns:
            ComprehensiveMarketingPackage with all results
        """
        try:
            self.logger.info(f"Starting enhanced marketing workflow for {company_info.name}")
            
            # Update workflow status
            self.current_status.status = "running"
            await self._update_progress(WorkflowProgress(
                stage="workflow_start",
                progress_percentage=5,
                current_activity=f"Starting workflow for {company_info.name}"
            ))
            
            # Execute agents based on mode
            if execution_mode == "parallel":
                agent_results = await self._execute_parallel_workflow(company_info)
            elif execution_mode == "sequential":
                agent_results = await self._execute_sequential_workflow(company_info)
            else:  # hybrid
                agent_results = await self._execute_hybrid_workflow(company_info)
            
            # Update status with agent results
            self.current_status.agent_results = list(agent_results.values())
            
            # Create comprehensive marketing package with fallback handling
            try:
                marketing_package = await self._create_comprehensive_package(
                    company_info, agent_results
                )
            except Exception as e:
                self.logger.error(f"Error creating comprehensive package: {e}")
                # Create a minimal package with available results
                marketing_package = await self._create_fallback_package(
                    company_info, agent_results
                )
            
            # Finalize workflow
            self.current_status.status = "completed"
            await self._update_progress(WorkflowProgress(
                stage="workflow_completed",
                progress_percentage=100,
                current_activity="Marketing workflow completed successfully"
            ))
            
            self.logger.info("Enhanced marketing workflow completed successfully")
            return marketing_package
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            self.current_status.status = "failed"
            await self._update_progress(WorkflowProgress(
                stage="workflow_error",
                progress_percentage=0,
                current_activity=f"Workflow failed: {str(e)}"
            ))
            raise
    
    async def _execute_parallel_workflow(self, company_info: CompanyInfo) -> Dict[str, AgentResponse]:
        """Execute all agents in parallel"""
        self.logger.info("Executing parallel workflow")
        
        agent_configs = {
            "BrandAnalyzer": {},
            "TrendResearcher": {},
            "ContentCreator": {},
            "MarketingAgent": {},
            "GeminiVisualGenerator": {}
        }
        
        return await self.orchestrator.execute_parallel(company_info, agent_configs)
    
    async def _execute_sequential_workflow(self, company_info: CompanyInfo) -> Dict[str, AgentResponse]:
        """Execute agents in sequential order"""
        self.logger.info("Executing sequential workflow")
        
        # Connect orchestrator's progress callbacks to workflow's callbacks
        self.orchestrator.add_progress_callback(self._update_progress)
        
        agent_configs = {
            "BrandAnalyzer": {},
            "TrendResearcher": {},
            "ContentCreator": {},
            "MarketingAgent": {},
            "GeminiVisualGenerator": {}
        }
        
        return await self.orchestrator.execute_sequential(company_info, agent_configs)
    
    async def _execute_hybrid_workflow(self, company_info: CompanyInfo) -> Dict[str, AgentResponse]:
        """Execute agents using hybrid approach (recommended)"""
        self.logger.info("Executing hybrid workflow (optimal performance)")
        
        # Connect orchestrator's progress callbacks to workflow's callbacks
        self.orchestrator.add_progress_callback(self._update_progress)
        
        agent_configs = {
            "BrandAnalyzer": {},
            "TrendResearcher": {},
            "ContentCreator": {},
            "MarketingAgent": {},
            "GeminiVisualGenerator": {}
        }
        
        return await self.orchestrator.execute_hybrid(company_info, agent_configs)
    
    async def _create_comprehensive_package(self, company_info: CompanyInfo,
                                          agent_results: Dict[str, AgentResponse]) -> ComprehensiveMarketingPackage:
        """Create comprehensive marketing package from agent results"""
        try:
            # Extract results from each agent
            brand_analyzer = agent_results.get("BrandAnalyzer")
            brand_analysis_result = brand_analyzer.result if brand_analyzer else None
            
            trend_researcher = agent_results.get("TrendResearcher")
            trend_research_result = trend_researcher.result if trend_researcher else None
            
            content_creator = agent_results.get("ContentCreator")
            content_creation_result = content_creator.result if content_creator else None
            
            marketing_agent = agent_results.get("MarketingAgent")
            marketing_strategy_result = marketing_agent.result if marketing_agent else None
            
            visual_generator = agent_results.get("GeminiVisualGenerator")
            visual_content_result = visual_generator.result if visual_generator else None
            
            # Create workflow metadata
            workflow_metadata = {
                "workflow_id": self.workflow_id,
                "execution_time_seconds": sum([
                    result.execution_time for result in agent_results.values() 
                    if hasattr(result, 'execution_time')
                ]),
                "agents_executed": list(agent_results.keys()),
                "success_count": sum([
                    1 for result in agent_results.values() 
                    if hasattr(result, 'success') and result.success
                ]),
                "total_agents": len(agent_results),
                "execution_mode": "hybrid",
                "configuration": self._get_workflow_configuration()
            }
            
            # Create comprehensive package
            package = ComprehensiveMarketingPackage(
                company_info=company_info,
                brand_analysis=brand_analysis_result,
                content_creation=content_creation_result,
                trend_research=trend_research_result,
                marketing_strategy=marketing_strategy_result,
                visual_content=visual_content_result,
                workflow_metadata=workflow_metadata
            )
            
            self.logger.info("Comprehensive marketing package created successfully")
            return package
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive package: {e}")
            raise
    
    async def _create_fallback_package(self, company_info: CompanyInfo,
                                     agent_results: Dict[str, AgentResponse]) -> ComprehensiveMarketingPackage:
        """Create a minimal marketing package when full package creation fails"""
        try:
            self.logger.info("Creating fallback marketing package with available results")
            
            # Extract what we can, use defaults for missing
            brand_analysis_result = None
            if "BrandAnalyzer" in agent_results and agent_results["BrandAnalyzer"].success:
                brand_analysis_result = agent_results["BrandAnalyzer"].result
            
            content_creation_result = None  # Will be None, which should be handled by the model
            marketing_strategy_result = None  # Will be None, which should be handled by the model
            visual_content_result = None  # Will be None, which should be handled by the model
            trend_research_result = None  # Will be None, which should be handled by the model
            
            # Try to get other successful results
            for agent_name, result in agent_results.items():
                if result.success and result.result:
                    if agent_name == "ContentCreator":
                        content_creation_result = result.result
                    elif agent_name == "MarketingAgent":
                        marketing_strategy_result = result.result
                    elif agent_name == "GeminiVisualGenerator":
                        visual_content_result = result.result
                    elif agent_name == "TrendResearcher":
                        trend_research_result = result.result
            
            # Create workflow metadata
            workflow_metadata = {
                "workflow_id": self.workflow_id,
                "execution_mode": "hybrid",
                "fallback_mode": True,
                "successful_agents": [name for name, result in agent_results.items() if result.success],
                "failed_agents": [name for name, result in agent_results.items() if not result.success],
                "total_agents": len(agent_results)
            }
            
            # Create package with optional fields as None if not available
            package = ComprehensiveMarketingPackage(
                company_info=company_info,
                brand_analysis=brand_analysis_result,
                content_creation=content_creation_result,
                trend_research=trend_research_result,
                marketing_strategy=marketing_strategy_result,
                visual_content=visual_content_result,
                workflow_metadata=workflow_metadata
            )
            
            self.logger.info("Fallback marketing package created successfully")
            return package
            
        except Exception as e:
            self.logger.error(f"Error creating fallback package: {e}")
            # Last resort: create package with minimal data
            package = ComprehensiveMarketingPackage(
                company_info=company_info,
                brand_analysis={"error": "Package creation failed", "fallback": True},
                content_creation=None,
                trend_research=None,
                marketing_strategy=None,
                visual_content=None,
                workflow_metadata={"error": str(e), "fallback_mode": True}
            )
            return package
    
    def _get_workflow_configuration(self) -> Dict[str, Any]:
        """Get workflow configuration summary"""
        return {
            "openai_model": self.config.get("OPENAI_MODEL", "gpt-4o-mini"),
            "gemini_enabled": bool(self.config.get("GEMINI_API_KEY")),
            "tavily_enabled": bool(self.config.get("TAVILY_API_KEY")),
            "vector_store_enabled": bool(self.config.get("TIDB_HOST")),
            "workflow_version": "1.0.0"
        }
    
    async def get_workflow_status(self) -> WorkflowStatus:
        """Get current workflow status"""
        return self.current_status
    
    async def get_agent_capabilities(self) -> Dict[str, Dict[str, str]]:
        """Get capabilities of all agents"""
        return self.orchestrator.get_agent_capabilities()
    
    async def cancel_workflow(self):
        """Cancel the current workflow execution"""
        self.logger.info("Workflow cancellation requested")
        self.current_status.status = "cancelled"
        await self._update_progress(WorkflowProgress(
            stage="workflow_cancelled",
            progress_percentage=0,
            current_activity="Workflow cancelled by user"
        ))


class WorkflowManager:
    """
    Manager for multiple workflow instances
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_workflows: Dict[str, EnhancedMarketingWorkflow] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        
    async def create_workflow(self, company_info: CompanyInfo) -> str:
        """Create a new workflow instance"""
        workflow = EnhancedMarketingWorkflow(self.config)
        workflow_id = workflow.workflow_id
        
        self.active_workflows[workflow_id] = workflow
        
        # Add to history
        self.workflow_history.append({
            "workflow_id": workflow_id,
            "company_name": company_info.name,
            "created_at": datetime.now(timezone.utc),
            "status": "created"
        })
        
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str, company_info: CompanyInfo,
                              execution_mode: str = "hybrid") -> ComprehensiveMarketingPackage:
        """Execute a specific workflow"""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        
        try:
            result = await workflow.execute_workflow(company_info, execution_mode)
            
            # Update history
            for item in self.workflow_history:
                if item["workflow_id"] == workflow_id:
                    item["status"] = "completed"
                    item["completed_at"] = datetime.now(timezone.utc)
                    break
            
            return result
            
        except Exception as e:
            # Update history
            for item in self.workflow_history:
                if item["workflow_id"] == workflow_id:
                    item["status"] = "failed"
                    item["error"] = str(e)
                    item["failed_at"] = datetime.now(timezone.utc)
                    break
            raise
    
    async def get_workflow_status(self, workflow_id: str) -> WorkflowStatus:
        """Get status of a specific workflow"""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        return await self.active_workflows[workflow_id].get_workflow_status()
    
    async def cancel_workflow(self, workflow_id: str):
        """Cancel a specific workflow"""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        await self.active_workflows[workflow_id].cancel_workflow()
        
        # Update history
        for item in self.workflow_history:
            if item["workflow_id"] == workflow_id:
                item["status"] = "cancelled"
                item["cancelled_at"] = datetime.now(timezone.utc)
                break
    
    def cleanup_completed_workflows(self):
        """Remove completed workflows from active list"""
        completed_workflows = []
        
        for workflow_id, workflow in self.active_workflows.items():
            if workflow.current_status.status in ["completed", "failed", "cancelled"]:
                completed_workflows.append(workflow_id)
        
        for workflow_id in completed_workflows:
            del self.active_workflows[workflow_id]
        
        return len(completed_workflows)
    
    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """Get history of all workflows"""
        return self.workflow_history.copy()
    
    def get_active_workflows(self) -> List[str]:
        """Get list of active workflow IDs"""
        return list(self.active_workflows.keys())


# Convenience function for single workflow execution
async def execute_enhanced_marketing_workflow(company_info: CompanyInfo, 
                                            config: Dict[str, Any],
                                            execution_mode: str = "hybrid") -> ComprehensiveMarketingPackage:
    """
    Convenience function to execute a single marketing workflow
    
    Args:
        company_info: Company information
        config: Configuration dictionary
        execution_mode: "parallel", "sequential", or "hybrid"
        
    Returns:
        ComprehensiveMarketingPackage
    """
    workflow = EnhancedMarketingWorkflow(config)
    return await workflow.execute_workflow(company_info, execution_mode)