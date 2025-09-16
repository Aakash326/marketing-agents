"""
Quick Marketing Workflow - Fast 2-4 minute analysis
Simplified workflow with core agents for rapid insights
"""
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging

from src.agents.base_agent import AgentOrchestrator
from src.agents.brand_analyzer import BrandAnalyzer
from src.agents.content_creator import ContentCreator
from src.agents.marketing_agent import MarketingAgent
from src.models.data_models import (
    CompanyInfo,
    WorkflowStatus,
    WorkflowProgress,
    ComprehensiveMarketingPackage,
    AgentResponse
)


class QuickMarketingWorkflow:
    """
    Quick Marketing Workflow - Fast 2-4 minute analysis
    
    Runs only core agents with simplified processing for rapid insights:
    1. BrandAnalyzer (simplified)
    2. ContentCreator (basic content only)
    3. MarketingAgent (essential strategy)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflow_id = str(uuid.uuid4())
        self.logger = self._setup_logging()
        
        # Initialize agent orchestrator
        self.orchestrator = AgentOrchestrator()
        
        # Initialize only core agents
        self._initialize_agents()
        
        # Progress tracking
        self.progress_callbacks: List[Callable] = []
        self.current_status = WorkflowStatus(
            workflow_id=self.workflow_id,
            status="initialized",
            progress=WorkflowProgress(
                stage="initialization",
                progress_percentage=0,
                current_activity="Quick workflow initialized"
            )
        )
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the workflow"""
        logger = logging.getLogger(f"quick_workflow_{self.workflow_id}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_agents(self):
        """Initialize core marketing agents for quick analysis"""
        try:
            # Simplified agent configurations - faster processing
            agent_configs = {
                "openai_model": self.config.get("OPENAI_MODEL", "gpt-4o-mini"),
                "temperature": self.config.get("OPENAI_TEMPERATURE", 0.3),  # Lower temperature for speed
                "max_tokens": self.config.get("OPENAI_MAX_TOKENS", 1000),  # Reduced tokens for speed
            }
            
            # Create and register core agents only
            brand_analyzer = BrandAnalyzer(agent_configs)
            content_creator = ContentCreator(agent_configs)
            marketing_agent = MarketingAgent(agent_configs)
            
            self.orchestrator.register_agent(brand_analyzer, 0)
            self.orchestrator.register_agent(content_creator, 1)
            self.orchestrator.register_agent(marketing_agent, 2)
            
            self.logger.info("Quick workflow agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
    
    def add_progress_callback(self, callback: Callable):
        """Add a callback function to receive progress updates"""
        self.progress_callbacks.append(callback)
    
    async def _update_progress(self, progress: WorkflowProgress):
        """Update progress and notify callbacks"""
        self.current_status.progress = progress
        
        for callback in self.progress_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress)
                else:
                    callback(progress)
            except Exception as e:
                self.logger.error(f"Progress callback error: {e}")
    
    async def execute(self, company_info: CompanyInfo, execution_mode: str = "sequential") -> ComprehensiveMarketingPackage:
        """
        Execute quick marketing workflow
        
        Args:
            company_info: Company information
            execution_mode: Always uses sequential for reliability
            
        Returns:
            Simplified ComprehensiveMarketingPackage with core insights
        """
        try:
            self.logger.info(f"Starting quick marketing workflow for {company_info.name}")
            
            # Update workflow status
            self.current_status.status = "running"
            await self._update_progress(WorkflowProgress(
                stage="workflow_start",
                progress_percentage=5,
                current_activity=f"Starting quick analysis for {company_info.name}"
            ))
            
            # Execute core agents sequentially
            agent_results = await self._execute_quick_workflow(company_info)
            
            # Update status with agent results
            self.current_status.agent_results = list(agent_results.values())
            
            # Create simplified marketing package
            marketing_package = await self._create_quick_package(company_info, agent_results)
            
            # Finalize workflow
            self.current_status.status = "completed"
            await self._update_progress(WorkflowProgress(
                stage="workflow_completed",
                progress_percentage=100,
                current_activity="Quick marketing analysis completed"
            ))
            
            self.logger.info("Quick marketing workflow completed successfully")
            return marketing_package
            
        except Exception as e:
            self.logger.error(f"Quick workflow execution failed: {e}")
            self.current_status.status = "failed"
            await self._update_progress(WorkflowProgress(
                stage="workflow_error",
                progress_percentage=0,
                current_activity=f"Quick workflow failed: {str(e)}"
            ))
            raise
    
    async def _execute_quick_workflow(self, company_info: CompanyInfo) -> Dict[str, AgentResponse]:
        """Execute core agents with simplified configurations"""
        self.logger.info("Executing quick workflow with core agents")
        
        # Connect orchestrator's progress callbacks to workflow's callbacks
        self.orchestrator.add_progress_callback(self._update_progress)
        
        # Simplified agent configurations for speed
        agent_configs = {
            "BrandAnalyzer": {
                "quick_mode": True,
                "skip_detailed_analysis": True
            },
            "ContentCreator": {
                "quick_mode": True,
                "limit_content_types": ["post", "caption"],
                "max_posts_per_platform": 1
            },
            "MarketingAgent": {
                "quick_mode": True,
                "focus_areas": ["positioning", "target_audience", "key_strategies"]
            }
        }
        
        return await self.orchestrator.execute_sequential(company_info, agent_configs)
    
    async def _create_quick_package(self, company_info: CompanyInfo,
                                  agent_results: Dict[str, AgentResponse]) -> ComprehensiveMarketingPackage:
        """Create simplified marketing package from core agent results"""
        try:
            self.logger.info("Creating quick marketing package")
            
            # Extract results with fallback handling
            brand_analysis_result = agent_results.get("BrandAnalyzer", {}).get("result", {})
            content_creation_result = agent_results.get("ContentCreator", {}).get("result", {})
            marketing_strategy_result = agent_results.get("MarketingAgent", {}).get("result", {})
            
            # Create simplified workflow metadata
            successful_agents = sum(1 for result in agent_results.values() if result.get("success", False))
            total_execution_time = sum(result.get("execution_time", 0) for result in agent_results.values())
            
            workflow_metadata = {
                "workflow_type": "quick_analysis",
                "workflow_id": self.workflow_id,
                "execution_time_seconds": total_execution_time,
                "completed_at": datetime.utcnow().isoformat(),
                "agents_executed": list(agent_results.keys()),
                "successful_agents": successful_agents,
                "total_agents": len(agent_results),
                "success_rate": f"{(successful_agents/len(agent_results)*100):.0f}%",
                "execution_mode": "quick_sequential",
                "configuration": {
                    "mode": "quick",
                    "agents": ["BrandAnalyzer", "ContentCreator", "MarketingAgent"],
                    "estimated_duration": "2-4 minutes"
                }
            }
            
            # Create simplified package (some fields will be empty/default)
            package = ComprehensiveMarketingPackage(
                company_info=company_info,
                brand_analysis=brand_analysis_result,
                content_creation=content_creation_result,
                trend_research={},  # Not included in quick workflow
                marketing_strategy=marketing_strategy_result,
                visual_content={},  # Not included in quick workflow
                workflow_metadata=workflow_metadata
            )
            
            self.logger.info("Quick marketing package created successfully")
            return package
            
        except Exception as e:
            self.logger.error(f"Error creating quick package: {e}")
            # Create minimal fallback package
            return await self._create_fallback_package(company_info, agent_results)
    
    async def _create_fallback_package(self, company_info: CompanyInfo,
                                     agent_results: Dict[str, AgentResponse]) -> ComprehensiveMarketingPackage:
        """Create minimal fallback package when main package creation fails"""
        workflow_metadata = {
            "workflow_type": "quick_analysis_fallback",
            "workflow_id": self.workflow_id,
            "error": "Main package creation failed, using fallback",
            "completed_at": datetime.utcnow().isoformat(),
            "agents_executed": list(agent_results.keys()),
            "execution_mode": "quick_sequential_fallback"
        }
        
        return ComprehensiveMarketingPackage(
            company_info=company_info,
            brand_analysis={},
            content_creation={},
            trend_research={},
            marketing_strategy={},
            visual_content={},
            workflow_metadata=workflow_metadata
        )
    
    def get_status(self) -> WorkflowStatus:
        """Get current workflow status"""
        return self.current_status
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get workflow information"""
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": "quick_marketing_analysis",
            "estimated_duration": "2-4 minutes",
            "agents": ["BrandAnalyzer", "ContentCreator", "MarketingAgent"],
            "features": [
                "Core brand analysis",
                "Essential content strategy",
                "Key marketing recommendations",
                "Fast execution (2-4 min)",
                "Simplified reporting"
            ]
        }