"""
Base agent interface and abstract classes for the marketing workflow system
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import time
import asyncio
from datetime import datetime

from ..models.data_models import (
    CompanyInfo, 
    AgentResponse,
    WorkflowProgress
)


class BaseAgent(ABC):
    """Abstract base class for all marketing workflow agents"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.execution_start_time: Optional[float] = None
        self.execution_end_time: Optional[float] = None
        
    @abstractmethod
    async def execute(self, company_info: CompanyInfo, **kwargs) -> AgentResponse:
        """
        Execute the agent's main functionality
        
        Args:
            company_info: Company information and context
            **kwargs: Additional parameters specific to each agent
            
        Returns:
            AgentResponse with results or error information
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, str]:
        """
        Return a description of the agent's capabilities
        
        Returns:
            Dictionary describing agent capabilities
        """
        pass
    
    async def _execute_with_tracking(self, company_info: CompanyInfo, **kwargs) -> AgentResponse:
        """
        Wrapper method that tracks execution time and handles errors
        """
        self.execution_start_time = time.time()
        
        try:
            # Execute the agent's main logic with timeout
            result = await asyncio.wait_for(
                self.execute(company_info, **kwargs), 
                timeout=300  # 5 minute timeout
            )
            
            self.execution_end_time = time.time()
            execution_time = self.execution_end_time - self.execution_start_time
            
            # If execute returns AgentResponse, update it with timing
            if isinstance(result, AgentResponse):
                result.execution_time = execution_time
                return result
            
            # Otherwise, wrap the result
            return AgentResponse(
                agent_name=self.name,
                execution_time=execution_time,
                success=True,
                result=result.dict() if hasattr(result, 'dict') else result,
                metadata={
                    "started_at": datetime.fromtimestamp(self.execution_start_time),
                    "completed_at": datetime.fromtimestamp(self.execution_end_time),
                    "config_used": self.config
                }
            )
            
        except asyncio.TimeoutError:
            self.execution_end_time = time.time()
            execution_time = self.execution_end_time - self.execution_start_time
            
            return AgentResponse(
                agent_name=self.name,
                execution_time=execution_time,
                success=False,
                error_message=f"Agent execution timed out after 300 seconds",
                metadata={
                    "started_at": datetime.fromtimestamp(self.execution_start_time),
                    "failed_at": datetime.fromtimestamp(self.execution_end_time),
                    "config_used": self.config,
                    "error_type": "TimeoutError"
                }
            )
        except Exception as e:
            self.execution_end_time = time.time()
            execution_time = self.execution_end_time - self.execution_start_time
            
            return AgentResponse(
                agent_name=self.name,
                execution_time=execution_time,
                success=False,
                error_message=str(e),
                metadata={
                    "started_at": datetime.fromtimestamp(self.execution_start_time),
                    "failed_at": datetime.fromtimestamp(self.execution_end_time),
                    "config_used": self.config,
                    "error_type": type(e).__name__
                }
            )
    
    def get_progress_update(self, stage: str, progress: float, activity: str) -> WorkflowProgress:
        """
        Generate a progress update for the workflow tracker
        """
        return WorkflowProgress(
            stage=stage,
            progress_percentage=progress,
            current_activity=activity,
            estimated_completion=None  # Can be calculated based on remaining work
        )


class AgentOrchestrator:
    """
    Orchestrates the execution of multiple agents in the marketing workflow
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_order: List[str] = []
        self.progress_callbacks: List[callable] = []
        
    def register_agent(self, agent: BaseAgent, execution_order: Optional[int] = None):
        """
        Register an agent with the orchestrator
        
        Args:
            agent: The agent instance to register
            execution_order: Optional order for sequential execution
        """
        self.agents[agent.name] = agent
        
        if execution_order is not None:
            # Insert agent at specific position in execution order
            if len(self.execution_order) <= execution_order:
                self.execution_order.extend([None] * (execution_order + 1 - len(self.execution_order)))
            self.execution_order[execution_order] = agent.name
        else:
            # Append to end of execution order
            self.execution_order.append(agent.name)
    
    def add_progress_callback(self, callback: callable):
        """Add a callback function to receive progress updates"""
        self.progress_callbacks.append(callback)
    
    async def _notify_progress(self, progress: WorkflowProgress):
        """Notify all registered callbacks of progress updates"""
        for callback in self.progress_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress)
                else:
                    callback(progress)
            except Exception as e:
                print(f"Progress callback error: {e}")
    
    async def execute_parallel(self, company_info: CompanyInfo, agent_configs: Dict[str, Dict[str, Any]]) -> Dict[str, AgentResponse]:
        """
        Execute multiple agents in parallel
        
        Args:
            company_info: Company information for all agents
            agent_configs: Configuration specific to each agent
            
        Returns:
            Dictionary of agent responses keyed by agent name
        """
        tasks = []
        agent_names = []
        
        for agent_name, agent in self.agents.items():
            config = agent_configs.get(agent_name, {})
            task = agent._execute_with_tracking(company_info, **config)
            tasks.append(task)
            agent_names.append(agent_name)
            
            # Notify progress - agent started
            await self._notify_progress(
                WorkflowProgress(
                    stage=f"{agent_name}_started",
                    progress_percentage=0,
                    current_activity=f"Starting {agent_name} execution"
                )
            )
        
        # Execute all agents in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        agent_responses = {}
        for agent_name, result in zip(agent_names, results):
            if isinstance(result, Exception):
                agent_responses[agent_name] = AgentResponse(
                    agent_name=agent_name,
                    execution_time=0,
                    success=False,
                    error_message=str(result)
                )
            else:
                agent_responses[agent_name] = result
                
            # Notify progress - agent completed
            await self._notify_progress(
                WorkflowProgress(
                    stage=f"{agent_name}_completed",
                    progress_percentage=100,
                    current_activity=f"Completed {agent_name} execution"
                )
            )
        
        return agent_responses
    
    async def execute_sequential(self, company_info: CompanyInfo, agent_configs: Dict[str, Dict[str, Any]]) -> Dict[str, AgentResponse]:
        """
        Execute agents in sequential order, allowing for data dependencies
        
        Args:
            company_info: Company information for all agents
            agent_configs: Configuration specific to each agent
            
        Returns:
            Dictionary of agent responses keyed by agent name
        """
        agent_responses = {}
        total_agents = len([name for name in self.execution_order if name is not None])
        
        for idx, agent_name in enumerate(self.execution_order):
            if agent_name is None or agent_name not in self.agents:
                continue
                
            agent = self.agents[agent_name]
            config = agent_configs.get(agent_name, {})
            
            # Add results from previous agents to config
            config['previous_results'] = agent_responses
            
            # Notify progress - agent starting
            progress = (idx / total_agents) * 100
            await self._notify_progress(
                WorkflowProgress(
                    stage=f"{agent_name}_executing",
                    progress_percentage=progress,
                    current_activity=f"Executing {agent_name}"
                )
            )
            
            # Execute agent
            result = await agent._execute_with_tracking(company_info, **config)
            agent_responses[agent_name] = result
            
            # Notify progress - agent completed
            progress = ((idx + 1) / total_agents) * 100
            await self._notify_progress(
                WorkflowProgress(
                    stage=f"{agent_name}_completed",
                    progress_percentage=progress,
                    current_activity=f"Completed {agent_name}"
                )
            )
        
        return agent_responses
    
    async def execute_hybrid(self, company_info: CompanyInfo, agent_configs: Dict[str, Dict[str, Any]]) -> Dict[str, AgentResponse]:
        """
        Execute agents using a hybrid approach:
        1. Run independent agents in parallel (BrandAnalyzer, TrendResearcher)
        2. Run dependent agents sequentially (ContentCreator, MarketingAgent)
        
        Args:
            company_info: Company information for all agents
            agent_configs: Configuration specific to each agent
            
        Returns:
            Dictionary of agent responses keyed by agent name
        """
        agent_responses = {}
        
        # Phase 1: Independent agents in parallel
        independent_agents = ['BrandAnalyzer', 'TrendResearcher']
        phase1_tasks = []
        phase1_names = []
        
        for agent_name in independent_agents:
            if agent_name in self.agents:
                config = agent_configs.get(agent_name, {})
                task = self.agents[agent_name]._execute_with_tracking(company_info, **config)
                phase1_tasks.append(task)
                phase1_names.append(agent_name)
        
        # Notify progress - starting phase 1
        await self._notify_progress(
            WorkflowProgress(
                stage="phase1_started",
                progress_percentage=0,
                current_activity="Starting parallel execution of independent agents"
            )
        )
        
        # Execute phase 1
        if phase1_tasks:
            phase1_results = await asyncio.gather(*phase1_tasks, return_exceptions=True)
            
            for agent_name, result in zip(phase1_names, phase1_results):
                if isinstance(result, Exception):
                    agent_responses[agent_name] = AgentResponse(
                        agent_name=agent_name,
                        execution_time=0,
                        success=False,
                        error_message=str(result)
                    )
                else:
                    agent_responses[agent_name] = result
        
        # Phase 1 complete
        await self._notify_progress(
            WorkflowProgress(
                stage="phase1_completed",
                progress_percentage=40,
                current_activity="Completed independent agents execution"
            )
        )
        
        # Phase 2: Content Creator (depends on Phase 1 results)
        if 'ContentCreator' in self.agents:
            config = agent_configs.get('ContentCreator', {})
            config['previous_results'] = agent_responses
            
            await self._notify_progress(
                WorkflowProgress(
                    stage="content_creator_executing",
                    progress_percentage=50,
                    current_activity="Executing ContentCreator agent"
                )
            )
            
            result = await self.agents['ContentCreator']._execute_with_tracking(company_info, **config)
            agent_responses['ContentCreator'] = result
        
        # Phase 3: Visual Generator (can run in parallel with Phase 4 prep)
        visual_task = None
        if 'GeminiVisualGenerator' in self.agents:
            config = agent_configs.get('GeminiVisualGenerator', {})
            config['previous_results'] = agent_responses
            visual_task = self.agents['GeminiVisualGenerator']._execute_with_tracking(company_info, **config)
        
        # Phase 4: Marketing Agent (depends on all previous results)
        if 'MarketingAgent' in self.agents:
            config = agent_configs.get('MarketingAgent', {})
            config['previous_results'] = agent_responses
            
            await self._notify_progress(
                WorkflowProgress(
                    stage="marketing_agent_executing",
                    progress_percentage=75,
                    current_activity="Executing MarketingAgent for strategy synthesis"
                )
            )
            
            result = await self.agents['MarketingAgent']._execute_with_tracking(company_info, **config)
            agent_responses['MarketingAgent'] = result
        
        # Complete visual generation if it was started
        if visual_task:
            await self._notify_progress(
                WorkflowProgress(
                    stage="visual_generator_executing",
                    progress_percentage=85,
                    current_activity="Completing visual content generation"
                )
            )
            
            result = await visual_task
            agent_responses['GeminiVisualGenerator'] = result
        
        # Workflow complete
        await self._notify_progress(
            WorkflowProgress(
                stage="workflow_completed",
                progress_percentage=100,
                current_activity="Marketing workflow completed successfully"
            )
        )
        
        return agent_responses
    
    def get_agent_capabilities(self) -> Dict[str, Dict[str, str]]:
        """
        Get capabilities of all registered agents
        
        Returns:
            Dictionary of agent capabilities keyed by agent name
        """
        return {
            agent_name: agent.get_capabilities() 
            for agent_name, agent in self.agents.items()
        }