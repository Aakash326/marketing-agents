"""
LangGraph Marketing Strategy Workflow Implementation.

This module implements the main workflow orchestration for the marketing strategy agent,
coordinating multiple specialized agents through a state-managed graph execution.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Literal
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from src.utils.logging import get_component_logger
from src.utils.exceptions import (
    WorkflowException,
    WorkflowStateException,
    WorkflowTimeoutException,
    handle_exception
)
from src.utils.helpers import timing_context
from src.workflows.workflow_states import (
    MarketingWorkflowState,
    NodeStatus,
    WorkflowStatus,
    create_initial_workflow_state,
    update_node_status,
    validate_workflow_state,
    get_workflow_progress
)
from src.agents.marketing_agent import MarketingAgent
from src.agents.brand_analyzer import BrandAnalyzer
from src.agents.content_creator import ContentCreator
from src.agents.trend_researcher import TrendResearcher


class MarketingWorkflow:
    """
    Main workflow orchestrator for marketing strategy generation.
    
    This class implements a LangGraph-based workflow that coordinates
    multiple specialized agents to create comprehensive marketing strategies.
    """
    
    def __init__(
        self,
        vector_store=None,
        enable_checkpointing: bool = True,
        workflow_timeout: int = 3600  # 1 hour default timeout
    ):
        """Initialize the marketing workflow."""
        self.logger = get_component_logger("marketing_workflow", __name__)
        
        # Initialize agents
        self.brand_analyzer = BrandAnalyzer(vector_store=vector_store)
        self.content_creator = ContentCreator(vector_store=vector_store)
        self.trend_researcher = TrendResearcher(vector_store=vector_store)
        self.marketing_agent = MarketingAgent(
            brand_analyzer=self.brand_analyzer,
            content_creator=self.content_creator,
            trend_researcher=self.trend_researcher,
            vector_store=vector_store
        )
        
        # Workflow configuration
        self.workflow_timeout = workflow_timeout
        self.enable_checkpointing = enable_checkpointing
        
        # Create workflow graph
        self.graph = self._create_workflow_graph()
        
        # Setup checkpointing if enabled
        if enable_checkpointing:
            memory = MemorySaver()
            self.app = self.graph.compile(checkpointer=memory)
        else:
            self.app = self.graph.compile()
        
        self.logger.info("Marketing workflow initialized successfully", extra={
            "enable_checkpointing": enable_checkpointing,
            "timeout_seconds": workflow_timeout
        })
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow definition."""
        # Create graph with workflow state
        workflow = StateGraph(MarketingWorkflowState)
        
        # Add workflow nodes
        workflow.add_node("validate_input", self._validate_input_node)
        workflow.add_node("brand_analysis", self._brand_analysis_node)
        workflow.add_node("trend_research", self._trend_research_node)
        workflow.add_node("content_creation", self._content_creation_node)
        workflow.add_node("strategy_synthesis", self._strategy_synthesis_node)
        workflow.add_node("quality_validation", self._quality_validation_node)
        workflow.add_node("finalize_results", self._finalize_results_node)
        
        # Define workflow edges and routing
        workflow.set_entry_point("validate_input")
        
        # Sequential flow with conditional routing
        workflow.add_edge("validate_input", "brand_analysis")
        workflow.add_edge("brand_analysis", "trend_research")
        workflow.add_edge("trend_research", "content_creation")
        workflow.add_edge("content_creation", "strategy_synthesis")
        workflow.add_edge("strategy_synthesis", "quality_validation")
        
        # Conditional routing from quality validation
        workflow.add_conditional_edges(
            "quality_validation",
            self._should_finalize,
            {
                "finalize": "finalize_results",
                "retry_brand_analysis": "brand_analysis",
                "retry_strategy": "strategy_synthesis"
            }
        )
        
        workflow.add_edge("finalize_results", END)
        
        return workflow
    
    async def execute_workflow(
        self,
        brand_name: str,
        industry: str,
        goals: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the complete marketing strategy workflow.
        
        Args:
            brand_name: Name of the brand to analyze
            industry: Industry context
            goals: Marketing goals and objectives
            **kwargs: Additional workflow parameters
            
        Returns:
            Dictionary containing workflow results
        """
        request_id = kwargs.get("request_id", str(uuid.uuid4()))
        
        # Remove request_id from kwargs to avoid duplicate parameter error
        workflow_kwargs = {k: v for k, v in kwargs.items() if k != "request_id"}
        
        try:
            with timing_context("marketing_workflow_execution"):
                self.logger.info("Starting marketing workflow execution", extra={
                    "request_id": request_id,
                    "brand_name": brand_name,
                    "industry": industry,
                    "goals": goals
                })
                
                # Create initial state
                initial_state = create_initial_workflow_state(
                    request_id=request_id,
                    brand_name=brand_name,
                    industry=industry,
                    goals=goals,
                    **workflow_kwargs
                )
                
                # Validate initial state
                validation_errors = validate_workflow_state(initial_state)
                if validation_errors:
                    raise WorkflowStateException(
                        f"Invalid initial workflow state: {', '.join(validation_errors)}"
                    )
                
                # Execute workflow with timeout
                start_time = time.time()
                
                if self.enable_checkpointing:
                    # Use thread ID for checkpointing
                    thread_id = kwargs.get("thread_id", request_id)
                    config = {
                        "configurable": {"thread_id": thread_id},
                        "recursion_limit": 50  # Increase recursion limit
                    }
                    
                    result = await asyncio.wait_for(
                        self.app.ainvoke(initial_state, config=config),
                        timeout=self.workflow_timeout
                    )
                else:
                    config = {"recursion_limit": 50}  # Increase recursion limit
                    result = await asyncio.wait_for(
                        self.app.ainvoke(initial_state, config=config),
                        timeout=self.workflow_timeout
                    )
                
                processing_time = time.time() - start_time
                
                # Update processing time in result
                if isinstance(result, dict):
                    result["processing_time"] = processing_time
                
                self.logger.info("Marketing workflow completed successfully", extra={
                    "request_id": request_id,
                    "processing_time": processing_time,
                    "workflow_status": result.get("workflow_status") if isinstance(result, dict) else None
                })
                
                return result
                
        except asyncio.TimeoutError:
            self.logger.error(f"Workflow execution timed out after {self.workflow_timeout} seconds")
            raise WorkflowTimeoutException(
                f"Workflow execution timed out after {self.workflow_timeout} seconds",
                timeout_seconds=self.workflow_timeout,
                workflow_name="marketing_strategy"
            )
        except WorkflowException:
            raise
        except Exception as e:
            self.logger.error(f"Error in workflow execution: {e}", exc_info=True)
            raise handle_exception(self.logger, e, "marketing workflow execution")
    
    async def _validate_input_node(self, state: MarketingWorkflowState) -> MarketingWorkflowState:
        """Validate and prepare input data."""
        try:
            self.logger.info("Executing input validation node", extra={
                "request_id": state["request_id"]
            })
            
            # Update node status
            state = update_node_status(state, "validate_input", NodeStatus.RUNNING)
            
            # Validate required fields
            validation_errors = validate_workflow_state(state)
            if validation_errors:
                error_msg = f"Input validation failed: {', '.join(validation_errors)}"
                state = update_node_status(state, "validate_input", NodeStatus.FAILED, error=error_msg)
                raise WorkflowStateException(error_msg)
            
            # Prepare brand data for analysis
            brand_data = {
                "brand_name": state["brand_name"],
                "industry": state["industry"],
                "target_audience": state["target_audience"],
                "description": state.get("additional_context", {}).get("description"),
                "mission": state.get("additional_context", {}).get("mission"),
                "values": state.get("additional_context", {}).get("values"),
                "competitors": state.get("additional_context", {}).get("competitors", [])
            }
            
            # Update brand analysis state
            state["brand_analysis"]["brand_data"] = brand_data
            state["brand_analysis"]["analysis_scope"] = ["positioning", "voice", "competitive", "value_proposition"]
            state["brand_analysis"]["competitive_context"] = brand_data["competitors"]
            
            # Update trend research state
            state["trend_research"]["industry_focus"] = state["industry"]
            state["trend_research"]["trend_categories"] = ["technology", "consumer_behavior", "marketing_channels", "competitive"]
            
            # Update content creation state
            state["content_creation"]["content_types"] = ["social_media", "blog_post", "email_campaign"]
            state["content_creation"]["platforms"] = ["linkedin", "twitter", "facebook", "instagram"]
            state["content_creation"]["campaign_goals"] = state["goals"]
            
            # Complete validation
            state = update_node_status(state, "validate_input", NodeStatus.COMPLETED, 
                                     result={"validation": "passed", "data_prepared": True})
            
            self.logger.info("Input validation completed successfully", extra={
                "request_id": state["request_id"]
            })
            
            return state
            
        except Exception as e:
            error_msg = f"Input validation failed: {str(e)}"
            state = update_node_status(state, "validate_input", NodeStatus.FAILED, error=error_msg)
            self.logger.error(error_msg, exc_info=True)
            raise WorkflowException(error_msg)
    
    async def _brand_analysis_node(self, state: MarketingWorkflowState) -> MarketingWorkflowState:
        """Execute brand analysis using the BrandAnalyzer agent."""
        try:
            self.logger.info("Executing brand analysis node", extra={
                "request_id": state["request_id"],
                "brand_name": state["brand_name"]
            })
            
            # Update node status
            state = update_node_status(state, "brand_analysis", NodeStatus.RUNNING)
            state["brand_analysis"]["status"] = NodeStatus.RUNNING.value
            state["brand_analysis"]["started_at"] = datetime.utcnow().isoformat()
            
            # Execute brand analysis
            analysis_result = await self.brand_analyzer.analyze_brand(
                state["brand_analysis"]["brand_data"]
            )
            
            # Update brand analysis state with results
            state["brand_analysis"]["positioning_analysis"] = analysis_result.get("positioning")
            state["brand_analysis"]["voice_analysis"] = analysis_result.get("voice_and_messaging")
            state["brand_analysis"]["competitive_analysis"] = analysis_result.get("positioning", {}).get("competitive_frame")
            state["brand_analysis"]["value_proposition"] = analysis_result.get("value_proposition")
            
            # Extract insights
            if analysis_result.get("key_insights"):
                state["brand_analysis"]["brand_strengths"] = analysis_result["key_insights"][:5]
            
            if analysis_result.get("strategic_recommendations"):
                state["brand_analysis"]["improvement_areas"] = analysis_result["strategic_recommendations"][:3]
            
            # Calculate quality metrics
            state["brand_analysis"]["analysis_confidence"] = analysis_result.get("overall_strength", 0.0)
            state["brand_analysis"]["completeness_score"] = self._calculate_analysis_completeness(analysis_result)
            state["brand_analysis"]["insights_generated"] = len(analysis_result.get("key_insights", []))
            
            # Complete brand analysis
            state["brand_analysis"]["status"] = NodeStatus.COMPLETED.value
            state["brand_analysis"]["completed_at"] = datetime.utcnow().isoformat()
            
            state = update_node_status(state, "brand_analysis", NodeStatus.COMPLETED, 
                                     result=analysis_result)
            
            self.logger.info("Brand analysis completed successfully", extra={
                "request_id": state["request_id"],
                "confidence_score": state["brand_analysis"]["analysis_confidence"],
                "insights_count": state["brand_analysis"]["insights_generated"]
            })
            
            return state
            
        except Exception as e:
            error_msg = f"Brand analysis failed: {str(e)}"
            state["brand_analysis"]["status"] = NodeStatus.FAILED.value
            state["brand_analysis"]["error_details"] = error_msg
            state = update_node_status(state, "brand_analysis", NodeStatus.FAILED, error=error_msg)
            self.logger.error(error_msg, exc_info=True)
            raise WorkflowException(error_msg)
    
    async def _trend_research_node(self, state: MarketingWorkflowState) -> MarketingWorkflowState:
        """Execute trend research using the TrendResearcher agent."""
        try:
            self.logger.info("Executing trend research node", extra={
                "request_id": state["request_id"],
                "industry": state["industry"]
            })
            
            # Update node status
            state = update_node_status(state, "trend_research", NodeStatus.RUNNING)
            state["trend_research"]["status"] = NodeStatus.RUNNING.value
            state["trend_research"]["research_started_at"] = datetime.utcnow().isoformat()
            
            # Execute trend research
            research_result = await self.trend_researcher.research_trends(
                industry=state["trend_research"]["industry_focus"],
                time_frame=state["trend_research"]["time_frame"]
            )
            
            # Update trend research state with results
            if research_result.get("trends"):
                # Categorize trends by type
                all_trends = research_result["trends"]
                state["trend_research"]["technology_trends"] = [t for t in all_trends if "technology" in t.get("keywords", [])]
                state["trend_research"]["consumer_behavior_trends"] = [t for t in all_trends if "consumer" in str(t).lower()]
                state["trend_research"]["marketing_channel_trends"] = [t for t in all_trends if "marketing" in str(t).lower()]
                state["trend_research"]["competitive_trends"] = [t for t in all_trends if "competitive" in str(t).lower()]
            
            # Extract insights and opportunities
            state["trend_research"]["key_insights"] = research_result.get("key_insights", [])
            state["trend_research"]["opportunities"] = research_result.get("opportunities", [])
            
            # Calculate quality metrics
            state["trend_research"]["trends_identified"] = len(research_result.get("trends", []))
            state["trend_research"]["data_freshness_score"] = 0.9  # Placeholder - would be calculated based on data sources
            state["trend_research"]["relevance_score"] = self._calculate_trend_relevance(research_result, state["industry"])
            
            # Complete trend research
            state["trend_research"]["status"] = NodeStatus.COMPLETED.value
            state["trend_research"]["research_completed_at"] = datetime.utcnow().isoformat()
            
            state = update_node_status(state, "trend_research", NodeStatus.COMPLETED, 
                                     result=research_result)
            
            self.logger.info("Trend research completed successfully", extra={
                "request_id": state["request_id"],
                "trends_found": state["trend_research"]["trends_identified"],
                "opportunities": len(state["trend_research"]["opportunities"])
            })
            
            return state
            
        except Exception as e:
            error_msg = f"Trend research failed: {str(e)}"
            state["trend_research"]["status"] = NodeStatus.FAILED.value
            state["trend_research"]["error_details"] = error_msg
            state = update_node_status(state, "trend_research", NodeStatus.FAILED, error=error_msg)
            self.logger.error(error_msg, exc_info=True)
            raise WorkflowException(error_msg)
    
    async def _content_creation_node(self, state: MarketingWorkflowState) -> MarketingWorkflowState:
        """Execute content creation using the ContentCreator agent."""
        try:
            self.logger.info("Executing content creation node", extra={
                "request_id": state["request_id"],
                "content_types": state["content_creation"]["content_types"]
            })
            
            # Update node status
            state = update_node_status(state, "content_creation", NodeStatus.RUNNING)
            state["content_creation"]["status"] = NodeStatus.RUNNING.value
            state["content_creation"]["creation_started_at"] = datetime.utcnow().isoformat()
            
            # Prepare brand voice guidelines from brand analysis
            brand_voice = {}
            if state["brand_analysis"]["voice_analysis"]:
                brand_voice = {
                    "voice_attributes": state["brand_analysis"]["voice_analysis"].get("voice_attributes", []),
                    "tone": state["brand_analysis"]["voice_analysis"].get("tone_consistency", 0.7),
                    "messaging_themes": state["brand_analysis"]["voice_analysis"].get("messaging_themes", [])
                }
            
            state["content_creation"]["brand_voice_guidelines"] = brand_voice
            
            # Create content for different types and platforms
            content_results = {}
            
            # Generate social media content
            for platform in state["content_creation"]["platforms"]:
                try:
                    social_request = {
                        "content_type": "social_media",
                        "platform": platform,
                        "brand_name": state["brand_name"],
                        "topic": f"{state['brand_name']} marketing strategy",
                        "tone": "professional",
                        "target_audience": state["target_audience"]
                    }
                    
                    social_content = await self.content_creator.create_content(social_request)
                    content_results[f"social_{platform}"] = social_content
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create {platform} content: {e}")
            
            # Generate blog content
            try:
                blog_request = {
                    "content_type": "blog_post",
                    "brand_name": state["brand_name"],
                    "topic": f"Marketing trends for {state['industry']} industry",
                    "target_audience": state["target_audience"],
                    "keywords": state["industry"]
                }
                
                blog_content = await self.content_creator.create_content(blog_request)
                content_results["blog"] = blog_content
                
            except Exception as e:
                self.logger.warning(f"Failed to create blog content: {e}")
            
            # Generate email campaign
            try:
                email_request = {
                    "content_type": "email_campaign",
                    "brand_name": state["brand_name"],
                    "topic": f"New marketing strategy for {state['brand_name']}",
                    "target_audience": state["target_audience"],
                    "goal": "engagement"
                }
                
                email_content = await self.content_creator.create_content(email_request)
                content_results["email"] = email_content
                
            except Exception as e:
                self.logger.warning(f"Failed to create email content: {e}")
            
            # Update content creation state
            state["content_creation"]["social_media_content"] = {
                k: v for k, v in content_results.items() if k.startswith("social_")
            }
            state["content_creation"]["blog_content"] = content_results.get("blog", {})
            state["content_creation"]["email_campaigns"] = content_results.get("email", {})
            
            # Calculate quality metrics
            quality_scores = {}
            for content_type, content in content_results.items():
                if isinstance(content, dict) and "quality_score" in content:
                    quality_scores[content_type] = content["quality_score"]
            
            state["content_creation"]["quality_scores"] = quality_scores
            state["content_creation"]["average_quality_score"] = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0.0
            state["content_creation"]["content_pieces_generated"] = len(content_results)
            state["content_creation"]["platforms_covered"] = len([k for k in content_results.keys() if k.startswith("social_")])
            
            # Complete content creation
            state["content_creation"]["status"] = NodeStatus.COMPLETED.value
            state["content_creation"]["creation_completed_at"] = datetime.utcnow().isoformat()
            
            state = update_node_status(state, "content_creation", NodeStatus.COMPLETED, 
                                     result=content_results)
            
            self.logger.info("Content creation completed successfully", extra={
                "request_id": state["request_id"],
                "content_pieces": state["content_creation"]["content_pieces_generated"],
                "average_quality": state["content_creation"]["average_quality_score"]
            })
            
            return state
            
        except Exception as e:
            error_msg = f"Content creation failed: {str(e)}"
            state["content_creation"]["status"] = NodeStatus.FAILED.value
            state["content_creation"]["error_details"] = error_msg
            state = update_node_status(state, "content_creation", NodeStatus.FAILED, error=error_msg)
            self.logger.error(error_msg, exc_info=True)
            raise WorkflowException(error_msg)
    
    async def _strategy_synthesis_node(self, state: MarketingWorkflowState) -> MarketingWorkflowState:
        """Synthesize comprehensive marketing strategy from all analysis results."""
        try:
            self.logger.info("Executing strategy synthesis node", extra={
                "request_id": state["request_id"]
            })
            
            # Update node status
            state = update_node_status(state, "strategy_synthesis", NodeStatus.RUNNING)
            state["strategy_state"]["status"] = NodeStatus.RUNNING.value
            state["strategy_state"]["synthesis_started_at"] = datetime.utcnow().isoformat()
            
            # Use the marketing agent to synthesize strategy
            strategy_request = {
                "brand_name": state["brand_name"],
                "industry": state["industry"],
                "target_audience": state["target_audience"],
                "goals": state["goals"],
                "budget_range": state["budget_range"],
                "timeline": state["timeline"]
            }
            
            strategy_result = await self.marketing_agent.generate_strategy(strategy_request)
            
            # Update strategy state
            if strategy_result.get("strategy"):
                strategy_data = strategy_result["strategy"]
                state["strategy_state"]["executive_summary"] = strategy_data.get("executive_summary")
                state["strategy_state"]["market_positioning"] = self._extract_market_positioning(state)
                state["strategy_state"]["target_audience_strategy"] = self._extract_audience_strategy(state)
                state["strategy_state"]["marketing_mix"] = strategy_data.get("marketing_mix")
                state["strategy_state"]["channel_strategy"] = self._extract_channel_strategy(state)
                state["strategy_state"]["content_strategy"] = self._extract_content_strategy(state)
            
            # Generate strategic recommendations
            state["strategy_state"]["key_recommendations"] = self._generate_key_recommendations(state)
            state["strategy_state"]["strategic_priorities"] = self._generate_strategic_priorities(state)
            state["strategy_state"]["competitive_advantages"] = self._extract_competitive_advantages(state)
            state["strategy_state"]["growth_opportunities"] = self._extract_growth_opportunities(state)
            
            # Calculate quality scores
            state["strategy_state"]["strategy_confidence"] = self._calculate_strategy_confidence(state)
            state["strategy_state"]["completeness_score"] = self._calculate_strategy_completeness(state)
            state["strategy_state"]["feasibility_score"] = self._assess_strategy_feasibility(state)
            state["strategy_state"]["alignment_score"] = self._assess_goal_alignment(state)
            
            # Complete strategy synthesis
            state["strategy_state"]["status"] = NodeStatus.COMPLETED.value
            state["strategy_state"]["synthesis_completed_at"] = datetime.utcnow().isoformat()
            
            state = update_node_status(state, "strategy_synthesis", NodeStatus.COMPLETED, 
                                     result=strategy_result)
            
            self.logger.info("Strategy synthesis completed successfully", extra={
                "request_id": state["request_id"],
                "confidence_score": state["strategy_state"]["strategy_confidence"],
                "recommendations_count": len(state["strategy_state"]["key_recommendations"])
            })
            
            return state
            
        except Exception as e:
            error_msg = f"Strategy synthesis failed: {str(e)}"
            state["strategy_state"]["status"] = NodeStatus.FAILED.value
            state["strategy_state"]["error_details"] = error_msg
            state = update_node_status(state, "strategy_synthesis", NodeStatus.FAILED, error=error_msg)
            self.logger.error(error_msg, exc_info=True)
            raise WorkflowException(error_msg)
    
    async def _quality_validation_node(self, state: MarketingWorkflowState) -> MarketingWorkflowState:
        """Validate the quality of generated strategy and results."""
        try:
            self.logger.info("Executing quality validation node", extra={
                "request_id": state["request_id"]
            })
            
            # Update node status
            state = update_node_status(state, "quality_validation", NodeStatus.RUNNING)
            
            # Calculate overall quality scores
            quality_scores = {
                "brand_analysis": state["brand_analysis"]["analysis_confidence"],
                "trend_research": state["trend_research"]["relevance_score"],
                "content_creation": state["content_creation"]["average_quality_score"],
                "strategy_synthesis": state["strategy_state"]["strategy_confidence"]
            }
            
            # Calculate overall confidence score
            overall_confidence = sum(quality_scores.values()) / len(quality_scores)
            state["confidence_score"] = round(overall_confidence, 2)
            state["quality_scores"] = quality_scores
            
            # Validation thresholds
            min_confidence_threshold = 0.6
            min_completeness_threshold = 0.7
            
            # Check quality gates
            quality_issues = []
            
            if overall_confidence < min_confidence_threshold:
                quality_issues.append(f"Overall confidence score ({overall_confidence:.2f}) below threshold ({min_confidence_threshold})")
            
            if state["brand_analysis"]["completeness_score"] < min_completeness_threshold:
                quality_issues.append("Brand analysis completeness below threshold")
            
            if state["strategy_state"]["completeness_score"] < min_completeness_threshold:
                quality_issues.append("Strategy completeness below threshold")
            
            if len(state["strategy_state"]["key_recommendations"]) < 3:
                quality_issues.append("Insufficient strategic recommendations generated")
            
            # Determine validation result
            if quality_issues:
                self.logger.warning("Quality validation identified issues", extra={
                    "request_id": state["request_id"],
                    "issues": quality_issues
                })
                
                validation_result = {
                    "passed": False,
                    "issues": quality_issues,
                    "overall_confidence": overall_confidence,
                    "recommendation": "retry_strategy" if overall_confidence > 0.4 else "retry_brand_analysis"
                }
            else:
                validation_result = {
                    "passed": True,
                    "issues": [],
                    "overall_confidence": overall_confidence,
                    "recommendation": "finalize"
                }
            
            # Complete quality validation
            state = update_node_status(state, "quality_validation", NodeStatus.COMPLETED, 
                                     result=validation_result)
            
            self.logger.info("Quality validation completed", extra={
                "request_id": state["request_id"],
                "validation_passed": validation_result["passed"],
                "overall_confidence": overall_confidence
            })
            
            return state
            
        except Exception as e:
            error_msg = f"Quality validation failed: {str(e)}"
            state = update_node_status(state, "quality_validation", NodeStatus.FAILED, error=error_msg)
            self.logger.error(error_msg, exc_info=True)
            raise WorkflowException(error_msg)
    
    async def _finalize_results_node(self, state: MarketingWorkflowState) -> MarketingWorkflowState:
        """Finalize and package the workflow results."""
        try:
            self.logger.info("Executing results finalization node", extra={
                "request_id": state["request_id"]
            })
            
            # Update node status
            state = update_node_status(state, "finalize_results", NodeStatus.RUNNING)
            
            # Package final strategy
            final_strategy = {
                "brand_name": state["brand_name"],
                "industry": state["industry"],
                "executive_summary": state["strategy_state"]["executive_summary"],
                "brand_analysis": {
                    "positioning": state["brand_analysis"]["positioning_analysis"],
                    "voice_analysis": state["brand_analysis"]["voice_analysis"],
                    "value_proposition": state["brand_analysis"]["value_proposition"],
                    "competitive_advantages": state["strategy_state"]["competitive_advantages"]
                },
                "market_insights": {
                    "key_trends": state["trend_research"]["key_insights"],
                    "opportunities": state["trend_research"]["opportunities"],
                    "strategic_priorities": state["strategy_state"]["strategic_priorities"]
                },
                "marketing_strategy": {
                    "target_audience": state["strategy_state"]["target_audience_strategy"],
                    "marketing_mix": state["strategy_state"]["marketing_mix"],
                    "channel_strategy": state["strategy_state"]["channel_strategy"],
                    "content_strategy": state["strategy_state"]["content_strategy"]
                },
                "recommendations": {
                    "key_recommendations": state["strategy_state"]["key_recommendations"],
                    "growth_opportunities": state["strategy_state"]["growth_opportunities"],
                    "next_steps": self._generate_next_steps(state)
                }
            }
            
            # Package content recommendations
            content_recommendations = {
                "social_media": state["content_creation"]["social_media_content"],
                "blog_content": state["content_creation"]["blog_content"],
                "email_campaigns": state["content_creation"]["email_campaigns"],
                "content_calendar": state["content_creation"]["content_calendar"],
                "optimization_suggestions": state["content_creation"]["optimization_suggestions"]
            }
            
            # Create implementation plan
            implementation_plan = {
                "timeline": self._create_implementation_timeline(state),
                "budget_allocation": self._create_budget_allocation(state),
                "success_metrics": self._define_success_metrics(state),
                "risk_mitigation": self._create_risk_mitigation_plan(state)
            }
            
            # Update final outputs
            state["final_strategy"] = final_strategy
            state["content_recommendations"] = content_recommendations
            state["implementation_plan"] = implementation_plan
            
            # Mark workflow as completed
            state["workflow_status"] = WorkflowStatus.COMPLETED.value
            state["completed_at"] = datetime.utcnow().isoformat()
            
            # Complete finalization
            state = update_node_status(state, "finalize_results", NodeStatus.COMPLETED, 
                                     result={"finalization": "success"})
            
            self.logger.info("Results finalization completed successfully", extra={
                "request_id": state["request_id"],
                "final_confidence": state["confidence_score"]
            })
            
            return state
            
        except Exception as e:
            error_msg = f"Results finalization failed: {str(e)}"
            state = update_node_status(state, "finalize_results", NodeStatus.FAILED, error=error_msg)
            self.logger.error(error_msg, exc_info=True)
            raise WorkflowException(error_msg)
    
    def _should_finalize(self, state: MarketingWorkflowState) -> Literal["finalize", "retry_brand_analysis", "retry_strategy"]:
        """Determine the next action based on quality validation results."""
        validation_result = state["node_results"].get("quality_validation", {})
        
        # Check retry counter to prevent infinite loops
        retry_count = state.get("retry_count", 0)
        max_retries = 2  # Maximum number of retries
        
        if validation_result.get("passed", False) or retry_count >= max_retries:
            return "finalize"
        
        # Increment retry counter
        state["retry_count"] = retry_count + 1
        
        # Determine retry strategy based on confidence and specific issues
        recommendation = validation_result.get("recommendation", "finalize")
        
        if recommendation == "retry_brand_analysis" and retry_count < max_retries:
            return "retry_brand_analysis"
        elif recommendation == "retry_strategy" and retry_count < max_retries:
            return "retry_strategy"
        else:
            return "finalize"  # Default to finalize if unclear or max retries reached
    
    # Helper methods for data extraction and calculation
    
    def _calculate_analysis_completeness(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate completeness score for brand analysis."""
        required_sections = ["positioning", "voice_and_messaging", "value_proposition", "key_insights"]
        completed_sections = sum(1 for section in required_sections if analysis_result.get(section))
        return round(completed_sections / len(required_sections), 2)
    
    def _calculate_trend_relevance(self, research_result: Dict[str, Any], industry: str) -> float:
        """Calculate relevance score for trend research."""
        trends = research_result.get("trends", [])
        if not trends:
            return 0.0
        
        # Simple relevance calculation based on industry keyword matches
        relevant_trends = 0
        for trend in trends:
            trend_text = str(trend).lower()
            if industry.lower() in trend_text:
                relevant_trends += 1
        
        return round(relevant_trends / len(trends), 2) if trends else 0.0
    
    def _extract_market_positioning(self, state: MarketingWorkflowState) -> Dict[str, Any]:
        """Extract market positioning from brand analysis."""
        positioning = state["brand_analysis"]["positioning_analysis"]
        if positioning:
            return {
                "positioning_statement": positioning.get("positioning_statement"),
                "target_audience": positioning.get("target_audience"),
                "competitive_frame": positioning.get("competitive_frame"),
                "differentiation": positioning.get("differentiation")
            }
        return {}
    
    def _extract_audience_strategy(self, state: MarketingWorkflowState) -> Dict[str, Any]:
        """Extract target audience strategy."""
        return {
            "primary_audience": state["target_audience"],
            "audience_insights": state["trend_research"]["consumer_behavior_trends"][:3],
            "engagement_preferences": self._extract_engagement_preferences(state),
            "messaging_approach": state["brand_analysis"]["voice_analysis"]
        }
    
    def _extract_engagement_preferences(self, state: MarketingWorkflowState) -> List[str]:
        """Extract audience engagement preferences from research."""
        # Simplified extraction - in production would be more sophisticated
        return [
            "Social media engagement",
            "Educational content consumption", 
            "Mobile-first interactions",
            "Personalized experiences"
        ]
    
    def _extract_channel_strategy(self, state: MarketingWorkflowState) -> Dict[str, Any]:
        """Extract marketing channel strategy."""
        return {
            "primary_channels": state["content_creation"]["platforms"],
            "channel_priorities": self._rank_channels_by_effectiveness(state),
            "content_distribution": state["content_creation"]["social_media_content"],
            "optimization_recommendations": state["content_creation"]["optimization_suggestions"]
        }
    
    def _rank_channels_by_effectiveness(self, state: MarketingWorkflowState) -> List[Dict[str, Any]]:
        """Rank marketing channels by predicted effectiveness."""
        channels = state["content_creation"]["platforms"]
        
        # Simplified ranking based on industry and audience
        channel_rankings = []
        for channel in channels:
            effectiveness_score = 0.7  # Default score
            
            # Adjust based on industry
            if state["industry"].lower() in ["technology", "software"] and channel == "linkedin":
                effectiveness_score = 0.9
            elif channel == "instagram" and state["target_audience"] and "young" in str(state["target_audience"]).lower():
                effectiveness_score = 0.85
            
            channel_rankings.append({
                "channel": channel,
                "effectiveness_score": effectiveness_score,
                "recommended_content_types": ["educational", "promotional"],
                "posting_frequency": "daily"
            })
        
        return sorted(channel_rankings, key=lambda x: x["effectiveness_score"], reverse=True)
    
    def _extract_content_strategy(self, state: MarketingWorkflowState) -> Dict[str, Any]:
        """Extract content strategy from creation results."""
        return {
            "content_themes": state["brand_analysis"]["voice_analysis"].get("messaging_themes", []),
            "content_calendar": state["content_creation"]["content_calendar"],
            "quality_standards": {
                "minimum_quality_score": 0.7,
                "brand_alignment_threshold": 0.8,
                "engagement_target": 0.05
            },
            "content_optimization": state["content_creation"]["optimization_suggestions"]
        }
    
    def _generate_key_recommendations(self, state: MarketingWorkflowState) -> List[str]:
        """Generate key strategic recommendations."""
        recommendations = []
        
        # Brand-based recommendations
        if state["brand_analysis"]["analysis_confidence"] < 0.7:
            recommendations.append("Strengthen brand positioning and messaging clarity")
        
        # Trend-based recommendations
        if state["trend_research"]["opportunities"]:
            top_opportunity = state["trend_research"]["opportunities"][0]
            if isinstance(top_opportunity, dict):
                recommendations.append(f"Capitalize on trend opportunity: {top_opportunity.get('description', 'market opportunity')}")
        
        # Content-based recommendations
        if state["content_creation"]["average_quality_score"] < 0.8:
            recommendations.append("Improve content quality and brand voice consistency")
        
        # Generic strategic recommendations
        recommendations.extend([
            "Implement data-driven marketing measurement framework",
            "Develop integrated multi-channel campaign approach",
            "Focus on customer experience optimization across touchpoints"
        ])
        
        return recommendations[:6]  # Limit to 6 key recommendations
    
    def _generate_strategic_priorities(self, state: MarketingWorkflowState) -> List[Dict[str, Any]]:
        """Generate strategic priorities with timelines."""
        priorities = [
            {
                "priority": "Brand positioning optimization",
                "timeline": "30 days",
                "impact": "high",
                "effort": "medium",
                "description": "Refine brand messaging and positioning based on analysis"
            },
            {
                "priority": "Content strategy implementation",
                "timeline": "60 days", 
                "impact": "high",
                "effort": "high",
                "description": "Launch multi-channel content marketing program"
            },
            {
                "priority": "Performance measurement setup",
                "timeline": "45 days",
                "impact": "medium",
                "effort": "low",
                "description": "Implement tracking and analytics framework"
            }
        ]
        
        return priorities
    
    def _extract_competitive_advantages(self, state: MarketingWorkflowState) -> List[str]:
        """Extract competitive advantages from analysis."""
        advantages = []
        
        # From brand analysis
        if state["brand_analysis"]["brand_strengths"]:
            advantages.extend(state["brand_analysis"]["brand_strengths"][:3])
        
        # From trend research
        if state["trend_research"]["opportunities"]:
            advantages.append("Early identification of market trends and opportunities")
        
        # From differentiation analysis
        positioning = state["brand_analysis"]["positioning_analysis"]
        if positioning and positioning.get("differentiation"):
            advantages.extend(positioning["differentiation"][:2] if isinstance(positioning["differentiation"], list) else [positioning["differentiation"]])
        
        return list(set(advantages))[:5]  # Remove duplicates and limit to 5
    
    def _extract_growth_opportunities(self, state: MarketingWorkflowState) -> List[str]:
        """Extract growth opportunities from research."""
        opportunities = []
        
        # From trend research
        if state["trend_research"]["opportunities"]:
            for opp in state["trend_research"]["opportunities"][:3]:
                if isinstance(opp, dict):
                    opportunities.append(opp.get("description", str(opp)))
                else:
                    opportunities.append(str(opp))
        
        # From market analysis
        opportunities.extend([
            "Content marketing expansion across new platforms",
            "Partnership development with industry leaders",
            "Customer experience enhancement initiatives"
        ])
        
        return opportunities[:5]
    
    def _calculate_strategy_confidence(self, state: MarketingWorkflowState) -> float:
        """Calculate overall strategy confidence score."""
        component_scores = [
            state["brand_analysis"]["analysis_confidence"],
            state["trend_research"]["relevance_score"],
            state["content_creation"]["average_quality_score"]
        ]
        
        # Weight components
        weights = [0.4, 0.3, 0.3]  # Brand analysis weighted higher
        weighted_score = sum(score * weight for score, weight in zip(component_scores, weights))
        
        return round(weighted_score, 2)
    
    def _calculate_strategy_completeness(self, state: MarketingWorkflowState) -> float:
        """Calculate strategy completeness score."""
        required_components = [
            "executive_summary",
            "market_positioning", 
            "target_audience_strategy",
            "marketing_mix",
            "channel_strategy",
            "content_strategy"
        ]
        
        completed_components = sum(
            1 for component in required_components 
            if state["strategy_state"].get(component)
        )
        
        return round(completed_components / len(required_components), 2)
    
    def _assess_strategy_feasibility(self, state: MarketingWorkflowState) -> float:
        """Assess strategy feasibility based on resources and constraints."""
        # Simplified feasibility assessment
        feasibility_score = 0.8  # Base score
        
        # Adjust based on content complexity
        if state["content_creation"]["content_pieces_generated"] > 20:
            feasibility_score -= 0.1  # More complex to execute
        
        # Adjust based on channel count
        if len(state["content_creation"]["platforms"]) > 5:
            feasibility_score -= 0.1  # More channels = more resources needed
        
        return max(0.0, round(feasibility_score, 2))
    
    def _assess_goal_alignment(self, state: MarketingWorkflowState) -> float:
        """Assess how well strategy aligns with stated goals."""
        goals = state["goals"]
        alignment_score = 0.8  # Base alignment score
        
        # Check if strategy addresses common goals
        strategy_text = str(state["strategy_state"]).lower()
        
        goal_keywords = {
            "awareness": ["brand", "visibility", "recognition"],
            "engagement": ["engagement", "interaction", "community"],
            "leads": ["lead", "conversion", "acquisition"],
            "sales": ["sales", "revenue", "purchase"]
        }
        
        goal_matches = 0
        for goal in goals:
            goal_lower = goal.lower()
            for goal_type, keywords in goal_keywords.items():
                if any(keyword in goal_lower for keyword in keywords):
                    if any(keyword in strategy_text for keyword in keywords):
                        goal_matches += 1
                    break
        
        if goals:
            alignment_score = goal_matches / len(goals)
        
        return round(alignment_score, 2)
    
    def _generate_next_steps(self, state: MarketingWorkflowState) -> List[Dict[str, str]]:
        """Generate specific next steps for implementation."""
        return [
            {
                "step": "Review and approve marketing strategy",
                "timeline": "1 week",
                "owner": "Marketing leadership",
                "priority": "high"
            },
            {
                "step": "Set up tracking and measurement framework",
                "timeline": "2 weeks",
                "owner": "Analytics team",
                "priority": "high"
            },
            {
                "step": "Begin content creation and approval process",
                "timeline": "2 weeks",
                "owner": "Content team",
                "priority": "medium"
            },
            {
                "step": "Launch pilot campaigns on priority channels",
                "timeline": "4 weeks",
                "owner": "Marketing team",
                "priority": "medium"
            },
            {
                "step": "Monitor performance and optimize based on results",
                "timeline": "Ongoing",
                "owner": "Marketing team",
                "priority": "medium"
            }
        ]
    
    def _create_implementation_timeline(self, state: MarketingWorkflowState) -> Dict[str, List[Dict[str, str]]]:
        """Create implementation timeline by phase."""
        return {
            "phase_1_setup": [
                {"task": "Strategy review and approval", "duration": "1 week"},
                {"task": "Team alignment and training", "duration": "1 week"},
                {"task": "Tool and system setup", "duration": "2 weeks"}
            ],
            "phase_2_content": [
                {"task": "Content creation and approval", "duration": "3 weeks"},
                {"task": "Content calendar finalization", "duration": "1 week"},
                {"task": "Creative asset development", "duration": "2 weeks"}
            ],
            "phase_3_launch": [
                {"task": "Pilot campaign launch", "duration": "1 week"},
                {"task": "Performance monitoring setup", "duration": "1 week"},
                {"task": "Full campaign rollout", "duration": "2 weeks"}
            ]
        }
    
    def _create_budget_allocation(self, state: MarketingWorkflowState) -> Dict[str, Dict[str, Any]]:
        """Create budget allocation recommendations."""
        return {
            "content_creation": {
                "percentage": 30,
                "estimated_cost": "30% of marketing budget",
                "justification": "High-quality content is foundation of strategy"
            },
            "paid_advertising": {
                "percentage": 40,
                "estimated_cost": "40% of marketing budget",
                "justification": "Paid channels provide immediate reach and measurable results"
            },
            "tools_and_technology": {
                "percentage": 15,
                "estimated_cost": "15% of marketing budget",
                "justification": "Analytics and automation tools essential for optimization"
            },
            "team_and_training": {
                "percentage": 15,
                "estimated_cost": "15% of marketing budget", 
                "justification": "Team development ensures successful execution"
            }
        }
    
    def _define_success_metrics(self, state: MarketingWorkflowState) -> Dict[str, Dict[str, Any]]:
        """Define success metrics and KPIs."""
        return {
            "awareness_metrics": {
                "brand_mentions": {"target": "20% increase", "timeframe": "3 months"},
                "website_traffic": {"target": "30% increase", "timeframe": "3 months"},
                "social_reach": {"target": "40% increase", "timeframe": "3 months"}
            },
            "engagement_metrics": {
                "social_engagement_rate": {"target": "5% minimum", "timeframe": "Monthly"},
                "email_open_rate": {"target": "25% minimum", "timeframe": "Monthly"},
                "content_shares": {"target": "15% increase", "timeframe": "3 months"}
            },
            "conversion_metrics": {
                "lead_generation": {"target": "50% increase", "timeframe": "6 months"},
                "conversion_rate": {"target": "3% minimum", "timeframe": "Ongoing"},
                "customer_acquisition_cost": {"target": "20% reduction", "timeframe": "6 months"}
            }
        }
    
    def _create_risk_mitigation_plan(self, state: MarketingWorkflowState) -> Dict[str, Dict[str, str]]:
        """Create risk mitigation plan."""
        return {
            "low_engagement_risk": {
                "probability": "Medium",
                "impact": "High",
                "mitigation": "A/B test content formats and optimize based on performance data"
            },
            "budget_overrun_risk": {
                "probability": "Low",
                "impact": "Medium", 
                "mitigation": "Implement strict budget monitoring and approval processes"
            },
            "competitive_response_risk": {
                "probability": "High",
                "impact": "Medium",
                "mitigation": "Maintain competitive intelligence and agile response capabilities"
            },
            "technology_failure_risk": {
                "probability": "Low",
                "impact": "High",
                "mitigation": "Implement backup systems and manual processes where possible"
            }
        }
    
    async def get_workflow_status(self, request_id: str) -> Dict[str, Any]:
        """Get current status of workflow execution."""
        # This would typically query a database or cache
        # For now, return a placeholder response
        return {
            "request_id": request_id,
            "status": "in_progress",
            "progress": {
                "overall_progress": 0.0,
                "current_node": "unknown",
                "nodes_completed": 0,
                "total_nodes": 7
            },
            "estimated_completion": "Unknown"
        }
    
    async def cancel_workflow(self, request_id: str) -> Dict[str, Any]:
        """Cancel running workflow."""
        # Implementation would depend on checkpointing system
        return {
            "request_id": request_id,
            "status": "cancelled",
            "message": "Workflow cancellation requested"
        }