"""
Workflow state definitions for LangGraph orchestration.

This module defines the state structures used throughout the marketing
strategy workflow to manage data flow between agents and workflow nodes.
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
from enum import Enum

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(Enum):
    """Individual node execution status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MarketingWorkflowState(TypedDict):
    """
    Main state for the marketing workflow.
    
    This is the primary state that flows through all nodes in the workflow,
    containing all necessary information for marketing strategy generation.
    """
    
    # Request Information
    request_id: str
    brand_name: str
    industry: str
    target_audience: Optional[str]
    goals: List[str]
    budget_range: Optional[str]
    timeline: Optional[str]
    additional_context: Optional[Dict[str, Any]]
    
    # Workflow Management
    current_node: str
    workflow_status: str
    started_at: str
    completed_at: Optional[str]
    error_message: Optional[str]
    
    # Node Execution Tracking
    node_statuses: Dict[str, str]
    node_results: Dict[str, Any]
    node_errors: Dict[str, str]
    execution_log: List[Dict[str, Any]]
    
    # Agent States
    brand_analysis: "BrandAnalysisState"
    trend_research: "TrendResearchState"
    content_creation: "ContentCreationState"
    strategy_state: "StrategyState"
    
    # Shared Resources
    messages: Annotated[List[BaseMessage], add_messages]
    knowledge_base_results: List[Dict[str, Any]]
    external_data: Dict[str, Any]
    
    # Final Outputs
    final_strategy: Optional[Dict[str, Any]]
    content_recommendations: Optional[Dict[str, Any]]
    implementation_plan: Optional[Dict[str, Any]]
    
    # Quality and Metrics
    confidence_score: float
    quality_scores: Dict[str, float]
    retry_count: int
    processing_time: Optional[float]


class BrandAnalysisState(TypedDict):
    """State for brand analysis node."""
    
    # Input Data
    brand_data: Dict[str, Any]
    competitive_context: List[str]
    analysis_scope: List[str]
    
    # Analysis Results
    positioning_analysis: Optional[Dict[str, Any]]
    voice_analysis: Optional[Dict[str, Any]]
    competitive_analysis: Optional[Dict[str, Any]]
    value_proposition: Optional[Dict[str, Any]]
    brand_strengths: List[str]
    improvement_areas: List[str]
    
    # Processing Status
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error_details: Optional[str]
    
    # Quality Metrics
    analysis_confidence: float
    completeness_score: float
    insights_generated: int


class TrendResearchState(TypedDict):
    """State for trend research node."""
    
    # Research Parameters
    industry_focus: str
    time_frame: str
    trend_categories: List[str]
    competitive_focus: List[str]
    
    # Research Results
    technology_trends: List[Dict[str, Any]]
    consumer_behavior_trends: List[Dict[str, Any]]
    marketing_channel_trends: List[Dict[str, Any]]
    competitive_trends: List[Dict[str, Any]]
    seasonal_opportunities: List[Dict[str, Any]]
    
    # Analysis Outputs
    key_insights: List[str]
    opportunities: List[Dict[str, Any]]
    threats: List[Dict[str, Any]]
    trend_priorities: List[Dict[str, Any]]
    
    # Processing Status
    status: str
    research_started_at: Optional[str]
    research_completed_at: Optional[str]
    error_details: Optional[str]
    
    # Quality Metrics
    trends_identified: int
    data_freshness_score: float
    relevance_score: float


class ContentCreationState(TypedDict):
    """State for content creation node."""
    
    # Content Requirements
    content_types: List[str]
    platforms: List[str]
    campaign_goals: List[str]
    brand_voice_guidelines: Dict[str, Any]
    target_messaging: Dict[str, Any]
    
    # Generated Content
    social_media_content: Dict[str, Any]
    blog_content: Dict[str, Any]
    email_campaigns: Dict[str, Any]
    ad_copy: Dict[str, Any]
    content_calendar: Dict[str, Any]
    
    # Content Analysis
    quality_scores: Dict[str, float]
    brand_alignment_scores: Dict[str, float]
    engagement_predictions: Dict[str, float]
    optimization_suggestions: Dict[str, List[str]]
    
    # Processing Status
    status: str
    creation_started_at: Optional[str]
    creation_completed_at: Optional[str]
    error_details: Optional[str]
    
    # Performance Metrics
    content_pieces_generated: int
    platforms_covered: int
    average_quality_score: float


class StrategyState(TypedDict):
    """State for strategy synthesis and planning."""
    
    # Strategy Components
    executive_summary: Optional[str]
    market_positioning: Optional[Dict[str, Any]]
    target_audience_strategy: Optional[Dict[str, Any]]
    marketing_mix: Optional[Dict[str, Any]]
    channel_strategy: Optional[Dict[str, Any]]
    content_strategy: Optional[Dict[str, Any]]
    campaign_recommendations: List[Dict[str, Any]]
    
    # Implementation Planning
    timeline: Optional[Dict[str, Any]]
    budget_allocation: Optional[Dict[str, Any]]
    success_metrics: Optional[Dict[str, Any]]
    risk_assessment: Optional[Dict[str, Any]]
    
    # Strategic Insights
    key_recommendations: List[str]
    strategic_priorities: List[Dict[str, Any]]
    competitive_advantages: List[str]
    growth_opportunities: List[str]
    
    # Quality and Validation
    strategy_confidence: float
    completeness_score: float
    feasibility_score: float
    alignment_score: float
    
    # Processing Status
    status: str
    synthesis_started_at: Optional[str]
    synthesis_completed_at: Optional[str]
    error_details: Optional[str]


class WorkflowConfiguration(TypedDict):
    """Configuration for workflow execution."""
    
    # Execution Settings
    timeout_seconds: int
    max_retries: int
    enable_parallel_execution: bool
    enable_human_in_loop: bool
    
    # Node Configuration
    node_timeouts: Dict[str, int]
    node_retry_policies: Dict[str, Dict[str, Any]]
    node_dependencies: Dict[str, List[str]]
    
    # Quality Thresholds
    minimum_confidence_scores: Dict[str, float]
    quality_gate_thresholds: Dict[str, float]
    
    # Monitoring and Logging
    enable_detailed_logging: bool
    log_intermediate_states: bool
    performance_monitoring: bool


def create_initial_workflow_state(
    request_id: str,
    brand_name: str,
    industry: str,
    goals: List[str],
    **kwargs
) -> MarketingWorkflowState:
    """
    Create initial workflow state from request parameters.
    
    Args:
        request_id: Unique identifier for the workflow execution
        brand_name: Name of the brand to analyze
        industry: Industry context
        goals: Marketing goals and objectives
        **kwargs: Additional optional parameters
        
    Returns:
        Initial MarketingWorkflowState instance
    """
    now = datetime.utcnow().isoformat()
    
    return MarketingWorkflowState(
        # Request Information
        request_id=request_id,
        brand_name=brand_name,
        industry=industry,
        target_audience=kwargs.get("target_audience"),
        goals=goals,
        budget_range=kwargs.get("budget_range"),
        timeline=kwargs.get("timeline"),
        additional_context=kwargs.get("additional_context", {}),
        
        # Workflow Management
        current_node="start",
        workflow_status=WorkflowStatus.PENDING.value,
        started_at=now,
        completed_at=None,
        error_message=None,
        
        # Node Execution Tracking
        node_statuses={},
        node_results={},
        node_errors={},
        execution_log=[],
        
        # Agent States
        brand_analysis=BrandAnalysisState(
            brand_data={},
            competitive_context=[],
            analysis_scope=[],
            positioning_analysis=None,
            voice_analysis=None,
            competitive_analysis=None,
            value_proposition=None,
            brand_strengths=[],
            improvement_areas=[],
            status=NodeStatus.NOT_STARTED.value,
            started_at=None,
            completed_at=None,
            error_details=None,
            analysis_confidence=0.0,
            completeness_score=0.0,
            insights_generated=0
        ),
        
        trend_research=TrendResearchState(
            industry_focus=industry,
            time_frame="current",
            trend_categories=[],
            competitive_focus=[],
            technology_trends=[],
            consumer_behavior_trends=[],
            marketing_channel_trends=[],
            competitive_trends=[],
            seasonal_opportunities=[],
            key_insights=[],
            opportunities=[],
            threats=[],
            trend_priorities=[],
            status=NodeStatus.NOT_STARTED.value,
            research_started_at=None,
            research_completed_at=None,
            error_details=None,
            trends_identified=0,
            data_freshness_score=0.0,
            relevance_score=0.0
        ),
        
        content_creation=ContentCreationState(
            content_types=[],
            platforms=[],
            campaign_goals=goals,
            brand_voice_guidelines={},
            target_messaging={},
            social_media_content={},
            blog_content={},
            email_campaigns={},
            ad_copy={},
            content_calendar={},
            quality_scores={},
            brand_alignment_scores={},
            engagement_predictions={},
            optimization_suggestions={},
            status=NodeStatus.NOT_STARTED.value,
            creation_started_at=None,
            creation_completed_at=None,
            error_details=None,
            content_pieces_generated=0,
            platforms_covered=0,
            average_quality_score=0.0
        ),
        
        strategy_state=StrategyState(
            executive_summary=None,
            market_positioning=None,
            target_audience_strategy=None,
            marketing_mix=None,
            channel_strategy=None,
            content_strategy=None,
            campaign_recommendations=[],
            timeline=None,
            budget_allocation=None,
            success_metrics=None,
            risk_assessment=None,
            key_recommendations=[],
            strategic_priorities=[],
            competitive_advantages=[],
            growth_opportunities=[],
            strategy_confidence=0.0,
            completeness_score=0.0,
            feasibility_score=0.0,
            alignment_score=0.0,
            status=NodeStatus.NOT_STARTED.value,
            synthesis_started_at=None,
            synthesis_completed_at=None,
            error_details=None
        ),
        
        # Shared Resources
        messages=[],
        knowledge_base_results=[],
        external_data={},
        
        # Final Outputs
        final_strategy=None,
        content_recommendations=None,
        implementation_plan=None,
        
        # Quality and Metrics
        confidence_score=0.0,
        quality_scores={},
        retry_count=0,
        processing_time=None
    )


def update_node_status(
    state: MarketingWorkflowState,
    node_name: str,
    status: NodeStatus,
    result: Optional[Any] = None,
    error: Optional[str] = None
) -> MarketingWorkflowState:
    """
    Update the status of a specific node in the workflow state.
    
    Args:
        state: Current workflow state
        node_name: Name of the node to update
        status: New status for the node
        result: Optional result data from the node
        error: Optional error message if node failed
        
    Returns:
        Updated workflow state
    """
    now = datetime.utcnow().isoformat()
    
    # Update node status
    state["node_statuses"][node_name] = status.value
    state["current_node"] = node_name
    
    # Store result if provided
    if result is not None:
        state["node_results"][node_name] = result
    
    # Store error if provided
    if error is not None:
        state["node_errors"][node_name] = error
    
    # Add execution log entry
    log_entry = {
        "timestamp": now,
        "node": node_name,
        "status": status.value,
        "message": error if error else f"Node {node_name} status updated to {status.value}"
    }
    state["execution_log"].append(log_entry)
    
    # Update workflow status based on node statuses
    state["workflow_status"] = _calculate_workflow_status(state["node_statuses"])
    
    return state


def _calculate_workflow_status(node_statuses: Dict[str, str]) -> str:
    """Calculate overall workflow status based on individual node statuses."""
    if not node_statuses:
        return WorkflowStatus.PENDING.value
    
    status_values = list(node_statuses.values())
    
    # Check for failures
    if NodeStatus.FAILED.value in status_values:
        return WorkflowStatus.FAILED.value
    
    # Check if all nodes are completed
    if all(status == NodeStatus.COMPLETED.value for status in status_values):
        return WorkflowStatus.COMPLETED.value
    
    # Check if any nodes are running
    if NodeStatus.RUNNING.value in status_values:
        return WorkflowStatus.IN_PROGRESS.value
    
    # Default to pending
    return WorkflowStatus.PENDING.value


def validate_workflow_state(state: MarketingWorkflowState) -> List[str]:
    """
    Validate workflow state for consistency and completeness.
    
    Args:
        state: Workflow state to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Required fields validation
    if not state.get("request_id"):
        errors.append("request_id is required")
    
    if not state.get("brand_name"):
        errors.append("brand_name is required")
    
    if not state.get("industry"):
        errors.append("industry is required")
    
    if not state.get("goals"):
        errors.append("goals are required")
    
    # Status consistency validation
    workflow_status = state.get("workflow_status")
    if workflow_status not in [status.value for status in WorkflowStatus]:
        errors.append(f"Invalid workflow_status: {workflow_status}")
    
    # Node status validation
    for node_name, status in state.get("node_statuses", {}).items():
        if status not in [node_status.value for node_status in NodeStatus]:
            errors.append(f"Invalid status for node {node_name}: {status}")
    
    # Timestamp validation
    if state.get("completed_at") and not state.get("started_at"):
        errors.append("completed_at cannot be set without started_at")
    
    return errors


def get_workflow_progress(state: MarketingWorkflowState) -> Dict[str, Any]:
    """
    Calculate workflow progress statistics.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary containing progress information
    """
    node_statuses = state.get("node_statuses", {})
    
    if not node_statuses:
        return {
            "overall_progress": 0.0,
            "nodes_completed": 0,
            "nodes_running": 0,
            "nodes_failed": 0,
            "total_nodes": 0
        }
    
    total_nodes = len(node_statuses)
    completed_nodes = sum(1 for status in node_statuses.values() if status == NodeStatus.COMPLETED.value)
    running_nodes = sum(1 for status in node_statuses.values() if status == NodeStatus.RUNNING.value)
    failed_nodes = sum(1 for status in node_statuses.values() if status == NodeStatus.FAILED.value)
    
    overall_progress = (completed_nodes / total_nodes) * 100 if total_nodes > 0 else 0.0
    
    return {
        "overall_progress": round(overall_progress, 2),
        "nodes_completed": completed_nodes,
        "nodes_running": running_nodes,
        "nodes_failed": failed_nodes,
        "total_nodes": total_nodes,
        "workflow_status": state.get("workflow_status"),
        "current_node": state.get("current_node"),
        "processing_time": state.get("processing_time")
    }