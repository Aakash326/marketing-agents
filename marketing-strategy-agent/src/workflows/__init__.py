"""
Workflows package for the Marketing Strategy Agent.

This package contains LangGraph workflow implementations for orchestrating
multi-agent marketing strategy processes with state management.
"""

from .marketing_workflow import MarketingWorkflow
from .workflow_states import (
    MarketingWorkflowState,
    BrandAnalysisState,
    TrendResearchState,
    ContentCreationState,
    StrategyState
)

__all__ = [
    "MarketingWorkflow",
    "MarketingWorkflowState",
    "BrandAnalysisState", 
    "TrendResearchState",
    "ContentCreationState",
    "StrategyState"
]