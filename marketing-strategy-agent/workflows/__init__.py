"""
Main Marketing Workflows

This folder contains the core workflows used by the frontend and backend:
- enhanced_marketing_workflow.py: Complete analysis with all agents
- quick_marketing_workflow.py: Fast analysis with core agents only
- workflow_states.py: Shared workflow state management
"""

from .enhanced_marketing_workflow import EnhancedMarketingWorkflow, WorkflowManager
from .quick_marketing_workflow import QuickMarketingWorkflow
from .workflow_states import WorkflowStatus, WorkflowProgress

__all__ = [
    'EnhancedMarketingWorkflow',
    'WorkflowManager', 
    'QuickMarketingWorkflow',
    'WorkflowStatus',
    'WorkflowProgress'
]