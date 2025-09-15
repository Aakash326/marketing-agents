"""
Analytics and insights API routes.

This module contains all analytics-related endpoints including
performance metrics, recommendations, and knowledge search.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional

from src.models.marketing_models import KnowledgeSearchRequest, KnowledgeSearchResponse
from src.models.response_models import APIResponse, create_success_response, create_error_response, ErrorCode
from src.utils.logging import get_component_logger

# Create router
router = APIRouter(
    prefix="/analytics",
    tags=["analytics"],
    responses={404: {"description": "Not found"}}
)

logger = get_component_logger("analytics_routes", __name__)


@router.get("/performance", response_model=APIResponse[Dict[str, Any]])
async def get_performance_metrics(
    campaign_id: Optional[str] = Query(None, description="Campaign ID to analyze"),
    time_period: str = Query("30d", description="Time period for metrics (7d, 30d, 90d)")
):
    """
    Get campaign performance metrics and analytics.
    
    This endpoint provides:
    - Campaign performance data
    - Engagement metrics
    - ROI analysis
    - Trend analysis
    """
    try:
        logger.info("Performance metrics requested", extra={
            "campaign_id": campaign_id,
            "time_period": time_period
        })
        
        # Mock performance data
        mock_response = {
            "campaign_id": campaign_id or "overall",
            "time_period": time_period,
            "generated_at": "2025-01-15T10:30:00Z",
            "metrics": {
                "impressions": 125000,
                "clicks": 3750,
                "click_through_rate": 3.0,
                "conversions": 185,
                "conversion_rate": 4.93,
                "cost_per_click": 2.50,
                "cost_per_conversion": 50.68,
                "return_on_ad_spend": 4.2
            },
            "engagement": {
                "likes": 1250,
                "shares": 320,
                "comments": 185,
                "saves": 95,
                "engagement_rate": 5.2
            },
            "trends": {
                "impressions_trend": "+12%",
                "clicks_trend": "+8%",
                "conversion_trend": "+15%",
                "engagement_trend": "+6%"
            },
            "top_performing_content": [
                {
                    "content_id": "post_001",
                    "type": "educational",
                    "platform": "linkedin",
                    "engagement_rate": 8.5,
                    "conversions": 45
                },
                {
                    "content_id": "post_002", 
                    "type": "promotional",
                    "platform": "facebook",
                    "engagement_rate": 6.2,
                    "conversions": 32
                }
            ],
            "insights": [
                "Educational content performs 40% better than promotional content",
                "LinkedIn generates highest quality leads with 65% higher conversion rate",
                "Optimal posting time is 10-11 AM for maximum engagement"
            ]
        }
        
        return create_success_response(
            data=mock_response,
            message="Performance metrics retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Performance metrics retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                ErrorCode.INTERNAL_SERVER_ERROR,
                "Failed to retrieve performance metrics"
            ).dict()
        )


@router.post("/recommendations", response_model=APIResponse[Dict[str, Any]])
async def get_optimization_recommendations(
    data: Dict[str, Any]
):
    """
    Get AI-powered optimization recommendations.
    
    This endpoint analyzes performance data and provides:
    - Content optimization suggestions
    - Budget allocation recommendations  
    - Audience targeting improvements
    - Campaign timing optimization
    """
    try:
        logger.info("Optimization recommendations requested")
        
        # Mock recommendations
        mock_response = {
            "generated_at": "2025-01-15T10:30:00Z",
            "analysis_period": "Last 30 days",
            "overall_score": 7.5,
            "recommendations": [
                {
                    "category": "content_optimization",
                    "priority": "high",
                    "title": "Increase Educational Content",
                    "description": "Educational content shows 40% higher engagement. Increase from 30% to 50% of content mix.",
                    "expected_impact": "25% increase in engagement rate",
                    "implementation_effort": "medium",
                    "action_items": [
                        "Create 2 additional educational posts per week",
                        "Focus on industry insights and how-to content",
                        "Include more data and statistics in posts"
                    ]
                },
                {
                    "category": "audience_targeting",
                    "priority": "high", 
                    "title": "Refine LinkedIn Targeting",
                    "description": "LinkedIn campaigns show highest ROI. Expand targeting to similar professional segments.",
                    "expected_impact": "30% increase in qualified leads",
                    "implementation_effort": "low",
                    "action_items": [
                        "Expand to adjacent job titles",
                        "Target similar company sizes",
                        "Test lookalike audiences"
                    ]
                },
                {
                    "category": "timing_optimization",
                    "priority": "medium",
                    "title": "Optimize Posting Schedule",
                    "description": "Shift more posts to 10-11 AM time slot for maximum engagement.",
                    "expected_impact": "15% increase in organic reach",
                    "implementation_effort": "low",
                    "action_items": [
                        "Schedule 70% of posts between 10-11 AM",
                        "Test afternoon slots for different content types",
                        "A/B test optimal days of week"
                    ]
                }
            ],
            "budget_optimization": {
                "current_allocation": {"linkedin": 45, "facebook": 35, "google": 20},
                "recommended_allocation": {"linkedin": 55, "facebook": 25, "google": 20},
                "rationale": "LinkedIn showing highest ROI and conversion quality"
            },
            "next_review_date": "2025-02-15T10:30:00Z"
        }
        
        return create_success_response(
            data=mock_response,
            message="Optimization recommendations generated successfully"
        )
        
    except Exception as e:
        logger.error(f"Recommendations generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                ErrorCode.INTERNAL_SERVER_ERROR,
                "Failed to generate recommendations"
            ).dict()
        )


@router.get("/knowledge/search", response_model=APIResponse[Dict[str, Any]])
async def search_knowledge_base(
    query: str = Query(..., description="Search query"),
    category: Optional[str] = Query(None, description="Knowledge category filter"),
    max_results: int = Query(5, ge=1, le=20, description="Maximum results to return")
):
    """
    Search the marketing knowledge base.
    
    This endpoint provides:
    - Semantic search across marketing knowledge
    - Categorized results
    - Relevance scoring
    - Related suggestions
    """
    try:
        logger.info("Knowledge search requested", extra={
            "query": query,
            "category": category,
            "max_results": max_results
        })
        
        # Mock search results
        mock_results = [
            {
                "content": f"Comprehensive insights about {query} in marketing strategy...",
                "title": f"Marketing Strategy Guide: {query}",
                "category": category or "marketing_strategies",
                "relevance_score": 0.95,
                "metadata": {
                    "source": "frameworks_guide.json",
                    "last_updated": "2025-01-15",
                    "content_type": "guide"
                }
            },
            {
                "content": f"Case study analysis showing how {query} impacts business results...",
                "title": f"Case Study: {query} Implementation",
                "category": "campaign_examples",
                "relevance_score": 0.87,
                "metadata": {
                    "source": "successful_campaigns.json",
                    "last_updated": "2025-01-15",
                    "content_type": "case_study"
                }
            }
        ]
        
        mock_response = {
            "query": query,
            "total_results": len(mock_results),
            "results": mock_results[:max_results],
            "search_time": 0.15,
            "suggested_queries": [
                f"{query} best practices",
                f"{query} implementation guide",
                f"{query} case studies"
            ],
            "categories_found": ["marketing_strategies", "campaign_examples"],
            "related_topics": [
                "digital marketing",
                "content strategy", 
                "brand positioning"
            ]
        }
        
        return create_success_response(
            data=mock_response,
            message="Knowledge search completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Knowledge search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                ErrorCode.INTERNAL_SERVER_ERROR,
                "Failed to search knowledge base"
            ).dict()
        )


@router.get("/health")
async def analytics_health_check():
    """Health check for analytics services."""
    return {"status": "healthy", "service": "analytics"}