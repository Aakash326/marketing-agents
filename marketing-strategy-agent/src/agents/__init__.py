"""
Agents package for the Marketing Strategy Agent.

This package contains specialized AI agents for different marketing functions:
- MarketingAgent: Main orchestrator for marketing activities
- BrandAnalyzer: Brand positioning and competitive analysis
- ContentCreator: Content generation across multiple channels
- TrendResearcher: Market trends and opportunity research
"""

from .marketing_agent import MarketingAgent
from .brand_analyzer import BrandAnalyzer
from .content_creator import ContentCreator
from .trend_researcher import TrendResearcher

__all__ = [
    "MarketingAgent",
    "BrandAnalyzer", 
    "ContentCreator",
    "TrendResearcher"
]