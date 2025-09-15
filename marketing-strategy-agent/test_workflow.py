#!/usr/bin/env python3
"""
Test script to verify the marketing workflow with all agents.

This script tests the complete marketing strategy workflow including:
- Brand analysis
- Trend research  
- Content creation
- Strategy synthesis
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.workflows.marketing_workflow import MarketingWorkflow
from src.utils.logging import get_component_logger

logger = get_component_logger("workflow_test", __name__)


async def test_marketing_workflow():
    """Test the complete marketing workflow."""
    try:
        print("🚀 Testing Marketing Strategy Workflow")
        print("=" * 60)
        
        # Initialize workflow
        workflow = MarketingWorkflow()
        
        # Test input parameters
        test_inputs = {
            "brand_name": "TechNova",
            "industry": "Technology",
            "goals": [
                "Increase brand awareness",
                "Generate leads", 
                "Expand into new markets"
            ],
            "target_audience": "Tech startups and developers",
            "budget_range": "$50,000 - $100,000",
            "timeline": "Q4 2025",
            "request_id": "test_workflow_001"
        }
        
        print(f"📋 Test Parameters:")
        print(f"   Brand: {test_inputs['brand_name']}")
        print(f"   Industry: {test_inputs['industry']}")
        print(f"   Goals: {', '.join(test_inputs['goals'])}")
        print(f"   Target Audience: {test_inputs['target_audience']}")
        print(f"   Budget: {test_inputs['budget_range']}")
        print(f"   Timeline: {test_inputs['timeline']}")
        
        print("\n🔄 Executing workflow...")
        
        # Execute the workflow
        result = await workflow.execute_workflow(**test_inputs)
        
        # Display results
        print("\n✅ Workflow completed successfully!")
        print("=" * 60)
        
        print(f"📊 Workflow Status: {result.get('workflow_status', 'Unknown')}")
        print(f"⏱️  Processing Time: {result.get('processing_time_seconds', 'N/A')} seconds")
        print(f"🎯 Confidence Score: {result.get('confidence_score', 'N/A')}")
        
        # Show key results
        if 'brand_analysis' in result:
            print(f"\n🏢 Brand Analysis:")
            brand_analysis = result['brand_analysis']
            if 'positioning' in brand_analysis:
                print(f"   • Positioning: {brand_analysis['positioning'].get('statement', 'N/A')}")
            if 'target_audience' in brand_analysis:
                audiences = brand_analysis['target_audience']
                if isinstance(audiences, list) and audiences:
                    print(f"   • Target Audience: {', '.join(audiences[:3])}")
        
        if 'trend_research' in result:
            print(f"\n📈 Trend Research:")
            trends = result['trend_research']
            if 'emerging_trends' in trends:
                trend_list = trends['emerging_trends']
                if isinstance(trend_list, list) and trend_list:
                    print(f"   • Key Trends: {', '.join(trend_list[:3])}")
        
        if 'content_creation' in result:
            print(f"\n📝 Content Creation:")
            content = result['content_creation']
            content_types = []
            if 'social_media_content' in content:
                content_types.append("Social Media")
            if 'blog_content' in content:
                content_types.append("Blog Posts")
            if 'email_campaigns' in content:
                content_types.append("Email Campaigns")
            print(f"   • Generated Content: {', '.join(content_types)}")
        
        if 'strategy_synthesis' in result:
            print(f"\n🎯 Strategy Synthesis:")
            strategy = result['strategy_synthesis']
            if 'executive_summary' in strategy:
                summary = strategy['executive_summary']
                if isinstance(summary, str):
                    print(f"   • Summary: {summary[:150]}...")
            if 'key_recommendations' in strategy:
                recommendations = strategy['key_recommendations']
                if isinstance(recommendations, list) and recommendations:
                    print(f"   • Top Recommendation: {recommendations[0]}")
        
        print(f"\n🎉 All agents successfully collaborated to generate a comprehensive marketing strategy!")
        print(f"📁 Full results available in the returned workflow object")
        
        return True
        
    except Exception as e:
        logger.error(f"Workflow test failed: {e}", exc_info=True)
        print(f"\n❌ Workflow test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🧪 Marketing Strategy Agent - Workflow Test")
    print("=" * 60)
    
    # Test the workflow
    success = await test_marketing_workflow()
    
    if success:
        print(f"\n✅ Workflow test completed successfully!")
        print(f"\n🎯 Your marketing strategy agent is fully operational!")
        print(f"\nNext steps:")
        print(f"1. The FastAPI server is running at: http://localhost:8000")
        print(f"2. View API docs at: http://localhost:8000/docs")
        print(f"3. Test the API endpoints with real data")
        print(f"4. Integrate with your applications")
    else:
        print(f"\n❌ Workflow test failed - check the logs for details")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
