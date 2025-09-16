"""
TrendResearcher Agent - Web-based trend research and timing optimization
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from tavily import TavilyClient
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .base_agent import BaseAgent
from ..models.data_models import (
    CompanyInfo,
    TrendResearchResult,
    TrendingTopic,
    HashtagStrategy,
    OptimalPostingSchedule,
    CompetitorActivity,
    ViralContentPattern,
    Platform,
    AgentResponse
)


class TrendResearcher(BaseAgent):
    """
    Agent responsible for web-based trend research including:
    - Real-time trending topics analysis
    - Hashtag strategy optimization
    - Optimal posting schedule analysis
    - Competitor activity monitoring
    - Viral content pattern identification
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("TrendResearcher", config)
        self.llm = ChatOpenAI(
            model=config.get("openai_model", "gpt-4o-mini"),
            temperature=config.get("temperature", 0.4),
            max_tokens=config.get("max_tokens", 2000)
        )
        
        # Initialize Tavily client for web search
        self.tavily_client = TavilyClient(
            api_key=config.get("tavily_api_key")
        )
        self.tavily_config = {
            "max_results": config.get("tavily_max_results", 10),
            "search_depth": config.get("tavily_search_depth", "advanced"),
            "include_domains": config.get("tavily_include_domains", []),
            "exclude_domains": config.get("tavily_exclude_domains", [])
        }
        
    def get_capabilities(self) -> Dict[str, str]:
        return {
            "trending_topics": "Identifies current trending topics relevant to industry",
            "hashtag_research": "Researches optimal hashtag strategies for maximum reach",
            "posting_schedule": "Analyzes optimal posting times for each platform",
            "competitor_monitoring": "Monitors competitor activities and campaigns",
            "viral_patterns": "Identifies patterns in viral content for strategic insights",
            "market_opportunities": "Discovers emerging market opportunities from trends"
        }
    
    def _extract_json_from_response(self, content: str) -> str:
        """Extract JSON content from response, handling markdown code blocks"""
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
        elif content.startswith('```'):
            lines = content.split('\n')
            if len(lines) > 1:
                content = '\n'.join(lines[1:-1]) if lines[-1].strip() == '```' else '\n'.join(lines[1:])
                content = content.strip()
        return content
    
    async def execute(self, company_info: CompanyInfo, **kwargs) -> AgentResponse:
        """
        Execute comprehensive trend research
        """
        try:
            # Research trending topics
            trending_topics = await self._research_trending_topics(company_info)
            
            # Analyze hashtag strategies
            hashtag_strategies = await self._analyze_hashtag_strategies(company_info, trending_topics)
            
            # Determine optimal posting schedules
            optimal_posting_schedule = await self._analyze_optimal_posting_schedule(company_info)
            
            # Monitor competitor activities
            competitor_activities = await self._monitor_competitor_activities(company_info)
            
            # Identify viral content patterns
            viral_content_patterns = await self._identify_viral_content_patterns(company_info)
            
            # Discover market opportunities
            market_opportunities = await self._discover_market_opportunities(
                company_info, trending_topics, competitor_activities
            )
            
            # Create comprehensive result
            result = TrendResearchResult(
                trending_topics=trending_topics,
                hashtag_strategies=hashtag_strategies,
                optimal_posting_schedule=optimal_posting_schedule,
                competitor_activities=competitor_activities,
                viral_content_patterns=viral_content_patterns,
                market_opportunities=market_opportunities
            )
            
            return AgentResponse(
                agent_name=self.name,
                execution_time=0,  # Will be set by base class
                success=True,
                result=result.dict()
            )
            
        except Exception as e:
            raise Exception(f"TrendResearcher execution failed: {str(e)}")
    
    async def _research_trending_topics(self, company_info: CompanyInfo) -> List[TrendingTopic]:
        """
        Research current trending topics using Tavily web search
        """
        trending_topics = []
        
        # Search queries for different aspects
        search_queries = [
            f"{company_info.industry.value} trends 2024",
            f"{company_info.industry.value} latest news",
            f"trending topics {company_info.industry.value}",
            f"{company_info.industry.value} social media trends",
            f"viral content {company_info.industry.value}"
        ]
        
        for query in search_queries:
            try:
                # Perform web search
                search_results = await self._tavily_search(query)
                
                # Extract and analyze trends from search results
                topics = await self._extract_trending_topics_from_results(
                    search_results, company_info, query
                )
                trending_topics.extend(topics)
                
            except Exception as e:
                print(f"Error searching for '{query}': {e}")
                continue
        
        # Remove duplicates and rank by relevance
        unique_topics = await self._deduplicate_and_rank_topics(trending_topics, company_info)
        
        return unique_topics[:10]  # Return top 10 trending topics
    
    async def _tavily_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform web search using Tavily API
        """
        try:
            response = self.tavily_client.search(
                query=query,
                max_results=self.tavily_config["max_results"],
                search_depth=self.tavily_config["search_depth"],
                include_domains=self.tavily_config["include_domains"],
                exclude_domains=self.tavily_config["exclude_domains"]
            )
            return response.get("results", [])
        except Exception as e:
            print(f"Tavily search error: {e}")
            return []
    
    async def _extract_trending_topics_from_results(self, search_results: List[Dict], 
                                                  company_info: CompanyInfo, query: str) -> List[TrendingTopic]:
        """
        Extract trending topics from search results using AI analysis
        """
        if not search_results:
            return []
        
        # Combine search results content
        content_summary = []
        for result in search_results[:5]:  # Analyze top 5 results
            title = result.get("title", "")
            snippet = result.get("content", "")[:500]  # Limit content length
            content_summary.append(f"Title: {title}\nContent: {snippet}")
        
        combined_content = "\n\n".join(content_summary)
        
        prompt = f"""
        Analyze the following search results for trending topics relevant to {company_info.name} in the {company_info.industry.value} industry.
        
        Search Query: {query}
        Company Description: {company_info.description}
        
        Search Results:
        {combined_content}
        
        Extract 3-5 trending topics and provide:
        1. Topic name/description
        2. Relevance score (0.0-1.0) to the company
        3. Platforms where it's trending (instagram, linkedin, twitter, tiktok, etc.)
        4. Trend direction (rising, stable, declining)
        5. Actionable insights for the company
        
        Format as JSON array:
        [
            {{
                "topic": "topic description",
                "relevance_score": 0.8,
                "platforms": ["instagram", "linkedin"],
                "trend_direction": "rising",
                "actionable_insights": ["insight1", "insight2"]
            }}
        ]
        """
        
        messages = [
            SystemMessage(content="You are a trend research analyst who identifies relevant market trends and opportunities."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Extract JSON from response content
            content = self._extract_json_from_response(response.content)
            topics_data = json.loads(content)
            
            trending_topics = []
            for topic_data in topics_data:
                try:
                    platforms = [Platform(p) for p in topic_data.get("platforms", []) if p in Platform.__members__.values()]
                    
                    topic = TrendingTopic(
                        topic=topic_data["topic"],
                        relevance_score=float(topic_data.get("relevance_score", 0.5)),
                        platforms=platforms,
                        trend_direction=topic_data.get("trend_direction", "stable"),
                        actionable_insights=topic_data.get("actionable_insights", [])
                    )
                    trending_topics.append(topic)
                except Exception as e:
                    print(f"Error parsing topic data: {e}")
                    continue
            
            return trending_topics
            
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return []
    
    async def _deduplicate_and_rank_topics(self, topics: List[TrendingTopic], 
                                         company_info: CompanyInfo) -> List[TrendingTopic]:
        """
        Remove duplicate topics and rank by relevance
        """
        # Simple deduplication by topic similarity
        unique_topics = []
        seen_topics = set()
        
        for topic in topics:
            # Create a normalized version for comparison
            normalized_topic = topic.topic.lower().strip()
            
            if normalized_topic not in seen_topics:
                seen_topics.add(normalized_topic)
                unique_topics.append(topic)
        
        # Sort by relevance score
        unique_topics.sort(key=lambda t: t.relevance_score, reverse=True)
        
        return unique_topics
    
    async def _analyze_hashtag_strategies(self, company_info: CompanyInfo, 
                                        trending_topics: List[TrendingTopic]) -> HashtagStrategy:
        """
        Analyze optimal hashtag strategies
        """
        # Search for hashtag trends
        hashtag_search_queries = [
            f"#{company_info.industry.value} hashtags trending",
            f"best hashtags for {company_info.industry.value}",
            f"viral hashtags {company_info.industry.value} 2024"
        ]
        
        hashtag_data = []
        for query in hashtag_search_queries:
            try:
                results = await self._tavily_search(query)
                hashtag_data.extend(results[:3])  # Limit results per query
            except Exception as e:
                print(f"Error searching hashtags: {e}")
                continue
        
        # Extract trending topics for additional hashtag context
        topic_keywords = [topic.topic for topic in trending_topics[:5]]
        
        # Analyze hashtags using AI
        hashtag_strategy = await self._generate_hashtag_strategy(
            company_info, hashtag_data, topic_keywords
        )
        
        return hashtag_strategy
    
    async def _generate_hashtag_strategy(self, company_info: CompanyInfo, 
                                       hashtag_data: List[Dict], 
                                       trending_keywords: List[str]) -> HashtagStrategy:
        """
        Generate comprehensive hashtag strategy using AI
        """
        # Combine hashtag research data
        content_summary = []
        for result in hashtag_data[:5]:
            title = result.get("title", "")
            snippet = result.get("content", "")[:300]
            content_summary.append(f"{title}: {snippet}")
        
        research_content = "\n".join(content_summary)
        
        prompt = f"""
        Create a comprehensive hashtag strategy for {company_info.name} in the {company_info.industry.value} industry.
        
        Company: {company_info.name}
        Industry: {company_info.industry.value}
        Description: {company_info.description}
        
        Trending Keywords: {', '.join(trending_keywords)}
        
        Research Data: {research_content}
        
        Provide hashtag recommendations in these categories:
        1. Trending hashtags (currently viral, high volume)
        2. Niche hashtags (industry-specific, medium volume)
        3. Branded hashtags (company/campaign specific)
        
        Format as JSON:
        {{
            "trending_hashtags": ["#hashtag1", "#hashtag2", ...],
            "niche_hashtags": ["#nichehashtag1", "#nichehashtag2", ...],
            "branded_hashtags": ["#brandhashtag1", "#brandhashtag2", ...]
        }}
        
        Include 5-10 hashtags per category. Make hashtags relevant and actionable.
        """
        
        messages = [
            SystemMessage(content="You are a social media hashtag strategist who creates effective hashtag campaigns."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Extract JSON from response content
            content = self._extract_json_from_response(response.content)
            hashtag_data = json.loads(content)
            
            return HashtagStrategy(
                trending_hashtags=hashtag_data.get("trending_hashtags", []),
                niche_hashtags=hashtag_data.get("niche_hashtags", []),
                branded_hashtags=hashtag_data.get("branded_hashtags", [])
            )
            
        except Exception as e:
            print(f"Error generating hashtag strategy: {e}")
            # Fallback strategy
            return HashtagStrategy(
                trending_hashtags=[f"#{company_info.industry.value}", "#trending", "#viral"],
                niche_hashtags=[f"#{company_info.industry.value}business", f"#{company_info.industry.value}tips"],
                branded_hashtags=[f"#{company_info.name.replace(' ', '').lower()}", f"#{company_info.name.replace(' ', '').lower()}community"]
            )
    
    async def _analyze_optimal_posting_schedule(self, company_info: CompanyInfo) -> List[OptimalPostingSchedule]:
        """
        Analyze optimal posting schedules for different platforms
        """
        platforms = [Platform.INSTAGRAM, Platform.LINKEDIN, Platform.TWITTER, Platform.TIKTOK]
        schedules = []
        
        for platform in platforms:
            schedule = await self._get_platform_posting_schedule(platform, company_info)
            schedules.append(schedule)
        
        return schedules
    
    async def _get_platform_posting_schedule(self, platform: Platform, 
                                           company_info: CompanyInfo) -> OptimalPostingSchedule:
        """
        Get optimal posting schedule for a specific platform
        """
        # Search for platform-specific posting time data
        query = f"best times to post on {platform.value} {company_info.industry.value} 2024"
        
        try:
            search_results = await self._tavily_search(query)
            research_content = ""
            
            for result in search_results[:3]:
                title = result.get("title", "")
                snippet = result.get("content", "")[:300]
                research_content += f"{title}: {snippet}\n"
            
        except Exception:
            research_content = "No specific research data available."
        
        # Use AI to analyze and recommend posting schedule
        prompt = f"""
        Based on the research data and best practices, recommend optimal posting times for {platform.value}.
        
        Company: {company_info.name}
        Industry: {company_info.industry.value}
        Target Audience: {', '.join(company_info.target_audience) if company_info.target_audience else 'General business audience'}
        
        Research Data: {research_content}
        
        Provide recommendations for:
        1. Best times to post (list 2-3 optimal times)
        2. Best days of the week (list 2-3 best days)
        3. Recommended posting frequency
        
        Consider the target audience's likely online behavior and platform-specific engagement patterns.
        
        Format as JSON:
        {{
            "best_times": ["9am", "1pm", "7pm"],
            "best_days": ["tuesday", "wednesday", "thursday"],
            "frequency_recommendation": "2-3 times per week"
        }}
        """
        
        messages = [
            SystemMessage(content="You are a social media timing expert who optimizes posting schedules for maximum engagement."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = self._extract_json_from_response(response.content)
            schedule_data = json.loads(content)
            
            return OptimalPostingSchedule(
                platform=platform,
                best_times=schedule_data.get("best_times", ["9am", "12pm", "5pm"]),
                best_days=schedule_data.get("best_days", ["tuesday", "wednesday", "thursday"]),
                frequency_recommendation=schedule_data.get("frequency_recommendation", "2-3 times per week")
            )
            
        except Exception as e:
            print(f"Error analyzing posting schedule for {platform.value}: {e}")
            # Fallback schedule
            return OptimalPostingSchedule(
                platform=platform,
                best_times=["9am", "12pm", "5pm"],
                best_days=["tuesday", "wednesday", "thursday"],
                frequency_recommendation="2-3 times per week"
            )
    
    async def _monitor_competitor_activities(self, company_info: CompanyInfo) -> List[CompetitorActivity]:
        """
        Monitor competitor activities and campaigns
        """
        competitor_activities = []
        
        for competitor in company_info.competitors[:3]:  # Monitor top 3 competitors
            try:
                activity = await self._analyze_competitor_activity(competitor, company_info)
                competitor_activities.append(activity)
            except Exception as e:
                print(f"Error analyzing competitor {competitor}: {e}")
                continue
        
        return competitor_activities
    
    async def _analyze_competitor_activity(self, competitor: str, 
                                         company_info: CompanyInfo) -> CompetitorActivity:
        """
        Analyze a specific competitor's recent activities
        """
        # Search for competitor's recent activities
        search_queries = [
            f"{competitor} recent campaigns marketing",
            f"{competitor} social media strategy 2024",
            f"{competitor} latest news announcements"
        ]
        
        competitor_data = []
        for query in search_queries:
            try:
                results = await self._tavily_search(query)
                competitor_data.extend(results[:2])  # Limit results per query
            except Exception:
                continue
        
        # Analyze competitor data
        content_summary = []
        for result in competitor_data[:5]:
            title = result.get("title", "")
            snippet = result.get("content", "")[:300]
            content_summary.append(f"{title}: {snippet}")
        
        research_content = "\n".join(content_summary)
        
        prompt = f"""
        Analyze the competitor {competitor} based on the following research data.
        
        Our Company: {company_info.name}
        Our Industry: {company_info.industry.value}
        
        Competitor Research: {research_content}
        
        Provide analysis on:
        1. Recent campaigns or marketing activities
        2. Engagement patterns or strategies
        3. Content gaps or weaknesses
        4. Opportunities for our company to differentiate
        
        Format as JSON:
        {{
            "recent_campaigns": ["campaign1", "campaign2"],
            "engagement_patterns": {{"pattern1": "description"}},
            "content_gaps": ["gap1", "gap2"],
            "opportunities": ["opportunity1", "opportunity2"]
        }}
        """
        
        messages = [
            SystemMessage(content="You are a competitive intelligence analyst who monitors competitor activities."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = self._extract_json_from_response(response.content)
            activity_data = json.loads(content)
            
            return CompetitorActivity(
                competitor=competitor,
                recent_campaigns=activity_data.get("recent_campaigns", []),
                engagement_patterns=activity_data.get("engagement_patterns", {}),
                content_gaps=activity_data.get("content_gaps", []),
                opportunities=activity_data.get("opportunities", [])
            )
            
        except Exception as e:
            print(f"Error analyzing competitor activity: {e}")
            # Fallback activity
            return CompetitorActivity(
                competitor=competitor,
                recent_campaigns=["Standard marketing activities"],
                engagement_patterns={},
                content_gaps=["Limited digital presence"],
                opportunities=["Increase digital engagement"]
            )
    
    async def _identify_viral_content_patterns(self, company_info: CompanyInfo) -> List[ViralContentPattern]:
        """
        Identify patterns in viral content for the industry
        """
        # Search for viral content patterns
        search_queries = [
            f"viral content {company_info.industry.value} 2024",
            f"most shared content {company_info.industry.value}",
            f"viral marketing campaigns {company_info.industry.value}"
        ]
        
        viral_data = []
        for query in search_queries:
            try:
                results = await self._tavily_search(query)
                viral_data.extend(results[:3])
            except Exception:
                continue
        
        # Analyze viral patterns using AI
        patterns = await self._analyze_viral_patterns(viral_data, company_info)
        
        return patterns
    
    async def _analyze_viral_patterns(self, viral_data: List[Dict], 
                                    company_info: CompanyInfo) -> List[ViralContentPattern]:
        """
        Analyze viral content patterns using AI
        """
        content_summary = []
        for result in viral_data[:5]:
            title = result.get("title", "")
            snippet = result.get("content", "")[:300]
            content_summary.append(f"{title}: {snippet}")
        
        research_content = "\n".join(content_summary)
        
        prompt = f"""
        Analyze viral content patterns for the {company_info.industry.value} industry.
        
        Research Data: {research_content}
        
        Identify 3-5 viral content patterns and provide:
        1. Content format (e.g., "how-to videos", "behind-the-scenes", "user-generated content")
        2. Key elements that make content viral
        3. Platform-specific tips for Instagram, LinkedIn, TikTok, Twitter
        4. Success metrics or benchmarks
        
        Format as JSON array:
        [
            {{
                "content_format": "format description",
                "key_elements": ["element1", "element2"],
                "platform_specific_tips": {{
                    "instagram": ["tip1", "tip2"],
                    "linkedin": ["tip1", "tip2"]
                }},
                "success_metrics": {{"metric1": 1000, "metric2": 500}}
            }}
        ]
        """
        
        messages = [
            SystemMessage(content="You are a viral content analyst who identifies patterns in successful social media content."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = self._extract_json_from_response(response.content)
            patterns_data = json.loads(content)
            
            viral_patterns = []
            for pattern_data in patterns_data:
                # Convert platform tips to use Platform enum
                platform_tips = {}
                for platform_str, tips in pattern_data.get("platform_specific_tips", {}).items():
                    try:
                        platform = Platform(platform_str.lower())
                        platform_tips[platform] = tips
                    except ValueError:
                        continue
                
                pattern = ViralContentPattern(
                    content_format=pattern_data.get("content_format", ""),
                    key_elements=pattern_data.get("key_elements", []),
                    platform_specific_tips=platform_tips,
                    success_metrics=pattern_data.get("success_metrics", {})
                )
                viral_patterns.append(pattern)
            
            return viral_patterns
            
        except Exception as e:
            print(f"Error analyzing viral patterns: {e}")
            # Fallback patterns
            return [
                ViralContentPattern(
                    content_format="Educational how-to content",
                    key_elements=["Clear value proposition", "Step-by-step format", "Visual elements"],
                    platform_specific_tips={
                        Platform.INSTAGRAM: ["Use carousel posts", "Add engaging captions"],
                        Platform.LINKEDIN: ["Professional tone", "Industry insights"]
                    },
                    success_metrics={"engagement_rate": 5, "shares": 100}
                )
            ]
    
    async def _discover_market_opportunities(self, company_info: CompanyInfo,
                                           trending_topics: List[TrendingTopic],
                                           competitor_activities: List[CompetitorActivity]) -> List[str]:
        """
        Discover market opportunities based on trends and competitor analysis
        """
        # Collect trend insights and competitor gaps
        trend_insights = []
        for topic in trending_topics[:5]:
            trend_insights.extend(topic.actionable_insights)
        
        competitor_gaps = []
        for activity in competitor_activities:
            competitor_gaps.extend(activity.content_gaps)
            competitor_gaps.extend(activity.opportunities)
        
        prompt = f"""
        Based on the trend analysis and competitive intelligence, identify 5 key market opportunities for {company_info.name}.
        
        Company: {company_info.name}
        Industry: {company_info.industry.value}
        Description: {company_info.description}
        
        Trending Topic Insights: {', '.join(trend_insights[:10])}
        Competitor Gaps: {', '.join(competitor_gaps[:10])}
        
        Identify specific, actionable market opportunities that:
        1. Leverage current trends
        2. Address competitor weaknesses
        3. Align with company strengths
        4. Offer potential for growth
        5. Are realistic to execute
        
        Provide 5 opportunities as a simple list, one per line.
        """
        
        messages = [
            SystemMessage(content="You are a market opportunity analyst who identifies growth opportunities from market intelligence."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            opportunities = [line.strip().lstrip('1234567890.-) ') for line in response.content.split('\n') if line.strip()]
            return opportunities[:5]
            
        except Exception as e:
            print(f"Error discovering market opportunities: {e}")
            # Fallback opportunities
            return [
                "Develop thought leadership content around industry trends",
                "Create educational content addressing market knowledge gaps",
                "Implement social media strategy focusing on emerging platforms",
                "Launch customer success story campaign",
                "Develop partnership opportunities with complementary businesses"
            ]