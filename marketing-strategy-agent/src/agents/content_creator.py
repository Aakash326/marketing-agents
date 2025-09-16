"""
ContentCreator Agent - Comprehensive content strategy and generation
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .base_agent import BaseAgent
from ..models.data_models import (
    CompanyInfo,
    ContentCreationResult,
    ContentStrategy,
    ContentAssets,
    PromotionalStrategy,
    ContentPillar,
    PlatformStrategy,
    SocialMediaPost,
    ReelScript,
    BlogPost,
    Platform,
    ContentType,
    AgentResponse
)


class ContentCreator(BaseAgent):
    """
    Agent responsible for comprehensive content strategy and creation including:
    - Content strategy development
    - Platform-specific content creation
    - Social media posts and campaigns
    - Reel scripts and video concepts
    - Blog post outlines
    - Promotional strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ContentCreator", config)
        self.llm = ChatOpenAI(
            model=config.get("openai_model", "gpt-4o-mini"),
            temperature=config.get("temperature", 0.4),
            max_tokens=config.get("max_tokens", 2000)
        )
        
    def get_capabilities(self) -> Dict[str, str]:
        return {
            "content_strategy": "Develops comprehensive content pillars and platform strategies",
            "social_media_content": "Creates platform-specific social media posts and captions",
            "video_content": "Generates reel scripts and video content concepts",
            "blog_content": "Creates blog post outlines and content structures",
            "promotional_campaigns": "Designs promotional strategies and engagement tactics",
            "content_calendar": "Plans strategic content calendar with optimal timing"
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
        Execute comprehensive content creation
        """
        try:
            # Get previous agent results for context
            previous_results = kwargs.get('previous_results', {})
            brand_analysis = previous_results.get('BrandAnalyzer', {}).get('result', {})
            trend_research = previous_results.get('TrendResearcher', {}).get('result', {})
            
            # Develop content strategy
            content_strategy = await self._develop_content_strategy(
                company_info, brand_analysis, trend_research
            )
            
            # Create content assets
            content_assets = await self._create_content_assets(
                company_info, content_strategy, brand_analysis, trend_research
            )
            
            # Design promotional strategies
            promotional_strategies = await self._design_promotional_strategies(
                company_info, content_strategy, brand_analysis
            )
            
            # Generate content calendar preview
            content_calendar_preview = await self._generate_content_calendar_preview(
                company_info, content_strategy, content_assets
            )
            
            # Create comprehensive result
            result = ContentCreationResult(
                content_strategy=content_strategy,
                content_assets=content_assets,
                promotional_strategies=promotional_strategies,
                content_calendar_preview=content_calendar_preview
            )
            
            return AgentResponse(
                agent_name=self.name,
                execution_time=0,  # Will be set by base class
                success=True,
                result=result.dict()
            )
            
        except Exception as e:
            raise Exception(f"ContentCreator execution failed: {str(e)}")
    
    async def _develop_content_strategy(self, company_info: CompanyInfo, 
                                      brand_analysis: Dict, trend_research: Dict) -> ContentStrategy:
        """
        Develop comprehensive content strategy
        """
        # Extract relevant data from previous analyses
        brand_positioning = brand_analysis.get('brand_positioning', {})
        trending_topics = trend_research.get('trending_topics', [])
        viral_patterns = trend_research.get('viral_content_patterns', [])
        
        # Generate content pillars
        content_pillars = await self._generate_content_pillars(
            company_info, brand_positioning, trending_topics
        )
        
        # Develop platform strategies
        platform_strategies = await self._develop_platform_strategies(
            company_info, brand_positioning, viral_patterns
        )
        
        # Create overall strategy description
        overall_strategy = await self._create_overall_strategy_description(
            company_info, content_pillars, platform_strategies
        )
        
        return ContentStrategy(
            content_pillars=content_pillars,
            platform_strategies=platform_strategies,
            overall_strategy=overall_strategy
        )
    
    async def _generate_content_pillars(self, company_info: CompanyInfo, 
                                      brand_positioning: Dict, trending_topics: List) -> List[ContentPillar]:
        """
        Generate strategic content pillars
        """
        # Extract trending topic insights
        topic_insights = []
        for topic in trending_topics[:5]:
            if isinstance(topic, dict):
                topic_insights.extend(topic.get('actionable_insights', []))
        
        prompt = f"""
        Create 4-5 strategic content pillars for {company_info.name} in the {company_info.industry.value} industry.
        
        Company: {company_info.name}
        Industry: {company_info.industry.value}
        Description: {company_info.description}
        Values: {', '.join(company_info.values) if company_info.values else 'Not specified'}
        
        Brand Voice Attributes: {', '.join(brand_positioning.get('voice_attributes', []))}
        Target Audience: {', '.join(brand_positioning.get('target_audience', []))}
        
        Trending Insights: {', '.join(topic_insights[:10])}
        
        Create content pillars that:
        1. Align with brand values and voice
        2. Address target audience needs
        3. Leverage trending opportunities
        4. Support business goals
        5. Provide consistent value
        
        For each pillar, provide:
        - Name (2-3 words)
        - Description (1-2 sentences)
        - Percentage of content (should total 100%)
        - 3-5 example topics
        
        Format as JSON array:
        [
            {{
                "name": "pillar name",
                "description": "pillar description",
                "percentage": 25.0,
                "example_topics": ["topic1", "topic2", "topic3"]
            }}
        ]
        """
        
        messages = [
            SystemMessage(content="You are a content strategist who creates compelling content pillar frameworks."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = self._extract_json_from_response(response.content)
            pillars_data = json.loads(content)
            
            content_pillars = []
            for pillar_data in pillars_data:
                pillar = ContentPillar(
                    name=pillar_data["name"],
                    description=pillar_data["description"],
                    percentage=float(pillar_data["percentage"]),
                    example_topics=pillar_data["example_topics"]
                )
                content_pillars.append(pillar)
            
            return content_pillars
            
        except Exception as e:
            print(f"Error generating content pillars: {e}")
            # Fallback pillars
            return [
                ContentPillar(
                    name="Educational",
                    description="Providing valuable insights and knowledge to our audience",
                    percentage=30.0,
                    example_topics=["Industry best practices", "How-to guides", "Tips and tricks"]
                ),
                ContentPillar(
                    name="Behind the Scenes",
                    description="Sharing company culture and processes",
                    percentage=25.0,
                    example_topics=["Team spotlights", "Office culture", "Day in the life"]
                ),
                ContentPillar(
                    name="Industry Insights",
                    description="Commentary on industry trends and news",
                    percentage=25.0,
                    example_topics=["Market analysis", "Trend discussions", "Future predictions"]
                ),
                ContentPillar(
                    name="Customer Success",
                    description="Showcasing customer achievements and testimonials",
                    percentage=20.0,
                    example_topics=["Case studies", "Customer testimonials", "Success stories"]
                )
            ]
    
    async def _develop_platform_strategies(self, company_info: CompanyInfo,
                                         brand_positioning: Dict, viral_patterns: List) -> List[PlatformStrategy]:
        """
        Develop platform-specific strategies
        """
        platforms = [Platform.INSTAGRAM, Platform.LINKEDIN, Platform.TWITTER, Platform.TIKTOK]
        platform_strategies = []
        
        for platform in platforms:
            strategy = await self._create_platform_strategy(
                platform, company_info, brand_positioning, viral_patterns
            )
            platform_strategies.append(strategy)
        
        return platform_strategies
    
    async def _create_platform_strategy(self, platform: Platform, company_info: CompanyInfo,
                                      brand_positioning: Dict, viral_patterns: List) -> PlatformStrategy:
        """
        Create strategy for a specific platform
        """
        # Extract viral pattern insights for this platform
        platform_insights = []
        for pattern in viral_patterns:
            if isinstance(pattern, dict):
                platform_tips = pattern.get('platform_specific_tips', {})
                if platform.value in platform_tips:
                    platform_insights.extend(platform_tips[platform.value])
        
        prompt = f"""
        Create a comprehensive strategy for {platform.value} for {company_info.name}.
        
        Company: {company_info.name}
        Industry: {company_info.industry.value}
        Target Audience: {', '.join(brand_positioning.get('target_audience', []))}
        Brand Voice: {', '.join(brand_positioning.get('voice_attributes', []))}
        
        Platform Insights: {', '.join(platform_insights)}
        
        For {platform.value}, recommend:
        1. Content types (post, reel, story, video, article, carousel)
        2. Posting frequency (daily, 3x/week, etc.)
        3. Optimal posting times (2-3 times)
        4. Engagement strategy description
        5. Budget allocation percentage (if paid promotion recommended)
        
        Consider platform-specific best practices and audience behavior.
        
        Format as JSON:
        {{
            "content_types": ["post", "reel"],
            "posting_frequency": "daily",
            "optimal_times": ["9am", "1pm", "7pm"],
            "engagement_strategy": "strategy description",
            "budget_allocation_percentage": 25.0
        }}
        """
        
        messages = [
            SystemMessage(content="You are a social media strategist specializing in platform-specific optimization."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = self._extract_json_from_response(response.content)
            strategy_data = json.loads(content)
            
            # Convert content types to enum
            content_types = []
            for ct in strategy_data.get("content_types", []):
                try:
                    content_types.append(ContentType(ct))
                except ValueError:
                    continue
            
            return PlatformStrategy(
                platform=platform,
                content_types=content_types,
                posting_frequency=strategy_data.get("posting_frequency", "3x/week"),
                optimal_times=strategy_data.get("optimal_times", ["9am", "1pm", "6pm"]),
                engagement_strategy=strategy_data.get("engagement_strategy", "Engage authentically with audience"),
                budget_allocation_percentage=strategy_data.get("budget_allocation_percentage")
            )
            
        except Exception as e:
            print(f"Error creating strategy for {platform.value}: {e}")
            # Fallback strategy
            return PlatformStrategy(
                platform=platform,
                content_types=[ContentType.POST],
                posting_frequency="3x/week",
                optimal_times=["9am", "1pm", "6pm"],
                engagement_strategy="Regular posting with authentic engagement",
                budget_allocation_percentage=25.0 if platform in [Platform.INSTAGRAM, Platform.LINKEDIN] else 15.0
            )
    
    async def _create_overall_strategy_description(self, company_info: CompanyInfo,
                                                 content_pillars: List[ContentPillar],
                                                 platform_strategies: List[PlatformStrategy]) -> str:
        """
        Create overall content strategy description
        """
        pillar_names = [pillar.name for pillar in content_pillars]
        platform_names = [strategy.platform.value for strategy in platform_strategies]
        
        prompt = f"""
        Create a comprehensive overall content strategy description for {company_info.name}.
        
        Company: {company_info.name}
        Industry: {company_info.industry.value}
        
        Content Pillars: {', '.join(pillar_names)}
        Platforms: {', '.join(platform_names)}
        
        Write a 2-3 paragraph strategy overview that explains:
        1. How the content strategy aligns with business goals
        2. How different pillars work together
        3. Platform-specific approach rationale
        4. Expected outcomes and benefits
        
        Keep it strategic, actionable, and compelling.
        """
        
        messages = [
            SystemMessage(content="You are a senior content strategist who creates compelling strategy overviews."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"Error creating strategy description: {e}")
            return f"Our content strategy for {company_info.name} focuses on delivering value through {len(content_pillars)} key content pillars across {len(platform_strategies)} platforms, ensuring consistent brand messaging while maximizing audience engagement and business growth."
    
    async def _create_content_assets(self, company_info: CompanyInfo, content_strategy: ContentStrategy,
                                   brand_analysis: Dict, trend_research: Dict) -> ContentAssets:
        """
        Create actual content assets
        """
        # Generate social media posts
        social_media_posts = await self._generate_social_media_posts(
            company_info, content_strategy, brand_analysis, trend_research
        )
        
        # Generate reel scripts
        reel_scripts = await self._generate_reel_scripts(
            company_info, content_strategy, brand_analysis
        )
        
        # Generate blog posts
        blog_posts = await self._generate_blog_posts(
            company_info, content_strategy, brand_analysis
        )
        
        return ContentAssets(
            social_media_posts=social_media_posts,
            reel_scripts=reel_scripts,
            blog_posts=blog_posts
        )
    
    async def _generate_social_media_posts(self, company_info: CompanyInfo, content_strategy: ContentStrategy,
                                         brand_analysis: Dict, trend_research: Dict) -> List[SocialMediaPost]:
        """
        Generate social media posts for different platforms
        """
        posts = []
        
        # Get hashtag strategies from trend research
        hashtag_strategies = trend_research.get('hashtag_strategies', {})
        trending_hashtags = hashtag_strategies.get('trending_hashtags', [])
        niche_hashtags = hashtag_strategies.get('niche_hashtags', [])
        
        brand_voice = ', '.join(brand_analysis.get('brand_positioning', {}).get('voice_attributes', []))
        
        # Generate posts for each platform and content pillar combination
        for platform_strategy in content_strategy.platform_strategies[:2]:  # Limit to 2 platforms for initial generation
            for content_pillar in content_strategy.content_pillars[:2]:  # 2 pillars per platform
                post = await self._generate_single_social_media_post(
                    platform_strategy.platform, content_pillar, company_info, 
                    brand_voice, trending_hashtags, niche_hashtags
                )
                if post:
                    posts.append(post)
        
        return posts
    
    async def _generate_single_social_media_post(self, platform: Platform, content_pillar: ContentPillar,
                                               company_info: CompanyInfo, brand_voice: str,
                                               trending_hashtags: List, niche_hashtags: List) -> Optional[SocialMediaPost]:
        """
        Generate a single social media post
        """
        # Select appropriate hashtags
        selected_hashtags = (trending_hashtags[:3] + niche_hashtags[:5])[:8]  # Max 8 hashtags
        
        prompt = f"""
        Create a {platform.value} post for {company_info.name} focusing on the "{content_pillar.name}" content pillar.
        
        Company: {company_info.name}
        Industry: {company_info.industry.value}
        Content Pillar: {content_pillar.name} - {content_pillar.description}
        Brand Voice: {brand_voice}
        Platform: {platform.value}
        
        Example Topics: {', '.join(content_pillar.example_topics[:3])}
        Available Hashtags: {', '.join(selected_hashtags)}
        
        Create:
        1. Engaging caption (appropriate length for {platform.value})
        2. Relevant hashtags from the provided list
        3. Visual prompt for AI image generation
        4. Call to action
        5. Target audience segment
        
        Make it authentic, valuable, and platform-appropriate.
        
        Format as JSON:
        {{
            "caption": "post caption text",
            "hashtags": ["#hashtag1", "#hashtag2"],
            "visual_prompt": "description for AI image generation",
            "call_to_action": "specific CTA",
            "target_audience": "audience segment"
        }}
        """
        
        messages = [
            SystemMessage(content="You are a social media content creator who crafts engaging, platform-specific posts."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = self._extract_json_from_response(response.content)
            post_data = json.loads(content)
            
            return SocialMediaPost(
                platform=platform,
                content_type=ContentType.POST,
                caption=post_data["caption"],
                hashtags=post_data.get("hashtags", []),
                visual_prompt=post_data.get("visual_prompt"),
                call_to_action=post_data.get("call_to_action"),
                target_audience=post_data.get("target_audience")
            )
            
        except Exception as e:
            print(f"Error generating post for {platform.value}: {e}")
            return None
    
    async def _generate_reel_scripts(self, company_info: CompanyInfo, content_strategy: ContentStrategy,
                                   brand_analysis: Dict) -> List[ReelScript]:
        """
        Generate reel scripts for video content
        """
        scripts = []
        brand_voice = ', '.join(brand_analysis.get('brand_positioning', {}).get('voice_attributes', []))
        
        # Generate 2-3 reel scripts from different content pillars
        for content_pillar in content_strategy.content_pillars[:3]:
            script = await self._generate_single_reel_script(
                content_pillar, company_info, brand_voice
            )
            if script:
                scripts.append(script)
        
        return scripts
    
    async def _generate_single_reel_script(self, content_pillar: ContentPillar,
                                         company_info: CompanyInfo, brand_voice: str) -> Optional[ReelScript]:
        """
        Generate a single reel script
        """
        prompt = f"""
        Create a reel script for {company_info.name} based on the "{content_pillar.name}" content pillar.
        
        Company: {company_info.name}
        Industry: {company_info.industry.value}
        Content Pillar: {content_pillar.name} - {content_pillar.description}
        Brand Voice: {brand_voice}
        
        Example Topics: {', '.join(content_pillar.example_topics)}
        
        Create a 30-60 second reel script with:
        1. Hook (first 3 seconds to grab attention)
        2. Body (main content with value)
        3. Call to action (ending CTA)
        4. Visual concept (how it should look)
        5. Music style suggestion
        
        Make it engaging, valuable, and shareable.
        
        Format as JSON:
        {{
            "hook": "opening line/visual",
            "body": "main content script",
            "call_to_action": "ending CTA",
            "visual_concept": "visual storytelling approach",
            "music_style": "suggested music type",
            "duration": 30
        }}
        """
        
        messages = [
            SystemMessage(content="You are a video content creator who writes engaging reel scripts that go viral."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = self._extract_json_from_response(response.content)
            script_data = json.loads(content)
            
            return ReelScript(
                hook=script_data["hook"],
                body=script_data["body"],
                call_to_action=script_data["call_to_action"],
                visual_concept=script_data["visual_concept"],
                music_style=script_data.get("music_style"),
                duration=script_data.get("duration", 30)
            )
            
        except Exception as e:
            print(f"Error generating reel script: {e}")
            return None
    
    async def _generate_blog_posts(self, company_info: CompanyInfo, content_strategy: ContentStrategy,
                                 brand_analysis: Dict) -> List[BlogPost]:
        """
        Generate blog post outlines
        """
        posts = []
        brand_voice = ', '.join(brand_analysis.get('brand_positioning', {}).get('voice_attributes', []))
        
        # Generate 2-3 blog posts from different content pillars
        for content_pillar in content_strategy.content_pillars[:3]:
            post = await self._generate_single_blog_post(
                content_pillar, company_info, brand_voice
            )
            if post:
                posts.append(post)
        
        return posts
    
    async def _generate_single_blog_post(self, content_pillar: ContentPillar,
                                       company_info: CompanyInfo, brand_voice: str) -> Optional[BlogPost]:
        """
        Generate a single blog post outline
        """
        prompt = f"""
        Create a blog post outline for {company_info.name} based on the "{content_pillar.name}" content pillar.
        
        Company: {company_info.name}
        Industry: {company_info.industry.value}
        Content Pillar: {content_pillar.name} - {content_pillar.description}
        Brand Voice: {brand_voice}
        
        Example Topics: {', '.join(content_pillar.example_topics)}
        
        Create:
        1. Compelling title
        2. Structured outline (5-7 main points)
        3. SEO keywords to target
        4. Meta description
        5. Target word count
        
        Make it valuable, SEO-friendly, and aligned with the content pillar.
        
        Format as JSON:
        {{
            "title": "blog post title",
            "outline": ["point1", "point2", "point3", "point4", "point5"],
            "seo_keywords": ["keyword1", "keyword2", "keyword3"],
            "meta_description": "meta description",
            "target_word_count": 1500
        }}
        """
        
        messages = [
            SystemMessage(content="You are a content marketing specialist who creates SEO-optimized blog content."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = self._extract_json_from_response(response.content)
            post_data = json.loads(content)
            
            return BlogPost(
                title=post_data["title"],
                outline=post_data["outline"],
                seo_keywords=post_data["seo_keywords"],
                meta_description=post_data["meta_description"],
                target_word_count=post_data.get("target_word_count", 1500),
                content_pillar=content_pillar.name
            )
            
        except Exception as e:
            print(f"Error generating blog post: {e}")
            return None
    
    async def _design_promotional_strategies(self, company_info: CompanyInfo, content_strategy: ContentStrategy,
                                           brand_analysis: Dict) -> PromotionalStrategy:
        """
        Design promotional strategies
        """
        brand_positioning = brand_analysis.get('brand_positioning', {})
        target_audience = brand_positioning.get('target_audience', [])
        
        prompt = f"""
        Design comprehensive promotional strategies for {company_info.name}.
        
        Company: {company_info.name}
        Industry: {company_info.industry.value}
        Target Audience: {', '.join(target_audience)}
        Content Pillars: {', '.join([pillar.name for pillar in content_strategy.content_pillars])}
        
        Create strategies for:
        1. Influencer collaboration approach
        2. Paid advertising strategy
        3. Organic engagement tactics (5-7 tactics)
        4. Budget recommendations (percentage allocation)
        
        Make recommendations specific, actionable, and budget-conscious.
        
        Format as JSON:
        {{
            "influencer_collaboration": "strategy description",
            "paid_advertising": "advertising approach",
            "engagement_tactics": ["tactic1", "tactic2", "tactic3"],
            "budget_recommendations": {{
                "influencer_marketing": 30.0,
                "paid_advertising": 40.0,
                "content_creation": 20.0,
                "tools_and_software": 10.0
            }}
        }}
        """
        
        messages = [
            SystemMessage(content="You are a digital marketing strategist who designs comprehensive promotional campaigns."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            content = self._extract_json_from_response(response.content)
            strategy_data = json.loads(content)
            
            return PromotionalStrategy(
                influencer_collaboration=strategy_data["influencer_collaboration"],
                paid_advertising=strategy_data["paid_advertising"],
                engagement_tactics=strategy_data["engagement_tactics"],
                budget_recommendations=strategy_data.get("budget_recommendations", {})
            )
            
        except Exception as e:
            print(f"Error designing promotional strategies: {e}")
            # Fallback strategy
            return PromotionalStrategy(
                influencer_collaboration="Partner with micro-influencers in the industry for authentic endorsements",
                paid_advertising="Focus on targeted social media ads with strong creative assets",
                engagement_tactics=[
                    "Respond to comments within 2 hours",
                    "Share user-generated content",
                    "Host live Q&A sessions",
                    "Create interactive polls and stories",
                    "Collaborate with other brands"
                ],
                budget_recommendations={
                    "paid_advertising": 40.0,
                    "influencer_marketing": 30.0,
                    "content_creation": 20.0,
                    "tools_and_software": 10.0
                }
            )
    
    async def _generate_content_calendar_preview(self, company_info: CompanyInfo, content_strategy: ContentStrategy,
                                               content_assets: ContentAssets) -> List[Dict[str, Any]]:
        """
        Generate a preview of the content calendar
        """
        calendar_entries = []
        
        # Create sample calendar entries for the next 2 weeks
        start_date = datetime.now()
        
        for i in range(14):  # 2 weeks
            current_date = start_date + timedelta(days=i)
            
            # Determine if we should post today based on platform strategies
            for platform_strategy in content_strategy.platform_strategies:
                should_post = await self._should_post_today(current_date, platform_strategy)
                
                if should_post:
                    # Select content from assets
                    content_item = await self._select_content_for_date(
                        current_date, platform_strategy, content_assets, content_strategy
                    )
                    
                    if content_item:
                        calendar_entries.append(content_item)
        
        return calendar_entries[:10]  # Return first 10 entries
    
    async def _should_post_today(self, date: datetime, platform_strategy: PlatformStrategy) -> bool:
        """
        Determine if we should post on a given date based on platform strategy
        """
        frequency = platform_strategy.posting_frequency.lower()
        day_of_week = date.strftime('%A').lower()
        
        if "daily" in frequency:
            return day_of_week not in ['saturday', 'sunday']  # Weekdays only
        elif "3x" in frequency or "3 times" in frequency:
            return day_of_week in ['monday', 'wednesday', 'friday']
        elif "2x" in frequency or "2 times" in frequency:
            return day_of_week in ['tuesday', 'thursday']
        else:
            return day_of_week in ['wednesday']  # Default to once a week
    
    async def _select_content_for_date(self, date: datetime, platform_strategy: PlatformStrategy,
                                     content_assets: ContentAssets, content_strategy: ContentStrategy) -> Optional[Dict[str, Any]]:
        """
        Select appropriate content for a given date and platform
        """
        # Cycle through content pillars
        pillar_index = date.day % len(content_strategy.content_pillars)
        content_pillar = content_strategy.content_pillars[pillar_index]
        
        # Select content type based on platform strategy
        if platform_strategy.content_types:
            content_type = platform_strategy.content_types[date.day % len(platform_strategy.content_types)]
        else:
            content_type = ContentType.POST
        
        # Find matching content from assets
        if content_type == ContentType.POST:
            for post in content_assets.social_media_posts:
                if post.platform == platform_strategy.platform:
                    return {
                        "date": date.isoformat(),
                        "platform": platform_strategy.platform.value,
                        "content_type": content_type.value,
                        "title": post.caption[:50] + "...",
                        "content_pillar": content_pillar.name,
                        "status": "planned"
                    }
        elif content_type == ContentType.REEL and content_assets.reel_scripts:
            reel = content_assets.reel_scripts[0]  # Use first reel script
            return {
                "date": date.isoformat(),
                "platform": platform_strategy.platform.value,
                "content_type": content_type.value,
                "title": reel.hook[:50] + "...",
                "content_pillar": content_pillar.name,
                "status": "planned"
            }
        
        # Default content entry
        return {
            "date": date.isoformat(),
            "platform": platform_strategy.platform.value,
            "content_type": content_type.value,
            "title": f"{content_pillar.name} content for {platform_strategy.platform.value}",
            "content_pillar": content_pillar.name,
            "status": "planned"
        }