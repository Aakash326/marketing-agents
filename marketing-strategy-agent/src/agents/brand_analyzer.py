"""
BrandAnalyzer Agent - Comprehensive brand analysis and positioning
"""
import asyncio
from typing import Dict, Any, List
import openai
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .base_agent import BaseAgent
from ..models.data_models import (
    CompanyInfo, 
    BrandAnalysisResult,
    BrandPositioning,
    CompetitorAnalysis,
    BrandArchetype,
    AgentResponse
)
from ..database.simple_vector_store import VectorStore


class BrandAnalyzer(BaseAgent):
    """
    Agent responsible for comprehensive brand analysis including:
    - Brand positioning analysis
    - Competitive landscape assessment
    - Brand archetype identification
    - Market positioning evaluation
    - Brand health scoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("BrandAnalyzer", config)
        self.llm = ChatOpenAI(
            model=config.get("openai_model", "gpt-4o-mini"),
            temperature=config.get("temperature", 0.4),
            max_tokens=config.get("max_tokens", 2000)
        )
        self.vector_store = VectorStore(config.get("vector_config", {}))
        
    def get_capabilities(self) -> Dict[str, str]:
        return {
            "brand_positioning": "Analyzes and defines brand positioning strategy",
            "competitive_analysis": "Evaluates competitive landscape and identifies opportunities", 
            "brand_archetype": "Identifies appropriate brand archetype and personality",
            "market_positioning": "Assesses current market position and growth opportunities",
            "brand_health": "Calculates comprehensive brand health score",
            "voice_guidelines": "Develops brand voice and tone guidelines"
        }
    
    async def execute(self, company_info: CompanyInfo, **kwargs) -> AgentResponse:
        """
        Execute comprehensive brand analysis
        """
        try:
            # Get relevant brand frameworks from knowledge base
            brand_frameworks = await self._get_brand_frameworks(company_info.industry.value)
            
            # Analyze brand positioning
            brand_positioning = await self._analyze_brand_positioning(company_info, brand_frameworks)
            
            # Perform competitive analysis
            competitive_analysis = await self._analyze_competitors(company_info)
            
            # Determine market positioning
            market_positioning = await self._determine_market_positioning(company_info, competitive_analysis)
            
            # Identify differentiation opportunities
            differentiation_opportunities = await self._identify_differentiation_opportunities(
                company_info, competitive_analysis
            )
            
            # Calculate brand health score
            brand_health_score = await self._calculate_brand_health_score(
                company_info, brand_positioning, competitive_analysis
            )
            
            # Generate strategic recommendations
            recommendations = await self._generate_recommendations(
                company_info, brand_positioning, competitive_analysis, brand_health_score
            )
            
            # Create comprehensive result
            result = BrandAnalysisResult(
                brand_positioning=brand_positioning,
                competitive_analysis=competitive_analysis,
                market_positioning=market_positioning,
                differentiation_opportunities=differentiation_opportunities,
                brand_health_score=brand_health_score,
                recommendations=recommendations
            )
            
            return AgentResponse(
                agent_name=self.name,
                execution_time=0,  # Will be set by base class
                success=True,
                result=result.dict()
            )
            
        except Exception as e:
            raise Exception(f"BrandAnalyzer execution failed: {str(e)}")
    
    async def _get_brand_frameworks(self, industry: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant brand frameworks from knowledge base
        """
        try:
            query = f"brand positioning frameworks for {industry} industry"
            results = await self.vector_store.similarity_search(query, k=5)
            return [result.get("content", {}) for result in results]
        except Exception:
            # Return default frameworks if vector store fails
            return [
                {
                    "framework": "Brand Positioning Canvas",
                    "elements": ["target_audience", "category", "differentiation", "reason_to_believe"]
                },
                {
                    "framework": "Brand Archetype Model", 
                    "archetypes": ["hero", "explorer", "creator", "ruler", "caregiver", "innocent"]
                }
            ]
    
    async def _analyze_brand_positioning(self, company_info: CompanyInfo, frameworks: List[Dict]) -> BrandPositioning:
        """
        Analyze and define brand positioning using AI and frameworks
        """
        prompt = PromptTemplate(
            input_variables=["company_name", "industry", "description", "mission", "values", "target_audience", "frameworks"],
            template="""
            As a brand strategist, analyze the following company and create a comprehensive brand positioning strategy.
            
            Company: {company_name}
            Industry: {industry}
            Description: {description}
            Mission: {mission}
            Values: {values}
            Current Target Audience: {target_audience}
            
            Available Frameworks: {frameworks}
            
            Please provide:
            1. A clear positioning statement in the format: "For [audience], [brand] is the [category] that [differentiation] because [reason to believe]"
            2. Refined target audience segments (be specific)
            3. Top 3 competitive advantages
            4. Most appropriate brand archetype from: innocent, explorer, sage, hero, outlaw, magician, regular_guy, lover, jester, caregiver, creator, ruler
            5. 5 key brand voice attributes
            6. Tone guidelines for different contexts (professional, social, crisis, celebration)
            
            Format your response as JSON with the following structure:
            {{
                "positioning_statement": "...",
                "target_audience": ["segment1", "segment2", "segment3"],
                "competitive_advantages": ["advantage1", "advantage2", "advantage3"],
                "brand_archetype": "archetype_name",
                "voice_attributes": ["attribute1", "attribute2", "attribute3", "attribute4", "attribute5"],
                "tone_guidelines": {{
                    "professional": "description",
                    "social": "description", 
                    "crisis": "description",
                    "celebration": "description"
                }}
            }}
            """
        )
        
        messages = [
            SystemMessage(content="You are an expert brand strategist with 20 years of experience helping companies define their brand positioning."),
            HumanMessage(content=prompt.format(
                company_name=company_info.name,
                industry=company_info.industry.value,
                description=company_info.description,
                mission=company_info.mission or "Not specified",
                values=", ".join(company_info.values) if company_info.values else "Not specified",
                target_audience=", ".join(company_info.target_audience) if company_info.target_audience else "Not specified",
                frameworks=str(frameworks)
            ))
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            import json
            result = json.loads(response.content)
            
            return BrandPositioning(
                positioning_statement=result["positioning_statement"],
                target_audience=result["target_audience"],
                competitive_advantages=result["competitive_advantages"],
                brand_archetype=BrandArchetype(result["brand_archetype"]),
                voice_attributes=result["voice_attributes"],
                tone_guidelines=result["tone_guidelines"]
            )
        except Exception as e:
            # Fallback if JSON parsing fails
            return BrandPositioning(
                positioning_statement=f"For {company_info.target_audience[0] if company_info.target_audience else 'businesses'}, {company_info.name} is the {company_info.industry.value} company that delivers exceptional value through innovation.",
                target_audience=company_info.target_audience or ["General market"],
                competitive_advantages=company_info.unique_selling_points or ["Quality", "Innovation", "Service"],
                brand_archetype=BrandArchetype.HERO,
                voice_attributes=["Professional", "Trustworthy", "Innovative", "Approachable", "Expert"],
                tone_guidelines={
                    "professional": "Confident and knowledgeable",
                    "social": "Friendly and engaging",
                    "crisis": "Calm and reassuring", 
                    "celebration": "Enthusiastic and grateful"
                }
            )
    
    async def _analyze_competitors(self, company_info: CompanyInfo) -> List[CompetitorAnalysis]:
        """
        Analyze competitors using AI and available data
        """
        if not company_info.competitors:
            # If no competitors specified, use AI to identify them
            competitors = await self._identify_competitors(company_info)
        else:
            competitors = company_info.competitors
        
        competitor_analyses = []
        
        for competitor in competitors[:5]:  # Limit to top 5 competitors
            analysis = await self._analyze_single_competitor(competitor, company_info)
            competitor_analyses.append(analysis)
        
        return competitor_analyses
    
    async def _identify_competitors(self, company_info: CompanyInfo) -> List[str]:
        """
        Use AI to identify main competitors if not provided
        """
        prompt = f"""
        Identify the top 5 main competitors for a {company_info.industry.value} company called {company_info.name}.
        
        Company description: {company_info.description}
        
        Please provide only the company names, one per line, without additional explanation.
        Focus on direct competitors in the same industry and market segment.
        """
        
        messages = [
            SystemMessage(content="You are a market research expert specializing in competitive analysis."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        competitors = [line.strip() for line in response.content.split('\n') if line.strip()]
        return competitors[:5]
    
    async def _analyze_single_competitor(self, competitor: str, company_info: CompanyInfo) -> CompetitorAnalysis:
        """
        Analyze a single competitor
        """
        prompt = f"""
        Analyze the competitor "{competitor}" in the {company_info.industry.value} industry.
        
        Our company: {company_info.name}
        Our description: {company_info.description}
        
        Please analyze this competitor and provide:
        1. Top 3 strengths
        2. Top 3 weaknesses  
        3. Their market position description
        4. Their apparent content strategy (if observable)
        5. Estimated engagement rate (if known, otherwise "Unknown")
        
        Format as JSON:
        {{
            "strengths": ["strength1", "strength2", "strength3"],
            "weaknesses": ["weakness1", "weakness2", "weakness3"],
            "market_position": "description of their market position",
            "content_strategy": "description of their content approach",
            "engagement_rate": null
        }}
        """
        
        messages = [
            SystemMessage(content="You are a competitive intelligence analyst with expertise in market research."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            import json
            result = json.loads(response.content)
            
            return CompetitorAnalysis(
                name=competitor,
                strengths=result["strengths"],
                weaknesses=result["weaknesses"],
                market_position=result["market_position"],
                content_strategy=result.get("content_strategy"),
                engagement_rate=result.get("engagement_rate")
            )
        except Exception:
            # Fallback analysis
            return CompetitorAnalysis(
                name=competitor,
                strengths=["Market presence", "Brand recognition", "Resources"],
                weaknesses=["Innovation gap", "Customer service", "Pricing"],
                market_position="Established player in the market",
                content_strategy="Traditional marketing approach",
                engagement_rate=None
            )
    
    async def _determine_market_positioning(self, company_info: CompanyInfo, competitive_analysis: List[CompetitorAnalysis]) -> str:
        """
        Determine overall market positioning relative to competitors
        """
        competitor_summaries = []
        for comp in competitive_analysis:
            competitor_summaries.append(f"{comp.name}: {comp.market_position}")
        
        prompt = f"""
        Based on the competitive landscape analysis, determine the market positioning for {company_info.name}.
        
        Company: {company_info.name}
        Industry: {company_info.industry.value}
        Description: {company_info.description}
        Size: {company_info.size.value}
        
        Competitor positions:
        {chr(10).join(competitor_summaries)}
        
        Provide a clear, concise description of where this company fits in the market landscape
        and their positioning relative to competitors. Focus on differentiation and market opportunity.
        """
        
        messages = [
            SystemMessage(content="You are a market positioning expert who helps companies identify their unique place in the market."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content.strip()
    
    async def _identify_differentiation_opportunities(self, company_info: CompanyInfo, competitive_analysis: List[CompetitorAnalysis]) -> List[str]:
        """
        Identify opportunities for differentiation based on competitive gaps
        """
        # Collect all competitor weaknesses
        all_weaknesses = []
        for comp in competitive_analysis:
            all_weaknesses.extend(comp.weaknesses)
        
        prompt = f"""
        Based on the competitive analysis, identify 5 key differentiation opportunities for {company_info.name}.
        
        Company strengths: {', '.join(company_info.unique_selling_points) if company_info.unique_selling_points else 'Not specified'}
        Company values: {', '.join(company_info.values) if company_info.values else 'Not specified'}
        
        Identified competitor weaknesses across the market:
        {', '.join(all_weaknesses)}
        
        Provide 5 specific, actionable differentiation opportunities that this company could pursue.
        Focus on areas where competitors are weak and the company could excel.
        
        Format as a simple list, one opportunity per line.
        """
        
        messages = [
            SystemMessage(content="You are a strategic differentiation consultant who helps companies find unique market opportunities."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        opportunities = [line.strip().lstrip('1234567890.-) ') for line in response.content.split('\n') if line.strip()]
        return opportunities[:5]
    
    async def _calculate_brand_health_score(self, company_info: CompanyInfo, brand_positioning: BrandPositioning, competitive_analysis: List[CompetitorAnalysis]) -> float:
        """
        Calculate a brand health score based on multiple factors
        """
        score = 0.0
        max_score = 1.0
        
        # Factor 1: Brand clarity (0.2 weight)
        clarity_score = 0.8 if brand_positioning.positioning_statement else 0.4
        if company_info.mission and company_info.values:
            clarity_score = min(1.0, clarity_score + 0.2)
        score += clarity_score * 0.2
        
        # Factor 2: Competitive position (0.3 weight)
        competitive_score = 0.5  # Base score
        if len(brand_positioning.competitive_advantages) >= 3:
            competitive_score += 0.3
        if len(competitive_analysis) >= 3:
            competitive_score += 0.2
        score += min(1.0, competitive_score) * 0.3
        
        # Factor 3: Target audience clarity (0.2 weight)
        audience_score = 0.3 if company_info.target_audience else 0.1
        if len(brand_positioning.target_audience) >= 2:
            audience_score += 0.4
        if len(brand_positioning.target_audience) >= 3:
            audience_score += 0.3
        score += min(1.0, audience_score) * 0.2
        
        # Factor 4: Brand differentiation (0.2 weight)
        diff_score = 0.5  # Base score
        if len(company_info.unique_selling_points) >= 2:
            diff_score += 0.3
        if company_info.unique_selling_points and len(company_info.unique_selling_points) >= 3:
            diff_score += 0.2
        score += min(1.0, diff_score) * 0.2
        
        # Factor 5: Strategic foundation (0.1 weight)
        foundation_score = 0.3
        if company_info.mission:
            foundation_score += 0.3
        if company_info.values and len(company_info.values) >= 3:
            foundation_score += 0.4
        score += min(1.0, foundation_score) * 0.1
        
        return min(1.0, score)
    
    async def _generate_recommendations(self, company_info: CompanyInfo, brand_positioning: BrandPositioning, 
                                      competitive_analysis: List[CompetitorAnalysis], brand_health_score: float) -> List[str]:
        """
        Generate strategic recommendations based on analysis
        """
        competitor_weaknesses = []
        for comp in competitive_analysis:
            competitor_weaknesses.extend(comp.weaknesses)
        
        prompt = f"""
        Based on the comprehensive brand analysis, provide 5 strategic recommendations for {company_info.name}.
        
        Current brand health score: {brand_health_score:.2f}/1.0
        
        Brand positioning: {brand_positioning.positioning_statement}
        Competitive advantages: {', '.join(brand_positioning.competitive_advantages)}
        
        Market gaps (competitor weaknesses): {', '.join(set(competitor_weaknesses))}
        
        Provide 5 specific, actionable strategic recommendations that will:
        1. Improve brand health score
        2. Strengthen market position
        3. Capitalize on competitive opportunities
        4. Enhance brand differentiation
        5. Support business growth
        
        Format as a numbered list with clear, actionable recommendations.
        """
        
        messages = [
            SystemMessage(content="You are a senior brand strategist providing actionable recommendations to improve brand performance."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        recommendations = [line.strip().lstrip('1234567890.-) ') for line in response.content.split('\n') if line.strip()]
        return recommendations[:5]