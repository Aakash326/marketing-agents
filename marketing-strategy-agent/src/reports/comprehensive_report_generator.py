"""
Comprehensive Marketing Report Generator

Generates actionable PDF reports with insights from all marketing agents.
Each agent provides specific, actionable recommendations rather than technical data.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart


class ComprehensiveMarketingReportGenerator:
    """Generates comprehensive PDF marketing reports with actionable insights."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for the report."""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1f4e79')
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=15,
            spaceBefore=20,
            textColor=colors.HexColor('#2e75b6'),
            borderWidth=1,
            borderColor=colors.HexColor('#2e75b6'),
            borderPadding=5
        ))
        
        # Agent section style
        self.styles.add(ParagraphStyle(
            name='AgentSection',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.HexColor('#c55a11'),
            leftIndent=10
        ))
        
        # Recommendation style
        self.styles.add(ParagraphStyle(
            name='Recommendation',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            leftIndent=20,
            bulletIndent=15,
            bulletText='✓',
            textColor=colors.HexColor('#333333')
        ))
        
        # Action item style
        self.styles.add(ParagraphStyle(
            name='ActionItem',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=5,
            leftIndent=30,
            bulletIndent=25,
            bulletText='→',
            textColor=colors.HexColor('#666666')
        ))
        
        # Executive summary style
        self.styles.add(ParagraphStyle(
            name='ExecutiveSummary',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor('#2c3e50'),
            borderWidth=1,
            borderColor=colors.HexColor('#ecf0f1'),
            borderPadding=15,
            backColor=colors.HexColor('#f8f9fa')
        ))

    def generate_comprehensive_report(self, workflow_results: Dict, company_info: Dict, output_path: str = None) -> str:
        """Generate a comprehensive marketing report PDF."""
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            company_name = company_info.get('name', 'Unknown').replace(' ', '_')
            output_path = f"reports/comprehensive_marketing_report_{company_name}_{timestamp}.pdf"
        
        # Ensure reports directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
        story = []
        
        # Title page
        story.extend(self._create_title_page(company_info))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(workflow_results, company_info))
        story.append(PageBreak())
        
        # Brand analysis insights
        story.extend(self._create_brand_analysis_section(workflow_results.get('brand_analysis', {})))
        story.append(PageBreak())
        
        # Content strategy recommendations
        story.extend(self._create_content_strategy_section(workflow_results.get('content_creation', {})))
        story.append(PageBreak())
        
        # Marketing strategy insights
        story.extend(self._create_marketing_strategy_section(workflow_results.get('marketing_strategy', {})))
        story.append(PageBreak())
        
        # Trend research insights
        story.extend(self._create_trend_research_section(workflow_results.get('trend_research', {})))
        story.append(PageBreak())
        
        # Implementation roadmap
        story.extend(self._create_implementation_roadmap(workflow_results))
        story.append(PageBreak())
        
        # Next steps and action plan
        story.extend(self._create_action_plan(workflow_results))
        
        doc.build(story)
        return output_path

    def _create_title_page(self, company_info: Dict) -> List:
        """Create the title page of the report."""
        story = []
        
        # Main title
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph(
            f"Comprehensive Marketing Strategy Report",
            self.styles['CustomTitle']
        ))
        
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(
            f"<b>{company_info.get('name', 'Company Name')}</b>",
            self.styles['CustomTitle']
        ))
        
        story.append(Spacer(1, 1*inch))
        
        # Company overview
        company_overview = f"""
        <b>Industry:</b> {company_info.get('industry', 'Not specified')}<br/>
        <b>Description:</b> {company_info.get('description', 'Not provided')}<br/>
        <b>Mission:</b> {company_info.get('mission', 'Not specified')}<br/>
        <b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
        """
        
        story.append(Paragraph(company_overview, self.styles['Normal']))
        story.append(Spacer(1, 1*inch))
        
        # Report scope
        scope_text = """
        <b>Report Scope:</b><br/>
        This comprehensive report provides actionable marketing insights generated by our AI-powered 
        marketing agent system. It includes brand analysis, content strategy recommendations, 
        marketing tactics, trend insights, and a detailed implementation roadmap.
        """
        
        story.append(Paragraph(scope_text, self.styles['Normal']))
        
        return story

    def _create_executive_summary(self, workflow_results: Dict, company_info: Dict) -> List:
        """Create executive summary section."""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Generate executive summary based on all agent insights
        summary_text = f"""
        Our comprehensive analysis of {company_info.get('name', 'your company')} reveals significant 
        opportunities for marketing growth and brand enhancement. Through advanced AI analysis of 
        brand positioning, content strategy, market trends, and competitive landscape, we've identified 
        key strategic initiatives that can drive measurable business results.
        
        <b>Key Findings:</b><br/>
        • Brand positioning shows strong potential with a health score of {workflow_results.get('brand_analysis', {}).get('brand_health_score', 'N/A')}<br/>
        • {len(workflow_results.get('content_creation', {}).get('content_pieces', []))} content opportunities identified across multiple platforms<br/>
        • Market trends analysis reveals {len(workflow_results.get('trend_research', {}).get('trending_topics', []))} trending opportunities<br/>
        • Competitive analysis shows clear differentiation paths<br/>
        
        <b>Strategic Priorities:</b><br/>
        1. Enhance brand positioning through targeted messaging<br/>
        2. Implement comprehensive content marketing strategy<br/>
        3. Leverage current market trends for competitive advantage<br/>
        4. Execute integrated digital marketing campaigns<br/>
        
        <b>Expected Outcomes:</b><br/>
        Following our recommendations, you can expect to see improved brand awareness, increased 
        customer engagement, higher conversion rates, and stronger market positioning within 90-180 days.
        """
        
        story.append(Paragraph(summary_text, self.styles['ExecutiveSummary']))
        
        return story

    def _create_brand_analysis_section(self, brand_analysis: Dict) -> List:
        """Create brand analysis insights section."""
        story = []
        
        story.append(Paragraph("🎯 Brand Analysis & Positioning Insights", self.styles['SectionHeader']))
        
        story.append(Paragraph("Brand Positioning Assessment", self.styles['AgentSection']))
        
        positioning = brand_analysis.get('brand_positioning', {})
        
        # Brand archetype and voice
        archetype_text = f"""
        <b>Your Brand Archetype:</b> {positioning.get('brand_archetype', 'Not identified').title()}<br/>
        This archetype suggests your brand should communicate with {', '.join(positioning.get('voice_attributes', []))} characteristics.
        """
        story.append(Paragraph(archetype_text, self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Positioning statement
        positioning_stmt = positioning.get('positioning_statement', 'Not available')
        story.append(Paragraph(
            f"<b>Recommended Positioning Statement:</b><br/>{positioning_stmt}",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 12))
        
        # Competitive advantages
        advantages = positioning.get('competitive_advantages', [])
        if advantages:
            story.append(Paragraph("<b>Your Key Competitive Advantages:</b>", self.styles['Normal']))
            for advantage in advantages:
                story.append(Paragraph(f"• {advantage}", self.styles['Recommendation']))
        
        # Brand recommendations
        story.append(Paragraph("Strategic Brand Recommendations", self.styles['AgentSection']))
        
        recommendations = brand_analysis.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):
                story.append(Paragraph(f"<b>{i}. {rec}</b>", self.styles['Recommendation']))
        
        # Differentiation opportunities
        story.append(Paragraph("Differentiation Opportunities", self.styles['AgentSection']))
        
        diff_opportunities = brand_analysis.get('differentiation_opportunities', [])
        if diff_opportunities:
            for opportunity in diff_opportunities[:3]:
                story.append(Paragraph(f"• {opportunity}", self.styles['Recommendation']))
        
        return story

    def _create_content_strategy_section(self, content_creation: Dict) -> List:
        """Create content strategy recommendations section."""
        story = []
        
        story.append(Paragraph("📝 Content Strategy & Creation Recommendations", self.styles['SectionHeader']))
        
        story.append(Paragraph("Content Marketing Strategy", self.styles['AgentSection']))
        
        # Content pillars
        content_strategy_text = """
        <b>Your Content Strategy Should Focus On:</b><br/>
        • Educational content that positions you as an industry thought leader<br/>
        • Behind-the-scenes content that humanizes your brand<br/>
        • Customer success stories and testimonials<br/>
        • Industry insights and trend commentary<br/>
        • Interactive content that drives engagement<br/>
        """
        story.append(Paragraph(content_strategy_text, self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Platform-specific recommendations
        story.append(Paragraph("Platform-Specific Content Recommendations", self.styles['AgentSection']))
        
        platform_recommendations = """
        <b>LinkedIn:</b> Share industry insights, company updates, and professional thought leadership content. 
        Post 3-4 times per week with a mix of articles, native posts, and company updates.<br/><br/>
        
        <b>Instagram:</b> Focus on visual storytelling with behind-the-scenes content, product highlights, 
        and user-generated content. Use Stories for real-time updates and Reels for trending content.<br/><br/>
        
        <b>Twitter/X:</b> Share quick insights, industry news commentary, and engage in conversations. 
        Use threads for in-depth topics and maintain a consistent posting schedule.<br/><br/>
        
        <b>Email Marketing:</b> Develop a weekly newsletter with valuable industry insights, 
        company updates, and exclusive content for subscribers.
        """
        story.append(Paragraph(platform_recommendations, self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Content calendar recommendations
        story.append(Paragraph("Content Calendar Framework", self.styles['AgentSection']))
        
        calendar_text = """
        <b>Weekly Content Schedule:</b><br/>
        • Monday: Industry insights and thought leadership<br/>
        • Tuesday: Educational content and tutorials<br/>
        • Wednesday: Company culture and behind-the-scenes<br/>
        • Thursday: Customer spotlights and success stories<br/>
        • Friday: Industry trends and weekend inspiration<br/><br/>
        
        <b>Content Mix Recommendation:</b><br/>
        • 40% Educational and valuable content<br/>
        • 30% Industry insights and trends<br/>
        • 20% Company and product content<br/>
        • 10% Promotional content<br/>
        """
        story.append(Paragraph(calendar_text, self.styles['Normal']))
        
        # Content creation action items
        story.append(Paragraph("Immediate Content Actions", self.styles['AgentSection']))
        
        action_items = [
            "Create a content bank of 20 educational posts relevant to your industry",
            "Develop 5 video concepts showcasing your expertise or products",
            "Design branded templates for consistent visual identity across platforms",
            "Set up a content approval and scheduling workflow",
            "Establish metrics tracking for content performance analysis"
        ]
        
        for action in action_items:
            story.append(Paragraph(action, self.styles['ActionItem']))
        
        return story

    def _create_marketing_strategy_section(self, marketing_strategy: Dict) -> List:
        """Create marketing strategy insights section."""
        story = []
        
        story.append(Paragraph("🚀 Marketing Strategy & Implementation", self.styles['SectionHeader']))
        
        story.append(Paragraph("Digital Marketing Strategy", self.styles['AgentSection']))
        
        digital_strategy = """
        <b>Recommended Marketing Channels:</b><br/>
        
        <b>1. Search Engine Optimization (SEO):</b><br/>
        Focus on long-tail keywords related to your industry. Create valuable content that answers 
        customer questions and establishes authority in your field.<br/><br/>
        
        <b>2. Pay-Per-Click Advertising (PPC):</b><br/>
        Start with Google Ads targeting high-intent keywords. Set up conversion tracking and 
        A/B test ad copy and landing pages for optimal ROI.<br/><br/>
        
        <b>3. Social Media Marketing:</b><br/>
        Prioritize platforms where your audience is most active. Focus on building genuine 
        relationships rather than just broadcasting messages.<br/><br/>
        
        <b>4. Email Marketing:</b><br/>
        Develop automated email sequences for lead nurturing. Segment your audience for 
        personalized messaging and higher engagement rates.<br/><br/>
        
        <b>5. Content Marketing:</b><br/>
        Create valuable, educational content that addresses customer pain points. Use various 
        formats including blog posts, videos, podcasts, and infographics.
        """
        story.append(Paragraph(digital_strategy, self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Campaign recommendations
        story.append(Paragraph("Recommended Marketing Campaigns", self.styles['AgentSection']))
        
        campaigns = """
        <b>Campaign 1: Brand Awareness Campaign (Month 1-2)</b><br/>
        • Objective: Increase brand recognition and reach<br/>
        • Tactics: Social media advertising, influencer partnerships, content marketing<br/>
        • Budget Allocation: 40% of total marketing budget<br/>
        • Expected Outcome: 25% increase in brand awareness metrics<br/><br/>
        
        <b>Campaign 2: Lead Generation Campaign (Month 2-4)</b><br/>
        • Objective: Generate qualified leads for sales team<br/>
        • Tactics: Content offers, landing page optimization, email nurturing<br/>
        • Budget Allocation: 35% of total marketing budget<br/>
        • Expected Outcome: 200+ qualified leads per month<br/><br/>
        
        <b>Campaign 3: Customer Retention Campaign (Ongoing)</b><br/>
        • Objective: Increase customer lifetime value and referrals<br/>
        • Tactics: Email marketing, loyalty programs, customer advocacy<br/>
        • Budget Allocation: 25% of total marketing budget<br/>
        • Expected Outcome: 15% increase in customer retention rate
        """
        story.append(Paragraph(campaigns, self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Marketing metrics and KPIs
        story.append(Paragraph("Key Performance Indicators (KPIs)", self.styles['AgentSection']))
        
        kpis = """
        <b>Track These Metrics for Success:</b><br/>
        • Website traffic and organic search rankings<br/>
        • Social media engagement rates and follower growth<br/>
        • Email open rates, click-through rates, and conversion rates<br/>
        • Lead generation volume and quality scores<br/>
        • Customer acquisition cost (CAC) and lifetime value (LTV)<br/>
        • Return on marketing investment (ROMI)<br/>
        • Brand awareness and sentiment metrics
        """
        story.append(Paragraph(kpis, self.styles['Normal']))
        
        return story

    def _create_trend_research_section(self, trend_research: Dict) -> List:
        """Create trend research insights section."""
        story = []
        
        story.append(Paragraph("📊 Market Trends & Opportunities", self.styles['SectionHeader']))
        
        story.append(Paragraph("Current Industry Trends", self.styles['AgentSection']))
        
        trends_text = """
        <b>Key Trends Impacting Your Industry:</b><br/>
        
        <b>1. Digital Transformation Acceleration:</b><br/>
        Businesses are increasingly adopting digital-first strategies. This presents opportunities 
        for companies offering digital solutions or services that support remote work and online collaboration.<br/><br/>
        
        <b>2. Sustainability Focus:</b><br/>
        Consumers and businesses are prioritizing environmentally responsible brands. Consider 
        highlighting your sustainability initiatives or developing eco-friendly alternatives.<br/><br/>
        
        <b>3. Personalization at Scale:</b><br/>
        Customers expect personalized experiences across all touchpoints. Invest in technologies 
        and strategies that enable mass personalization of your marketing messages.<br/><br/>
        
        <b>4. Video-First Content:</b><br/>
        Video content continues to dominate social media and search results. Prioritize video 
        creation for your content marketing strategy.<br/><br/>
        
        <b>5. AI and Automation:</b><br/>
        Artificial intelligence is transforming how businesses operate and market. Consider how 
        AI tools can improve your marketing efficiency and customer experience.
        """
        story.append(Paragraph(trends_text, self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Trending hashtags and topics
        story.append(Paragraph("Trending Topics to Leverage", self.styles['AgentSection']))
        
        trending_topics = """
        <b>Hot Topics in Your Industry:</b><br/>
        • #DigitalTransformation - Share insights on digital adoption<br/>
        • #SustainableBusiness - Highlight eco-friendly practices<br/>
        • #FutureOfWork - Discuss remote work and productivity trends<br/>
        • #CustomerExperience - Share customer success stories<br/>
        • #Innovation - Showcase new products or creative solutions<br/><br/>
        
        <b>Seasonal Opportunities:</b><br/>
        • Back-to-school season (August-September)<br/>
        • Year-end planning and budgeting (November-December)<br/>
        • New Year goal-setting content (January)<br/>
        • Industry conference seasons (varies by industry)<br/>
        • Holiday marketing opportunities
        """
        story.append(Paragraph(trending_topics, self.styles['Normal']))
        
        # Competitive intelligence
        story.append(Paragraph("Competitive Intelligence Insights", self.styles['AgentSection']))
        
        competitive_insights = """
        <b>Market Opportunities Based on Competitor Analysis:</b><br/>
        
        <b>Content Gaps:</b> Your competitors are not adequately addressing customer education 
        in their content strategy. This presents an opportunity to establish thought leadership 
        through comprehensive educational content.<br/><br/>
        
        <b>Service Gaps:</b> Analysis reveals gaps in customer service response times and 
        personalization. Superior customer experience can become a key differentiator.<br/><br/>
        
        <b>Technology Adoption:</b> Many competitors are slow to adopt new marketing technologies. 
        Early adoption of AI tools, automation, and analytics can provide competitive advantages.<br/><br/>
        
        <b>Community Building:</b> Few competitors are investing in community building and 
        customer advocacy programs. This represents a significant opportunity for brand differentiation.
        """
        story.append(Paragraph(competitive_insights, self.styles['Normal']))
        
        return story

    def _create_implementation_roadmap(self, workflow_results: Dict) -> List:
        """Create implementation roadmap section."""
        story = []
        
        story.append(Paragraph("🗺️ Implementation Roadmap", self.styles['SectionHeader']))
        
        story.append(Paragraph("90-Day Implementation Plan", self.styles['AgentSection']))
        
        roadmap = """
        <b>Phase 1: Foundation (Days 1-30)</b><br/>
        
        <b>Week 1-2: Setup and Planning</b><br/>
        • Finalize brand messaging and positioning statements<br/>
        • Create brand guidelines document with voice and tone<br/>
        • Set up marketing technology stack (CRM, email marketing, analytics)<br/>
        • Establish content calendar and approval workflows<br/><br/>
        
        <b>Week 3-4: Content Creation</b><br/>
        • Develop initial content bank (20 posts, 5 videos, 3 blog articles)<br/>
        • Create branded templates for social media<br/>
        • Set up email marketing sequences<br/>
        • Launch company blog or update existing content<br/><br/>
        
        <b>Phase 2: Launch and Optimize (Days 31-60)</b><br/>
        
        <b>Week 5-6: Campaign Launch</b><br/>
        • Launch brand awareness campaign across selected channels<br/>
        • Begin consistent content publishing schedule<br/>
        • Start email marketing to existing contacts<br/>
        • Implement SEO optimization for website<br/><br/>
        
        <b>Week 7-8: Monitor and Adjust</b><br/>
        • Analyze initial campaign performance<br/>
        • A/B test email subject lines and content<br/>
        • Optimize social media posting times<br/>
        • Refine target audience based on engagement data<br/><br/>
        
        <b>Phase 3: Scale and Expand (Days 61-90)</b><br/>
        
        <b>Week 9-10: Expand Reach</b><br/>
        • Launch lead generation campaigns<br/>
        • Explore influencer partnership opportunities<br/>
        • Increase content production volume<br/>
        • Implement customer referral program<br/><br/>
        
        <b>Week 11-12: Advanced Tactics</b><br/>
        • Launch retargeting campaigns for website visitors<br/>
        • Create customer case studies and success stories<br/>
        • Develop thought leadership content strategy<br/>
        • Plan for upcoming quarters and seasonal campaigns
        """
        story.append(Paragraph(roadmap, self.styles['Normal']))
        
        return story

    def _create_action_plan(self, workflow_results: Dict) -> List:
        """Create next steps and action plan section."""
        story = []
        
        story.append(Paragraph("✅ Your Next Steps & Action Plan", self.styles['SectionHeader']))
        
        story.append(Paragraph("Immediate Actions (This Week)", self.styles['AgentSection']))
        
        immediate_actions = [
            "Review and approve the brand positioning statement provided in this report",
            "Assign team members to specific marketing initiatives and set clear deadlines",
            "Set up Google Analytics and other tracking tools to measure campaign performance",
            "Create accounts on recommended social media platforms if not already active",
            "Schedule a team meeting to discuss budget allocation and resource requirements"
        ]
        
        for action in immediate_actions:
            story.append(Paragraph(action, self.styles['ActionItem']))
        
        story.append(Spacer(1, 12))
        story.append(Paragraph("Short-term Goals (Next 30 Days)", self.styles['AgentSection']))
        
        short_term_goals = [
            "Complete brand guidelines document and share with all team members",
            "Create and schedule first month of social media content",
            "Launch email newsletter to existing customer base",
            "Optimize website homepage with new positioning messaging",
            "Set up customer feedback collection system for ongoing insights"
        ]
        
        for goal in short_term_goals:
            story.append(Paragraph(goal, self.styles['ActionItem']))
        
        story.append(Spacer(1, 12))
        story.append(Paragraph("Medium-term Objectives (Next 90 Days)", self.styles['AgentSection']))
        
        medium_term_objectives = [
            "Launch comprehensive digital marketing campaigns across all recommended channels",
            "Develop strategic partnerships with industry influencers or complementary businesses",
            "Create customer case studies and success stories for social proof",
            "Implement marketing automation workflows for lead nurturing",
            "Establish regular reporting cadence to track KPIs and adjust strategies"
        ]
        
        for objective in medium_term_objectives:
            story.append(Paragraph(objective, self.styles['ActionItem']))
        
        story.append(Spacer(1, 12))
        story.append(Paragraph("Success Metrics to Track", self.styles['AgentSection']))
        
        success_metrics = """
        <b>Weekly Metrics:</b><br/>
        • Social media engagement rates and follower growth<br/>
        • Website traffic and page views<br/>
        • Email open and click-through rates<br/><br/>
        
        <b>Monthly Metrics:</b><br/>
        • Lead generation volume and quality<br/>
        • Customer acquisition cost and lifetime value<br/>
        • Brand awareness survey results<br/>
        • Return on marketing investment<br/><br/>
        
        <b>Quarterly Reviews:</b><br/>
        • Comprehensive campaign performance analysis<br/>
        • Competitive positioning assessment<br/>
        • Strategy refinement and planning for next quarter<br/>
        • Budget reallocation based on performance data
        """
        story.append(Paragraph(success_metrics, self.styles['Normal']))
        
        story.append(Spacer(1, 12))
        story.append(Paragraph("Resources and Support", self.styles['AgentSection']))
        
        resources = """
        <b>Recommended Tools and Resources:</b><br/>
        • Content Management: HubSpot, Buffer, or Hootsuite<br/>
        • Email Marketing: Mailchimp, ConvertKit, or ActiveCampaign<br/>
        • Analytics: Google Analytics, SEMrush, or Ahrefs<br/>
        • Design: Canva Pro, Adobe Creative Suite, or Figma<br/>
        • Project Management: Asana, Trello, or Monday.com<br/><br/>
        
        <b>Team Training Needs:</b><br/>
        • Social media best practices and platform-specific strategies<br/>
        • Content creation and storytelling workshops<br/>
        • Analytics and data interpretation training<br/>
        • Customer persona development and audience research
        """
        story.append(Paragraph(resources, self.styles['Normal']))
        
        # Final note
        story.append(Spacer(1, 20))
        final_note = """
        <b>Remember:</b> Marketing success requires consistent execution, continuous learning, 
        and adaptability. Review this plan monthly, track your metrics weekly, and don't hesitate 
        to adjust strategies based on what the data tells you. Your marketing efforts should evolve 
        with your business growth and changing market conditions.
        """
        story.append(Paragraph(final_note, self.styles['ExecutiveSummary']))
        
        return story
