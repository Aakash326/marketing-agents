"""
PDF Generator Agent - Creates professional marketing strategy PDF reports
"""
import asyncio
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import logging

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
        PageBreak, Image, KeepTogether
    )
    from reportlab.platypus.flowables import HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from .base_agent import BaseAgent
from ..models.data_models import CompanyInfo, AgentResponse, ComprehensiveMarketingPackage

logger = logging.getLogger(__name__)


class PDFGeneratorAgent(BaseAgent):
    """
    Agent responsible for generating professional PDF marketing reports
    from the combined outputs of all other agents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("PDFGeneratorAgent", config)
        self.output_path = Path(config.get("reports_path", "./data/reports"))
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")
        
        self._setup_styles()
        
    def _setup_styles(self):
        """Setup custom PDF styles"""
        self.styles = getSampleStyleSheet()
        
        # Custom title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1f2937'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Custom heading styles
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=15,
            spaceBefore=20,
            textColor=colors.HexColor('#2563eb'),
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.HexColor('#374151'),
            fontName='Helvetica-Bold'
        ))
        
        # Custom body styles
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            textColor=colors.HexColor('#4b5563'),
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        # Summary box style
        self.styles.add(ParagraphStyle(
            name='SummaryBox',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=10,
            textColor=colors.HexColor('#059669'),
            fontName='Helvetica',
            backColor=colors.HexColor('#f0fdf4'),
            borderColor=colors.HexColor('#bbf7d0'),
            borderWidth=1,
            borderPadding=10
        ))
    
    def get_capabilities(self) -> Dict[str, str]:
        return {
            "pdf_generation": "Creates professional PDF marketing strategy reports",
            "multi_format_output": "Generates both PDF and text versions",
            "comprehensive_layout": "Professional formatting with tables, sections, and styling",
            "agent_synthesis": "Combines outputs from all marketing agents into cohesive report"
        }
    
    async def execute(self, company_info: CompanyInfo, **kwargs) -> AgentResponse:
        """
        Generate comprehensive PDF report from all agent results
        """
        try:
            # Get all agent results
            agent_results = kwargs.get('agent_results', {})
            
            # Generate timestamp for file naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            company_name_clean = company_info.name.replace(" ", "_").replace("/", "_").replace(".", "")
            
            # Generate PDF report
            pdf_path = await self._generate_pdf_report(
                company_info, agent_results, timestamp, company_name_clean
            )
            
            # Generate text report
            text_path = await self._generate_text_report(
                company_info, agent_results, timestamp, company_name_clean
            )
            
            result = {
                "pdf_report_path": str(pdf_path),
                "text_report_path": str(text_path),
                "company_name": company_info.name,
                "generated_at": datetime.now().isoformat(),
                "file_size_pdf": pdf_path.stat().st_size if pdf_path.exists() else 0,
                "file_size_text": text_path.stat().st_size if text_path.exists() else 0,
                "agent_results_included": list(agent_results.keys()),
                "total_pages": "calculated_after_generation"
            }
            
            return AgentResponse(
                agent_name=self.name,
                execution_time=0,  # Will be set by base class
                success=True,
                result=result
            )
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return AgentResponse(
                agent_name=self.name,
                execution_time=0,
                success=False,
                error_message=str(e)
            )
    
    async def _generate_pdf_report(self, company_info: CompanyInfo, agent_results: Dict[str, Any], 
                                 timestamp: str, company_name_clean: str) -> Path:
        """Generate the PDF report"""
        filename = f"marketing_strategy_{company_name_clean}_{timestamp}.pdf"
        filepath = self.output_path / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(filepath), 
            pagesize=letter, 
            rightMargin=72, 
            leftMargin=72, 
            topMargin=72, 
            bottomMargin=72
        )
        
        story = []
        
        # Title page
        story.extend(self._create_title_page(company_info))
        story.append(PageBreak())
        
        # Table of contents
        story.extend(self._create_table_of_contents())
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(company_info, agent_results))
        story.append(PageBreak())
        
        # Agent results sections
        if "BrandAnalyzer" in agent_results:
            story.extend(self._create_brand_analysis_section(agent_results["BrandAnalyzer"]))
            story.append(PageBreak())
        
        if "TrendResearcher" in agent_results:
            story.extend(self._create_trend_research_section(agent_results["TrendResearcher"]))
            story.append(PageBreak())
        
        if "ContentCreator" in agent_results:
            story.extend(self._create_content_strategy_section(agent_results["ContentCreator"]))
            story.append(PageBreak())
        
        if "MarketingAgent" in agent_results:
            story.extend(self._create_marketing_strategy_section(agent_results["MarketingAgent"]))
            story.append(PageBreak())
        
        if "GeminiVisualGenerator" in agent_results:
            story.extend(self._create_visual_content_section(agent_results["GeminiVisualGenerator"]))
            story.append(PageBreak())
        
        # Recommendations and next steps
        story.extend(self._create_recommendations_section(agent_results))
        
        # Build PDF
        doc.build(story)
        
        return filepath
    
    def _create_title_page(self, company_info: CompanyInfo) -> List:
        """Create professional title page"""
        story = []
        
        # Main title
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph(
            "COMPREHENSIVE MARKETING STRATEGY REPORT",
            self.styles['CustomTitle']
        ))
        
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(
            f"<b>{company_info.name}</b>",
            ParagraphStyle(
                name='CompanyTitle',
                parent=self.styles['CustomTitle'],
                fontSize=20,
                textColor=colors.HexColor('#2563eb')
            )
        ))
        
        story.append(Spacer(1, 1*inch))
        
        # Company details table
        company_data = [
            ["Industry:", company_info.industry.title()],
            ["Company Size:", company_info.size.title()],
            ["Location:", company_info.location],
            ["Analysis Date:", datetime.now().strftime("%B %d, %Y")],
            ["Generated By:", "AI Marketing Strategy System"]
        ]
        
        company_table = Table(company_data, colWidths=[2*inch, 3*inch])
        company_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0'))
        ]))
        
        story.append(company_table)
        
        story.append(Spacer(1, 1*inch))
        story.append(Paragraph(
            "This report contains proprietary analysis and strategic recommendations "
            "generated by advanced AI marketing agents. The insights provided are "
            "based on comprehensive market research, competitive analysis, and "
            "industry best practices.",
            self.styles['CustomBody']
        ))
        
        return story
    
    def _create_table_of_contents(self) -> List:
        """Create table of contents"""
        story = []
        
        story.append(Paragraph("TABLE OF CONTENTS", self.styles['CustomHeading1']))
        story.append(Spacer(1, 20))
        
        toc_data = [
            ["Executive Summary", "3"],
            ["Brand Analysis", "4"],
            ["Market Trends & Research", "5"],
            ["Content Strategy", "6"],
            ["Marketing Strategy", "7"],
            ["Visual Content Strategy", "8"],
            ["Recommendations & Next Steps", "9"]
        ]
        
        toc_table = Table(toc_data, colWidths=[5*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0'))
        ]))
        
        story.append(toc_table)
        
        return story
    
    def _create_executive_summary(self, company_info: CompanyInfo, agent_results: Dict) -> List:
        """Create executive summary from all agent results"""
        story = []
        
        story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2563eb')))
        story.append(Spacer(1, 15))
        
        # Summary overview
        summary_text = f"""
        This comprehensive marketing analysis for {company_info.name} was conducted using 
        advanced AI agents specializing in brand analysis, market research, content strategy, 
        marketing strategy, and visual content creation. The analysis provides actionable 
        insights and strategic recommendations to enhance {company_info.name}'s market position 
        and marketing effectiveness.
        """
        
        story.append(Paragraph(summary_text, self.styles['CustomBody']))
        story.append(Spacer(1, 15))
        
        # Key findings summary
        story.append(Paragraph("Key Findings:", self.styles['CustomHeading2']))
        
        findings = []
        if "BrandAnalyzer" in agent_results and agent_results["BrandAnalyzer"].get("success"):
            findings.append("• Brand positioning analysis completed with strategic recommendations")
        if "TrendResearcher" in agent_results and agent_results["TrendResearcher"].get("success"):
            findings.append("• Market trends and competitive landscape thoroughly researched")
        if "ContentCreator" in agent_results and agent_results["ContentCreator"].get("success"):
            findings.append("• Comprehensive content strategy developed across multiple platforms")
        if "MarketingAgent" in agent_results and agent_results["MarketingAgent"].get("success"):
            findings.append("• Integrated marketing strategy synthesized from all analysis")
        if "GeminiVisualGenerator" in agent_results and agent_results["GeminiVisualGenerator"].get("success"):
            findings.append("• Visual content strategy and asset recommendations provided")
        
        for finding in findings:
            story.append(Paragraph(finding, self.styles['CustomBody']))
        
        story.append(Spacer(1, 15))
        
        # Analysis metrics
        successful_agents = sum(1 for result in agent_results.values() if result.get("success", False))
        total_agents = len(agent_results)
        
        metrics_data = [
            ["Total Agents Executed", str(total_agents)],
            ["Successful Analyses", str(successful_agents)],
            ["Success Rate", f"{(successful_agents/total_agents*100):.0f}%" if total_agents > 0 else "N/A"],
            ["Analysis Completion", datetime.now().strftime("%Y-%m-%d %H:%M")]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbeafe')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#93c5fd'))
        ]))
        
        story.append(Paragraph("Analysis Overview:", self.styles['CustomHeading2']))
        story.append(metrics_table)
        
        return story
    
    def _create_brand_analysis_section(self, brand_result: Dict) -> List:
        """Create brand analysis section"""
        story = []
        
        story.append(Paragraph("BRAND ANALYSIS", self.styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2563eb')))
        story.append(Spacer(1, 15))
        
        if brand_result.get("success") and brand_result.get("result"):
            result_data = brand_result["result"]
            
            # Format the brand analysis data nicely
            formatted_content = self._format_agent_result(result_data, "brand analysis")
            story.append(Paragraph(formatted_content, self.styles['CustomBody']))
            
        else:
            story.append(Paragraph(
                "Brand analysis was not completed successfully.", 
                self.styles['CustomBody']
            ))
        
        return story
    
    def _create_trend_research_section(self, trend_result: Dict) -> List:
        """Create trend research section"""
        story = []
        
        story.append(Paragraph("MARKET TRENDS & RESEARCH", self.styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2563eb')))
        story.append(Spacer(1, 15))
        
        if trend_result.get("success") and trend_result.get("result"):
            result_data = trend_result["result"]
            formatted_content = self._format_agent_result(result_data, "market trends and research")
            story.append(Paragraph(formatted_content, self.styles['CustomBody']))
        else:
            story.append(Paragraph("Market trend research was not completed successfully.", self.styles['CustomBody']))
        
        return story
    
    def _create_content_strategy_section(self, content_result: Dict) -> List:
        """Create content strategy section"""
        story = []
        
        story.append(Paragraph("CONTENT STRATEGY", self.styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2563eb')))
        story.append(Spacer(1, 15))
        
        if content_result.get("success") and content_result.get("result"):
            result_data = content_result["result"]
            formatted_content = self._format_agent_result(result_data, "content strategy")
            story.append(Paragraph(formatted_content, self.styles['CustomBody']))
        else:
            story.append(Paragraph("Content strategy analysis was not completed successfully.", self.styles['CustomBody']))
        
        return story
    
    def _create_marketing_strategy_section(self, marketing_result: Dict) -> List:
        """Create marketing strategy section"""
        story = []
        
        story.append(Paragraph("MARKETING STRATEGY", self.styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2563eb')))
        story.append(Spacer(1, 15))
        
        if marketing_result.get("success") and marketing_result.get("result"):
            result_data = marketing_result["result"]
            formatted_content = self._format_agent_result(result_data, "marketing strategy")
            story.append(Paragraph(formatted_content, self.styles['CustomBody']))
        else:
            story.append(Paragraph("Marketing strategy synthesis was not completed successfully.", self.styles['CustomBody']))
        
        return story
    
    def _create_visual_content_section(self, visual_result: Dict) -> List:
        """Create visual content section"""
        story = []
        
        story.append(Paragraph("VISUAL CONTENT STRATEGY", self.styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2563eb')))
        story.append(Spacer(1, 15))
        
        if visual_result.get("success") and visual_result.get("result"):
            result_data = visual_result["result"]
            formatted_content = self._format_agent_result(result_data, "visual content strategy")
            story.append(Paragraph(formatted_content, self.styles['CustomBody']))
        else:
            story.append(Paragraph("Visual content strategy was not completed successfully.", self.styles['CustomBody']))
        
        return story
    
    def _create_recommendations_section(self, agent_results: Dict) -> List:
        """Create final recommendations section"""
        story = []
        
        story.append(Paragraph("RECOMMENDATIONS & NEXT STEPS", self.styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2563eb')))
        story.append(Spacer(1, 15))
        
        recommendations = [
            "1. Implement the brand positioning strategy outlined in the brand analysis section",
            "2. Execute the content strategy across recommended platforms with suggested frequency",
            "3. Monitor market trends and adjust strategies based on competitive landscape changes",
            "4. Deploy visual assets according to the visual content strategy recommendations",
            "5. Track performance metrics and iterate based on marketing strategy KPIs",
            "6. Regular review and updates every quarter to maintain competitive advantage"
        ]
        
        for rec in recommendations:
            story.append(Paragraph(rec, self.styles['CustomBody']))
            story.append(Spacer(1, 8))
        
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            "This analysis was generated using advanced AI marketing agents and should be "
            "reviewed by marketing professionals before implementation. Regular updates "
            "and monitoring are recommended for optimal results.",
            self.styles['CustomBody']
        ))
        
        return story
    
    async def _generate_text_report(self, company_info: CompanyInfo, agent_results: Dict[str, Any], 
                                  timestamp: str, company_name_clean: str) -> Path:
        """Generate text version of the report"""
        filename = f"marketing_strategy_{company_name_clean}_{timestamp}.txt"
        filepath = self.output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE MARKETING STRATEGY REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Company: {company_info.name}\n")
            f.write(f"Industry: {company_info.industry}\n")
            f.write(f"Size: {company_info.size}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "="*80 + "\n\n")
            
            # Write each agent's results
            for agent_name, result in agent_results.items():
                f.write(f"{agent_name.upper()} RESULTS\n")
                f.write("-" * 40 + "\n")
                
                if result.get("success"):
                    f.write(f"Status: SUCCESS\n")
                    f.write(f"Execution Time: {result.get('execution_time', 'N/A'):.2f}s\n")
                    f.write(f"Result:\n{str(result.get('result', 'No result data'))}\n\n")
                else:
                    f.write(f"Status: FAILED\n")
                    f.write(f"Error: {result.get('error_message', 'Unknown error')}\n\n")
                
                f.write("\n" + "="*80 + "\n\n")
        
        return filepath
    
    def _format_agent_result(self, result_data: Any, agent_type: str) -> str:
        """Format agent result data for human readability"""
        if not result_data:
            return f"No {agent_type} data available."
        
        try:
            # If it's a string, return it directly (might be formatted text)
            if isinstance(result_data, str):
                return result_data
                
            # If it's a dict, format it nicely
            if isinstance(result_data, dict):
                formatted_lines = []
                formatted_lines.append(f"<b>{agent_type.title()} Results:</b><br/><br/>")
                
                for key, value in result_data.items():
                    # Format the key to be more readable
                    readable_key = key.replace('_', ' ').title()
                    
                    if isinstance(value, list):
                        formatted_lines.append(f"<b>{readable_key}:</b><br/>")
                        for i, item in enumerate(value, 1):
                            if isinstance(item, dict):
                                formatted_lines.append(f"  {i}. {self._format_dict_item(item)}<br/>")
                            else:
                                formatted_lines.append(f"  • {str(item)}<br/>")
                        formatted_lines.append("<br/>")
                    elif isinstance(value, dict):
                        formatted_lines.append(f"<b>{readable_key}:</b><br/>")
                        formatted_lines.append(f"{self._format_dict_item(value)}<br/><br/>")
                    else:
                        formatted_lines.append(f"<b>{readable_key}:</b> {str(value)}<br/><br/>")
                
                return ''.join(formatted_lines)
            
            # If it's a list, format each item
            if isinstance(result_data, list):
                formatted_lines = []
                formatted_lines.append(f"<b>{agent_type.title()} Results:</b><br/><br/>")
                for i, item in enumerate(result_data, 1):
                    if isinstance(item, dict):
                        formatted_lines.append(f"{i}. {self._format_dict_item(item)}<br/>")
                    else:
                        formatted_lines.append(f"{i}. {str(item)}<br/>")
                return ''.join(formatted_lines)
            
            # Fallback to string representation
            return str(result_data)
            
        except Exception as e:
            return f"Error formatting {agent_type} data: {str(e)}"
    
    def _format_dict_item(self, item: dict) -> str:
        """Format a dictionary item for display"""
        if not item:
            return "No data"
        
        parts = []
        for key, value in item.items():
            readable_key = key.replace('_', ' ').title()
            if isinstance(value, list):
                value_str = ', '.join(str(v) for v in value)
            else:
                value_str = str(value)
            parts.append(f"{readable_key}: {value_str}")
        
        return '; '.join(parts)