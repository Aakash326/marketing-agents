"""
Report Generation Module for Marketing Strategy Agent
Generates comprehensive marketing reports in various formats
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
from io import BytesIO
import base64

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.platypus.flowables import HRFlowable
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from ..models.data_models import ComprehensiveMarketingPackage, CompanyInfo
from ..config.settings import load_config

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive marketing reports in multiple formats"""
    
    def __init__(self, output_path: Optional[str] = None):
        self.config = load_config()
        self.output_path = Path(output_path) if output_path else Path(self.config.get("REPORTS_PATH", "./data/reports"))
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup styles for PDF generation
        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for PDF generation"""
        if not REPORTLAB_AVAILABLE:
            return
            
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2563eb')
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=12,
            textColor=colors.HexColor('#1f2937'),
            borderWidth=0,
            borderColor=colors.HexColor('#e5e7eb'),
            borderPadding=0
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            textColor=colors.HexColor('#374151')
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            textColor=colors.HexColor('#4b5563')
        ))
    
    async def generate_comprehensive_report(
        self, 
        marketing_package: ComprehensiveMarketingPackage,
        format_type: str = "pdf",
        include_sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive marketing report
        
        Args:
            marketing_package: The marketing analysis results
            format_type: Output format ("pdf", "html", "json", "markdown")
            include_sections: List of sections to include (None for all)
            
        Returns:
            Dict with file_path, file_name, and metadata
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            company_name = marketing_package.company_info.name.replace(" ", "_").replace("/", "_")
            base_filename = f"marketing_report_{company_name}_{timestamp}"
            
            if format_type.lower() == "pdf":
                return await self._generate_pdf_report(marketing_package, base_filename, include_sections)
            elif format_type.lower() == "html":
                return await self._generate_html_report(marketing_package, base_filename, include_sections)
            elif format_type.lower() == "json":
                return await self._generate_json_report(marketing_package, base_filename, include_sections)
            elif format_type.lower() == "markdown":
                return await self._generate_markdown_report(marketing_package, base_filename, include_sections)
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
                
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    async def _generate_pdf_report(
        self, 
        marketing_package: ComprehensiveMarketingPackage,
        base_filename: str,
        include_sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate PDF report"""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")
        
        filename = f"{base_filename}.pdf"
        filepath = self.output_path / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(filepath), pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        story = []
        
        # Title page
        story.extend(self._create_pdf_title_page(marketing_package))
        story.append(PageBreak())
        
        # Executive summary
        if not include_sections or "executive_summary" in include_sections:
            story.extend(self._create_pdf_executive_summary(marketing_package))
            story.append(PageBreak())
        
        # Brand analysis
        if not include_sections or "brand_analysis" in include_sections:
            story.extend(self._create_pdf_brand_analysis(marketing_package))
            story.append(PageBreak())
        
        # Market trends
        if not include_sections or "trend_research" in include_sections:
            story.extend(self._create_pdf_trend_research(marketing_package))
            story.append(PageBreak())
        
        # Content strategy
        if not include_sections or "content_strategy" in include_sections:
            story.extend(self._create_pdf_content_strategy(marketing_package))
            story.append(PageBreak())
        
        # Marketing strategy
        if not include_sections or "marketing_strategy" in include_sections:
            story.extend(self._create_pdf_marketing_strategy(marketing_package))
            story.append(PageBreak())
        
        # Visual content
        if not include_sections or "visual_content" in include_sections:
            story.extend(self._create_pdf_visual_content(marketing_package))
        
        # Build PDF
        doc.build(story)
        
        return {
            "file_path": str(filepath),
            "file_name": filename,
            "format": "pdf",
            "size_bytes": filepath.stat().st_size,
            "generated_at": datetime.now().isoformat(),
            "sections_included": include_sections or ["all"]
        }
    
    def _create_pdf_title_page(self, marketing_package: ComprehensiveMarketingPackage) -> List:
        """Create PDF title page"""
        story = []
        
        # Title
        title = Paragraph(
            f"Marketing Strategy Report<br/>{marketing_package.company_info.name}",
            self.styles['CustomTitle']
        )
        story.append(title)
        story.append(Spacer(1, 0.3*inch))
        
        # Company info table
        company_data = [
            ["Company:", marketing_package.company_info.name],
            ["Industry:", marketing_package.company_info.industry],
            ["Size:", marketing_package.company_info.size],
            ["Location:", marketing_package.company_info.location],
            ["Generated:", datetime.now().strftime("%B %d, %Y at %I:%M %p")]
        ]
        
        company_table = Table(company_data, colWidths=[2*inch, 4*inch])
        company_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        story.append(company_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Report description
        description = Paragraph(
            "This comprehensive marketing strategy report was generated using AI-powered analysis "
            "of your brand, competitive landscape, market trends, and industry best practices. "
            "The report includes actionable insights and recommendations to enhance your marketing effectiveness.",
            self.styles['CustomBody']
        )
        story.append(description)
        
        return story
    
    def _create_pdf_executive_summary(self, marketing_package: ComprehensiveMarketingPackage) -> List:
        """Create PDF executive summary section"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e5e7eb')))
        story.append(Spacer(1, 12))
        
        # Key insights from marketing strategy
        if marketing_package.marketing_strategy and hasattr(marketing_package.marketing_strategy, 'executive_summary'):
            summary_text = marketing_package.marketing_strategy.executive_summary
        else:
            summary_text = f"""
            This report provides a comprehensive analysis of {marketing_package.company_info.name}'s 
            marketing opportunities and strategic recommendations. Our AI-powered analysis examined 
            brand positioning, competitive landscape, market trends, and content opportunities to 
            deliver actionable insights for marketing success.
            """
        
        story.append(Paragraph(summary_text, self.styles['CustomBody']))
        story.append(Spacer(1, 20))
        
        # Key metrics summary
        if marketing_package.workflow_metadata:
            metrics_data = [
                ["Analysis Duration", f"{marketing_package.workflow_metadata.get('execution_time_seconds', 0):.1f} seconds"],
                ["Agents Executed", str(marketing_package.workflow_metadata.get('total_agents', 'N/A'))],
                ["Success Rate", f"{(marketing_package.workflow_metadata.get('success_count', 0) / marketing_package.workflow_metadata.get('total_agents', 1) * 100):.0f}%"],
                ["Execution Mode", marketing_package.workflow_metadata.get('execution_mode', 'hybrid').title()]
            ]
            
            metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1f2937')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb'))
            ]))
            
            story.append(Paragraph("Analysis Metrics", self.styles['CustomHeading2']))
            story.append(metrics_table)
        
        return story
    
    def _create_pdf_brand_analysis(self, marketing_package: ComprehensiveMarketingPackage) -> List:
        """Create PDF brand analysis section"""
        story = []
        
        story.append(Paragraph("Brand Analysis", self.styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e5e7eb')))
        story.append(Spacer(1, 12))
        
        if marketing_package.brand_analysis:
            # Brand positioning
            if hasattr(marketing_package.brand_analysis, 'brand_positioning'):
                story.append(Paragraph("Brand Positioning", self.styles['CustomHeading2']))
                positioning_text = str(marketing_package.brand_analysis.brand_positioning)
                story.append(Paragraph(positioning_text, self.styles['CustomBody']))
                story.append(Spacer(1, 12))
            
            # Competitive analysis
            if hasattr(marketing_package.brand_analysis, 'competitive_analysis'):
                story.append(Paragraph("Competitive Analysis", self.styles['CustomHeading2']))
                competitive_text = str(marketing_package.brand_analysis.competitive_analysis)
                story.append(Paragraph(competitive_text, self.styles['CustomBody']))
                story.append(Spacer(1, 12))
            
            # Brand health score
            if hasattr(marketing_package.brand_analysis, 'brand_health_score'):
                story.append(Paragraph("Brand Health Assessment", self.styles['CustomHeading2']))
                health_text = f"Brand Health Score: {marketing_package.brand_analysis.brand_health_score}/100"
                story.append(Paragraph(health_text, self.styles['CustomBody']))
        else:
            story.append(Paragraph("Brand analysis data not available.", self.styles['CustomBody']))
        
        return story
    
    def _create_pdf_trend_research(self, marketing_package: ComprehensiveMarketingPackage) -> List:
        """Create PDF trend research section"""
        story = []
        
        story.append(Paragraph("Market Trends & Research", self.styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e5e7eb')))
        story.append(Spacer(1, 12))
        
        if marketing_package.trend_research:
            trend_text = str(marketing_package.trend_research)
            story.append(Paragraph(trend_text, self.styles['CustomBody']))
        else:
            story.append(Paragraph("Trend research data not available.", self.styles['CustomBody']))
        
        return story
    
    def _create_pdf_content_strategy(self, marketing_package: ComprehensiveMarketingPackage) -> List:
        """Create PDF content strategy section"""
        story = []
        
        story.append(Paragraph("Content Strategy", self.styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e5e7eb')))
        story.append(Spacer(1, 12))
        
        if marketing_package.content_creation:
            content_text = str(marketing_package.content_creation)
            story.append(Paragraph(content_text, self.styles['CustomBody']))
        else:
            story.append(Paragraph("Content strategy data not available.", self.styles['CustomBody']))
        
        return story
    
    def _create_pdf_marketing_strategy(self, marketing_package: ComprehensiveMarketingPackage) -> List:
        """Create PDF marketing strategy section"""
        story = []
        
        story.append(Paragraph("Marketing Strategy", self.styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e5e7eb')))
        story.append(Spacer(1, 12))
        
        if marketing_package.marketing_strategy:
            strategy_text = str(marketing_package.marketing_strategy)
            story.append(Paragraph(strategy_text, self.styles['CustomBody']))
        else:
            story.append(Paragraph("Marketing strategy data not available.", self.styles['CustomBody']))
        
        return story
    
    def _create_pdf_visual_content(self, marketing_package: ComprehensiveMarketingPackage) -> List:
        """Create PDF visual content section"""
        story = []
        
        story.append(Paragraph("Visual Content & Assets", self.styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e5e7eb')))
        story.append(Spacer(1, 12))
        
        if marketing_package.visual_content:
            visual_text = str(marketing_package.visual_content)
            story.append(Paragraph(visual_text, self.styles['CustomBody']))
        else:
            story.append(Paragraph("Visual content data not available.", self.styles['CustomBody']))
        
        return story
    
    async def _generate_html_report(
        self, 
        marketing_package: ComprehensiveMarketingPackage,
        base_filename: str,
        include_sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate HTML report"""
        filename = f"{base_filename}.html"
        filepath = self.output_path / filename
        
        html_content = self._create_html_template(marketing_package, include_sections)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return {
            "file_path": str(filepath),
            "file_name": filename,
            "format": "html",
            "size_bytes": filepath.stat().st_size,
            "generated_at": datetime.now().isoformat(),
            "sections_included": include_sections or ["all"]
        }
    
    def _create_html_template(self, marketing_package: ComprehensiveMarketingPackage, include_sections: Optional[List[str]] = None) -> str:
        """Create HTML template for the report"""
        sections = []
        
        # Title and company info
        sections.append(f"""
        <div class="hero-section">
            <h1>Marketing Strategy Report</h1>
            <h2>{marketing_package.company_info.name}</h2>
            <div class="company-info">
                <p><strong>Industry:</strong> {marketing_package.company_info.industry}</p>
                <p><strong>Size:</strong> {marketing_package.company_info.size}</p>
                <p><strong>Location:</strong> {marketing_package.company_info.location}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
        </div>
        """)
        
        # Add sections based on include_sections
        if not include_sections or "brand_analysis" in include_sections:
            sections.append(self._create_html_section("Brand Analysis", marketing_package.brand_analysis))
        
        if not include_sections or "trend_research" in include_sections:
            sections.append(self._create_html_section("Market Trends & Research", marketing_package.trend_research))
        
        if not include_sections or "content_strategy" in include_sections:
            sections.append(self._create_html_section("Content Strategy", marketing_package.content_creation))
        
        if not include_sections or "marketing_strategy" in include_sections:
            sections.append(self._create_html_section("Marketing Strategy", marketing_package.marketing_strategy))
        
        if not include_sections or "visual_content" in include_sections:
            sections.append(self._create_html_section("Visual Content & Assets", marketing_package.visual_content))
        
        # Combine with HTML template
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Marketing Strategy Report - {marketing_package.company_info.name}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .hero-section {{ text-align: center; margin-bottom: 40px; padding-bottom: 30px; border-bottom: 2px solid #e9ecef; }}
                h1 {{ color: #2563eb; font-size: 2.5em; margin-bottom: 10px; }}
                h2 {{ color: #1f2937; font-size: 2em; margin-bottom: 20px; }}
                h3 {{ color: #374151; font-size: 1.5em; margin-top: 30px; margin-bottom: 15px; }}
                .company-info {{ background: #f8f9fa; padding: 20px; border-radius: 6px; margin-top: 20px; }}
                .company-info p {{ margin: 5px 0; }}
                .section {{ margin-bottom: 30px; padding: 20px; border-left: 4px solid #2563eb; background: #f8f9fa; }}
                .content {{ line-height: 1.6; color: #4b5563; }}
                pre {{ background: #f1f3f4; padding: 15px; border-radius: 4px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                {''.join(sections)}
            </div>
        </body>
        </html>
        """
    
    def _create_html_section(self, title: str, content: Any) -> str:
        """Create HTML section"""
        content_str = str(content) if content else "Data not available."
        return f"""
        <div class="section">
            <h3>{title}</h3>
            <div class="content">
                <pre>{content_str}</pre>
            </div>
        </div>
        """
    
    async def _generate_json_report(
        self, 
        marketing_package: ComprehensiveMarketingPackage,
        base_filename: str,
        include_sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate JSON report"""
        filename = f"{base_filename}.json"
        filepath = self.output_path / filename
        
        # Convert to dict and filter sections
        report_data = marketing_package.dict()
        report_data["generated_at"] = datetime.now().isoformat()
        report_data["report_metadata"] = {
            "format": "json",
            "sections_included": include_sections or ["all"],
            "generator_version": "1.0.0"
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return {
            "file_path": str(filepath),
            "file_name": filename,
            "format": "json",
            "size_bytes": filepath.stat().st_size,
            "generated_at": datetime.now().isoformat(),
            "sections_included": include_sections or ["all"]
        }
    
    async def _generate_markdown_report(
        self, 
        marketing_package: ComprehensiveMarketingPackage,
        base_filename: str,
        include_sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate Markdown report"""
        filename = f"{base_filename}.md"
        filepath = self.output_path / filename
        
        markdown_content = self._create_markdown_content(marketing_package, include_sections)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return {
            "file_path": str(filepath),
            "file_name": filename,
            "format": "markdown",
            "size_bytes": filepath.stat().st_size,
            "generated_at": datetime.now().isoformat(),
            "sections_included": include_sections or ["all"]
        }
    
    def _create_markdown_content(self, marketing_package: ComprehensiveMarketingPackage, include_sections: Optional[List[str]] = None) -> str:
        """Create Markdown content for the report"""
        sections = []
        
        # Title
        sections.append(f"# Marketing Strategy Report: {marketing_package.company_info.name}")
        sections.append(f"*Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*\n")
        
        # Company Information
        sections.append("## Company Information")
        sections.append(f"- **Company:** {marketing_package.company_info.name}")
        sections.append(f"- **Industry:** {marketing_package.company_info.industry}")
        sections.append(f"- **Size:** {marketing_package.company_info.size}")
        sections.append(f"- **Location:** {marketing_package.company_info.location}")
        if marketing_package.company_info.website:
            sections.append(f"- **Website:** {marketing_package.company_info.website}")
        sections.append("")
        
        # Add sections
        if not include_sections or "brand_analysis" in include_sections:
            sections.append("## Brand Analysis")
            sections.append(str(marketing_package.brand_analysis) if marketing_package.brand_analysis else "Data not available.")
            sections.append("")
        
        if not include_sections or "trend_research" in include_sections:
            sections.append("## Market Trends & Research")
            sections.append(str(marketing_package.trend_research) if marketing_package.trend_research else "Data not available.")
            sections.append("")
        
        if not include_sections or "content_strategy" in include_sections:
            sections.append("## Content Strategy")
            sections.append(str(marketing_package.content_creation) if marketing_package.content_creation else "Data not available.")
            sections.append("")
        
        if not include_sections or "marketing_strategy" in include_sections:
            sections.append("## Marketing Strategy")
            sections.append(str(marketing_package.marketing_strategy) if marketing_package.marketing_strategy else "Data not available.")
            sections.append("")
        
        if not include_sections or "visual_content" in include_sections:
            sections.append("## Visual Content & Assets")
            sections.append(str(marketing_package.visual_content) if marketing_package.visual_content else "Data not available.")
            sections.append("")
        
        return "\n".join(sections)
    
    async def list_reports(self) -> List[Dict[str, Any]]:
        """List all generated reports"""
        reports = []
        
        for file_path in self.output_path.glob("marketing_report_*"):
            if file_path.is_file():
                stat = file_path.stat()
                reports.append({
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "format": file_path.suffix[1:] if file_path.suffix else "unknown"
                })
        
        # Sort by creation time, newest first
        reports.sort(key=lambda x: x["created_at"], reverse=True)
        return reports
    
    async def delete_report(self, file_name: str) -> bool:
        """Delete a specific report file"""
        try:
            file_path = self.output_path / file_name
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting report {file_name}: {e}")
            return False


# Convenience functions
async def generate_report(
    marketing_package: ComprehensiveMarketingPackage,
    format_type: str = "pdf",
    output_path: Optional[str] = None,
    include_sections: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate a marketing report
    
    Args:
        marketing_package: The marketing analysis results
        format_type: Output format ("pdf", "html", "json", "markdown")
        output_path: Optional custom output directory
        include_sections: Optional list of sections to include
        
    Returns:
        Dict with file information
    """
    generator = ReportGenerator(output_path)
    return await generator.generate_comprehensive_report(
        marketing_package, format_type, include_sections
    )