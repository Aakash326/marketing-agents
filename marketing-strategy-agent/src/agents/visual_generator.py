"""
GeminiVisualGenerator - AI-powered visual content generation
"""
import asyncio
import os
import base64
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

import google.generativeai as genai
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .base_agent import BaseAgent
from ..models.data_models import (
    CompanyInfo,
    VisualContentResult,
    VisualAsset,
    BrandVisualAssets,
    AgentResponse
)


class GeminiVisualGenerator(BaseAgent):
    """
    Agent responsible for AI-powered visual content generation including:
    - Social media visual assets
    - Brand-consistent graphics
    - Logo variations and templates
    - Marketing materials
    - Visual content for campaigns
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("GeminiVisualGenerator", config)
        self.llm = ChatOpenAI(
            model=config.get("openai_model", "gpt-4o-mini"),
            temperature=config.get("temperature", 0.4),
            max_tokens=config.get("max_tokens", 2000)
        )
        
        # Initialize Gemini client
        self.gemini_api_key = config.get("gemini_api_key")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = config.get("gemini_model", "gemini-1.5-pro-latest")
            self.gemini_temperature = config.get("gemini_temperature", 0.4)
            self.gemini_max_tokens = config.get("gemini_max_tokens", 2048)
        else:
            print("Warning: Gemini API key not provided. Visual generation will use placeholder content.")
            
        self.output_directory = config.get("output_directory", "data/generated_visuals")
        os.makedirs(self.output_directory, exist_ok=True)
        
    def get_capabilities(self) -> Dict[str, str]:
        return {
            "visual_generation": "Generates AI-powered visual content using Gemini",
            "brand_consistency": "Creates brand-consistent visual assets",
            "social_media_graphics": "Designs platform-specific social media graphics",
            "logo_variations": "Creates logo variations and brand templates",
            "marketing_materials": "Designs marketing collateral and promotional materials",
            "visual_guidelines": "Develops comprehensive visual brand guidelines"
        }
    
    async def execute(self, company_info: CompanyInfo, **kwargs) -> AgentResponse:
        """
        Execute comprehensive visual content generation
        """
        try:
            # Get previous agent results for context
            previous_results = kwargs.get('previous_results', {})
            brand_analysis = previous_results.get('BrandAnalyzer', {}).get('result', {})
            content_creation = previous_results.get('ContentCreator', {}).get('result', {})
            
            # Generate visual assets for content
            generated_visuals = await self._generate_content_visuals(
                company_info, content_creation, brand_analysis
            )
            
            # Create brand visual assets
            brand_visual_assets = await self._create_brand_visual_assets(
                company_info, brand_analysis
            )
            
            # Develop visual guidelines
            visual_guidelines = await self._develop_visual_guidelines(
                company_info, brand_analysis
            )
            
            # Generate usage recommendations
            usage_recommendations = await self._generate_usage_recommendations(
                company_info, generated_visuals, brand_visual_assets
            )
            
            # Create comprehensive result
            result = VisualContentResult(
                generated_visuals=generated_visuals,
                brand_visual_assets=brand_visual_assets,
                visual_guidelines=visual_guidelines,
                usage_recommendations=usage_recommendations
            )
            
            return AgentResponse(
                agent_name=self.name,
                execution_time=0,  # Will be set by base class
                success=True,
                result=result.dict()
            )
            
        except Exception as e:
            raise Exception(f"GeminiVisualGenerator execution failed: {str(e)}")
    
    async def _generate_content_visuals(self, company_info: CompanyInfo,
                                      content_creation: Dict, brand_analysis: Dict) -> List[VisualAsset]:
        """
        Generate visual assets for social media content
        """
        generated_visuals = []
        
        # Extract content assets that need visuals
        content_assets = content_creation.get('content_assets', {})
        social_media_posts = content_assets.get('social_media_posts', [])
        reel_scripts = content_assets.get('reel_scripts', [])
        
        brand_positioning = brand_analysis.get('brand_positioning', {})
        voice_attributes = brand_positioning.get('voice_attributes', [])
        
        # Generate visuals for social media posts (reduced for performance)
        for idx, post in enumerate(social_media_posts[:2]):  # Limit to first 2 posts for speed
            if isinstance(post, dict) and post.get('visual_prompt'):
                visual_asset = await self._generate_single_visual(
                    content_id=f"post_{idx+1}",
                    prompt=post['visual_prompt'],
                    platform=post.get('platform', 'instagram'),
                    company_info=company_info,
                    brand_attributes=voice_attributes,
                    asset_type="social_media_post"
                )
                if visual_asset:
                    generated_visuals.append(visual_asset)
        
        # Generate visuals for reel scripts (reduced for performance)
        for idx, reel in enumerate(reel_scripts[:1]):  # Limit to first 1 reel for speed
            if isinstance(reel, dict) and reel.get('visual_concept'):
                visual_asset = await self._generate_single_visual(
                    content_id=f"reel_{idx+1}",
                    prompt=reel['visual_concept'],
                    platform="instagram",
                    company_info=company_info,
                    brand_attributes=voice_attributes,
                    asset_type="reel_thumbnail"
                )
                if visual_asset:
                    generated_visuals.append(visual_asset)
        
        return generated_visuals
    
    async def _generate_single_visual(self, content_id: str, prompt: str, platform: str,
                                    company_info: CompanyInfo, brand_attributes: List[str],
                                    asset_type: str) -> Optional[VisualAsset]:
        """
        Generate a single visual asset using Gemini
        """
        try:
            # Enhance the prompt with brand context
            enhanced_prompt = await self._enhance_visual_prompt(
                prompt, company_info, brand_attributes, platform, asset_type
            )
            
            # Generate image using Gemini (placeholder implementation)
            image_path = await self._generate_image_with_gemini(enhanced_prompt, content_id)
            
            return VisualAsset(
                content_id=content_id,
                asset_type=asset_type,
                image_url=image_path,
                description=f"AI-generated visual for {asset_type}",
                generation_prompt=enhanced_prompt,
                brand_elements=brand_attributes
            )
            
        except Exception as e:
            print(f"Error generating visual for {content_id}: {e}")
            return None
    
    async def _enhance_visual_prompt(self, base_prompt: str, company_info: CompanyInfo,
                                   brand_attributes: List[str], platform: str, asset_type: str) -> str:
        """
        Enhance the visual prompt with brand and platform context
        """
        # Platform-specific dimensions and requirements
        platform_specs = {
            "instagram": {
                "post": "1080x1080px square format, Instagram-optimized",
                "reel": "1080x1920px vertical format, mobile-first design"
            },
            "linkedin": {
                "post": "1200x627px professional format",
                "article": "1200x630px header image"
            },
            "twitter": {
                "post": "1200x675px Twitter card format"
            },
            "tiktok": {
                "video": "1080x1920px vertical format, attention-grabbing"
            }
        }
        
        spec = platform_specs.get(platform, {}).get(asset_type.split('_')[-1], "standard social media format")
        
        enhanced_prompt = f"""
        Create a {spec} visual for {company_info.name} in the {company_info.industry.value} industry.
        
        Visual Description: {base_prompt}
        
        Brand Context:
        - Company: {company_info.name}
        - Industry: {company_info.industry.value}
        - Brand Attributes: {', '.join(brand_attributes)}
        - Style: Modern, professional, {', '.join(brand_attributes[:3])}
        
        Platform: {platform}
        Asset Type: {asset_type}
        
        Requirements:
        - High quality, professional appearance
        - Brand consistent colors and typography
        - Optimized for {platform} platform
        - Engaging and visually appealing
        - Clear focal point and composition
        """
        
        return enhanced_prompt.strip()
    
    async def _generate_image_with_gemini(self, prompt: str, content_id: str) -> str:
        """
        Generate image description and visual concept using Gemini API
        Note: Gemini is optimized for text generation and image analysis, not image creation.
        This method generates detailed visual concepts that could be used with image generation tools.
        """
        try:
            if not self.gemini_api_key:
                # Create placeholder path for demo purposes
                placeholder_path = os.path.join(self.output_directory, f"{content_id}_placeholder.txt")
                with open(placeholder_path, 'w') as f:
                    f.write(f"No Gemini API key provided. Visual concept:\n{prompt}")
                return placeholder_path
            
            # Use Gemini to create detailed visual specifications
            model = genai.GenerativeModel(self.gemini_model)
            
            enhanced_prompt = f"""
            Create a comprehensive visual specification for this image request:
            {prompt}
            
            Please provide:
            1. Detailed visual description (composition, colors, elements, style)
            2. Technical specifications (dimensions, resolution, format recommendations)
            3. Brand consistency notes (color palette, typography, style guidelines)
            4. Platform optimization suggestions
            5. Call-to-action or engagement elements to include
            6. Specific design elements that would make this visual effective
            
            Format the response as a detailed creative brief that could be used by a designer or image generation tool.
            """
            
            # Use asyncio to run the synchronous Gemini call with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    model.generate_content,
                    enhanced_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.gemini_temperature,
                        max_output_tokens=self.gemini_max_tokens,
                        top_p=0.8,
                        top_k=40
                    )
                ),
                timeout=30.0  # 30 second timeout per Gemini call
            )
            
            # Save comprehensive visual specification
            specification_path = os.path.join(self.output_directory, f"{content_id}_visual_spec.md")
            with open(specification_path, 'w', encoding='utf-8') as f:
                f.write(f"# Visual Specification for {content_id}\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Original Request:** {prompt}\n\n")
                f.write("## Detailed Visual Specification\n\n")
                f.write(response.text)
                f.write("\n\n## Implementation Notes\n\n")
                f.write("This specification was generated by Gemini AI and provides detailed guidance ")
                f.write("for creating the requested visual asset. Use this specification with:\n")
                f.write("- Professional design tools (Photoshop, Figma, Canva)\n")
                f.write("- AI image generation tools (DALL-E, Midjourney, Stable Diffusion)\n")
                f.write("- Design agencies or freelance designers\n\n")
                f.write("The specification includes brand consistency guidelines and platform optimization suggestions.")
            
            return specification_path
            
        except asyncio.TimeoutError:
            print(f"Gemini API timeout after 30 seconds for content_id: {content_id}")
            # Create timeout-specific error report
            timeout_path = os.path.join(self.output_directory, f"{content_id}_timeout.md")
            with open(timeout_path, 'w', encoding='utf-8') as f:
                f.write(f"# Gemini API Timeout Report for {content_id}\n\n")
                f.write(f"**Error:** Gemini API call timed out after 30 seconds\n\n")
                f.write(f"**Original Request:** {prompt}\n\n")
                f.write("## Fallback Visual Concept\n\n")
                f.write("Due to API timeout, use this general visual guidance:\n")
                f.write("1. Create modern, professional visual content\n")
                f.write("2. Use brand-consistent colors and fonts\n")
                f.write("3. Ensure high contrast and readability\n")
                f.write("4. Optimize for the target platform\n")
                f.write("5. Include clear call-to-action elements\n\n")
                f.write("## Troubleshooting\n\n")
                f.write("- Gemini API may be experiencing high load\n")
                f.write("- Try again later or use alternative image generation tools\n")
                f.write("- Consider simplifying the prompt for faster processing\n")
            return timeout_path
        except Exception as e:
            print(f"Error with Gemini visual specification generation: {e}")
            
            # Create detailed error report with fallback guidance
            error_path = os.path.join(self.output_directory, f"{content_id}_error_report.md")
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"# Visual Generation Error Report for {content_id}\n\n")
                f.write(f"**Error:** {str(e)}\n\n")
                f.write(f"**Original Request:** {prompt}\n\n")
                f.write("## Fallback Visual Concept\n\n")
                f.write("Based on the original request, consider creating a visual that:\n")
                f.write("1. Aligns with brand guidelines and color palette\n")
                f.write("2. Follows platform-specific sizing requirements\n") 
                f.write("3. Includes clear, readable text elements\n")
                f.write("4. Uses high-quality, professional imagery\n")
                f.write("5. Incorporates strong visual hierarchy and composition\n\n")
                f.write("## Troubleshooting\n\n")
                f.write("- Check Gemini API key configuration\n")
                f.write("- Verify internet connectivity\n")
                f.write("- Review API rate limits and quotas\n")
                f.write("- Consider using alternative image generation tools\n")
            
            return error_path
    
    async def _create_brand_visual_assets(self, company_info: CompanyInfo,
                                        brand_analysis: Dict) -> BrandVisualAssets:
        """
        Create brand-specific visual assets
        """
        brand_positioning = brand_analysis.get('brand_positioning', {})
        voice_attributes = brand_positioning.get('voice_attributes', [])
        brand_archetype = brand_positioning.get('brand_archetype', 'hero')
        
        # Generate logo variations
        logo_variations = await self._generate_logo_variations(
            company_info, voice_attributes, brand_archetype
        )
        
        # Generate color palette applications
        color_palette_applications = await self._generate_color_palette_applications(
            company_info, voice_attributes
        )
        
        # Generate typography samples
        typography_samples = await self._generate_typography_samples(
            company_info, voice_attributes
        )
        
        # Generate template designs
        template_designs = await self._generate_template_designs(
            company_info, voice_attributes
        )
        
        return BrandVisualAssets(
            logo_variations=logo_variations,
            color_palette_applications=color_palette_applications,
            typography_samples=typography_samples,
            template_designs=template_designs
        )
    
    async def _generate_logo_variations(self, company_info: CompanyInfo,
                                      voice_attributes: List[str], brand_archetype: str) -> List[VisualAsset]:
        """
        Generate logo variations for different use cases
        """
        logo_variations = []
        
        # Reduced logo types for faster generation
        logo_types = [
            {"type": "primary", "description": "Main logo for primary brand applications"},
            {"type": "icon", "description": "Icon version for social media profiles and favicons"}
        ]
        
        for logo_type in logo_types:
            prompt = f"""
            Design a {logo_type['type']} logo variation for {company_info.name}.
            
            Company: {company_info.name}
            Industry: {company_info.industry.value}
            Brand Archetype: {brand_archetype}
            Brand Attributes: {', '.join(voice_attributes)}
            
            Logo Type: {logo_type['type']}
            Purpose: {logo_type['description']}
            
            Design a modern, professional logo that reflects the brand personality and industry.
            """
            
            asset = await self._generate_single_visual(
                content_id=f"logo_{logo_type['type']}",
                prompt=prompt,
                platform="brand",
                company_info=company_info,
                brand_attributes=voice_attributes,
                asset_type="logo"
            )
            
            if asset:
                logo_variations.append(asset)
        
        return logo_variations
    
    async def _generate_color_palette_applications(self, company_info: CompanyInfo,
                                                 voice_attributes: List[str]) -> List[VisualAsset]:
        """
        Generate color palette applications and examples
        """
        color_applications = []
        
        application_types = [
            {"type": "primary_palette", "description": "Primary brand color palette with hex codes"},
            {"type": "secondary_palette", "description": "Secondary colors for accent and variety"},
            {"type": "gradient_examples", "description": "Brand gradient applications and examples"},
            {"type": "color_combinations", "description": "Approved color combinations for different uses"}
        ]
        
        for app_type in application_types:
            prompt = f"""
            Create a color palette application showing {app_type['description']} for {company_info.name}.
            
            Company: {company_info.name}
            Industry: {company_info.industry.value}
            Brand Attributes: {', '.join(voice_attributes)}
            
            Show professional, modern colors that reflect the brand personality.
            Include color codes and usage examples.
            """
            
            asset = await self._generate_single_visual(
                content_id=f"colors_{app_type['type']}",
                prompt=prompt,
                platform="brand",
                company_info=company_info,
                brand_attributes=voice_attributes,
                asset_type="color_palette"
            )
            
            if asset:
                color_applications.append(asset)
        
        return color_applications
    
    async def _generate_typography_samples(self, company_info: CompanyInfo,
                                         voice_attributes: List[str]) -> List[VisualAsset]:
        """
        Generate typography samples and hierarchies
        """
        typography_samples = []
        
        typography_types = [
            {"type": "heading_fonts", "description": "Primary heading fonts and hierarchy"},
            {"type": "body_fonts", "description": "Body text fonts and sizes"},
            {"type": "accent_fonts", "description": "Accent fonts for special applications"}
        ]
        
        for typo_type in typography_types:
            prompt = f"""
            Create typography samples showing {typo_type['description']} for {company_info.name}.
            
            Show professional, readable fonts that match the brand personality: {', '.join(voice_attributes)}.
            Include font names, sizes, and usage examples.
            """
            
            asset = await self._generate_single_visual(
                content_id=f"typography_{typo_type['type']}",
                prompt=prompt,
                platform="brand",
                company_info=company_info,
                brand_attributes=voice_attributes,
                asset_type="typography"
            )
            
            if asset:
                typography_samples.append(asset)
        
        return typography_samples
    
    async def _generate_template_designs(self, company_info: CompanyInfo,
                                       voice_attributes: List[str]) -> List[VisualAsset]:
        """
        Generate design templates for various marketing materials
        """
        template_designs = []
        
        # Reduced template types for faster generation
        template_types = [
            {"type": "social_media_template", "description": "Template for Instagram and Facebook posts"},
            {"type": "presentation_template", "description": "PowerPoint/presentation slide template"}
        ]
        
        for template_type in template_types:
            prompt = f"""
            Design a {template_type['type']} template for {company_info.name}.
            
            Template Type: {template_type['description']}
            
            Create a professional, brand-consistent template that incorporates:
            - Company branding
            - Brand colors and typography
            - Professional layout
            - {', '.join(voice_attributes)} brand personality
            """
            
            asset = await self._generate_single_visual(
                content_id=f"template_{template_type['type']}",
                prompt=prompt,
                platform="brand",
                company_info=company_info,
                brand_attributes=voice_attributes,
                asset_type="template"
            )
            
            if asset:
                template_designs.append(asset)
        
        return template_designs
    
    async def _develop_visual_guidelines(self, company_info: CompanyInfo,
                                       brand_analysis: Dict) -> Dict[str, Any]:
        """
        Develop comprehensive visual brand guidelines
        """
        brand_positioning = brand_analysis.get('brand_positioning', {})
        voice_attributes = brand_positioning.get('voice_attributes', [])
        
        prompt = f"""
        Create comprehensive visual brand guidelines for {company_info.name}.
        
        Company: {company_info.name}
        Industry: {company_info.industry.value}
        Brand Attributes: {', '.join(voice_attributes)}
        
        Provide guidelines for:
        1. Logo usage and placement
        2. Color palette and applications
        3. Typography hierarchy and usage
        4. Photography style and tone
        5. Graphic elements and iconography
        6. Layout and composition principles
        7. Do's and don'ts for brand applications
        
        Format as structured guidelines with clear rules and examples.
        """
        
        messages = [
            SystemMessage(content="You are a brand guidelines specialist who creates comprehensive visual identity standards."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            guidelines_content = response.content.strip()
            
            # Structure the guidelines
            guidelines = {
                "logo_guidelines": "Proper logo usage, sizing, and placement rules",
                "color_guidelines": "Brand color palette and application standards",
                "typography_guidelines": "Font hierarchy and usage specifications",
                "photography_guidelines": "Photo style, tone, and content standards",
                "layout_guidelines": "Composition and design principles",
                "usage_examples": "Examples of correct brand applications",
                "restrictions": "What not to do with brand elements",
                "full_guidelines_text": guidelines_content
            }
            
            return guidelines
            
        except Exception as e:
            print(f"Error developing visual guidelines: {e}")
            return {
                "logo_guidelines": "Use logo consistently across all materials",
                "color_guidelines": "Maintain brand colors in all applications",
                "typography_guidelines": "Use approved fonts for all text",
                "photography_guidelines": "Use high-quality, professional imagery",
                "layout_guidelines": "Maintain consistent spacing and alignment",
                "usage_examples": "Follow established brand patterns",
                "restrictions": "Do not alter logo or use unapproved colors"
            }
    
    async def _generate_usage_recommendations(self, company_info: CompanyInfo,
                                            generated_visuals: List[VisualAsset],
                                            brand_visual_assets: BrandVisualAssets) -> List[str]:
        """
        Generate recommendations for using the visual assets
        """
        total_assets = len(generated_visuals) + len(brand_visual_assets.logo_variations) + \
                      len(brand_visual_assets.color_palette_applications) + \
                      len(brand_visual_assets.typography_samples) + \
                      len(brand_visual_assets.template_designs)
        
        recommendations = [
            f"Use the {len(generated_visuals)} generated content visuals for immediate social media posting",
            f"Implement the {len(brand_visual_assets.logo_variations)} logo variations across different touchpoints",
            "Apply brand colors consistently across all marketing materials",
            "Use typography samples as reference for all text-based designs",
            "Customize templates for ongoing marketing material creation",
            f"Total of {total_assets} visual assets created for comprehensive brand implementation",
            "Review visual guidelines before creating new materials",
            "Test visual assets on different platforms for optimal display",
            "Update visuals quarterly to maintain freshness and relevance",
            "Train team members on proper usage of brand visual assets"
        ]
        
        return recommendations