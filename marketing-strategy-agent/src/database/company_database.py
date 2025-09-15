"""
Company Database Management System
Handles pre-loaded company database and custom company input
"""
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from ..models.data_models import CompanyInfo, Industry, CompanySize


class CompanyDatabase:
    """
    Manages company data including pre-loaded companies and custom entries
    """
    
    def __init__(self, data_path: str = "data/companies"):
        self.data_path = data_path
        self.companies: Dict[str, CompanyInfo] = {}
        self.company_file = os.path.join(data_path, "companies.json")
        
        # Ensure data directory exists
        os.makedirs(data_path, exist_ok=True)
        
        # Load existing companies
        self._load_companies()
        
        # If no companies exist, create initial database
        if not self.companies:
            self._create_initial_database()
    
    def _load_companies(self):
        """Load companies from JSON file"""
        try:
            if os.path.exists(self.company_file):
                with open(self.company_file, 'r') as f:
                    companies_data = json.load(f)
                    
                for company_id, company_data in companies_data.items():
                    try:
                        # Convert dict back to CompanyInfo object
                        self.companies[company_id] = CompanyInfo(**company_data)
                    except Exception as e:
                        print(f"Error loading company {company_id}: {e}")
                        continue
                        
                print(f"Loaded {len(self.companies)} companies from database")
        except Exception as e:
            print(f"Error loading companies database: {e}")
    
    def _save_companies(self):
        """Save companies to JSON file"""
        try:
            companies_data = {}
            for company_id, company in self.companies.items():
                companies_data[company_id] = company.dict()
            
            with open(self.company_file, 'w') as f:
                json.dump(companies_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error saving companies database: {e}")
    
    def _create_initial_database(self):
        """Create initial database with major companies"""
        initial_companies = [
            {
                "name": "Apple Inc.",
                "industry": Industry.TECHNOLOGY,
                "size": CompanySize.ENTERPRISE,
                "description": "Technology company that designs, develops, and sells consumer electronics, computer software, and online services.",
                "mission": "To bring the best personal computing experience to students, educators, creative professionals and consumers around the world through its innovative hardware, software and Internet offerings.",
                "values": ["Innovation", "Quality", "Simplicity", "Privacy", "Environmental Responsibility"],
                "website": "https://www.apple.com",
                "target_audience": ["Creative professionals", "Tech enthusiasts", "Premium consumers"],
                "competitors": ["Samsung", "Google", "Microsoft", "Amazon"],
                "unique_selling_points": ["Premium design", "Integrated ecosystem", "Privacy focus", "Brand prestige"],
                "geographic_markets": ["North America", "Europe", "Asia Pacific", "China"]
            },
            {
                "name": "Tesla, Inc.",
                "industry": Industry.AUTOMOTIVE,
                "size": CompanySize.LARGE,
                "description": "Electric vehicle and clean energy company that designs, manufactures, and sells electric cars, energy generation and storage systems.",
                "mission": "To accelerate the world's transition to sustainable transport and energy.",
                "values": ["Sustainability", "Innovation", "Efficiency", "Safety", "Accessibility"],
                "website": "https://www.tesla.com",
                "target_audience": ["Environmentally conscious consumers", "Tech-savvy car buyers", "Early adopters"],
                "competitors": ["Ford", "General Motors", "Volkswagen", "BYD"],
                "unique_selling_points": ["Leading EV technology", "Autopilot features", "Supercharger network", "Direct sales model"],
                "geographic_markets": ["North America", "Europe", "China", "Australia"]
            },
            {
                "name": "Shopify Inc.",
                "industry": Industry.TECHNOLOGY,
                "size": CompanySize.LARGE,
                "description": "E-commerce platform that provides a suite of services including payments, marketing, shipping and customer engagement tools.",
                "mission": "To make commerce better for everyone by democratizing commerce for small businesses.",
                "values": ["Entrepreneurship", "Innovation", "Inclusivity", "Sustainability", "Customer Success"],
                "website": "https://www.shopify.com",
                "target_audience": ["Small business owners", "Entrepreneurs", "E-commerce merchants"],
                "competitors": ["WooCommerce", "BigCommerce", "Magento", "Square"],
                "unique_selling_points": ["Easy setup", "Comprehensive features", "App ecosystem", "Scalability"],
                "geographic_markets": ["North America", "Europe", "Asia Pacific", "Latin America"]
            },
            {
                "name": "Airbnb, Inc.",
                "industry": Industry.TRAVEL,
                "size": CompanySize.LARGE,
                "description": "Online marketplace for short-term homestays and experiences in various countries and regions.",
                "mission": "To create a world where anyone can belong anywhere.",
                "values": ["Belonging", "Trust", "Inclusion", "Community", "Authenticity"],
                "website": "https://www.airbnb.com",
                "target_audience": ["Travelers", "Property owners", "Experience seekers"],
                "competitors": ["Booking.com", "Expedia", "VRBO", "Hotels.com"],
                "unique_selling_points": ["Unique accommodations", "Local experiences", "Community-driven", "Global reach"],
                "geographic_markets": ["Global presence in 220+ countries"]
            },
            {
                "name": "Zoom Video Communications",
                "industry": Industry.TECHNOLOGY,
                "size": CompanySize.LARGE,
                "description": "Video conferencing and communications platform providing cloud-based peer-to-peer software platform.",
                "mission": "To make video communications frictionless and secure.",
                "values": ["Care", "Community", "Courage", "Curiosity", "Collaboration"],
                "website": "https://www.zoom.us",
                "target_audience": ["Businesses", "Educational institutions", "Healthcare providers", "Remote workers"],
                "competitors": ["Microsoft Teams", "Google Meet", "Skype", "Cisco Webex"],
                "unique_selling_points": ["Reliable video quality", "Easy to use", "Scalable", "Cross-platform"],
                "geographic_markets": ["Global", "North America", "Europe", "Asia Pacific"]
            },
            {
                "name": "Peloton Interactive",
                "industry": Industry.ENTERTAINMENT,
                "size": CompanySize.MEDIUM,
                "description": "Interactive fitness platform that provides connected fitness equipment and virtual fitness classes.",
                "mission": "To empower people to be the best version of themselves anywhere, anytime.",
                "values": ["Excellence", "Empowerment", "Community", "Innovation", "Authenticity"],
                "website": "https://www.onepeloton.com",
                "target_audience": ["Fitness enthusiasts", "Busy professionals", "Health-conscious consumers"],
                "competitors": ["NordicTrack", "Mirror", "SoulCycle", "Equinox"],
                "unique_selling_points": ["At-home fitness experience", "Live and on-demand classes", "Community features", "Premium equipment"],
                "geographic_markets": ["North America", "Europe", "Australia"]
            },
            {
                "name": "Stripe, Inc.",
                "industry": Industry.FINANCE,
                "size": CompanySize.LARGE,
                "description": "Financial services and software as a service company that provides payment processing software and APIs for e-commerce websites and mobile applications.",
                "mission": "To increase the GDP of the internet.",
                "values": ["Focus on users", "Optimism", "Rigor", "Craft", "Transparency"],
                "website": "https://www.stripe.com",
                "target_audience": ["Developers", "E-commerce businesses", "Online platforms", "Startups"],
                "competitors": ["PayPal", "Square", "Adyen", "Braintree"],
                "unique_selling_points": ["Developer-friendly APIs", "Global reach", "Advanced fraud prevention", "Easy integration"],
                "geographic_markets": ["Global", "North America", "Europe", "Asia Pacific"]
            }
        ]
        
        for company_data in initial_companies:
            company = CompanyInfo(**company_data)
            company_id = self._generate_company_id(company.name)
            self.companies[company_id] = company
        
        self._save_companies()
        print(f"Created initial database with {len(initial_companies)} companies")
    
    def _generate_company_id(self, company_name: str) -> str:
        """Generate a unique company ID from company name"""
        return company_name.lower().replace(' ', '_').replace(',', '').replace('.', '')
    
    def add_company(self, company: CompanyInfo) -> str:
        """Add a new company to the database"""
        company_id = self._generate_company_id(company.name)
        
        # Check if company already exists
        if company_id in self.companies:
            raise ValueError(f"Company {company.name} already exists in database")
        
        self.companies[company_id] = company
        self._save_companies()
        
        return company_id
    
    def update_company(self, company_id: str, company: CompanyInfo) -> bool:
        """Update an existing company"""
        if company_id not in self.companies:
            return False
        
        self.companies[company_id] = company
        self._save_companies()
        return True
    
    def get_company(self, company_id: str) -> Optional[CompanyInfo]:
        """Get a company by ID"""
        return self.companies.get(company_id)
    
    def search_companies(self, query: str) -> List[Dict[str, Any]]:
        """Search companies by name or industry"""
        results = []
        query_lower = query.lower()
        
        for company_id, company in self.companies.items():
            if (query_lower in company.name.lower() or 
                query_lower in company.industry.value.lower() or
                query_lower in company.description.lower()):
                
                results.append({
                    "id": company_id,
                    "name": company.name,
                    "industry": company.industry.value,
                    "size": company.size.value,
                    "description": company.description[:150] + "..." if len(company.description) > 150 else company.description
                })
        
        return results
    
    def list_companies(self, industry: Optional[Industry] = None, 
                      size: Optional[CompanySize] = None,
                      limit: int = 50) -> List[Dict[str, Any]]:
        """List companies with optional filters"""
        results = []
        
        for company_id, company in self.companies.items():
            # Apply filters
            if industry and company.industry != industry:
                continue
            if size and company.size != size:
                continue
            
            results.append({
                "id": company_id,
                "name": company.name,
                "industry": company.industry.value,
                "size": company.size.value,
                "description": company.description[:150] + "..." if len(company.description) > 150 else company.description,
                "target_audience": company.target_audience[:3] if company.target_audience else []
            })
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_companies_by_industry(self, industry: Industry) -> List[Dict[str, Any]]:
        """Get all companies in a specific industry"""
        return self.list_companies(industry=industry)
    
    def delete_company(self, company_id: str) -> bool:
        """Delete a company from the database"""
        if company_id not in self.companies:
            return False
        
        del self.companies[company_id]
        self._save_companies()
        return True
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database"""
        if not self.companies:
            return {
                "total_companies": 0,
                "by_industry": {},
                "by_size": {}
            }
        
        by_industry = {}
        by_size = {}
        
        for company in self.companies.values():
            # Count by industry
            industry = company.industry.value
            by_industry[industry] = by_industry.get(industry, 0) + 1
            
            # Count by size
            size = company.size.value
            by_size[size] = by_size.get(size, 0) + 1
        
        return {
            "total_companies": len(self.companies),
            "by_industry": by_industry,
            "by_size": by_size,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the company database"""
        if backup_path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.data_path, f"companies_backup_{timestamp}.json")
        
        try:
            companies_data = {}
            for company_id, company in self.companies.items():
                companies_data[company_id] = company.dict()
            
            with open(backup_path, 'w') as f:
                json.dump(companies_data, f, indent=2, default=str)
            
            return backup_path
        except Exception as e:
            raise Exception(f"Failed to create backup: {e}")
    
    def restore_database(self, backup_path: str) -> int:
        """Restore database from backup"""
        try:
            with open(backup_path, 'r') as f:
                companies_data = json.load(f)
            
            restored_count = 0
            for company_id, company_data in companies_data.items():
                try:
                    company = CompanyInfo(**company_data)
                    self.companies[company_id] = company
                    restored_count += 1
                except Exception as e:
                    print(f"Error restoring company {company_id}: {e}")
                    continue
            
            self._save_companies()
            return restored_count
            
        except Exception as e:
            raise Exception(f"Failed to restore backup: {e}")


class CompanySelector:
    """
    Interface for selecting and managing company information for workflows
    """
    
    def __init__(self, company_db: CompanyDatabase):
        self.company_db = company_db
    
    async def select_company(self, selection_method: str, **kwargs) -> CompanyInfo:
        """
        Select a company using different methods
        
        Args:
            selection_method: "by_id", "by_search", "custom", "random"
            **kwargs: Method-specific parameters
            
        Returns:
            CompanyInfo object
        """
        if selection_method == "by_id":
            return await self._select_by_id(kwargs.get("company_id"))
        elif selection_method == "by_search":
            return await self._select_by_search(kwargs.get("query"))
        elif selection_method == "custom":
            return await self._create_custom_company(kwargs)
        elif selection_method == "random":
            return await self._select_random_company(kwargs.get("industry"), kwargs.get("size"))
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
    
    async def _select_by_id(self, company_id: str) -> CompanyInfo:
        """Select company by ID"""
        if not company_id:
            raise ValueError("Company ID is required")
        
        company = self.company_db.get_company(company_id)
        if not company:
            raise ValueError(f"Company with ID {company_id} not found")
        
        return company
    
    async def _select_by_search(self, query: str) -> CompanyInfo:
        """Select company by search query"""
        if not query:
            raise ValueError("Search query is required")
        
        results = self.company_db.search_companies(query)
        if not results:
            raise ValueError(f"No companies found matching '{query}'")
        
        # Return the first result's full company info
        first_result = results[0]
        return self.company_db.get_company(first_result["id"])
    
    async def _create_custom_company(self, company_data: Dict[str, Any]) -> CompanyInfo:
        """Create a custom company from provided data"""
        try:
            # Validate required fields
            required_fields = ["name", "industry", "size", "description"]
            for field in required_fields:
                if field not in company_data:
                    raise ValueError(f"Required field '{field}' is missing")
            
            # Convert industry and size strings to enums if needed
            if isinstance(company_data["industry"], str):
                company_data["industry"] = Industry(company_data["industry"])
            
            if isinstance(company_data["size"], str):
                company_data["size"] = CompanySize(company_data["size"])
            
            # Create CompanyInfo object
            company = CompanyInfo(**company_data)
            
            # Optionally save to database
            if company_data.get("save_to_database", False):
                self.company_db.add_company(company)
            
            return company
            
        except Exception as e:
            raise ValueError(f"Error creating custom company: {e}")
    
    async def _select_random_company(self, industry: Optional[Industry] = None, 
                                   size: Optional[CompanySize] = None) -> CompanyInfo:
        """Select a random company with optional filters"""
        import random
        
        companies = self.company_db.list_companies(industry=industry, size=size)
        if not companies:
            raise ValueError("No companies found matching the criteria")
        
        random_company = random.choice(companies)
        return self.company_db.get_company(random_company["id"])
    
    async def get_selection_options(self) -> Dict[str, Any]:
        """Get available options for company selection"""
        stats = self.company_db.get_database_stats()
        
        return {
            "total_companies": stats["total_companies"],
            "industries": list(stats["by_industry"].keys()),
            "company_sizes": list(stats["by_size"].keys()),
            "recent_companies": self.company_db.list_companies(limit=10),
            "popular_industries": sorted(stats["by_industry"].items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    async def validate_company_data(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate company data and return validation results"""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ["name", "industry", "size", "description"]
        for field in required_fields:
            if field not in company_data or not company_data[field]:
                errors.append(f"Required field '{field}' is missing or empty")
        
        # Validate industry
        if "industry" in company_data:
            try:
                if isinstance(company_data["industry"], str):
                    Industry(company_data["industry"])
            except ValueError:
                errors.append(f"Invalid industry: {company_data['industry']}")
        
        # Validate company size
        if "size" in company_data:
            try:
                if isinstance(company_data["size"], str):
                    CompanySize(company_data["size"])
            except ValueError:
                errors.append(f"Invalid company size: {company_data['size']}")
        
        # Check for optional but recommended fields
        recommended_fields = ["mission", "values", "target_audience", "unique_selling_points"]
        for field in recommended_fields:
            if field not in company_data or not company_data[field]:
                warnings.append(f"Recommended field '{field}' is missing - this may affect analysis quality")
        
        # Check description length
        if "description" in company_data and len(company_data["description"]) < 50:
            warnings.append("Description is quite short - a more detailed description would improve analysis")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }