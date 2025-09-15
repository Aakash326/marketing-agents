# Marketing Strategy Agent

A comprehensive AI-powered marketing workflow automation system that leverages multiple specialized agents to generate complete marketing strategies, content plans, and visual assets for companies.

## 🚀 Features

### Multi-Agent System
- **BrandAnalyzer**: Brand positioning and competitive analysis
- **TrendResearcher**: Market trends and industry insights using web research
- **ContentCreator**: Content strategy and social media planning
- **MarketingAgent**: Comprehensive marketing strategy synthesis
- **GeminiVisualGenerator**: AI-powered visual content generation

### Interactive Dashboard
- Modern web interface with real-time progress tracking
- Company selection and management
- Workflow orchestration and monitoring
- Report generation and download

### Report Generation
- Multiple output formats (PDF, HTML, JSON, Markdown)
- Comprehensive marketing analysis reports
- Downloadable assets and strategies

### API Integration
- OpenAI GPT-4o-mini for content generation
- Google Gemini for visual content creation
- Tavily for web-based market research
- TiDB Vector Database for data persistence

## 📋 Prerequisites

- Python 3.8 or higher
- API keys for:
  - OpenAI (required)
  - Google Gemini (optional, for visual generation)
  - Tavily (optional, for web research)
  - TiDB Vector Database (optional, for data persistence)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd marketing-strategy-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the root directory:
   ```env
   # Required
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional - for enhanced features
   GEMINI_API_KEY=your_gemini_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   
   # Optional - for data persistence
   TIDB_HOST=your_tidb_host
   TIDB_PORT=4000
   TIDB_USER=your_tidb_user
   TIDB_PASSWORD=your_tidb_password
   TIDB_DATABASE=your_tidb_database
   
   # Application settings
   HOST=0.0.0.0
   PORT=8000
   DEBUG=False
   ```

## 🚀 Quick Start

1. **Start the dashboard**
   ```bash
   python run_dashboard.py
   ```

2. **Access the web interface**
   Open your browser and navigate to `http://localhost:8000`

3. **Select or add a company**
   - Choose from pre-loaded companies (Apple, Tesla, Shopify, etc.)
   - Or add your own custom company

4. **Run marketing analysis**
   - Click "Start Marketing Analysis"
   - Monitor real-time progress
   - Generate and download reports when complete

## 📊 System Architecture

```
marketing-strategy-agent/
├── src/
│   ├── agents/           # AI agent implementations
│   ├── api/             # FastAPI web application
│   ├── config/          # Configuration management
│   ├── database/        # Company database and vector store
│   ├── models/          # Pydantic data models
│   ├── reports/         # Report generation system
│   ├── static/          # Web assets
│   ├── templates/       # HTML templates
│   └── workflows/       # Workflow orchestration
├── data/                # Storage directories
└── run_dashboard.py     # Application launcher
```

## 🤖 Agent Capabilities

### BrandAnalyzer
- Brand positioning analysis
- Competitive landscape assessment
- Brand health scoring
- Market differentiation strategies

### TrendResearcher
- Industry trend analysis
- Market research and insights
- Competitive intelligence
- Emerging opportunity identification

### ContentCreator
- Content strategy development
- Social media planning
- Campaign ideation
- Content calendar creation

### MarketingAgent
- Strategy synthesis and integration
- Channel optimization
- Budget allocation recommendations
- ROI projections

### GeminiVisualGenerator
- Brand visual asset creation
- Marketing material design
- Social media graphics
- Campaign visual concepts

## 📊 Workflow Execution Modes

- **Parallel**: All agents run simultaneously (fastest)
- **Sequential**: Agents run one after another (most reliable)
- **Hybrid**: Optimized balance of speed and reliability (recommended)

## 📋 API Endpoints

### Companies
- `GET /api/companies` - List all companies
- `POST /api/companies` - Add new company
- `GET /api/companies/{name}` - Get company details

### Workflows
- `POST /api/workflows` - Create and start workflow
- `GET /api/workflows` - List all workflows
- `GET /api/workflows/{id}/status` - Get workflow status
- `DELETE /api/workflows/{id}` - Cancel workflow

### Reports
- `POST /api/reports/generate` - Generate report
- `GET /api/reports` - List generated reports
- `GET /api/reports/download/{filename}` - Download report
- `DELETE /api/reports/{filename}` - Delete report

### WebSocket
- `WS /ws/{workflow_id}` - Real-time workflow updates

## 🎯 Usage Examples

### Python API Usage
```python
from src.workflows.enhanced_marketing_workflow import execute_enhanced_marketing_workflow
from src.models.data_models import CompanyInfo
from src.config.settings import load_config

# Setup
config = load_config()
company_info = CompanyInfo(
    name="Your Company",
    industry="Technology",
    size="Medium",
    location="Global"
)

# Execute workflow
result = await execute_enhanced_marketing_workflow(
    company_info, 
    config, 
    execution_mode="hybrid"
)

# Generate report
from src.reports.report_generator import generate_report
report = await generate_report(result, format_type="pdf")
```

### REST API Usage
```bash
# Create workflow
curl -X POST "http://localhost:8000/api/workflows" \
  -H "Content-Type: application/json" \
  -d '{
    "company_info": {
      "name": "Your Company",
      "industry": "Technology"
    },
    "execution_mode": "hybrid"
  }'

# Generate report
curl -X POST "http://localhost:8000/api/reports/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "your-workflow-id",
    "format": "pdf"
  }'
```

## 🔧 Configuration Options

### Environment Variables
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | Yes |
| `GEMINI_API_KEY` | Google Gemini API key | - | No |
| `TAVILY_API_KEY` | Tavily API key | - | No |
| `HOST` | Server host | 0.0.0.0 | No |
| `PORT` | Server port | 8000 | No |
| `DEBUG` | Debug mode | False | No |

### Workflow Settings
| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_EXECUTION_MODE` | Default workflow mode | hybrid |
| `MAX_CONCURRENT_WORKFLOWS` | Max simultaneous workflows | 5 |
| `WORKFLOW_TIMEOUT_SECONDS` | Workflow timeout | 1800 |

## 📈 Monitoring and Logging

- Real-time workflow progress via WebSocket
- Comprehensive logging system
- Error tracking and recovery
- Performance metrics collection

## 🛡️ Security Features

- API key validation
- Secure file handling
- Input validation and sanitization
- CORS protection

## 🚧 Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
isort src/
```

### Type Checking
```bash
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support and questions, please create an issue in the repository.

## 🔄 Version History

- **v1.0.0** - Initial release with full multi-agent system, web dashboard, and report generation

---

**Built with ❤️ using FastAPI, OpenAI, and modern web technologies**