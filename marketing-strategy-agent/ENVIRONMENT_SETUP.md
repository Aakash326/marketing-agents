# Environment Configuration Guide

This guide explains all environment variables used in the Marketing Strategy Agent and how to configure them properly.

  1. Test the installation:
  python quick_test.py

  2. Initialize the knowledge base:
  python
  scripts/initialize_knowledge_base.py

  3. Start the server:
  python start_server.py

## 🔧 Required Configuration

### 1. OpenAI Configuration

**Required for AI functionality:**

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=2000
```

**How to get OpenAI API Key:**
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Create a new API key
4. Copy the key to your `.env` file

### 2. TiDB Database Configuration

**Required for vector storage:**

```env
TIDB_HOST=gateway01.us-west-2.prod.aws.tidbcloud.com
TIDB_PORT=4000
TIDB_USER=your_tidb_username
TIDB_PASSWORD=your_tidb_password
TIDB_DATABASE=your_database_name
TIDB_SSL_DISABLED=false
```

**How to get TiDB credentials:**
1. Visit [TiDB Cloud](https://tidbcloud.com/)
2. Create a free account
3. Create a new Serverless cluster
4. Get connection details from the cluster dashboard
5. Update your `.env` file with the credentials

## ⚙️ Core Configuration

### Application Settings

```env
APP_NAME="Marketing Strategy Agent"
APP_VERSION=1.0.0
APP_HOST=0.0.0.0
APP_PORT=8000
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
```

### Vector Store Settings

```env
VECTOR_TABLE_NAME=marketing_knowledge
VECTOR_DIMENSION=1536
VECTOR_DISTANCE_STRATEGY=cosine
```

### Security Configuration

```env
SECRET_KEY=your_secret_key_here_change_in_production
API_KEY_HEADER=X-API-Key
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8000"]
ALLOWED_HOSTS=["localhost", "127.0.0.1", "0.0.0.0"]
```

**Generate a secure secret key:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 🔗 Optional Integrations

### Social Media APIs

For enhanced content optimization and social media integration:

```env
# Twitter/X API
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# LinkedIn API
LINKEDIN_CLIENT_ID=your_linkedin_client_id
LINKEDIN_CLIENT_SECRET=your_linkedin_client_secret

# Facebook/Meta API
FACEBOOK_ACCESS_TOKEN=your_facebook_access_token

# Google APIs
GOOGLE_API_KEY=your_google_api_key
SERPAPI_KEY=your_serpapi_key
```

### Email Configuration

For notifications and alerts:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM_EMAIL=noreply@yourcompany.com
```

### LangChain Tracing (Optional)

For debugging and monitoring AI workflows:

```env
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=marketing-strategy-agent
```

## 🚀 Performance & Scaling

### Rate Limiting

```env
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_REQUESTS_PER_HOUR=1000
```

### Caching Configuration

```env
CACHE_ENABLED=true
CACHE_TTL_SECONDS=3600
CACHE_MAX_SIZE=1000
```

### Database Connection Pool

```env
DB_POOL_MIN_SIZE=5
DB_POOL_MAX_SIZE=20
DB_POOL_TIMEOUT_SECONDS=30
```

### Workflow Limits

```env
MAX_WORKFLOW_DURATION_MINUTES=30
WORKFLOW_TIMEOUT_SECONDS=1800
MAX_CONCURRENT_WORKFLOWS=10
```

## 📊 Content Generation Limits

```env
MAX_CONTENT_LENGTH=5000
MAX_CALENDAR_DAYS=90
MAX_SOCIAL_POSTS_PER_REQUEST=50
```

## 📁 File & Storage Configuration

```env
MAX_FILE_SIZE_MB=10
ALLOWED_FILE_TYPES=["json", "txt", "csv", "md", "pdf"]
UPLOAD_DIR=data/uploads
EXPORT_DIR=data/exports
KNOWLEDGE_BASE_PATH=data/knowledge_base
```

## 🔒 Advanced Security & Compliance

```env
DATA_PRIVACY_MODE=standard
PII_DETECTION_ENABLED=true
AUDIT_LOGGING_ENABLED=true
```

## 🧪 Development & Testing

```env
TESTING_MODE=false
MOCK_EXTERNAL_APIS=false
SEED_DATA_ENABLED=true
```

## 📈 Monitoring & Analytics

```env
ANALYTICS_ENABLED=true
METRICS_RETENTION_DAYS=90
PERFORMANCE_MONITORING=true
HEALTH_CHECK_ENABLED=true
```

## 💾 Data Retention

```env
DATA_RETENTION_DAYS=365
LOG_RETENTION_DAYS=30
TEMP_FILE_CLEANUP_HOURS=24
```

## 🎛️ Feature Flags

```env
FEATURE_ADVANCED_ANALYTICS=true
FEATURE_REAL_TIME_COLLABORATION=false
FEATURE_A_B_TESTING=true
FEATURE_CUSTOM_TEMPLATES=true
```

## 🚨 Environment-Specific Settings

### Development Environment

```env
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
APP_RELOAD=true
TESTING_MODE=false
```

### Production Environment

```env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
APP_RELOAD=false
TESTING_MODE=false
```

## 📋 Setup Checklist

- [ ] Copy `.env.example` to `.env`
- [ ] Set `OPENAI_API_KEY` with your OpenAI API key
- [ ] Configure TiDB database credentials
- [ ] Generate and set a secure `SECRET_KEY`
- [ ] Update `CORS_ORIGINS` with your frontend URLs
- [ ] Configure optional integrations as needed
- [ ] Test configuration with `python quick_test.py`

## 🆘 Troubleshooting

### Common Issues

1. **OpenAI API Key Invalid**
   - Verify the key is correct and has sufficient credits
   - Check if the key has proper permissions

2. **TiDB Connection Failed**
   - Verify host, port, username, and password
   - Check if SSL is properly configured
   - Ensure the database exists

3. **Permission Errors**
   - Check file system permissions for upload/export directories
   - Verify log directory is writable

4. **Rate Limiting Issues**
   - Adjust rate limiting settings if needed
   - Monitor API usage to avoid exceeding limits

### Validation Script

Run this to validate your environment:

```bash
python scripts/validate_environment.py
```

## 🔗 Useful Links

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [TiDB Cloud Documentation](https://docs.pingcap.com/tidbcloud/)
- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Note:** Always keep your `.env` file secure and never commit it to version control. Use different configurations for development, staging, and production environments.