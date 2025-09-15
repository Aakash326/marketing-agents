"""
Custom exception classes for the Marketing Strategy Agent.

This module defines comprehensive exception hierarchy for different
components and error scenarios in the application.
"""

from typing import Any, Dict, Optional


class MarketingAgentException(Exception):
    """Base exception class for all Marketing Agent errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "MARKETING_AGENT_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class BrandAnalysisException(MarketingAgentException):
    """Exception raised during brand analysis operations."""
    
    def __init__(
        self,
        message: str = "Brand analysis failed",
        error_code: str = "BRAND_ANALYSIS_ERROR",
        status_code: int = 422,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize brand analysis exception."""
        super().__init__(message, error_code, status_code, details)


class InvalidBrandDataException(BrandAnalysisException):
    """Exception raised when brand data is invalid or incomplete."""
    
    def __init__(
        self,
        message: str = "Invalid or incomplete brand data provided",
        missing_fields: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize invalid brand data exception."""
        if details is None:
            details = {}
        
        if missing_fields:
            details["missing_fields"] = missing_fields
        
        super().__init__(
            message=message,
            error_code="INVALID_BRAND_DATA",
            status_code=400,
            details=details
        )


class CompetitorAnalysisException(BrandAnalysisException):
    """Exception raised during competitor analysis."""
    
    def __init__(
        self,
        message: str = "Competitor analysis failed",
        competitor: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize competitor analysis exception."""
        if details is None:
            details = {}
        
        if competitor:
            details["competitor"] = competitor
        
        super().__init__(
            message=message,
            error_code="COMPETITOR_ANALYSIS_ERROR",
            status_code=422,
            details=details
        )


class ContentGenerationException(MarketingAgentException):
    """Exception raised during content generation operations."""
    
    def __init__(
        self,
        message: str = "Content generation failed",
        error_code: str = "CONTENT_GENERATION_ERROR",
        status_code: int = 422,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize content generation exception."""
        super().__init__(message, error_code, status_code, details)


class ContentTypeNotSupportedException(ContentGenerationException):
    """Exception raised when requested content type is not supported."""
    
    def __init__(
        self,
        content_type: str,
        supported_types: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize content type not supported exception."""
        if details is None:
            details = {}
        
        details["requested_type"] = content_type
        if supported_types:
            details["supported_types"] = supported_types
        
        message = f"Content type '{content_type}' is not supported"
        
        super().__init__(
            message=message,
            error_code="CONTENT_TYPE_NOT_SUPPORTED",
            status_code=400,
            details=details
        )


class ContentQualityException(ContentGenerationException):
    """Exception raised when generated content doesn't meet quality standards."""
    
    def __init__(
        self,
        message: str = "Generated content doesn't meet quality standards",
        quality_issues: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize content quality exception."""
        if details is None:
            details = {}
        
        if quality_issues:
            details["quality_issues"] = quality_issues
        
        super().__init__(
            message=message,
            error_code="CONTENT_QUALITY_ERROR",
            status_code=422,
            details=details
        )


class TrendResearchException(MarketingAgentException):
    """Exception raised during trend research operations."""
    
    def __init__(
        self,
        message: str = "Trend research failed",
        error_code: str = "TREND_RESEARCH_ERROR",
        status_code: int = 422,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize trend research exception."""
        super().__init__(message, error_code, status_code, details)


class MarketDataUnavailableException(TrendResearchException):
    """Exception raised when market data is unavailable."""
    
    def __init__(
        self,
        market: str,
        source: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize market data unavailable exception."""
        if details is None:
            details = {}
        
        details["market"] = market
        if source:
            details["source"] = source
        
        message = f"Market data for '{market}' is not available"
        
        super().__init__(
            message=message,
            error_code="MARKET_DATA_UNAVAILABLE",
            status_code=404,
            details=details
        )


class DatabaseException(MarketingAgentException):
    """Exception raised during database operations."""
    
    def __init__(
        self,
        message: str = "Database operation failed",
        error_code: str = "DATABASE_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize database exception."""
        super().__init__(message, error_code, status_code, details)


class VectorStoreException(DatabaseException):
    """Exception raised during vector store operations."""
    
    def __init__(
        self,
        message: str = "Vector store operation failed",
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize vector store exception."""
        if details is None:
            details = {}
        
        if operation:
            details["operation"] = operation
        
        super().__init__(
            message=message,
            error_code="VECTOR_STORE_ERROR",
            status_code=500,
            details=details
        )


class EmbeddingException(VectorStoreException):
    """Exception raised during embedding operations."""
    
    def __init__(
        self,
        message: str = "Embedding operation failed",
        text_length: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize embedding exception."""
        if details is None:
            details = {}
        
        if text_length:
            details["text_length"] = text_length
        
        super().__init__(
            message=message,
            operation="embedding",
            details=details
        )


class APIException(MarketingAgentException):
    """Exception raised during external API calls."""
    
    def __init__(
        self,
        message: str = "External API call failed",
        error_code: str = "API_ERROR",
        status_code: int = 502,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize API exception."""
        super().__init__(message, error_code, status_code, details)


class OpenAIException(APIException):
    """Exception raised during OpenAI API calls."""
    
    def __init__(
        self,
        message: str = "OpenAI API call failed",
        api_error_type: Optional[str] = None,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize OpenAI exception."""
        if details is None:
            details = {}
        
        if api_error_type:
            details["api_error_type"] = api_error_type
        if model:
            details["model"] = model
        
        super().__init__(
            message=message,
            error_code="OPENAI_API_ERROR",
            status_code=502,
            details=details
        )


class RateLimitException(APIException):
    """Exception raised when API rate limits are exceeded."""
    
    def __init__(
        self,
        message: str = "API rate limit exceeded",
        api_provider: Optional[str] = None,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize rate limit exception."""
        if details is None:
            details = {}
        
        if api_provider:
            details["api_provider"] = api_provider
        if retry_after:
            details["retry_after"] = retry_after
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details=details
        )


class WorkflowException(MarketingAgentException):
    """Exception raised during LangGraph workflow execution."""
    
    def __init__(
        self,
        message: str = "Workflow execution failed",
        error_code: str = "WORKFLOW_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize workflow exception."""
        super().__init__(message, error_code, status_code, details)


class WorkflowStateException(WorkflowException):
    """Exception raised when workflow state is invalid."""
    
    def __init__(
        self,
        message: str = "Invalid workflow state",
        state_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize workflow state exception."""
        if details is None:
            details = {}
        
        if state_key:
            details["state_key"] = state_key
        if expected_type:
            details["expected_type"] = expected_type
        if actual_type:
            details["actual_type"] = actual_type
        
        super().__init__(
            message=message,
            error_code="WORKFLOW_STATE_ERROR",
            status_code=422,
            details=details
        )


class WorkflowTimeoutException(WorkflowException):
    """Exception raised when workflow execution times out."""
    
    def __init__(
        self,
        message: str = "Workflow execution timed out",
        timeout_seconds: Optional[int] = None,
        workflow_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize workflow timeout exception."""
        if details is None:
            details = {}
        
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if workflow_name:
            details["workflow_name"] = workflow_name
        
        super().__init__(
            message=message,
            error_code="WORKFLOW_TIMEOUT",
            status_code=408,
            details=details
        )


class ValidationException(MarketingAgentException):
    """Exception raised during input validation."""
    
    def __init__(
        self,
        message: str = "Input validation failed",
        field_errors: Optional[Dict[str, str]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize validation exception."""
        if details is None:
            details = {}
        
        if field_errors:
            details["field_errors"] = field_errors
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details
        )


class ConfigurationException(MarketingAgentException):
    """Exception raised when configuration is invalid."""
    
    def __init__(
        self,
        message: str = "Configuration error",
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize configuration exception."""
        if details is None:
            details = {}
        
        if config_key:
            details["config_key"] = config_key
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            details=details
        )


# Utility functions for exception handling

def handle_exception(logger, exception: Exception, context: str = "") -> MarketingAgentException:
    """Convert generic exceptions to MarketingAgentException."""
    if isinstance(exception, MarketingAgentException):
        return exception
    
    message = f"Unexpected error in {context}: {str(exception)}" if context else str(exception)
    
    logger.error(message, extra={
        "exception_type": type(exception).__name__,
        "context": context
    }, exc_info=True)
    
    return MarketingAgentException(
        message=message,
        error_code="UNEXPECTED_ERROR",
        details={"original_exception": type(exception).__name__}
    )


def log_and_raise(logger, exception: MarketingAgentException) -> None:
    """Log and raise a MarketingAgentException."""
    logger.error(f"{exception.error_code}: {exception.message}", extra={
        "error_code": exception.error_code,
        "status_code": exception.status_code,
        "details": exception.details
    })
    raise exception