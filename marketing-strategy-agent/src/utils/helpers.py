"""
Helper utility functions for the Marketing Strategy Agent.

This module contains common utility functions used throughout the application
including text processing, data validation, formatting, and other utilities.
"""

import re
import json
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
import asyncio
from functools import wraps
import time

from src.utils.logging import get_logger
from src.utils.exceptions import ValidationException, MarketingAgentException

logger = get_logger(__name__)


def sanitize_text(text: str) -> str:
    """Sanitize text input by removing unwanted characters."""
    if not text:
        return ""
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def validate_email(email: str) -> bool:
    """Validate email address format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """Validate URL format."""
    pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])+)?(?:#(?:[\w.])+)?)?$'
    return bool(re.match(pattern, url))


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    # Convert to lowercase
    text = text.lower()
    
    # Replace spaces and special characters with hyphens
    text = re.sub(r'[^a-z0-9]+', '-', text)
    
    # Remove leading/trailing hyphens
    text = text.strip('-')
    
    # Remove consecutive hyphens
    text = re.sub(r'-+', '-', text)
    
    return text


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length with optional suffix."""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text."""
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter words by minimum length
    words = [word for word in words if len(word) >= min_length]
    
    # Count word frequency
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    keywords = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in keywords[:max_keywords]]


def calculate_readability_score(text: str) -> float:
    """Calculate simple readability score based on sentence and word length."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.0
    
    words = re.findall(r'\b\w+\b', text)
    
    if not words:
        return 0.0
    
    avg_sentence_length = len(words) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Simple readability formula (lower is better)
    score = (avg_sentence_length * 1.015) + (avg_word_length * 84.6) - 206.835
    
    # Normalize to 0-100 scale (higher is more readable)
    return max(0, min(100, 100 - (score / 2)))


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount."""
    if currency == "USD":
        return f"${amount:,.2f}"
    elif currency == "EUR":
        return f"€{amount:,.2f}"
    elif currency == "GBP":
        return f"£{amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format percentage value."""
    return f"{value:.{decimal_places}f}%"


def format_number(number: Union[int, float], abbreviate: bool = False) -> str:
    """Format number with optional abbreviation for large numbers."""
    if not abbreviate:
        return f"{number:,}"
    
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.1f}B"
    elif abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    elif abs(number) >= 1_000:
        return f"{number / 1_000:.1f}K"
    else:
        return str(number)


def generate_hash(data: Union[str, Dict, List]) -> str:
    """Generate SHA-256 hash for data."""
    if isinstance(data, (dict, list)):
        data = json.dumps(data, sort_keys=True)
    
    return hashlib.sha256(data.encode()).hexdigest()


def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary."""
    items = []
    
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def get_nested_value(data: Dict, key_path: str, default: Any = None) -> Any:
    """Get value from nested dictionary using dot notation."""
    keys = key_path.split('.')
    value = data
    
    try:
        for key in keys:
            if isinstance(value, dict):
                value = value[key]
            elif isinstance(value, (list, tuple)) and key.isdigit():
                value = value[int(key)]
            else:
                return default
        return value
    except (KeyError, IndexError, TypeError):
        return default


def set_nested_value(data: Dict, key_path: str, value: Any) -> Dict:
    """Set value in nested dictionary using dot notation."""
    keys = key_path.split('.')
    current = data
    
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    return data


def validate_required_fields(data: Dict, required_fields: List[str]) -> None:
    """Validate that required fields are present in data."""
    missing_fields = []
    
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == "":
            missing_fields.append(field)
    
    if missing_fields:
        raise ValidationException(
            message="Missing required fields",
            field_errors={field: "This field is required" for field in missing_fields}
        )


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters."""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x80-\x9f]', '', filename)
    
    # Limit length
    filename = filename[:255]
    
    # Ensure not empty
    if not filename.strip():
        filename = "untitled"
    
    return filename


def parse_duration(duration_str: str) -> int:
    """Parse duration string to seconds."""
    pattern = r'(\d+)\s*(s|sec|second|seconds|m|min|minute|minutes|h|hour|hours|d|day|days)'
    matches = re.findall(pattern, duration_str.lower())
    
    if not matches:
        raise ValidationException(f"Invalid duration format: {duration_str}")
    
    total_seconds = 0
    
    for value, unit in matches:
        value = int(value)
        
        if unit in ['s', 'sec', 'second', 'seconds']:
            total_seconds += value
        elif unit in ['m', 'min', 'minute', 'minutes']:
            total_seconds += value * 60
        elif unit in ['h', 'hour', 'hours']:
            total_seconds += value * 3600
        elif unit in ['d', 'day', 'days']:
            total_seconds += value * 86400
    
    return total_seconds


def retry_async(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
            
            logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            raise last_exception
        
        return wrapper
    return decorator


def timing_context(operation_name: str):
    """Context manager for timing operations."""
    class TimingContext:
        def __init__(self, name: str):
            self.name = name
            self.start_time = None
            self.duration = None
        
        def __enter__(self):
            self.start_time = time.time()
            logger.debug(f"Starting operation: {self.name}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.duration = time.time() - self.start_time
            if exc_type is None:
                logger.info(f"Completed operation: {self.name}", extra={
                    "operation": self.name,
                    "duration_ms": self.duration * 1000,
                    "success": True
                })
            else:
                logger.error(f"Failed operation: {self.name}", extra={
                    "operation": self.name,
                    "duration_ms": self.duration * 1000,
                    "success": False,
                    "error": str(exc_val)
                })
    
    return TimingContext(operation_name)


def get_utc_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO timestamp string to datetime object."""
    return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))


def batch_process(items: List[Any], batch_size: int = 100):
    """Generator for processing items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]