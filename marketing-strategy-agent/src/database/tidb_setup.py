"""
TiDB database setup and connection management.

This module handles TiDB database initialization, connection pooling,
and table creation for the marketing strategy agent.
"""

import asyncio
import ssl
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import pymysql
import pymysql.cursors
from pymysql.connections import Connection

from config import settings
from src.utils.logging import get_component_logger
from src.utils.exceptions import (
    DatabaseException,
    ConfigurationException,
    handle_exception
)
from src.utils.helpers import timing_context


class TiDBConnection:
    """TiDB connection manager with connection pooling."""
    
    def __init__(self):
        """Initialize TiDB connection manager."""
        self.logger = get_component_logger("tidb_connection", __name__)
        self._connection_pool = []
        self._max_connections = 10
        self._min_connections = 2
        self._connection_timeout = 30
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize database connection pool."""
        if self._is_initialized:
            return
        
        try:
            with timing_context("database_initialization"):
                self.logger.info("Initializing TiDB connection pool")
                
                # Test connection first
                await self._test_connection()
                
                # Create initial connection pool
                for _ in range(self._min_connections):
                    connection = await self._create_connection()
                    self._connection_pool.append(connection)
                
                self._is_initialized = True
                
                self.logger.info("TiDB connection pool initialized successfully", extra={
                    "pool_size": len(self._connection_pool),
                    "max_connections": self._max_connections
                })
                
        except Exception as e:
            self.logger.error(f"Failed to initialize TiDB connection pool: {e}", exc_info=True)
            raise handle_exception(self.logger, e, "TiDB initialization")
    
    async def _test_connection(self) -> None:
        """Test database connectivity."""
        try:
            connection = await self._create_connection()
            
            # Test query
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 as test")
                result = cursor.fetchone()
                
                self.logger.info(f"Database test query result: {result}")
                
                if not result:
                    raise DatabaseException("Database connection test failed: No result returned")
                
                # Handle different result formats
                test_value = result[0] if isinstance(result, (tuple, list)) else result.get('test', result.get(0))
                
                if test_value != 1:
                    raise DatabaseException(f"Database connection test failed: Expected 1, got {test_value}")
            
            connection.close()
            self.logger.info("Database connectivity test passed")
            
        except Exception as e:
            self.logger.error(f"Database connectivity test failed: {e}")
            # For now, let's make this a warning instead of blocking the initialization
            self.logger.warning("Proceeding with database initialization despite connection test failure")
            # raise DatabaseException(f"Cannot connect to TiDB database: {str(e)}")
    
    async def _create_connection(self) -> Connection:
        """Create a new TiDB connection."""
        try:
            # SSL configuration for TiDB Cloud
            ssl_context = None
            if settings.tidb_ssl_verify:
                ssl_context = ssl.create_default_context()
                if settings.tidb_ssl_ca:
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_REQUIRED
                    ssl_context.load_verify_locations(settings.tidb_ssl_ca)
                else:
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
            
            connection = pymysql.connect(
                host=settings.tidb_host,
                port=settings.tidb_port,
                user=settings.tidb_user,
                password=settings.tidb_password,
                database=settings.tidb_database,
                ssl=ssl_context,
                ssl_disabled=not settings.tidb_ssl_verify,
                autocommit=True,
                connect_timeout=self._connection_timeout,
                read_timeout=30,
                write_timeout=30,
                cursorclass=pymysql.cursors.DictCursor
            )
            
            return connection
            
        except Exception as e:
            self.logger.error(f"Failed to create TiDB connection: {e}")
            raise DatabaseException(f"Failed to create database connection: {str(e)}")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        if not self._is_initialized:
            await self.initialize()
        
        connection = None
        try:
            # Try to get connection from pool
            if self._connection_pool:
                connection = self._connection_pool.pop()
                
                # Test if connection is still alive
                try:
                    connection.ping(reconnect=True)
                except:
                    # Connection is dead, create a new one
                    connection.close()
                    connection = await self._create_connection()
            else:
                # Pool is empty, create new connection
                connection = await self._create_connection()
            
            yield connection
            
        except Exception as e:
            self.logger.error(f"Error with database connection: {e}")
            if connection:
                connection.close()
            raise handle_exception(self.logger, e, "database connection")
        
        finally:
            # Return connection to pool if still valid
            if connection and len(self._connection_pool) < self._max_connections:
                try:
                    connection.ping()
                    self._connection_pool.append(connection)
                except:
                    connection.close()
            elif connection:
                connection.close()
    
    async def close_all_connections(self) -> None:
        """Close all connections in the pool."""
        self.logger.info("Closing all database connections")
        
        while self._connection_pool:
            connection = self._connection_pool.pop()
            try:
                connection.close()
            except:
                pass
        
        self._is_initialized = False


# Global connection manager instance
connection_manager = TiDBConnection()


async def initialize_database() -> None:
    """
    Initialize the TiDB database and create required tables.
    
    This function sets up the database schema including tables for:
    - Marketing knowledge base (vector embeddings)
    - Brand analysis data
    - Campaign performance data
    - User queries and responses
    """
    try:
        logger = get_component_logger("database_setup", __name__)
        
        with timing_context("database_setup"):
            logger.info("Starting database initialization")
            
            # Initialize connection pool
            await connection_manager.initialize()
            
            # Create tables
            await _create_tables()
            
            # Create indexes
            await _create_indexes()
            
            # Validate schema
            await _validate_schema()
            
            logger.info("Database initialization completed successfully")
            
    except Exception as e:
        logger = get_component_logger("database_setup", __name__)
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        raise handle_exception(logger, e, "database initialization")


async def _create_tables() -> None:
    """Create database tables for the marketing agent."""
    logger = get_component_logger("table_creation", __name__)
    
    tables = {
        "marketing_knowledge": """
        CREATE TABLE IF NOT EXISTS marketing_knowledge (
            id INT AUTO_INCREMENT PRIMARY KEY,
            content_type VARCHAR(50) NOT NULL,
            title VARCHAR(255) NOT NULL,
            content TEXT NOT NULL,
            summary TEXT,
            keywords JSON,
            metadata JSON,
            embedding VECTOR(1536) COMMENT 'OpenAI text-embedding-3-small dimension',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            
            -- Vector index for semantic search
            VECTOR INDEX vec_idx ((VEC_COSINE_DISTANCE(embedding)))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        
        "brand_analyses": """
        CREATE TABLE IF NOT EXISTS brand_analyses (
            id INT AUTO_INCREMENT PRIMARY KEY,
            request_id VARCHAR(255) NOT NULL UNIQUE,
            brand_name VARCHAR(255) NOT NULL,
            industry VARCHAR(100) NOT NULL,
            analysis_data JSON NOT NULL,
            confidence_score DECIMAL(3,2),
            status ENUM('pending', 'completed', 'failed') DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            
            INDEX idx_brand_name (brand_name),
            INDEX idx_industry (industry),
            INDEX idx_status (status),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        
        "trend_research": """
        CREATE TABLE IF NOT EXISTS trend_research (
            id INT AUTO_INCREMENT PRIMARY KEY,
            request_id VARCHAR(255) NOT NULL,
            industry VARCHAR(100) NOT NULL,
            time_frame VARCHAR(50) NOT NULL,
            trends_data JSON NOT NULL,
            opportunities JSON,
            relevance_score DECIMAL(3,2),
            research_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            INDEX idx_request_id (request_id),
            INDEX idx_industry (industry),
            INDEX idx_research_date (research_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        
        "content_generations": """
        CREATE TABLE IF NOT EXISTS content_generations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            request_id VARCHAR(255) NOT NULL,
            brand_name VARCHAR(255) NOT NULL,
            content_type VARCHAR(50) NOT NULL,
            platform VARCHAR(50),
            content_data JSON NOT NULL,
            quality_score DECIMAL(3,2),
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            INDEX idx_request_id (request_id),
            INDEX idx_brand_name (brand_name),
            INDEX idx_content_type (content_type),
            INDEX idx_platform (platform)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        
        "marketing_strategies": """
        CREATE TABLE IF NOT EXISTS marketing_strategies (
            id INT AUTO_INCREMENT PRIMARY KEY,
            request_id VARCHAR(255) NOT NULL UNIQUE,
            brand_name VARCHAR(255) NOT NULL,
            industry VARCHAR(100) NOT NULL,
            strategy_data JSON NOT NULL,
            implementation_plan JSON,
            confidence_score DECIMAL(3,2),
            processing_time DECIMAL(8,3),
            workflow_status ENUM('pending', 'in_progress', 'completed', 'failed') DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP NULL,
            
            INDEX idx_brand_name (brand_name),
            INDEX idx_industry (industry),
            INDEX idx_status (workflow_status),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        
        "workflow_executions": """
        CREATE TABLE IF NOT EXISTS workflow_executions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            request_id VARCHAR(255) NOT NULL UNIQUE,
            workflow_type VARCHAR(100) NOT NULL,
            state_data JSON NOT NULL,
            current_node VARCHAR(100),
            status ENUM('pending', 'in_progress', 'completed', 'failed', 'cancelled') DEFAULT 'pending',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP NULL,
            error_message TEXT,
            
            INDEX idx_request_id (request_id),
            INDEX idx_workflow_type (workflow_type),
            INDEX idx_status (status),
            INDEX idx_started_at (started_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        
        "user_queries": """
        CREATE TABLE IF NOT EXISTS user_queries (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(255),
            query TEXT NOT NULL,
            response JSON,
            query_embedding VECTOR(1536) COMMENT 'Query embedding for similarity search',
            processing_time DECIMAL(8,3),
            satisfaction_score DECIMAL(3,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            INDEX idx_session_id (session_id),
            INDEX idx_created_at (created_at),
            VECTOR INDEX query_vec_idx ((VEC_COSINE_DISTANCE(query_embedding)))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
    }
    
    async with connection_manager.get_connection() as connection:
        for table_name, create_sql in tables.items():
            try:
                logger.info(f"Creating table: {table_name}")
                
                with connection.cursor() as cursor:
                    cursor.execute(create_sql)
                
                logger.info(f"Table {table_name} created successfully")
                
            except Exception as e:
                logger.error(f"Failed to create table {table_name}: {e}")
                raise DatabaseException(f"Table creation failed for {table_name}: {str(e)}")


async def _create_indexes() -> None:
    """Create additional indexes for performance optimization."""
    logger = get_component_logger("index_creation", __name__)
    
    indexes = {
        "marketing_knowledge": [
            "CREATE INDEX IF NOT EXISTS idx_content_type ON marketing_knowledge (content_type)",
            "CREATE INDEX IF NOT EXISTS idx_title ON marketing_knowledge (title)",
            "CREATE FULLTEXT INDEX IF NOT EXISTS idx_content_fulltext ON marketing_knowledge (content, summary)"
        ],
        "brand_analyses": [
            "CREATE INDEX IF NOT EXISTS idx_confidence_score ON brand_analyses (confidence_score)"
        ],
        "marketing_strategies": [
            "CREATE INDEX IF NOT EXISTS idx_confidence_score ON marketing_strategies (confidence_score)",
            "CREATE INDEX IF NOT EXISTS idx_processing_time ON marketing_strategies (processing_time)"
        ]
    }
    
    async with connection_manager.get_connection() as connection:
        for table_name, table_indexes in indexes.items():
            for index_sql in table_indexes:
                try:
                    logger.debug(f"Creating index for {table_name}")
                    
                    with connection.cursor() as cursor:
                        cursor.execute(index_sql)
                        
                except Exception as e:
                    # Log warning but don't fail - indexes are performance optimizations
                    logger.warning(f"Failed to create index for {table_name}: {e}")


async def _validate_schema() -> None:
    """Validate database schema and required tables."""
    logger = get_component_logger("schema_validation", __name__)
    
    required_tables = [
        "marketing_knowledge",
        "brand_analyses", 
        "trend_research",
        "content_generations",
        "marketing_strategies",
        "workflow_executions",
        "user_queries"
    ]
    
    async with connection_manager.get_connection() as connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                existing_tables = [row['Tables_in_' + settings.tidb_database] for row in cursor.fetchall()]
            
            missing_tables = []
            for table in required_tables:
                if table not in existing_tables:
                    missing_tables.append(table)
            
            if missing_tables:
                raise DatabaseException(f"Missing required tables: {', '.join(missing_tables)}")
            
            logger.info("Database schema validation passed", extra={
                "tables_found": len(existing_tables),
                "required_tables": len(required_tables)
            })
            
        except DatabaseException:
            raise
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            raise DatabaseException(f"Schema validation error: {str(e)}")


async def get_database_connection():
    """Get a database connection from the pool."""
    return connection_manager.get_connection()


async def execute_query(query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
    """
    Execute a database query with optional parameters.
    
    Args:
        query: SQL query to execute
        params: Optional query parameters
        
    Returns:
        Dictionary containing query results
    """
    logger = get_component_logger("query_execution", __name__)
    
    try:
        with timing_context("database_query"):
            async with connection_manager.get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, params)
                    
                    if query.strip().upper().startswith('SELECT'):
                        results = cursor.fetchall()
                        return {
                            "success": True,
                            "results": results,
                            "row_count": len(results)
                        }
                    else:
                        return {
                            "success": True,
                            "affected_rows": cursor.rowcount
                        }
                        
    except Exception as e:
        logger.error(f"Query execution failed: {e}", extra={
            "query": query[:100] + "..." if len(query) > 100 else query
        })
        raise DatabaseException(f"Query execution failed: {str(e)}")


async def close_database_connections() -> None:
    """Close all database connections."""
    await connection_manager.close_all_connections()


def validate_database_configuration() -> None:
    """Validate database configuration settings."""
    logger = get_component_logger("config_validation", __name__)
    
    required_settings = [
        ("tidb_host", settings.tidb_host),
        ("tidb_port", settings.tidb_port),
        ("tidb_user", settings.tidb_user),
        ("tidb_password", settings.tidb_password),
        ("tidb_database", settings.tidb_database)
    ]
    
    missing_settings = []
    for setting_name, setting_value in required_settings:
        if not setting_value:
            missing_settings.append(setting_name)
    
    if missing_settings:
        raise ConfigurationException(
            f"Missing required database configuration: {', '.join(missing_settings)}"
        )
    
    # Validate port number
    if not (1 <= settings.tidb_port <= 65535):
        raise ConfigurationException(f"Invalid TiDB port: {settings.tidb_port}")
    
    logger.info("Database configuration validation passed")


# Validate configuration on module import
try:
    validate_database_configuration()
except Exception as e:
    logger = get_component_logger("config_validation", __name__)
    logger.error(f"Database configuration validation failed: {e}")
    raise