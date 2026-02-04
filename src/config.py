"""
Database Configuration Module

Provides database connection settings from environment variables.
"""

import os

# PostgreSQL connection settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "academic_assistant")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# Embedding model settings
EMBEDDING_MODEL = os.getenv("ARXIV_ASSISTANT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Chunk settings
CHUNK_SIZE = int(os.getenv("ARXIV_ASSISTANT_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("ARXIV_ASSISTANT_CHUNK_OVERLAP", "200"))


def get_connection_string() -> str:
    """Get PostgreSQL connection string."""
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_connection_params() -> dict:
    """Get connection parameters as dictionary."""
    return {
        "host": DB_HOST,
        "port": DB_PORT,
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD
    }
