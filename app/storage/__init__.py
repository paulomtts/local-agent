"""Storage module for persistent data management.

Provides SQLite database with vector search (sqlite-vec) and BM25 full-text search.
"""

from .database import (
    Entity,
    Relationship,
    clear_database,
    get_engine,
    get_entity_by_id,
    get_relationship_by_id,
    index_entity_aliases,
    index_entity_embedding,
    index_relationship_embedding,
    init_database,
    search_aliases_bm25,
    vector_search,
)

__all__ = [
    # Models
    "Entity",
    "Relationship",
    # Database management
    "init_database",
    "clear_database",
    "get_engine",
    # Indexing
    "index_entity_aliases",
    "index_entity_embedding",
    "index_relationship_embedding",
    # Search
    "search_aliases_bm25",
    "vector_search",
    # Retrieval
    "get_entity_by_id",
    "get_relationship_by_id",
]
