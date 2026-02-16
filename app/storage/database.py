"""Database module with SQLite, vector search, and BM25 support."""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import event, text
from sqlmodel import Field, Session, SQLModel, create_engine, select


# Database file path
DB_DIR = Path(__file__).parent
DB_PATH = DB_DIR / "memory.db"


# ==================== SQLModel Table Definitions ====================


class Entity(SQLModel, table=True):
    """Entity table with embeddings and searchable aliases.

    The embeddings are stored as JSON-serialized list of floats.
    Aliases are stored in a separate FTS5 virtual table for BM25 search.
    """

    __tablename__ = "entities"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    type: str = Field(index=True)  # e.g., "person", "organization", "concept"
    description: Optional[str] = None

    # Embedding stored as JSON array
    embedding: str = Field(default="[]")  # JSON string of list[float]

    # Aliases stored as JSON array (for data integrity)
    # Actual search happens via FTS5 virtual table
    aliases: str = Field(default="[]")  # JSON string of list[str]

    # Metadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def set_embedding(self, embedding: list[float]) -> None:
        """Set embedding from a list of floats."""
        self.embedding = json.dumps(embedding)

    def get_embedding(self) -> list[float]:
        """Get embedding as a list of floats."""
        return json.loads(self.embedding) if self.embedding else []

    def set_aliases(self, aliases: list[str]) -> None:
        """Set aliases from a list of strings."""
        self.aliases = json.dumps(aliases)

    def get_aliases(self) -> list[str]:
        """Get aliases as a list of strings."""
        return json.loads(self.aliases) if self.aliases else []


class Relationship(SQLModel, table=True):
    """Relationship table connecting entities.

    Can also have embeddings for semantic relationship search.
    """

    __tablename__ = "relationships"

    id: Optional[int] = Field(default=None, primary_key=True)
    source_entity_id: int = Field(foreign_key="entities.id", index=True)
    target_entity_id: int = Field(foreign_key="entities.id", index=True)

    relationship_type: str = Field(index=True)  # e.g., "knows", "works_at", "related_to"
    description: Optional[str] = None

    # Optional embedding for the relationship itself
    embedding: str = Field(default="[]")  # JSON string of list[float]

    # Metadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def set_embedding(self, embedding: list[float]) -> None:
        """Set embedding from a list of floats."""
        self.embedding = json.dumps(embedding)

    def get_embedding(self) -> list[float]:
        """Get embedding as a list of floats."""
        return json.loads(self.embedding) if self.embedding else []


# ==================== Database Initialization ====================


def _enable_sqlite_extensions(dbapi_connection, connection_record):
    """Enable SQLite extensions like vec0 for vector search."""
    dbapi_connection.enable_load_extension(True)

    # Try to load sqlite-vec extension
    try:
        # Common paths where sqlite-vec might be installed
        vec_paths = [
            "vec0",  # If in system path
            str(DB_DIR / "vec0.so"),
            str(DB_DIR / "vec0.dylib"),
            str(DB_DIR / "vec0.dll"),
        ]

        for vec_path in vec_paths:
            try:
                dbapi_connection.load_extension(vec_path)
                break
            except sqlite3.OperationalError:
                continue
    except Exception as e:
        # Vector search won't be available but database will still work
        print(f"Warning: Could not load sqlite-vec extension: {e}")

    dbapi_connection.enable_load_extension(False)


def _create_fts5_tables(connection):
    """Create FTS5 virtual tables for BM25 search on aliases."""
    cursor = connection.cursor()

    # Create FTS5 virtual table for entity aliases
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS entity_aliases_fts
        USING fts5(
            entity_id UNINDEXED,
            alias,
            tokenize='porter unicode61'
        )
    """)

    connection.commit()


def _create_vector_tables(connection):
    """Create vector search tables using sqlite-vec."""
    cursor = connection.cursor()

    try:
        # Create virtual table for entity embeddings
        # Using vec0 for vector similarity search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS entity_vec
            USING vec0(
                entity_id INTEGER PRIMARY KEY,
                embedding FLOAT[768]
            )
        """)

        # Create virtual table for relationship embeddings
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS relationship_vec
            USING vec0(
                relationship_id INTEGER PRIMARY KEY,
                embedding FLOAT[768]
            )
        """)

        connection.commit()
    except sqlite3.OperationalError as e:
        # Vector extension not available
        print(f"Warning: Could not create vector tables: {e}")


def get_engine():
    """Get SQLModel engine with SQLite and extensions enabled."""
    engine = create_engine(
        f"sqlite:///{DB_PATH}",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    # Register listener to enable extensions on connect
    event.listen(engine, "connect", _enable_sqlite_extensions)

    return engine


def init_database() -> None:
    """Initialize database with all tables and extensions."""
    # Create database file if it doesn't exist
    engine = get_engine()

    # Create all SQLModel tables
    SQLModel.metadata.create_all(engine)

    # Create FTS5 and vector tables using raw SQLite connection
    with sqlite3.connect(DB_PATH) as conn:
        _create_fts5_tables(conn)
        _create_vector_tables(conn)

    print(f"Database initialized at {DB_PATH}")


def clear_database() -> None:
    """Clear database by deleting the database file."""
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Database cleared: {DB_PATH}")
    else:
        print(f"Database file not found: {DB_PATH}")


# ==================== BM25 Search Functions ====================


def search_aliases_bm25(query: str, limit: int = 10) -> list[int]:
    """Search entity aliases using BM25 ranking.

    Args:
        query: Search query string
        limit: Maximum number of results to return

    Returns:
        List of entity IDs ranked by BM25 score
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        # Search using FTS5 with BM25 ranking
        cursor.execute("""
            SELECT entity_id, rank
            FROM entity_aliases_fts
            WHERE entity_aliases_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))

        results = cursor.fetchall()
        return [row[0] for row in results]


def index_entity_aliases(entity_id: int, aliases: list[str]) -> None:
    """Index entity aliases in FTS5 table for BM25 search.

    Args:
        entity_id: ID of the entity
        aliases: List of alias strings to index
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        # Delete existing aliases for this entity
        cursor.execute(
            "DELETE FROM entity_aliases_fts WHERE entity_id = ?",
            (entity_id,)
        )

        # Insert new aliases
        for alias in aliases:
            cursor.execute(
                "INSERT INTO entity_aliases_fts (entity_id, alias) VALUES (?, ?)",
                (entity_id, alias)
            )

        conn.commit()


# ==================== Vector Search Functions ====================


def vector_search(
    query_embedding: list[float],
    table: str = "entity",
    limit: int = 10,
    threshold: float = 0.0,
) -> list[tuple[int, float]]:
    """Perform vector similarity search.

    Args:
        query_embedding: Query embedding as list of floats
        table: Table to search ("entity" or "relationship")
        limit: Maximum number of results to return
        threshold: Minimum similarity threshold (0.0 to 1.0)

    Returns:
        List of tuples (id, distance) ordered by similarity
    """
    # Determine which vector table to use
    vec_table = "entity_vec" if table == "entity" else "relationship_vec"
    id_column = "entity_id" if table == "entity" else "relationship_id"

    # Convert embedding to JSON for SQLite
    embedding_json = json.dumps(query_embedding)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        try:
            # Perform vector search using vec0
            # Lower distance = higher similarity
            cursor.execute(f"""
                SELECT {id_column}, distance
                FROM {vec_table}
                WHERE embedding MATCH ?
                AND distance >= ?
                ORDER BY distance
                LIMIT ?
            """, (embedding_json, threshold, limit))

            results = cursor.fetchall()
            return [(row[0], row[1]) for row in results]
        except sqlite3.OperationalError as e:
            print(f"Warning: Vector search not available: {e}")
            return []


def index_entity_embedding(entity_id: int, embedding: list[float]) -> None:
    """Index entity embedding in vector table.

    Args:
        entity_id: ID of the entity
        embedding: Embedding vector as list of floats
    """
    embedding_json = json.dumps(embedding)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        try:
            # Delete existing embedding
            cursor.execute(
                "DELETE FROM entity_vec WHERE entity_id = ?",
                (entity_id,)
            )

            # Insert new embedding
            cursor.execute(
                "INSERT INTO entity_vec (entity_id, embedding) VALUES (?, ?)",
                (entity_id, embedding_json)
            )

            conn.commit()
        except sqlite3.OperationalError as e:
            print(f"Warning: Could not index entity embedding: {e}")


def index_relationship_embedding(relationship_id: int, embedding: list[float]) -> None:
    """Index relationship embedding in vector table.

    Args:
        relationship_id: ID of the relationship
        embedding: Embedding vector as list of floats
    """
    embedding_json = json.dumps(embedding)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        try:
            # Delete existing embedding
            cursor.execute(
                "DELETE FROM relationship_vec WHERE relationship_id = ?",
                (relationship_id,)
            )

            # Insert new embedding
            cursor.execute(
                "INSERT INTO relationship_vec (relationship_id, embedding) VALUES (?, ?)",
                (relationship_id, embedding_json)
            )

            conn.commit()
        except sqlite3.OperationalError as e:
            print(f"Warning: Could not index relationship embedding: {e}")


# ==================== Helper Functions ====================


def get_entity_by_id(entity_id: int) -> Optional[Entity]:
    """Get entity by ID."""
    engine = get_engine()
    with Session(engine) as session:
        return session.get(Entity, entity_id)


def get_relationship_by_id(relationship_id: int) -> Optional[Relationship]:
    """Get relationship by ID."""
    engine = get_engine()
    with Session(engine) as session:
        return session.get(Relationship, relationship_id)
