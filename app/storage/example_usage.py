"""Example usage of the database module."""

from datetime import datetime

from sqlmodel import Session

from app.storage.database import (
    Entity,
    Relationship,
    clear_database,
    get_engine,
    index_entity_aliases,
    index_entity_embedding,
    init_database,
    search_aliases_bm25,
    vector_search,
)


def example_basic_usage():
    """Demonstrate basic database operations."""
    # Initialize database
    init_database()

    # Get engine for session management
    engine = get_engine()

    # Create an entity
    with Session(engine) as session:
        entity = Entity(
            name="Python",
            type="programming_language",
            description="A high-level programming language",
            created_at=datetime.now().isoformat(),
        )

        # Set embedding and aliases
        entity.set_embedding([0.1] * 768)  # Example 768-dim embedding
        entity.set_aliases(["Python", "py", "python3", "Python 3"])

        session.add(entity)
        session.commit()
        session.refresh(entity)

        entity_id = entity.id
        print(f"Created entity with ID: {entity_id}")

        # Index the entity for search
        index_entity_aliases(entity_id, entity.get_aliases())
        index_entity_embedding(entity_id, entity.get_embedding())

    # Create another entity
    with Session(engine) as session:
        entity2 = Entity(
            name="JavaScript",
            type="programming_language",
            description="A scripting language for web development",
            created_at=datetime.now().isoformat(),
        )
        entity2.set_embedding([0.2] * 768)
        entity2.set_aliases(["JavaScript", "JS", "js", "ECMAScript"])

        session.add(entity2)
        session.commit()
        session.refresh(entity2)

        entity2_id = entity2.id
        print(f"Created entity with ID: {entity2_id}")

        index_entity_aliases(entity2_id, entity2.get_aliases())
        index_entity_embedding(entity2_id, entity2.get_embedding())

    # Create a relationship
    with Session(engine) as session:
        relationship = Relationship(
            source_entity_id=entity_id,
            target_entity_id=entity2_id,
            relationship_type="similar_to",
            description="Both are programming languages",
            created_at=datetime.now().isoformat(),
        )
        relationship.set_embedding([0.15] * 768)

        session.add(relationship)
        session.commit()
        print(f"Created relationship between {entity_id} and {entity2_id}")

    # Search aliases using BM25
    print("\n=== BM25 Alias Search for 'python' ===")
    results = search_aliases_bm25("python", limit=5)
    print(f"Found entity IDs: {results}")

    # Vector search
    print("\n=== Vector Search ===")
    query_embedding = [0.1] * 768  # Similar to Python's embedding
    results = vector_search(query_embedding, table="entity", limit=5)
    print(f"Vector search results (id, distance): {results}")

    # Retrieve entities
    print("\n=== Retrieve Entities ===")
    with Session(engine) as session:
        entity = session.get(Entity, entity_id)
        if entity:
            print(f"Entity: {entity.name}")
            print(f"  Type: {entity.type}")
            print(f"  Aliases: {entity.get_aliases()}")
            print(f"  Embedding dimensions: {len(entity.get_embedding())}")


def example_clear_database():
    """Demonstrate clearing the database."""
    print("\n=== Clearing Database ===")
    clear_database()
    print("Database cleared successfully")


if __name__ == "__main__":
    # Run basic example
    example_basic_usage()

    # Uncomment to clear database
    # example_clear_database()
