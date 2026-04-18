"""
VectorEmbedding — stores learned representations for retrieval / similarity.

The 001 migration creates a `vector_embeddings` table with the vector column
declared as `Text` and later cast to pgvector's `vector(N)` via raw SQL. At
the SQLAlchemy level we model `embedding` as `Text` too; Phase 6 will
replace this with the proper pgvector `Vector` column type and typed
similarity queries. Keeping it as `Text` now avoids a hard dependency on
`pgvector` at import time.

Restored in Phase 1 / Session 1.2.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class VectorEmbedding(Base):
    """
    Per-entity learned embedding vector.

    Generic shape — the entity type is a free-form string so any table can
    have associated embeddings without new schema. For typed relationships
    (e.g. Structure.embeddings), use manual join clauses at the query level.
    """

    __tablename__ = "vector_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    entity_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True,
        comment="Entity table the embedding is attached to (e.g. 'structure').",
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True,
    )

    # Vector payload. Stored as text here; pgvector casts at the DB layer.
    # Phase 6 introduces a proper Vector(256) column + ANN index.
    embedding: Mapped[str] = mapped_column(Text, nullable=False)

    model_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Name of the embedding model (e.g. 'cgcnn', 'matminer-magpie').",
    )
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    embedding_dimension: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Dimensionality of the vector.",
    )

    extra_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=dict,
        comment="Free-form metadata (featurizer config, normalization, etc.).",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True,
    )

    def __repr__(self) -> str:
        return (
            f"<VectorEmbedding(id={self.id}, entity={self.entity_type}:"
            f"{self.entity_id}, model={self.model_name}, dim={self.embedding_dimension})>"
        )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "entity_type": self.entity_type,
            "entity_id": str(self.entity_id),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "embedding_dimension": self.embedding_dimension,
            "metadata": self.extra_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
