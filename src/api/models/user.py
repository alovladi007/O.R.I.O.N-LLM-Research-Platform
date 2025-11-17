"""
User model for authentication and authorization.

Supports:
- Email/password authentication
- Role-based access control (RBAC)
- OAuth2/SSO integration (future)
- API key authentication (future)
"""

import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import String, Boolean, DateTime, Text, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
import enum

from ..database import Base


class UserRole(str, enum.Enum):
    """User roles for RBAC."""
    ADMIN = "admin"          # Full system access
    RESEARCHER = "researcher"  # Can create materials, run simulations
    VIEWER = "viewer"         # Read-only access
    SERVICE = "service"       # For API services/integrations


class User(Base):
    """
    User model for authentication and profile management.

    Future sessions will add:
    - Organization/team management
    - Resource quotas and usage tracking
    - Fine-grained permissions
    """
    __tablename__ = "users"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Authentication
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    # Profile
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, native_enum=False),
        nullable=False,
        default=UserRole.RESEARCHER
    )

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # OAuth/SSO (future use)
    oauth_provider: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    oauth_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Soft delete
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Relationships
    design_campaigns: Mapped[List["DesignCampaign"]] = relationship(
        "DesignCampaign",
        back_populates="owner",
        cascade="all, delete-orphan"
    )
    # materials: Mapped[list["Material"]] = relationship(back_populates="created_by")
    # simulation_jobs: Mapped[list["SimulationJob"]] = relationship(back_populates="submitted_by")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"

    @property
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return self.role == UserRole.ADMIN or self.is_superuser

    def can_create_materials(self) -> bool:
        """Check if user can create materials."""
        return self.role in [UserRole.ADMIN, UserRole.RESEARCHER]

    def can_run_simulations(self) -> bool:
        """Check if user can submit simulation jobs."""
        return self.role in [UserRole.ADMIN, UserRole.RESEARCHER]
