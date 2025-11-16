"""
Authentication and user management router for NANO-OS API.

Provides:
- User registration
- Login with JWT tokens
- Token refresh
- Current user information
- OAuth2 password flow
"""

from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
from typing import Optional
import logging
import uuid

from ..database import get_db
from ..models import User
from ..schemas.auth import (
    UserCreate,
    UserLogin,
    UserResponse,
    Token,
    TokenRefresh,
)
from ..auth.security import (
    SecurityService,
    get_current_user,
    get_current_active_user
)
from ..exceptions import (
    AuthenticationError,
    ConflictError,
    NotFoundError,
    ValidationError
)
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
    responses={
        401: {"description": "Authentication failed"},
        403: {"description": "Not authorized"}
    }
)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="""
    Create a new user account.

    Requirements:
    - Unique email and username
    - Password must meet security requirements:
      - At least 8 characters
      - At least one uppercase letter
      - At least one lowercase letter
      - At least one digit

    Returns the created user (without password).
    """,
    responses={
        201: {
            "description": "User created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "email": "researcher@example.com",
                        "username": "researcher",
                        "full_name": "Jane Doe",
                        "role": "researcher",
                        "is_active": True,
                        "is_verified": False,
                        "created_at": "2024-01-15T10:30:00Z"
                    }
                }
            }
        },
        409: {"description": "Email or username already exists"}
    }
)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Register a new user account.
    """
    logger.info(f"Registering new user: {user_data.username}")

    # Check if email already exists
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    if result.scalar_one_or_none():
        raise ConflictError(
            message="Email already registered",
            details={"field": "email", "value": user_data.email}
        )

    # Check if username already exists
    result = await db.execute(
        select(User).where(User.username == user_data.username)
    )
    if result.scalar_one_or_none():
        raise ConflictError(
            message="Username already taken",
            details={"field": "username", "value": user_data.username}
        )

    # Hash password
    hashed_password = SecurityService.hash_password(user_data.password)

    # Create user
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        is_active=True,
        is_verified=False  # Future: email verification
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    logger.info(f"User registered successfully: {new_user.id}")

    return UserResponse.model_validate(new_user)


@router.post(
    "/login",
    response_model=Token,
    summary="User login",
    description="""
    Authenticate user and return JWT tokens.

    Accepts username or email with password.
    Returns access token (short-lived) and refresh token (long-lived).

    Access token expires in 30 minutes (default).
    Refresh token expires in 7 days (default).
    """,
    responses={
        200: {
            "description": "Login successful",
            "content": {
                "application/json": {
                    "example": {
                        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                        "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                        "token_type": "bearer",
                        "expires_in": 1800,
                        "user": {
                            "id": "123e4567-e89b-12d3-a456-426614174000",
                            "email": "researcher@example.com",
                            "username": "researcher",
                            "role": "researcher"
                        }
                    }
                }
            }
        },
        401: {"description": "Invalid credentials"}
    }
)
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db)
) -> Token:
    """
    Authenticate user and return JWT tokens.
    """
    logger.info(f"Login attempt for: {credentials.username}")

    # Find user by username or email
    result = await db.execute(
        select(User).where(
            (User.username == credentials.username) |
            (User.email == credentials.username)
        )
    )
    user = result.scalar_one_or_none()

    if not user:
        logger.warning(f"Login failed: user not found - {credentials.username}")
        raise AuthenticationError("Invalid username or password")

    # Verify password
    if not SecurityService.verify_password(credentials.password, user.hashed_password):
        logger.warning(f"Login failed: invalid password - {credentials.username}")
        raise AuthenticationError("Invalid username or password")

    # Check if user is active
    if not user.is_active:
        logger.warning(f"Login failed: inactive user - {credentials.username}")
        raise AuthenticationError("Account is inactive")

    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()

    # Create tokens
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    refresh_token_expires = timedelta(days=settings.refresh_token_expire_days)

    token_data = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role.value,
    }

    access_token = SecurityService.create_access_token(
        data=token_data,
        expires_delta=access_token_expires
    )

    refresh_token = SecurityService.create_refresh_token(
        data=token_data,
        expires_delta=refresh_token_expires
    )

    logger.info(f"Login successful: {user.username} ({user.id})")

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds()),
        user=UserResponse.model_validate(user)
    )


@router.post(
    "/token",
    response_model=Token,
    summary="OAuth2 token endpoint",
    description="OAuth2 password flow token endpoint (alternative to /login)",
    include_in_schema=True
)
async def token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
) -> Token:
    """
    OAuth2 password flow - standard token endpoint.

    This endpoint is used by OAuth2PasswordBearer scheme.
    """
    # Convert OAuth2 form to our UserLogin schema
    credentials = UserLogin(
        username=form_data.username,
        password=form_data.password
    )

    # Reuse login logic
    return await login(credentials, db)


@router.post(
    "/refresh",
    response_model=Token,
    summary="Refresh access token",
    description="""
    Refresh an expired access token using a refresh token.

    When the access token expires, use the refresh token to get
    a new access token without requiring the user to login again.

    The refresh token itself is also renewed.
    """,
    responses={
        200: {"description": "Token refreshed successfully"},
        401: {"description": "Invalid or expired refresh token"}
    }
)
async def refresh_token(
    token_data: TokenRefresh,
    db: AsyncSession = Depends(get_db)
) -> Token:
    """
    Refresh access token using refresh token.
    """
    logger.debug("Token refresh requested")

    try:
        # Decode refresh token
        payload = SecurityService.decode_token(token_data.refresh_token)

        # Validate token type
        if payload.get("type") != "refresh":
            raise AuthenticationError("Invalid token type")

        # Extract user info
        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationError("Invalid token")

        # Get user from database
        user = await db.get(User, uuid.UUID(user_id))
        if not user:
            raise NotFoundError("User", user_id)

        if not user.is_active:
            raise AuthenticationError("Account is inactive")

        # Create new tokens
        access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
        refresh_token_expires = timedelta(days=settings.refresh_token_expire_days)

        new_token_data = {
            "sub": str(user.id),
            "username": user.username,
            "role": user.role.value,
        }

        access_token = SecurityService.create_access_token(
            data=new_token_data,
            expires_delta=access_token_expires
        )

        refresh_token = SecurityService.create_refresh_token(
            data=new_token_data,
            expires_delta=refresh_token_expires
        )

        logger.info(f"Token refreshed for user: {user.username}")

        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=int(access_token_expires.total_seconds()),
            user=UserResponse.model_validate(user)
        )

    except Exception as e:
        logger.warning(f"Token refresh failed: {e}")
        raise AuthenticationError("Invalid or expired refresh token")


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="""
    Get information about the currently authenticated user.

    Requires valid JWT access token in Authorization header:
    ```
    Authorization: Bearer <access_token>
    ```
    """,
    responses={
        200: {
            "description": "Current user information",
            "content": {
                "application/json": {
                    "example": {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "email": "researcher@example.com",
                        "username": "researcher",
                        "full_name": "Jane Doe",
                        "role": "researcher",
                        "is_active": True,
                        "is_verified": True,
                        "created_at": "2024-01-15T10:30:00Z",
                        "last_login": "2024-01-15T14:30:00Z"
                    }
                }
            }
        },
        401: {"description": "Not authenticated"}
    }
)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
) -> UserResponse:
    """
    Get current authenticated user information.
    """
    return UserResponse.model_validate(current_user)
