"""
Authentication and user management router for NANO-OS API.

Provides:
- User registration
- Login with JWT tokens (bearer **or** httpOnly cookie — see ``mode=``)
- Token refresh
- Logout (clears cookies if cookie-mode was used)
- Current user information
- OAuth2 password flow

Auth modes
----------

The ``POST /auth/login`` endpoint supports two response modes via the
``mode`` query parameter (default ``bearer``):

- ``mode=bearer`` (default) — returns the access + refresh tokens in
  the response body, as before. Used by curl / Postman / the Python
  SDK / the OAuth2 password flow.
- ``mode=cookie`` (Phase 9 / Session 9.1) — returns the same
  :class:`Token` body **plus** sets the access + refresh tokens as
  ``httpOnly``, ``SameSite=Lax`` cookies. The frontend never sees
  the raw token (no ``localStorage``); the
  :func:`backend.common...security.get_current_user` dependency
  reads from the cookie when no ``Authorization`` header is present.

Backward-compatible — bearer clients keep working unchanged.
"""

from fastapi import APIRouter, Depends, Query, Request, Response, status, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
from typing import Literal, Optional
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
    ACCESS_COOKIE_NAME,
    REFRESH_COOKIE_NAME,
    SecurityService,
    get_current_user,
    get_current_active_user,
)
from ..config import settings as app_settings
from ..exceptions import (
    AuthenticationError,
    ConflictError,
    NotFoundError,
    ValidationError
)
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    # No prefix — app.py mounts this router under ``/api/v1/auth``.
    # (Pre-Phase-9 the router *also* had ``prefix="/auth"`` which made
    # every endpoint live at ``/api/v1/auth/auth/...``; the OAuth2
    # scheme's ``tokenUrl`` already pointed at the un-doubled
    # ``/api/v1/auth/token``, so the bug had been latent. Session 9.1
    # cleans it up.)
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
    response: Response,
    mode: Literal["bearer", "cookie"] = Query(
        "bearer",
        description=(
            "Token-delivery mode. 'bearer' returns tokens in the body "
            "(default). 'cookie' additionally sets httpOnly, SameSite=Lax "
            "access + refresh cookies that subsequent requests can use "
            "in place of the Authorization header."
        ),
    ),
    db: AsyncSession = Depends(get_db),
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

    if mode == "cookie":
        # httpOnly + SameSite=Lax to keep the access token out of JS
        # while still letting top-level navigations carry it. ``secure``
        # mirrors the production-environment flag — in dev (HTTP) we
        # leave it off so the browser will accept the cookie over
        # http://localhost.
        secure = app_settings.is_production
        response.set_cookie(
            key=ACCESS_COOKIE_NAME,
            value=access_token,
            max_age=int(access_token_expires.total_seconds()),
            httponly=True,
            secure=secure,
            samesite="lax",
            path="/",
        )
        response.set_cookie(
            key=REFRESH_COOKIE_NAME,
            value=refresh_token,
            max_age=int(refresh_token_expires.total_seconds()),
            httponly=True,
            secure=secure,
            samesite="lax",
            # Refresh-token endpoint lives at /api/v1/auth/refresh; we
            # could scope the cookie there, but FastAPI clients treat
            # path scoping inconsistently across browsers. Keep at /
            # for now and revisit in Session 11 (security hardening).
            path="/",
        )

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
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> Token:
    """
    OAuth2 password flow - standard token endpoint.

    This endpoint is used by OAuth2PasswordBearer scheme. Always uses
    ``mode=bearer`` — OAuth2 callers consume the body tokens directly.
    """
    # Convert OAuth2 form to our UserLogin schema
    credentials = UserLogin(
        username=form_data.username,
        password=form_data.password
    )

    # Reuse login logic with bearer mode (no cookies for OAuth2 flow).
    return await login(credentials, response, mode="bearer", db=db)


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
    request: Request,
    response: Response,
    token_data: Optional[TokenRefresh] = None,
    mode: Literal["bearer", "cookie"] = Query(
        "bearer",
        description=(
            "If 'cookie', read the refresh token from the "
            "orion_refresh_token httpOnly cookie instead of the body, "
            "and set fresh cookies on the response."
        ),
    ),
    db: AsyncSession = Depends(get_db),
) -> Token:
    """
    Refresh access token using refresh token.
    """
    logger.debug("Token refresh requested")

    try:
        # Resolve the refresh token: cookie mode reads from request.cookies,
        # bearer mode requires the body.
        if mode == "cookie":
            refresh_str = request.cookies.get(REFRESH_COOKIE_NAME)
            if not refresh_str:
                raise AuthenticationError("Missing refresh-token cookie")
        else:
            if token_data is None or not token_data.refresh_token:
                raise AuthenticationError("Missing refresh_token in body")
            refresh_str = token_data.refresh_token

        # Decode refresh token
        payload = SecurityService.decode_token(refresh_str)

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

        if mode == "cookie":
            secure = app_settings.is_production
            response.set_cookie(
                key=ACCESS_COOKIE_NAME,
                value=access_token,
                max_age=int(access_token_expires.total_seconds()),
                httponly=True,
                secure=secure,
                samesite="lax",
                path="/",
            )
            response.set_cookie(
                key=REFRESH_COOKIE_NAME,
                value=refresh_token,
                max_age=int(refresh_token_expires.total_seconds()),
                httponly=True,
                secure=secure,
                samesite="lax",
                path="/",
            )

        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=int(access_token_expires.total_seconds()),
            user=UserResponse.model_validate(user)
        )

    except AuthenticationError:
        raise
    except Exception as e:
        logger.warning(f"Token refresh failed: {e}")
        raise AuthenticationError("Invalid or expired refresh token")


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Log out — clears auth cookies",
    description=(
        "Clears the orion_access_token and orion_refresh_token cookies "
        "if they are present. Bearer-token clients can ignore this — "
        "JWT statelessness means there's no server-side state to clear; "
        "the client should drop its in-memory token. Returns 204."
    ),
)
async def logout() -> Response:
    """Clear the auth cookies. Idempotent — safe to call when not logged in.

    Returns a fresh 204 response with the two cookie-deletion
    Set-Cookie headers attached. We construct the response directly
    instead of mutating an injected ``Response`` because the latter
    pattern is incompatible with FastAPI's status_code=204 contract
    (which yields an empty body and ignores ``set_cookie`` calls
    routed through the dependency-injected response).
    """
    resp = Response(status_code=status.HTTP_204_NO_CONTENT)
    resp.delete_cookie(ACCESS_COOKIE_NAME, path="/")
    resp.delete_cookie(REFRESH_COOKIE_NAME, path="/")
    return resp


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
