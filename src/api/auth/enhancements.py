"""
Authentication Enhancements
===========================

Advanced authentication features:
- Token refresh and rotation
- Token blacklisting/revocation
- Password policy enforcement
- Account lockout after failed attempts
- Session management
- MFA (Multi-Factor Authentication) support
- Security event logging
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging
import hashlib
import pyotp
import qrcode
import io
import base64

from ..cache import cache_set, cache_get, cache_delete, cache_increment
from ..config import settings

logger = logging.getLogger(__name__)


# ========== Token Revocation (Blacklist) ==========


class TokenBlacklist:
    """
    Token blacklist using Redis.

    Stores revoked tokens to prevent their reuse.
    """

    @staticmethod
    async def revoke_token(token: str, expires_in_seconds: int):
        """
        Add token to blacklist.

        Args:
            token: JWT token to revoke
            expires_in_seconds: Time until token expires naturally
        """
        # Hash token for privacy
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        key = f"blacklist:token:{token_hash}"

        await cache_set(key, "revoked", ttl=expires_in_seconds)
        logger.info(f"Token revoked: {token_hash[:16]}...")

    @staticmethod
    async def is_token_revoked(token: str) -> bool:
        """
        Check if token has been revoked.

        Args:
            token: JWT token to check

        Returns:
            True if token is in blacklist
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        key = f"blacklist:token:{token_hash}"

        result = await cache_get(key)
        return result is not None

    @staticmethod
    async def revoke_all_user_tokens(user_id: str):
        """
        Revoke all tokens for a specific user.

        This is done by incrementing a user version counter.
        All tokens with older version become invalid.

        Args:
            user_id: User ID
        """
        key = f"user_token_version:{user_id}"
        await cache_increment(key)
        logger.info(f"All tokens revoked for user: {user_id}")

    @staticmethod
    async def get_user_token_version(user_id: str) -> int:
        """
        Get current token version for user.

        Args:
            user_id: User ID

        Returns:
            Current token version (0 if not set)
        """
        key = f"user_token_version:{user_id}"
        version = await cache_get(key)
        return int(version) if version else 0


# ========== Password Policy Enforcement ==========


class PasswordPolicy:
    """
    Enforce password complexity requirements.
    """

    MIN_LENGTH = 8
    MAX_LENGTH = 128
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGIT = True
    REQUIRE_SPECIAL = True

    SPECIAL_CHARACTERS = "!@#$%^&*()_+-=[]{}|;:,.<>?"

    @classmethod
    def validate_password(cls, password: str) -> Tuple[bool, Optional[str]]:
        """
        Validate password against policy.

        Args:
            password: Password to validate

        Returns:
            (is_valid, error_message)
        """
        # Length check
        if len(password) < cls.MIN_LENGTH:
            return False, f"Password must be at least {cls.MIN_LENGTH} characters"

        if len(password) > cls.MAX_LENGTH:
            return False, f"Password must not exceed {cls.MAX_LENGTH} characters"

        # Uppercase check
        if cls.REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"

        # Lowercase check
        if cls.REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"

        # Digit check
        if cls.REQUIRE_DIGIT and not any(c.isdigit() for c in password):
            return False, "Password must contain at least one digit"

        # Special character check
        if cls.REQUIRE_SPECIAL and not any(c in cls.SPECIAL_CHARACTERS for c in password):
            return False, f"Password must contain at least one special character ({cls.SPECIAL_CHARACTERS})"

        return True, None

    @staticmethod
    async def check_password_history(user_id: str, password_hash: str, history_count: int = 5) -> bool:
        """
        Check if password has been used recently.

        Args:
            user_id: User ID
            password_hash: Hash of new password
            history_count: Number of previous passwords to check

        Returns:
            True if password was used recently
        """
        key = f"password_history:{user_id}"
        history = await cache_get(key, default=[])

        if not isinstance(history, list):
            history = []

        return password_hash in history[:history_count]

    @staticmethod
    async def add_to_password_history(user_id: str, password_hash: str, max_history: int = 5):
        """
        Add password hash to user's password history.

        Args:
            user_id: User ID
            password_hash: Hash of password
            max_history: Maximum passwords to keep in history
        """
        key = f"password_history:{user_id}"
        history = await cache_get(key, default=[])

        if not isinstance(history, list):
            history = []

        # Add new password to front of list
        history.insert(0, password_hash)

        # Keep only max_history passwords
        history = history[:max_history]

        # Store with 1 year TTL
        await cache_set(key, history, ttl=365 * 24 * 3600)


# ========== Account Lockout ==========


class AccountLockout:
    """
    Implement account lockout after failed login attempts.
    """

    MAX_ATTEMPTS = 5
    LOCKOUT_DURATION_SECONDS = 900  # 15 minutes

    @staticmethod
    async def record_failed_attempt(identifier: str):
        """
        Record a failed login attempt.

        Args:
            identifier: User email or username
        """
        key = f"failed_login:{identifier}"

        # Increment counter
        attempts = await cache_increment(key)

        # Set expiry on first attempt
        if attempts == 1:
            await cache_set(key, attempts, ttl=AccountLockout.LOCKOUT_DURATION_SECONDS)

        logger.warning(f"Failed login attempt for {identifier}: {attempts} attempts")

        return attempts

    @staticmethod
    async def is_locked_out(identifier: str) -> Tuple[bool, int]:
        """
        Check if account is locked out.

        Args:
            identifier: User email or username

        Returns:
            (is_locked, attempts)
        """
        key = f"failed_login:{identifier}"
        attempts = await cache_get(key, default=0)

        if not isinstance(attempts, int):
            attempts = int(attempts) if attempts else 0

        is_locked = attempts >= AccountLockout.MAX_ATTEMPTS

        return is_locked, attempts

    @staticmethod
    async def reset_failed_attempts(identifier: str):
        """
        Reset failed login attempts (after successful login).

        Args:
            identifier: User email or username
        """
        key = f"failed_login:{identifier}"
        await cache_delete(key)
        logger.info(f"Failed login attempts reset for {identifier}")


# ========== Session Management ==========


class SessionManager:
    """
    Enhanced session management with device tracking.
    """

    @staticmethod
    async def create_session(
        user_id: str,
        device_info: dict,
        ip_address: str,
        ttl: int = 3600
    ) -> str:
        """
        Create a new session.

        Args:
            user_id: User ID
            device_info: Device information (user agent, etc.)
            ip_address: IP address
            ttl: Session TTL in seconds

        Returns:
            Session ID
        """
        import secrets
        session_id = secrets.token_urlsafe(32)

        session_data = {
            "user_id": user_id,
            "device": device_info,
            "ip_address": ip_address,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat()
        }

        key = f"session:{session_id}"
        await cache_set(key, session_data, ttl=ttl)

        # Track user sessions
        user_sessions_key = f"user_sessions:{user_id}"
        sessions = await cache_get(user_sessions_key, default=[])
        if not isinstance(sessions, list):
            sessions = []

        sessions.append(session_id)
        await cache_set(user_sessions_key, sessions, ttl=ttl)

        return session_id

    @staticmethod
    async def get_session(session_id: str) -> Optional[dict]:
        """Get session data"""
        key = f"session:{session_id}"
        return await cache_get(key)

    @staticmethod
    async def update_session_activity(session_id: str):
        """Update last activity timestamp"""
        session = await SessionManager.get_session(session_id)
        if session:
            session["last_activity"] = datetime.utcnow().isoformat()
            key = f"session:{session_id}"
            await cache_set(key, session, ttl=3600)

    @staticmethod
    async def revoke_session(session_id: str):
        """Revoke a specific session"""
        key = f"session:{session_id}"
        await cache_delete(key)
        logger.info(f"Session revoked: {session_id[:16]}...")

    @staticmethod
    async def revoke_all_user_sessions(user_id: str):
        """Revoke all sessions for a user"""
        user_sessions_key = f"user_sessions:{user_id}"
        sessions = await cache_get(user_sessions_key, default=[])

        if isinstance(sessions, list):
            for session_id in sessions:
                await SessionManager.revoke_session(session_id)

        await cache_delete(user_sessions_key)
        logger.info(f"All sessions revoked for user: {user_id}")


# ========== Multi-Factor Authentication (MFA) ==========


class MFAManager:
    """
    Multi-Factor Authentication using TOTP (Time-based One-Time Password).
    """

    @staticmethod
    def generate_totp_secret() -> str:
        """
        Generate a new TOTP secret.

        Returns:
            Base32-encoded secret
        """
        return pyotp.random_base32()

    @staticmethod
    def generate_totp_uri(secret: str, user_email: str, issuer: str = "ORION Platform") -> str:
        """
        Generate TOTP provisioning URI for QR code.

        Args:
            secret: TOTP secret
            user_email: User's email
            issuer: Application name

        Returns:
            TOTP URI
        """
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=user_email, issuer_name=issuer)

    @staticmethod
    def generate_qr_code(uri: str) -> str:
        """
        Generate QR code image from TOTP URI.

        Args:
            uri: TOTP provisioning URI

        Returns:
            Base64-encoded PNG image
        """
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return img_str

    @staticmethod
    def verify_totp_code(secret: str, code: str) -> bool:
        """
        Verify a TOTP code.

        Args:
            secret: TOTP secret
            code: 6-digit code from authenticator app

        Returns:
            True if code is valid
        """
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=1)  # Allow 1 time step tolerance

    @staticmethod
    async def enable_mfa(user_id: str, secret: str):
        """
        Enable MFA for a user.

        Args:
            user_id: User ID
            secret: TOTP secret
        """
        key = f"mfa:{user_id}"
        await cache_set(key, {"enabled": True, "secret": secret}, ttl=None)
        logger.info(f"MFA enabled for user: {user_id}")

    @staticmethod
    async def disable_mfa(user_id: str):
        """
        Disable MFA for a user.

        Args:
            user_id: User ID
        """
        key = f"mfa:{user_id}"
        await cache_delete(key)
        logger.info(f"MFA disabled for user: {user_id}")

    @staticmethod
    async def is_mfa_enabled(user_id: str) -> bool:
        """
        Check if MFA is enabled for user.

        Args:
            user_id: User ID

        Returns:
            True if MFA is enabled
        """
        key = f"mfa:{user_id}"
        mfa_data = await cache_get(key)
        return mfa_data is not None and mfa_data.get("enabled", False)

    @staticmethod
    async def get_mfa_secret(user_id: str) -> Optional[str]:
        """
        Get MFA secret for user.

        Args:
            user_id: User ID

        Returns:
            TOTP secret or None
        """
        key = f"mfa:{user_id}"
        mfa_data = await cache_get(key)
        return mfa_data.get("secret") if mfa_data else None


# ========== Security Event Logging ==========


class SecurityLogger:
    """
    Log security-related events for auditing.
    """

    @staticmethod
    async def log_event(
        event_type: str,
        user_id: Optional[str] = None,
        details: Optional[dict] = None,
        ip_address: Optional[str] = None
    ):
        """
        Log a security event.

        Args:
            event_type: Type of event (login, logout, password_change, etc.)
            user_id: User ID (if applicable)
            details: Additional event details
            ip_address: IP address
        """
        event = {
            "type": event_type,
            "user_id": user_id,
            "details": details or {},
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Log to application logs
        logger.info(f"Security event: {event_type}", extra=event)

        # Store in cache for recent events
        key = f"security_events:{user_id}" if user_id else "security_events:system"
        events = await cache_get(key, default=[])

        if not isinstance(events, list):
            events = []

        events.insert(0, event)
        events = events[:100]  # Keep last 100 events

        await cache_set(key, events, ttl=7 * 24 * 3600)  # 7 days

    @staticmethod
    async def get_user_security_events(user_id: str, limit: int = 20) -> list:
        """
        Get recent security events for a user.

        Args:
            user_id: User ID
            limit: Maximum number of events to return

        Returns:
            List of security events
        """
        key = f"security_events:{user_id}"
        events = await cache_get(key, default=[])

        if not isinstance(events, list):
            events = []

        return events[:limit]


# ========== Rate Limiting for Auth Endpoints ==========


async def check_auth_rate_limit(identifier: str, max_requests: int = 10) -> Tuple[bool, int]:
    """
    Check rate limit for authentication endpoints.

    More restrictive than general API rate limits.

    Args:
        identifier: User identifier (email, IP, etc.)
        max_requests: Maximum requests per minute

    Returns:
        (is_allowed, requests_remaining)
    """
    from ..cache import check_rate_limit
    return await check_rate_limit(
        f"auth:{identifier}",
        max_requests=max_requests,
        window_seconds=60
    )
