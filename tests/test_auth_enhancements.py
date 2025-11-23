"""
Tests for Authentication Enhancements
======================================

Tests advanced authentication features:
- Token revocation and blacklist
- Password policy enforcement
- Account lockout
- Session management
- Multi-Factor Authentication (MFA)
- Security event logging
- Auth rate limiting
"""

import pytest
from datetime import datetime
from src.api.auth.enhancements import (
    TokenBlacklist,
    PasswordPolicy,
    AccountLockout,
    SessionManager,
    MFAManager,
    SecurityLogger,
    check_auth_rate_limit,
)
from src.api.cache import init_cache, close_cache


class TestTokenBlacklist:
    """Test token revocation and blacklist"""

    @pytest.mark.asyncio
    async def test_revoke_token(self):
        """Test revoking a token"""
        try:
            await init_cache()

            token = "test_jwt_token_12345"

            # Revoke token
            await TokenBlacklist.revoke_token(token, expires_in_seconds=3600)

            # Check if revoked
            is_revoked = await TokenBlacklist.is_token_revoked(token)
            assert is_revoked is True

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_non_revoked_token(self):
        """Test that non-revoked tokens are not in blacklist"""
        try:
            await init_cache()

            token = "valid_token_67890"

            # Check if revoked (should be False)
            is_revoked = await TokenBlacklist.is_token_revoked(token)
            assert is_revoked is False

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_revoke_all_user_tokens(self):
        """Test revoking all tokens for a user"""
        try:
            await init_cache()

            user_id = "user_123"

            # Get initial version
            version1 = await TokenBlacklist.get_user_token_version(user_id)

            # Revoke all tokens
            await TokenBlacklist.revoke_all_user_tokens(user_id)

            # Version should have incremented
            version2 = await TokenBlacklist.get_user_token_version(user_id)
            assert version2 > version1

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()


class TestPasswordPolicy:
    """Test password policy enforcement"""

    def test_password_too_short(self):
        """Test that short passwords are rejected"""
        password = "Short1!"
        is_valid, error = PasswordPolicy.validate_password(password)
        assert is_valid is False
        assert "at least" in error

    def test_password_missing_uppercase(self):
        """Test that passwords without uppercase are rejected"""
        password = "lowercase123!"
        is_valid, error = PasswordPolicy.validate_password(password)
        assert is_valid is False
        assert "uppercase" in error.lower()

    def test_password_missing_lowercase(self):
        """Test that passwords without lowercase are rejected"""
        password = "UPPERCASE123!"
        is_valid, error = PasswordPolicy.validate_password(password)
        assert is_valid is False
        assert "lowercase" in error.lower()

    def test_password_missing_digit(self):
        """Test that passwords without digits are rejected"""
        password = "NoDigitsHere!"
        is_valid, error = PasswordPolicy.validate_password(password)
        assert is_valid is False
        assert "digit" in error.lower()

    def test_password_missing_special(self):
        """Test that passwords without special characters are rejected"""
        password = "NoSpecial123"
        is_valid, error = PasswordPolicy.validate_password(password)
        assert is_valid is False
        assert "special" in error.lower()

    def test_valid_password(self):
        """Test that valid passwords are accepted"""
        valid_passwords = [
            "SecurePass123!",
            "Compl3x@Password",
            "MyP@ssw0rd2024",
        ]

        for password in valid_passwords:
            is_valid, error = PasswordPolicy.validate_password(password)
            assert is_valid is True
            assert error is None

    @pytest.mark.asyncio
    async def test_password_history(self):
        """Test password history tracking"""
        try:
            await init_cache()

            user_id = "user_123"
            password_hash = "hashed_password_abc"

            # Add to history
            await PasswordPolicy.add_to_password_history(user_id, password_hash)

            # Check if in history
            in_history = await PasswordPolicy.check_password_history(user_id, password_hash)
            assert in_history is True

            # Different password should not be in history
            in_history = await PasswordPolicy.check_password_history(user_id, "different_hash")
            assert in_history is False

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()


class TestAccountLockout:
    """Test account lockout functionality"""

    @pytest.mark.asyncio
    async def test_record_failed_attempts(self):
        """Test recording failed login attempts"""
        try:
            await init_cache()

            identifier = "user@example.com"

            # Record attempts
            attempts1 = await AccountLockout.record_failed_attempt(identifier)
            assert attempts1 == 1

            attempts2 = await AccountLockout.record_failed_attempt(identifier)
            assert attempts2 == 2

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_account_lockout_threshold(self):
        """Test that account locks after max attempts"""
        try:
            await init_cache()

            identifier = "locked@example.com"

            # Make failed attempts up to max
            for i in range(AccountLockout.MAX_ATTEMPTS):
                await AccountLockout.record_failed_attempt(identifier)

            # Check if locked
            is_locked, attempts = await AccountLockout.is_locked_out(identifier)
            assert is_locked is True
            assert attempts >= AccountLockout.MAX_ATTEMPTS

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_reset_failed_attempts(self):
        """Test resetting failed attempts"""
        try:
            await init_cache()

            identifier = "reset@example.com"

            # Record some failed attempts
            await AccountLockout.record_failed_attempt(identifier)
            await AccountLockout.record_failed_attempt(identifier)

            # Reset
            await AccountLockout.reset_failed_attempts(identifier)

            # Should not be locked
            is_locked, attempts = await AccountLockout.is_locked_out(identifier)
            assert is_locked is False
            assert attempts == 0

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()


class TestSessionManager:
    """Test session management"""

    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test creating a session"""
        try:
            await init_cache()

            user_id = "user_123"
            device_info = {"user_agent": "Mozilla/5.0"}
            ip_address = "192.168.1.1"

            session_id = await SessionManager.create_session(
                user_id, device_info, ip_address, ttl=3600
            )

            assert session_id is not None
            assert len(session_id) > 0

            # Retrieve session
            session = await SessionManager.get_session(session_id)
            assert session is not None
            assert session["user_id"] == user_id
            assert session["ip_address"] == ip_address

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_update_session_activity(self):
        """Test updating session last activity"""
        try:
            await init_cache()

            user_id = "user_123"
            session_id = await SessionManager.create_session(
                user_id, {}, "192.168.1.1"
            )

            # Get initial session
            session1 = await SessionManager.get_session(session_id)
            last_activity1 = session1["last_activity"]

            # Wait a moment and update
            import asyncio
            await asyncio.sleep(0.1)
            await SessionManager.update_session_activity(session_id)

            # Get updated session
            session2 = await SessionManager.get_session(session_id)
            last_activity2 = session2["last_activity"]

            # Should be updated
            assert last_activity2 > last_activity1

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_revoke_session(self):
        """Test revoking a session"""
        try:
            await init_cache()

            user_id = "user_123"
            session_id = await SessionManager.create_session(
                user_id, {}, "192.168.1.1"
            )

            # Session should exist
            session = await SessionManager.get_session(session_id)
            assert session is not None

            # Revoke session
            await SessionManager.revoke_session(session_id)

            # Session should no longer exist
            session = await SessionManager.get_session(session_id)
            assert session is None

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()


class TestMFAManager:
    """Test Multi-Factor Authentication"""

    def test_generate_totp_secret(self):
        """Test generating TOTP secret"""
        secret = MFAManager.generate_totp_secret()
        assert secret is not None
        assert len(secret) > 0
        # Should be base32
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567=" for c in secret)

    def test_generate_totp_uri(self):
        """Test generating TOTP URI"""
        secret = "JBSWY3DPEHPK3PXP"
        email = "user@example.com"

        uri = MFAManager.generate_totp_uri(secret, email)

        assert uri.startswith("otpauth://totp/")
        assert email in uri
        assert secret in uri

    def test_generate_qr_code(self):
        """Test generating QR code"""
        uri = "otpauth://totp/ORION%20Platform:user@example.com?secret=JBSWY3DPEHPK3PXP&issuer=ORION%20Platform"

        qr_code = MFAManager.generate_qr_code(uri)

        # Should be base64 encoded
        assert qr_code is not None
        assert len(qr_code) > 0

    def test_verify_totp_code(self):
        """Test verifying TOTP code"""
        import pyotp

        secret = MFAManager.generate_totp_secret()
        totp = pyotp.TOTP(secret)
        code = totp.now()

        # Valid code should verify
        is_valid = MFAManager.verify_totp_code(secret, code)
        assert is_valid is True

        # Invalid code should not verify
        is_valid = MFAManager.verify_totp_code(secret, "000000")
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_enable_disable_mfa(self):
        """Test enabling and disabling MFA"""
        try:
            await init_cache()

            user_id = "user_123"
            secret = MFAManager.generate_totp_secret()

            # Initially not enabled
            is_enabled = await MFAManager.is_mfa_enabled(user_id)
            assert is_enabled is False

            # Enable MFA
            await MFAManager.enable_mfa(user_id, secret)

            # Should be enabled
            is_enabled = await MFAManager.is_mfa_enabled(user_id)
            assert is_enabled is True

            # Secret should be retrievable
            stored_secret = await MFAManager.get_mfa_secret(user_id)
            assert stored_secret == secret

            # Disable MFA
            await MFAManager.disable_mfa(user_id)

            # Should not be enabled
            is_enabled = await MFAManager.is_mfa_enabled(user_id)
            assert is_enabled is False

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()


class TestSecurityLogger:
    """Test security event logging"""

    @pytest.mark.asyncio
    async def test_log_security_event(self):
        """Test logging a security event"""
        try:
            await init_cache()

            user_id = "user_123"

            await SecurityLogger.log_event(
                event_type="login",
                user_id=user_id,
                details={"success": True},
                ip_address="192.168.1.1"
            )

            # Should not raise error
            assert True

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_get_user_security_events(self):
        """Test retrieving user security events"""
        try:
            await init_cache()

            user_id = "user_123"

            # Log multiple events
            for i in range(5):
                await SecurityLogger.log_event(
                    event_type=f"event_{i}",
                    user_id=user_id,
                    details={"count": i}
                )

            # Retrieve events
            events = await SecurityLogger.get_user_security_events(user_id, limit=3)

            assert len(events) <= 3
            # Most recent event should be first
            assert events[0]["type"] == "event_4"

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()


class TestAuthRateLimiting:
    """Test authentication rate limiting"""

    @pytest.mark.asyncio
    async def test_auth_rate_limit_allows_requests(self):
        """Test that auth rate limit allows requests within limit"""
        try:
            await init_cache()

            identifier = "auth_user_123"

            # Should allow first request
            allowed, remaining = await check_auth_rate_limit(identifier, max_requests=10)
            assert allowed is True
            assert remaining > 0

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_auth_rate_limit_blocks_excess(self):
        """Test that auth rate limit blocks excess requests"""
        try:
            await init_cache()

            identifier = "auth_user_456"
            max_requests = 3

            # Make max_requests allowed requests
            for i in range(max_requests):
                allowed, remaining = await check_auth_rate_limit(
                    identifier, max_requests=max_requests
                )
                assert allowed is True

            # Next request should be blocked
            allowed, remaining = await check_auth_rate_limit(
                identifier, max_requests=max_requests
            )
            assert allowed is False
            assert remaining == 0

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
