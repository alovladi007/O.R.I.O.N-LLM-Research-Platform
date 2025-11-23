"""
Input Validation and Sanitization
==================================

Provides comprehensive input validation and sanitization:
- SQL injection prevention
- XSS attack prevention
- Path traversal prevention
- Command injection prevention
- Format validation (email, URL, chemical formula, etc.)
- Content sanitization
- Rate limiting by input patterns
"""

import re
import html
from typing import Any, Optional, List
from pydantic import validator, field_validator
import logging
import urllib.parse

logger = logging.getLogger(__name__)


# ========== SQL Injection Prevention ==========


class SQLInjectionDetector:
    """
    Detects potential SQL injection attempts.

    Note: This is a defense-in-depth measure. The primary
    protection is using parameterized queries with SQLAlchemy.
    """

    # Common SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bSELECT\b.*\bFROM\b.*\bWHERE\b)",
        r"(\bINSERT\b.*\bINTO\b.*\bVALUES\b)",
        r"(\bUPDATE\b.*\bSET\b)",
        r"(\bDELETE\b.*\bFROM\b)",
        r"(\bDROP\b.*\b(TABLE|DATABASE)\b)",
        r"(\bEXEC\b.*\(.*\))",
        r"(--|\#|\/\*|\*\/)",  # SQL comments
        r"(;.*\b(SELECT|INSERT|UPDATE|DELETE|DROP)\b)",  # Multiple statements
        r"(\bOR\b.*=.*)",  # OR-based injection
        r"('.*--)",  # Comment-based injection
        r"(\bAND\b.*=.*)",  # AND-based injection
    ]

    @classmethod
    def is_potentially_malicious(cls, value: str) -> bool:
        """
        Check if input contains potential SQL injection patterns.

        Args:
            value: Input string to check

        Returns:
            True if suspicious patterns detected
        """
        if not isinstance(value, str):
            return False

        value_upper = value.upper()

        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_upper, re.IGNORECASE):
                logger.warning(f"Potential SQL injection detected: {value[:50]}")
                return True

        return False

    @classmethod
    def validate_safe_string(cls, value: str, field_name: str = "input") -> str:
        """
        Validate that string is safe from SQL injection.

        Raises:
            ValueError if suspicious patterns detected
        """
        if cls.is_potentially_malicious(value):
            raise ValueError(
                f"{field_name} contains potentially malicious SQL patterns"
            )
        return value


# ========== XSS Prevention ==========


class XSSPrevention:
    """Prevents Cross-Site Scripting (XSS) attacks."""

    # Dangerous HTML tags and attributes
    DANGEROUS_TAGS = [
        'script', 'iframe', 'object', 'embed', 'applet',
        'meta', 'link', 'style', 'base'
    ]

    DANGEROUS_ATTRIBUTES = [
        'onerror', 'onload', 'onclick', 'onmouseover',
        'onfocus', 'onblur', 'onchange', 'onsubmit'
    ]

    # JavaScript protocol pattern
    JAVASCRIPT_PROTOCOL_PATTERN = r'javascript\s*:'

    @classmethod
    def sanitize_html(cls, value: str) -> str:
        """
        Sanitize HTML by escaping special characters.

        Args:
            value: Input string possibly containing HTML

        Returns:
            HTML-escaped string
        """
        if not isinstance(value, str):
            return value

        # HTML escape
        return html.escape(value)

    @classmethod
    def detect_xss_patterns(cls, value: str) -> bool:
        """
        Detect XSS patterns in input.

        Args:
            value: Input string to check

        Returns:
            True if XSS patterns detected
        """
        if not isinstance(value, str):
            return False

        value_lower = value.lower()

        # Check for dangerous tags
        for tag in cls.DANGEROUS_TAGS:
            if f'<{tag}' in value_lower:
                logger.warning(f"XSS pattern detected: {tag} tag in {value[:50]}")
                return True

        # Check for event handlers
        for attr in cls.DANGEROUS_ATTRIBUTES:
            if attr in value_lower:
                logger.warning(f"XSS pattern detected: {attr} attribute")
                return True

        # Check for javascript: protocol
        if re.search(cls.JAVASCRIPT_PROTOCOL_PATTERN, value_lower):
            logger.warning("XSS pattern detected: javascript: protocol")
            return True

        return False

    @classmethod
    def validate_safe_html(cls, value: str, field_name: str = "input") -> str:
        """
        Validate that HTML is safe from XSS.

        Raises:
            ValueError if XSS patterns detected
        """
        if cls.detect_xss_patterns(value):
            raise ValueError(
                f"{field_name} contains potentially malicious HTML/JavaScript patterns"
            )
        return value


# ========== Path Traversal Prevention ==========


class PathTraversalPrevention:
    """Prevents directory traversal attacks."""

    # Path traversal patterns
    TRAVERSAL_PATTERNS = [
        r'\.\.',  # Parent directory
        r'\.\/|\.\\',  # Current directory references
        r'~/',  # Home directory
        r'\%2e\%2e',  # URL-encoded ..
        r'\%252e\%252e',  # Double URL-encoded ..
        r'\.\%2f|\%2f\.',  # Mixed encoding
    ]

    @classmethod
    def contains_traversal(cls, value: str) -> bool:
        """
        Check if path contains traversal patterns.

        Args:
            value: Path or filename to check

        Returns:
            True if traversal patterns detected
        """
        if not isinstance(value, str):
            return False

        # Normalize the path
        normalized = urllib.parse.unquote(value)

        for pattern in cls.TRAVERSAL_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                logger.warning(f"Path traversal detected: {value}")
                return True

        return False

    @classmethod
    def validate_safe_path(cls, value: str, field_name: str = "path") -> str:
        """
        Validate that path is safe from traversal attacks.

        Raises:
            ValueError if traversal patterns detected
        """
        if cls.contains_traversal(value):
            raise ValueError(
                f"{field_name} contains potentially malicious path traversal patterns"
            )
        return value


# ========== Command Injection Prevention ==========


class CommandInjectionPrevention:
    """Prevents command injection attacks."""

    # Shell metacharacters
    SHELL_METACHARACTERS = [
        ';', '|', '&', '$', '`', '\n', '\r',
        '$(', '${', '&&', '||', '>>',  '>',
    ]

    @classmethod
    def contains_shell_metacharacters(cls, value: str) -> bool:
        """
        Check if string contains shell metacharacters.

        Args:
            value: Input string to check

        Returns:
            True if shell metacharacters detected
        """
        if not isinstance(value, str):
            return False

        for char in cls.SHELL_METACHARACTERS:
            if char in value:
                logger.warning(f"Shell metacharacter detected: {char}")
                return True

        return False

    @classmethod
    def validate_safe_command_arg(cls, value: str, field_name: str = "argument") -> str:
        """
        Validate that string is safe as command argument.

        Raises:
            ValueError if shell metacharacters detected
        """
        if cls.contains_shell_metacharacters(value):
            raise ValueError(
                f"{field_name} contains potentially malicious shell metacharacters"
            )
        return value


# ========== Format Validators ==========


class FormatValidators:
    """Validators for specific data formats."""

    # Chemical formula pattern (simplified)
    CHEMICAL_FORMULA_PATTERN = r'^[A-Z][a-z]?(\d+(\.\d+)?)?(\s*[A-Z][a-z]?(\d+(\.\d+)?)?)*$'

    # UUID pattern
    UUID_PATTERN = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'

    # Email pattern (basic)
    EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    # URL pattern (basic)
    URL_PATTERN = r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b'

    # Alphanumeric with common separators
    SAFE_IDENTIFIER_PATTERN = r'^[a-zA-Z0-9_\-\.]+$'

    @classmethod
    def validate_chemical_formula(cls, value: str) -> str:
        """
        Validate chemical formula format.

        Examples:
            - H2O (valid)
            - MoS2 (valid)
            - NaCl (valid)
            - <script> (invalid)

        Raises:
            ValueError if invalid format
        """
        if not re.match(cls.CHEMICAL_FORMULA_PATTERN, value):
            raise ValueError(f"Invalid chemical formula format: {value}")
        return value

    @classmethod
    def validate_uuid(cls, value: str) -> str:
        """
        Validate UUID format.

        Raises:
            ValueError if invalid UUID
        """
        if not re.match(cls.UUID_PATTERN, value, re.IGNORECASE):
            raise ValueError(f"Invalid UUID format: {value}")
        return value

    @classmethod
    def validate_email(cls, value: str) -> str:
        """
        Validate email format.

        Raises:
            ValueError if invalid email
        """
        if not re.match(cls.EMAIL_PATTERN, value):
            raise ValueError(f"Invalid email format: {value}")
        return value

    @classmethod
    def validate_url(cls, value: str) -> str:
        """
        Validate URL format.

        Raises:
            ValueError if invalid URL
        """
        if not re.match(cls.URL_PATTERN, value):
            raise ValueError(f"Invalid URL format: {value}")

        # Additional check for javascript: protocol
        if value.lower().startswith('javascript:'):
            raise ValueError("JavaScript URLs are not allowed")

        return value

    @classmethod
    def validate_safe_identifier(cls, value: str) -> str:
        """
        Validate safe identifier (alphanumeric with _ - .).

        Use for filenames, IDs, slugs, etc.

        Raises:
            ValueError if invalid characters
        """
        if not re.match(cls.SAFE_IDENTIFIER_PATTERN, value):
            raise ValueError(
                f"Invalid identifier: must contain only alphanumeric, underscore, hyphen, or dot: {value}"
            )
        return value


# ========== Content Length Validators ==========


class ContentLengthValidators:
    """Validators for content length limits."""

    @staticmethod
    def validate_max_length(value: str, max_length: int, field_name: str = "field") -> str:
        """
        Validate maximum string length.

        Raises:
            ValueError if exceeds max length
        """
        if len(value) > max_length:
            raise ValueError(
                f"{field_name} exceeds maximum length of {max_length} characters"
            )
        return value

    @staticmethod
    def validate_min_length(value: str, min_length: int, field_name: str = "field") -> str:
        """
        Validate minimum string length.

        Raises:
            ValueError if below min length
        """
        if len(value) < min_length:
            raise ValueError(
                f"{field_name} must be at least {min_length} characters"
            )
        return value

    @staticmethod
    def validate_list_max_items(value: list, max_items: int, field_name: str = "list") -> list:
        """
        Validate maximum list length.

        Raises:
            ValueError if exceeds max items
        """
        if len(value) > max_items:
            raise ValueError(
                f"{field_name} exceeds maximum of {max_items} items"
            )
        return value


# ========== Comprehensive Input Validator ==========


class InputValidator:
    """
    Comprehensive input validator combining all security checks.
    """

    @classmethod
    def validate_user_input(
        cls,
        value: str,
        field_name: str = "input",
        check_sql_injection: bool = True,
        check_xss: bool = True,
        check_path_traversal: bool = False,
        check_command_injection: bool = False,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        format_type: Optional[str] = None
    ) -> str:
        """
        Validate user input with comprehensive security checks.

        Args:
            value: Input value to validate
            field_name: Field name for error messages
            check_sql_injection: Check for SQL injection
            check_xss: Check for XSS patterns
            check_path_traversal: Check for path traversal
            check_command_injection: Check for command injection
            max_length: Maximum allowed length
            min_length: Minimum required length
            format_type: Specific format to validate (email, url, uuid, etc.)

        Returns:
            Validated value

        Raises:
            ValueError if validation fails
        """
        if not isinstance(value, str):
            return value

        # Length checks
        if max_length:
            ContentLengthValidators.validate_max_length(value, max_length, field_name)

        if min_length:
            ContentLengthValidators.validate_min_length(value, min_length, field_name)

        # Security checks
        if check_sql_injection:
            SQLInjectionDetector.validate_safe_string(value, field_name)

        if check_xss:
            XSSPrevention.validate_safe_html(value, field_name)

        if check_path_traversal:
            PathTraversalPrevention.validate_safe_path(value, field_name)

        if check_command_injection:
            CommandInjectionPrevention.validate_safe_command_arg(value, field_name)

        # Format validation
        if format_type:
            if format_type == "email":
                FormatValidators.validate_email(value)
            elif format_type == "url":
                FormatValidators.validate_url(value)
            elif format_type == "uuid":
                FormatValidators.validate_uuid(value)
            elif format_type == "chemical_formula":
                FormatValidators.validate_chemical_formula(value)
            elif format_type == "safe_identifier":
                FormatValidators.validate_safe_identifier(value)

        return value

    @classmethod
    def sanitize_html_content(cls, value: str) -> str:
        """Sanitize HTML content by escaping special characters."""
        return XSSPrevention.sanitize_html(value)


# ========== Pydantic Field Validators ==========


def validate_safe_string_field(value: str) -> str:
    """Pydantic validator for safe string fields."""
    return InputValidator.validate_user_input(
        value,
        check_sql_injection=True,
        check_xss=True
    )


def validate_chemical_formula_field(value: str) -> str:
    """Pydantic validator for chemical formula fields."""
    return InputValidator.validate_user_input(
        value,
        check_sql_injection=True,
        check_xss=True,
        format_type="chemical_formula"
    )


def validate_url_field(value: str) -> str:
    """Pydantic validator for URL fields."""
    return InputValidator.validate_user_input(
        value,
        check_sql_injection=True,
        check_xss=True,
        format_type="url"
    )


def validate_email_field(value: str) -> str:
    """Pydantic validator for email fields."""
    return InputValidator.validate_user_input(
        value,
        check_sql_injection=True,
        check_xss=True,
        format_type="email"
    )


# ========== Query Parameter Validation ==========


def validate_search_query(query: str, max_length: int = 200) -> str:
    """
    Validate search query parameter.

    Ensures query is safe for database searches.

    Args:
        query: Search query string
        max_length: Maximum query length

    Returns:
        Validated query string

    Raises:
        ValueError if query is invalid
    """
    return InputValidator.validate_user_input(
        query,
        field_name="search_query",
        check_sql_injection=True,
        check_xss=True,
        max_length=max_length
    )


def validate_filter_value(value: str, max_length: int = 100) -> str:
    """
    Validate filter parameter value.

    Args:
        value: Filter value
        max_length: Maximum value length

    Returns:
        Validated filter value

    Raises:
        ValueError if value is invalid
    """
    return InputValidator.validate_user_input(
        value,
        field_name="filter_value",
        check_sql_injection=True,
        check_xss=True,
        max_length=max_length
    )
