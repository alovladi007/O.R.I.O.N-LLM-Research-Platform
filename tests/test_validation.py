"""
Tests for Input Validation and Sanitization
============================================

Tests comprehensive input validation:
- SQL injection detection and prevention
- XSS attack detection and prevention
- Path traversal detection
- Command injection detection
- Format validation (email, URL, chemical formula, etc.)
- Content length validation
- Query parameter validation
"""

import pytest
from src.api.validation import (
    SQLInjectionDetector,
    XSSPrevention,
    PathTraversalPrevention,
    CommandInjectionPrevention,
    FormatValidators,
    ContentLengthValidators,
    InputValidator,
    validate_safe_string_field,
    validate_chemical_formula_field,
    validate_url_field,
    validate_email_field,
    validate_search_query,
    validate_filter_value,
)


class TestSQLInjectionDetection:
    """Test SQL injection detection"""

    def test_detect_union_select(self):
        """Test detection of UNION SELECT injection"""
        malicious = "test' UNION SELECT * FROM users--"
        assert SQLInjectionDetector.is_potentially_malicious(malicious) is True

    def test_detect_comment_injection(self):
        """Test detection of comment-based injection"""
        malicious = "admin'--"
        assert SQLInjectionDetector.is_potentially_malicious(malicious) is True

    def test_detect_or_injection(self):
        """Test detection of OR-based injection"""
        malicious = "' OR '1'='1"
        assert SQLInjectionDetector.is_potentially_malicious(malicious) is True

    def test_detect_drop_table(self):
        """Test detection of DROP TABLE"""
        malicious = "test'; DROP TABLE users;--"
        assert SQLInjectionDetector.is_potentially_malicious(malicious) is True

    def test_safe_string_passes(self):
        """Test that safe strings pass validation"""
        safe = "Molybdenum Disulfide"
        assert SQLInjectionDetector.is_potentially_malicious(safe) is False

    def test_validate_safe_string_raises_on_injection(self):
        """Test that validator raises ValueError on injection"""
        malicious = "test' OR '1'='1"
        with pytest.raises(ValueError) as exc_info:
            SQLInjectionDetector.validate_safe_string(malicious, "test_field")
        assert "potentially malicious SQL patterns" in str(exc_info.value)

    def test_validate_safe_string_accepts_safe_input(self):
        """Test that validator accepts safe input"""
        safe = "MoS2"
        result = SQLInjectionDetector.validate_safe_string(safe, "formula")
        assert result == safe


class TestXSSPrevention:
    """Test XSS prevention"""

    def test_detect_script_tag(self):
        """Test detection of script tags"""
        malicious = "<script>alert('XSS')</script>"
        assert XSSPrevention.detect_xss_patterns(malicious) is True

    def test_detect_iframe_tag(self):
        """Test detection of iframe tags"""
        malicious = '<iframe src="http://evil.com"></iframe>'
        assert XSSPrevention.detect_xss_patterns(malicious) is True

    def test_detect_onerror_attribute(self):
        """Test detection of onerror attribute"""
        malicious = '<img src="x" onerror="alert(1)">'
        assert XSSPrevention.detect_xss_patterns(malicious) is True

    def test_detect_javascript_protocol(self):
        """Test detection of javascript: protocol"""
        malicious = '<a href="javascript:alert(1)">Click</a>'
        assert XSSPrevention.detect_xss_patterns(malicious) is True

    def test_safe_html_passes(self):
        """Test that safe HTML passes"""
        safe = "This is <b>bold</b> text"
        # Note: Bold is not in dangerous tags list, but would still be escaped
        assert XSSPrevention.detect_xss_patterns(safe) is False

    def test_sanitize_html(self):
        """Test HTML sanitization"""
        input_html = '<script>alert("XSS")</script>'
        sanitized = XSSPrevention.sanitize_html(input_html)
        assert '&lt;script&gt;' in sanitized
        assert '<script>' not in sanitized

    def test_validate_safe_html_raises_on_xss(self):
        """Test that validator raises ValueError on XSS"""
        malicious = '<script>alert(1)</script>'
        with pytest.raises(ValueError) as exc_info:
            XSSPrevention.validate_safe_html(malicious, "description")
        assert "potentially malicious HTML/JavaScript patterns" in str(exc_info.value)


class TestPathTraversalPrevention:
    """Test path traversal prevention"""

    def test_detect_parent_directory(self):
        """Test detection of .. pattern"""
        malicious = "../../etc/passwd"
        assert PathTraversalPrevention.contains_traversal(malicious) is True

    def test_detect_url_encoded_traversal(self):
        """Test detection of URL-encoded traversal"""
        malicious = "%2e%2e%2fetc%2fpasswd"
        assert PathTraversalPrevention.contains_traversal(malicious) is True

    def test_detect_double_encoded_traversal(self):
        """Test detection of double URL-encoded traversal"""
        malicious = "%252e%252e%252f"
        assert PathTraversalPrevention.contains_traversal(malicious) is True

    def test_safe_path_passes(self):
        """Test that safe paths pass"""
        safe = "data/structures/structure_001.cif"
        assert PathTraversalPrevention.contains_traversal(safe) is False

    def test_validate_safe_path_raises_on_traversal(self):
        """Test that validator raises ValueError on traversal"""
        malicious = "../../../etc/passwd"
        with pytest.raises(ValueError) as exc_info:
            PathTraversalPrevention.validate_safe_path(malicious, "file_path")
        assert "potentially malicious path traversal patterns" in str(exc_info.value)


class TestCommandInjectionPrevention:
    """Test command injection prevention"""

    def test_detect_semicolon(self):
        """Test detection of semicolon"""
        malicious = "test; rm -rf /"
        assert CommandInjectionPrevention.contains_shell_metacharacters(malicious) is True

    def test_detect_pipe(self):
        """Test detection of pipe"""
        malicious = "test | cat /etc/passwd"
        assert CommandInjectionPrevention.contains_shell_metacharacters(malicious) is True

    def test_detect_backticks(self):
        """Test detection of backticks"""
        malicious = "test`whoami`"
        assert CommandInjectionPrevention.contains_shell_metacharacters(malicious) is True

    def test_detect_command_substitution(self):
        """Test detection of command substitution"""
        malicious = "test$(whoami)"
        assert CommandInjectionPrevention.contains_shell_metacharacters(malicious) is True

    def test_safe_argument_passes(self):
        """Test that safe arguments pass"""
        safe = "test_file.txt"
        assert CommandInjectionPrevention.contains_shell_metacharacters(safe) is False

    def test_validate_safe_command_arg_raises(self):
        """Test that validator raises ValueError on shell metacharacters"""
        malicious = "test; rm -rf /"
        with pytest.raises(ValueError) as exc_info:
            CommandInjectionPrevention.validate_safe_command_arg(malicious, "filename")
        assert "potentially malicious shell metacharacters" in str(exc_info.value)


class TestFormatValidators:
    """Test format validators"""

    def test_validate_chemical_formula_valid(self):
        """Test valid chemical formulas"""
        valid_formulas = ["H2O", "MoS2", "NaCl", "Fe2O3", "Ca10P6O26"]
        for formula in valid_formulas:
            result = FormatValidators.validate_chemical_formula(formula)
            assert result == formula

    def test_validate_chemical_formula_invalid(self):
        """Test invalid chemical formulas"""
        invalid_formulas = [
            "<script>",
            "test'; DROP TABLE--",
            "123InvalidFormula",
        ]
        for formula in invalid_formulas:
            with pytest.raises(ValueError):
                FormatValidators.validate_chemical_formula(formula)

    def test_validate_uuid_valid(self):
        """Test valid UUIDs"""
        valid_uuid = "123e4567-e89b-12d3-a456-426614174000"
        result = FormatValidators.validate_uuid(valid_uuid)
        assert result == valid_uuid

    def test_validate_uuid_invalid(self):
        """Test invalid UUIDs"""
        invalid = "not-a-uuid"
        with pytest.raises(ValueError):
            FormatValidators.validate_uuid(invalid)

    def test_validate_email_valid(self):
        """Test valid emails"""
        valid_emails = [
            "user@example.com",
            "test.user@domain.co.uk",
            "admin+tag@company.org",
        ]
        for email in valid_emails:
            result = FormatValidators.validate_email(email)
            assert result == email

    def test_validate_email_invalid(self):
        """Test invalid emails"""
        invalid_emails = [
            "not-an-email",
            "@example.com",
            "user@",
            "<script>@example.com",
        ]
        for email in invalid_emails:
            with pytest.raises(ValueError):
                FormatValidators.validate_email(email)

    def test_validate_url_valid(self):
        """Test valid URLs"""
        valid_urls = [
            "https://example.com",
            "http://www.example.com/path",
            "https://sub.domain.example.org/path?query=value",
        ]
        for url in valid_urls:
            result = FormatValidators.validate_url(url)
            assert result == url

    def test_validate_url_invalid(self):
        """Test invalid URLs"""
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",
            "javascript:alert(1)",
        ]
        for url in invalid_urls:
            with pytest.raises(ValueError):
                FormatValidators.validate_url(url)

    def test_validate_safe_identifier_valid(self):
        """Test valid identifiers"""
        valid_identifiers = [
            "test_file",
            "structure-001",
            "data.csv",
            "MoS2_structure",
        ]
        for identifier in valid_identifiers:
            result = FormatValidators.validate_safe_identifier(identifier)
            assert result == identifier

    def test_validate_safe_identifier_invalid(self):
        """Test invalid identifiers"""
        invalid_identifiers = [
            "test/file",
            "structure;001",
            "data|csv",
            "../etc/passwd",
        ]
        for identifier in invalid_identifiers:
            with pytest.raises(ValueError):
                FormatValidators.validate_safe_identifier(identifier)


class TestContentLengthValidators:
    """Test content length validators"""

    def test_validate_max_length_passes(self):
        """Test that valid length passes"""
        value = "test"
        result = ContentLengthValidators.validate_max_length(value, 10, "field")
        assert result == value

    def test_validate_max_length_fails(self):
        """Test that exceeding max length fails"""
        value = "test string that is too long"
        with pytest.raises(ValueError) as exc_info:
            ContentLengthValidators.validate_max_length(value, 10, "field")
        assert "exceeds maximum length" in str(exc_info.value)

    def test_validate_min_length_passes(self):
        """Test that valid length passes"""
        value = "test string"
        result = ContentLengthValidators.validate_min_length(value, 5, "field")
        assert result == value

    def test_validate_min_length_fails(self):
        """Test that below min length fails"""
        value = "ab"
        with pytest.raises(ValueError) as exc_info:
            ContentLengthValidators.validate_min_length(value, 5, "field")
        assert "must be at least" in str(exc_info.value)

    def test_validate_list_max_items_passes(self):
        """Test that valid list passes"""
        value = [1, 2, 3]
        result = ContentLengthValidators.validate_list_max_items(value, 5, "list")
        assert result == value

    def test_validate_list_max_items_fails(self):
        """Test that exceeding max items fails"""
        value = [1, 2, 3, 4, 5, 6]
        with pytest.raises(ValueError) as exc_info:
            ContentLengthValidators.validate_list_max_items(value, 3, "list")
        assert "exceeds maximum of" in str(exc_info.value)


class TestInputValidator:
    """Test comprehensive input validator"""

    def test_validate_user_input_all_checks(self):
        """Test validation with all checks enabled"""
        safe = "MoS2"
        result = InputValidator.validate_user_input(
            safe,
            field_name="formula",
            check_sql_injection=True,
            check_xss=True,
            check_path_traversal=True,
            check_command_injection=True,
            max_length=100,
            min_length=1
        )
        assert result == safe

    def test_validate_user_input_sql_injection_fails(self):
        """Test that SQL injection is detected"""
        malicious = "test' OR '1'='1"
        with pytest.raises(ValueError):
            InputValidator.validate_user_input(
                malicious,
                check_sql_injection=True
            )

    def test_validate_user_input_xss_fails(self):
        """Test that XSS is detected"""
        malicious = "<script>alert(1)</script>"
        with pytest.raises(ValueError):
            InputValidator.validate_user_input(
                malicious,
                check_xss=True
            )

    def test_validate_user_input_with_format(self):
        """Test validation with format type"""
        email = "user@example.com"
        result = InputValidator.validate_user_input(
            email,
            format_type="email"
        )
        assert result == email

    def test_sanitize_html_content(self):
        """Test HTML content sanitization"""
        html = '<script>alert("XSS")</script>'
        sanitized = InputValidator.sanitize_html_content(html)
        assert '&lt;script&gt;' in sanitized
        assert '<script>' not in sanitized


class TestPydanticValidators:
    """Test Pydantic field validators"""

    def test_validate_safe_string_field(self):
        """Test safe string field validator"""
        safe = "Molybdenum Disulfide"
        result = validate_safe_string_field(safe)
        assert result == safe

    def test_validate_safe_string_field_fails(self):
        """Test that unsafe strings fail"""
        malicious = "test' OR '1'='1"
        with pytest.raises(ValueError):
            validate_safe_string_field(malicious)

    def test_validate_chemical_formula_field(self):
        """Test chemical formula field validator"""
        formula = "MoS2"
        result = validate_chemical_formula_field(formula)
        assert result == formula

    def test_validate_chemical_formula_field_fails(self):
        """Test that invalid formulas fail"""
        invalid = "<script>"
        with pytest.raises(ValueError):
            validate_chemical_formula_field(invalid)

    def test_validate_url_field(self):
        """Test URL field validator"""
        url = "https://example.com"
        result = validate_url_field(url)
        assert result == url

    def test_validate_url_field_fails(self):
        """Test that invalid URLs fail"""
        invalid = "javascript:alert(1)"
        with pytest.raises(ValueError):
            validate_url_field(invalid)

    def test_validate_email_field(self):
        """Test email field validator"""
        email = "user@example.com"
        result = validate_email_field(email)
        assert result == email

    def test_validate_email_field_fails(self):
        """Test that invalid emails fail"""
        invalid = "not-an-email"
        with pytest.raises(ValueError):
            validate_email_field(invalid)


class TestQueryParameterValidation:
    """Test query parameter validation"""

    def test_validate_search_query(self):
        """Test search query validation"""
        query = "molybdenum disulfide"
        result = validate_search_query(query)
        assert result == query

    def test_validate_search_query_fails_on_injection(self):
        """Test that search query rejects injection"""
        malicious = "test' OR '1'='1"
        with pytest.raises(ValueError):
            validate_search_query(malicious)

    def test_validate_search_query_fails_on_length(self):
        """Test that search query rejects excessive length"""
        long_query = "a" * 300
        with pytest.raises(ValueError):
            validate_search_query(long_query, max_length=200)

    def test_validate_filter_value(self):
        """Test filter value validation"""
        value = "NaCl"
        result = validate_filter_value(value)
        assert result == value

    def test_validate_filter_value_fails_on_injection(self):
        """Test that filter value rejects injection"""
        malicious = "test'; DROP TABLE--"
        with pytest.raises(ValueError):
            validate_filter_value(malicious)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
