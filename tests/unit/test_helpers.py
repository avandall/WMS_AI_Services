"""
Unit tests for helper utilities in WMS AI Engine.
Specifically tests TASK-079 / TECH-02:
Preservation of inch (") and measurement symbols in sanitize_text.
"""
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ai_engine.utils.helpers import sanitize_text


class TestSanitizeText:
    """Test text sanitization and preservation of critical characters."""

    def test_sanitize_preserves_inch_double_quote(self):
        """15\" Laptop must retain the double quote (inch measurement) after sanitization."""
        query = '15" Laptop'
        result = sanitize_text(query)
        assert '15"' in result
        assert result == '15" Laptop'

    def test_sanitize_preserves_other_measurements(self):
        """Various measurement notations with quotes and apostrophes must be preserved."""
        samples = [
            ('27" 4K Monitor', '27" 4K Monitor'),
            ('Storage box 12"x8"x6"', 'Storage box 12"x8"x6"'),
            ("Levi's 32\" Jeans", "Levi's 32\" Jeans"),
        ]
        for input_text, expected in samples:
            assert sanitize_text(input_text) == expected

    def test_sanitize_strips_dangerous_html_tags(self):
        """Sanitization must still strip harmful HTML/XML angle brackets and ampersands."""
        malicious = '<script>alert("xss")</script> 15" Laptop & accessories'
        result = sanitize_text(malicious)
        assert "<" not in result
        assert ">" not in result
        assert "&" not in result
        assert '15"' in result
        assert 'alert("xss")' in result

    def test_sanitize_normalizes_whitespace(self):
        """Excessive spaces, tabs, and newlines must be collapsed."""
        text = '   15"   Gaming    Laptop \n\t Core i7   '
        assert sanitize_text(text) == '15" Gaming Laptop Core i7'
