"""
Tests for the symbols module.
"""

from canto_nlp.tts.text.symbols import (
    symbols,
    language_tone_start_map,
    language_id_map,
)
from canto_nlp.tts.text import _symbol_to_id


class TestSymbols:
    """Test the symbols module constants and mappings."""

    def test_symbols_not_empty(self):
        """Test that symbols list is not empty."""
        assert len(symbols) > 0
        assert isinstance(symbols, list)

    def test_symbol_to_id_mapping(self):
        """Test symbol to ID mapping."""
        assert len(_symbol_to_id) == len(symbols)

        # Test that each symbol maps to its index
        for i, symbol in enumerate(symbols):
            assert _symbol_to_id[symbol] == i

    def test_language_tone_start_map(self):
        """Test language tone start mapping."""
        assert "EN" in language_tone_start_map
        assert "YUE" in language_tone_start_map

        # Test that values are integers
        assert isinstance(language_tone_start_map["EN"], int)
        assert isinstance(language_tone_start_map["YUE"], int)

    def test_language_id_map(self):
        """Test language ID mapping."""
        assert "EN" in language_id_map
        assert "YUE" in language_id_map

        # Test that values are integers
        assert isinstance(language_id_map["EN"], int)
        assert isinstance(language_id_map["YUE"], int)

    def test_symbol_uniqueness(self):
        """Test that all symbols are unique."""
        unique_symbols = set(symbols)
        assert len(unique_symbols) == len(symbols)

    def test_symbol_to_id_consistency(self):
        """Test consistency between symbols and _symbol_to_id."""
        # Test that all symbols are in the mapping
        for symbol in symbols:
            assert symbol in _symbol_to_id

        # Test that all mapping keys are in symbols
        for symbol in _symbol_to_id.keys():
            assert symbol in symbols
