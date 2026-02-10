"""
Unit Tests for Response Parser
"""
import unittest
from src.core.response_parser import ResponseParser
from src.models.analysis_result import ImageDescription, ImageTag


class TestResponseParser(unittest.TestCase):
    """Test cases for response parser"""

    def test_parse_descriptions(self):
        """Test parsing descriptions"""
        descriptions_data = [
            {'text': 'A dog in a park', 'confidence': 0.95},
            {'text': 'An animal outdoors', 'confidence': 0.85}
        ]

        result = ResponseParser._parse_descriptions(descriptions_data)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], ImageDescription)
        self.assertEqual(result[0].text, 'A dog in a park')
        self.assertEqual(result[0].confidence, 0.95)

        # Should be sorted by confidence
        self.assertGreater(result[0].confidence, result[1].confidence)

    def test_parse_tags(self):
        """Test parsing tags"""
        tags_data = [
            {'name': 'dog', 'confidence': 0.92, 'hint': None},
            {'name': 'outdoor', 'confidence': 0.87, 'hint': None}
        ]

        result = ResponseParser._parse_tags(tags_data)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], ImageTag)
        self.assertEqual(result[0].name, 'dog')

        # Should be sorted by confidence
        self.assertGreater(result[0].confidence, result[1].confidence)

    def test_filter_by_confidence(self):
        """Test filtering items by confidence"""
        items = [
            ImageTag(name='high', confidence=0.9, hint=None),
            ImageTag(name='medium', confidence=0.6, hint=None),
            ImageTag(name='low', confidence=0.3, hint=None)
        ]

        filtered = ResponseParser.filter_by_confidence(items, min_confidence=0.5)

        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0].name, 'high')
        self.assertEqual(filtered[1].name, 'medium')


if __name__ == '__main__':
    unittest.main()