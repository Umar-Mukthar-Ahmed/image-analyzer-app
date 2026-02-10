"""
Unit Tests for Confidence Engine
"""
import unittest
from src.decision.confidence_engine import (
    ConfidenceEngine,
    ConfidenceLevel,
    ConfidenceDecision
)


class TestConfidenceEngine(unittest.TestCase):
    """Test cases for confidence engine"""

    def test_categorize_confidence_high(self):
        """Test high confidence categorization"""
        level = ConfidenceEngine.categorize_confidence(0.9)
        self.assertEqual(level, ConfidenceLevel.HIGH)

    def test_categorize_confidence_medium(self):
        """Test medium confidence categorization"""
        level = ConfidenceEngine.categorize_confidence(0.6)
        self.assertEqual(level, ConfidenceLevel.MEDIUM)

    def test_categorize_confidence_low(self):
        """Test low confidence categorization"""
        level = ConfidenceEngine.categorize_confidence(0.4)
        self.assertEqual(level, ConfidenceLevel.LOW)

    def test_categorize_confidence_very_low(self):
        """Test very low confidence categorization"""
        level = ConfidenceEngine.categorize_confidence(0.2)
        self.assertEqual(level, ConfidenceLevel.VERY_LOW)

    def test_make_decision_high_confidence(self):
        """Test decision making with high confidence"""
        decision = ConfidenceEngine.make_decision(0.9, "test")

        self.assertIsInstance(decision, ConfidenceDecision)
        self.assertEqual(decision.level, ConfidenceLevel.HIGH)
        self.assertTrue(decision.should_trust)
        self.assertIsNone(decision.uncertainty_note)

    def test_make_decision_low_confidence(self):
        """Test decision making with low confidence"""
        decision = ConfidenceEngine.make_decision(0.4, "test")

        self.assertEqual(decision.level, ConfidenceLevel.LOW)
        self.assertFalse(decision.should_trust)
        self.assertIsNotNone(decision.uncertainty_note)

    def test_evaluate_aggregate_confidence(self):
        """Test aggregate confidence evaluation"""
        confidences = [0.9, 0.8, 0.7]

        aggregate, level = ConfidenceEngine.evaluate_aggregate_confidence(confidences)

        self.assertGreater(aggregate, 0)
        self.assertLess(aggregate, 1)
        self.assertIsInstance(level, ConfidenceLevel)

    def test_format_confidence_display(self):
        """Test confidence display formatting"""
        display = ConfidenceEngine.format_confidence_display(0.85)

        self.assertIn("85", display)
        self.assertIn("High", display)
        self.assertIn("🟢", display)

    def test_should_show_result(self):
        """Test result display threshold"""
        self.assertTrue(ConfidenceEngine.should_show_result(0.5, min_threshold=0.3))
        self.assertFalse(ConfidenceEngine.should_show_result(0.2, min_threshold=0.3))


if __name__ == '__main__':
    unittest.main()