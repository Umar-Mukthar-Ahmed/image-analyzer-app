"""
Confidence Engine - Probabilistic Decision Making
Acts on confidence scores, not certainties
Handles uncertainty and provides recommendations
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from config.settings import settings

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level categories"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    VERY_LOW = "Very Low"


@dataclass
class ConfidenceDecision:
    """Decision based on confidence score"""
    level: ConfidenceLevel
    score: float
    recommendation: str
    should_trust: bool
    uncertainty_note: Optional[str] = None


class ConfidenceEngine:
    """Engine for making probabilistic decisions based on confidence scores"""

    @staticmethod
    def categorize_confidence(confidence: float) -> ConfidenceLevel:
        """
        Categorize confidence score into levels

        Args:
            confidence: Confidence score (0.0 to 1.0)

        Returns:
            ConfidenceLevel enum
        """
        if confidence >= settings.HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.HIGH
        elif confidence >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        elif confidence >= settings.LOW_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    @staticmethod
    def make_decision(
            confidence: float,
            context: str = "result"
    ) -> ConfidenceDecision:
        """
        Make a decision based on confidence score

        Args:
            confidence: Confidence score (0.0 to 1.0)
            context: Context of what's being evaluated

        Returns:
            ConfidenceDecision object
        """
        level = ConfidenceEngine.categorize_confidence(confidence)

        if level == ConfidenceLevel.HIGH:
            return ConfidenceDecision(
                level=level,
                score=confidence,
                recommendation=f"High confidence in this {context}. Result is reliable.",
                should_trust=True,
                uncertainty_note=None
            )

        elif level == ConfidenceLevel.MEDIUM:
            return ConfidenceDecision(
                level=level,
                score=confidence,
                recommendation=f"Moderate confidence. This {context} is likely correct but verify if critical.",
                should_trust=True,
                uncertainty_note=f"Confidence is {confidence * 100:.1f}%. Consider alternative interpretations."
            )

        elif level == ConfidenceLevel.LOW:
            return ConfidenceDecision(
                level=level,
                score=confidence,
                recommendation=f"Low confidence. This {context} may not be accurate. Use with caution.",
                should_trust=False,
                uncertainty_note=f"Only {confidence * 100:.1f}% confident. Manual verification recommended."
            )

        else:  # VERY_LOW
            return ConfidenceDecision(
                level=level,
                score=confidence,
                recommendation=f"Very low confidence. This {context} is unreliable. Do not trust without verification.",
                should_trust=False,
                uncertainty_note=f"Confidence is only {confidence * 100:.1f}%. Consider retaking image with better quality."
            )

    @staticmethod
    def evaluate_aggregate_confidence(
            confidences: List[float],
            weights: Optional[List[float]] = None
    ) -> Tuple[float, ConfidenceLevel]:
        """
        Evaluate aggregate confidence from multiple scores

        Args:
            confidences: List of confidence scores
            weights: Optional weights for each confidence score

        Returns:
            Tuple of (aggregate_score, confidence_level)
        """
        if not confidences:
            return 0.0, ConfidenceLevel.VERY_LOW

        if weights:
            if len(weights) != len(confidences):
                logger.warning("Weights length doesn't match confidences, using uniform weights")
                weights = None

        if weights:
            # Weighted average
            total_weight = sum(weights)
            aggregate = sum(c * w for c, w in zip(confidences, weights)) / total_weight
        else:
            # Simple average
            aggregate = sum(confidences) / len(confidences)

        level = ConfidenceEngine.categorize_confidence(aggregate)

        return aggregate, level

    @staticmethod
    def generate_uncertainty_warnings(
            analysis_result
    ) -> List[str]:
        """
        Generate uncertainty warnings based on analysis results

        Args:
            analysis_result: AnalysisResult object

        Returns:
            List of warning messages
        """
        warnings = []

        # Check description confidence
        if analysis_result.descriptions:
            desc_confidences = [d.confidence for d in analysis_result.descriptions]
            if desc_confidences and max(desc_confidences) < settings.MEDIUM_CONFIDENCE_THRESHOLD:
                warnings.append(
                    f"⚠️ Image description has low confidence ({max(desc_confidences) * 100:.1f}%). "
                    "Consider uploading a clearer image."
                )

        # Check object detection confidence
        if analysis_result.objects:
            low_conf_objects = [
                obj for obj in analysis_result.objects
                if obj.confidence < settings.MEDIUM_CONFIDENCE_THRESHOLD
            ]
            if low_conf_objects:
                warnings.append(
                    f"⚠️ {len(low_conf_objects)} object(s) detected with low confidence. "
                    "Results may not be accurate."
                )

        # Check OCR confidence
        if analysis_result.ocr_text:
            low_conf_text = [
                ocr for ocr in analysis_result.ocr_text
                if ocr.confidence < settings.MEDIUM_CONFIDENCE_THRESHOLD
            ]
            if low_conf_text:
                warnings.append(
                    f"⚠️ {len(low_conf_text)} text region(s) have low OCR confidence. "
                    "Some text may be misread. Try better lighting or higher resolution."
                )

        # Check if very few features detected
        total_features = (
                len(analysis_result.descriptions) +
                len(analysis_result.tags) +
                len(analysis_result.objects)
        )
        if total_features < 3:
            warnings.append(
                "⚠️ Few features detected in image. Image may be unclear, too dark, or abstract."
            )

        # Check adult content uncertainty
        if (0.3 < analysis_result.adult_score < 0.7 or
                0.3 < analysis_result.racy_score < 0.7):
            warnings.append(
                "⚠️ Content moderation scores are uncertain. Manual review recommended."
            )

        return warnings

    @staticmethod
    def suggest_improvements(analysis_result) -> List[str]:
        """
        Suggest improvements based on analysis quality

        Args:
            analysis_result: AnalysisResult object

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        avg_confidence = analysis_result.average_confidence

        if avg_confidence < settings.HIGH_CONFIDENCE_THRESHOLD:
            suggestions.append("💡 Try uploading a higher resolution image")
            suggestions.append("💡 Ensure good lighting conditions")
            suggestions.append("💡 Avoid blurry or distorted images")

        if not analysis_result.has_objects and not analysis_result.has_descriptions:
            suggestions.append("💡 Image may be too abstract - try more concrete subjects")
            suggestions.append("💡 Ensure the subject is clearly visible and in focus")

        if analysis_result.has_text and analysis_result.ocr_text:
            low_ocr = [o for o in analysis_result.ocr_text if o.confidence < 0.7]
            if low_ocr:
                suggestions.append("💡 For better text extraction, ensure text is clearly readable")
                suggestions.append("💡 Avoid shadows or glare on text areas")

        if analysis_result.is_bw_image:
            suggestions.append("💡 Color images generally provide better analysis results")

        return suggestions

    @staticmethod
    def get_reliability_score(analysis_result) -> Dict[str, Any]:
        """
        Calculate overall reliability score for analysis

        Args:
            analysis_result: AnalysisResult object

        Returns:
            Dictionary with reliability metrics
        """
        # Collect all confidence scores
        all_confidences = []

        if analysis_result.descriptions:
            all_confidences.extend([d.confidence for d in analysis_result.descriptions])
        if analysis_result.tags:
            all_confidences.extend([t.confidence for t in analysis_result.tags[:5]])  # Top 5 tags
        if analysis_result.objects:
            all_confidences.extend([o.confidence for o in analysis_result.objects])

        if not all_confidences:
            return {
                'reliability_score': 0.0,
                'reliability_level': 'Unknown',
                'confidence': 'No features detected'
            }

        # Calculate metrics
        avg_confidence = sum(all_confidences) / len(all_confidences)
        min_confidence = min(all_confidences)
        max_confidence = max(all_confidences)

        # Reliability score considers average and minimum
        reliability_score = (avg_confidence * 0.7) + (min_confidence * 0.3)

        level = ConfidenceEngine.categorize_confidence(reliability_score)

        return {
            'reliability_score': round(reliability_score, 3),
            'reliability_level': level.value,
            'average_confidence': round(avg_confidence, 3),
            'min_confidence': round(min_confidence, 3),
            'max_confidence': round(max_confidence, 3),
            'total_features': len(all_confidences),
            'high_confidence_features': len([c for c in all_confidences if c >= settings.HIGH_CONFIDENCE_THRESHOLD]),
            'low_confidence_features': len([c for c in all_confidences if c < settings.MEDIUM_CONFIDENCE_THRESHOLD])
        }

    @staticmethod
    def format_confidence_display(confidence: float) -> str:
        """
        Format confidence score for display

        Args:
            confidence: Confidence score (0.0 to 1.0)

        Returns:
            Formatted string with emoji indicator
        """
        percentage = confidence * 100
        level = ConfidenceEngine.categorize_confidence(confidence)

        if level == ConfidenceLevel.HIGH:
            emoji = "🟢"
        elif level == ConfidenceLevel.MEDIUM:
            emoji = "🟡"
        elif level == ConfidenceLevel.LOW:
            emoji = "🟠"
        else:
            emoji = "🔴"

        return f"{emoji} {percentage:.1f}% ({level.value})"

    @staticmethod
    def should_show_result(confidence: float, min_threshold: float = 0.3) -> bool:
        """
        Determine if result should be shown based on confidence

        Args:
            confidence: Confidence score
            min_threshold: Minimum threshold to show

        Returns:
            True if result should be displayed
        """
        return confidence >= min_threshold


def evaluate_confidence(confidence: float, context: str = "result") -> ConfidenceDecision:
    """
    Convenience function to evaluate confidence

    Args:
        confidence: Confidence score
        context: Context description

    Returns:
        ConfidenceDecision object
    """
    engine = ConfidenceEngine()
    return engine.make_decision(confidence, context)