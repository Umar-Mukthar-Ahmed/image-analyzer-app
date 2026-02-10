"""
Main Image Analyzer
Orchestrates image analysis workflow with uncertainty handling
"""
import logging
import io
from typing import Optional, Dict, Any
from pathlib import Path
from src.core.image_handler import ImageHandler, ImageValidationError
from src.core.vision_client import get_vision_client, VisionAPIError
from src.core.response_parser import ResponseParser
from src.models.analysis_result import AnalysisResult
from src.decision.confidence_engine import ConfidenceEngine

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Custom exception for analysis errors"""
    pass


class ImageAnalyzer:
    """Main image analyzer orchestrator"""

    def __init__(self):
        """Initialize analyzer"""
        self.image_handler = ImageHandler()
        self.vision_client = get_vision_client()
        self.parser = ResponseParser()
        self.confidence_engine = ConfidenceEngine()
        logger.info("ImageAnalyzer initialized")

    def analyze_image(
            self,
            image_path: str,
            include_ocr: bool = True
    ) -> AnalysisResult:
        """
        Perform complete image analysis

        Args:
            image_path: Path to image file
            include_ocr: Whether to include OCR text extraction

        Returns:
            AnalysisResult object with all analysis data

        Raises:
            AnalysisError: If analysis fails
        """
        logger.info(f"Starting analysis for: {image_path}")

        try:
            # Load and validate image
            image_data, pil_image = self.image_handler.load_image(image_path)

            # Convert to stream for API
            image_stream = io.BytesIO(image_data)

            # Perform analysis
            logger.info("Calling Azure Vision API for analysis")
            response = self.vision_client.analyze_image(image_stream.getvalue())

            # Parse response
            result = self.parser.parse_full_analysis(response)

            # Add OCR if requested
            if include_ocr:
                logger.info("Extracting text (OCR)")
                try:
                    image_stream.seek(0)  # Reset stream
                    ocr_response = self.vision_client.read_text(image_stream.getvalue())
                    result.ocr_text = self.parser.parse_ocr_response(ocr_response)
                except Exception as e:
                    logger.warning(f"OCR failed: {str(e)}")
                    # Continue without OCR

            # Generate confidence-based insights
            self._add_confidence_insights(result)

            logger.info("Analysis completed successfully")

            return result

        except ImageValidationError as e:
            logger.error(f"Image validation failed: {str(e)}")
            raise AnalysisError(f"Invalid image: {str(e)}")

        except VisionAPIError as e:
            logger.error(f"Vision API error: {str(e)}")
            raise AnalysisError(f"API error: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error during analysis: {str(e)}")
            raise AnalysisError(f"Analysis failed: {str(e)}")

    def analyze_from_bytes(
            self,
            image_bytes: bytes,
            include_ocr: bool = True
    ) -> AnalysisResult:
        """
        Analyze image from bytes (useful for uploaded files)

        Args:
            image_bytes: Image binary data
            include_ocr: Whether to include OCR

        Returns:
            AnalysisResult object
        """
        logger.info("Analyzing image from bytes")

        try:
            # Perform analysis
            response = self.vision_client.analyze_image(image_bytes)
            result = self.parser.parse_full_analysis(response)

            # Add OCR if requested
            if include_ocr:
                try:
                    ocr_response = self.vision_client.read_text(image_bytes)
                    result.ocr_text = self.parser.parse_ocr_response(ocr_response)
                except Exception as e:
                    logger.warning(f"OCR failed: {str(e)}")

            # Generate confidence insights
            self._add_confidence_insights(result)

            return result

        except Exception as e:
            logger.error(f"Analysis from bytes failed: {str(e)}")
            raise AnalysisError(f"Analysis failed: {str(e)}")

    def extract_text_only(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text only (OCR)

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with OCR results
        """
        logger.info(f"Extracting text from: {image_path}")

        try:
            image_data, _ = self.image_handler.load_image(image_path)
            ocr_response = self.vision_client.read_text(image_data)
            ocr_results = self.parser.parse_ocr_response(ocr_response)

            # Extract full text
            full_text = self.parser.extract_text_content(ocr_results)

            # Evaluate confidence
            if ocr_results:
                avg_confidence = sum(o.confidence for o in ocr_results) / len(ocr_results)
            else:
                avg_confidence = 0.0

            decision = self.confidence_engine.make_decision(avg_confidence, "OCR text")

            return {
                'text': full_text,
                'lines': ocr_results,
                'confidence': avg_confidence,
                'decision': decision,
                'total_lines': len(ocr_results)
            }

        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            raise AnalysisError(f"OCR failed: {str(e)}")

    def detect_objects_only(self, image_path: str) -> Dict[str, Any]:
        """
        Detect objects only

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with object detection results
        """
        logger.info(f"Detecting objects in: {image_path}")

        try:
            image_data, _ = self.image_handler.load_image(image_path)
            response = self.vision_client.detect_objects(image_data)

            objects = []
            for obj_data in response.get('objects', []):
                obj = self.parser._parse_objects([obj_data])[0]
                decision = self.confidence_engine.make_decision(obj.confidence, f"object '{obj.name}'")
                objects.append({
                    'object': obj,
                    'decision': decision
                })

            return {
                'objects': objects,
                'total_objects': len(objects),
                'metadata': response.get('metadata')
            }

        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}")
            raise AnalysisError(f"Object detection failed: {str(e)}")

    def get_description_only(self, image_path: str) -> Dict[str, Any]:
        """
        Get image description only

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with description results
        """
        logger.info(f"Generating description for: {image_path}")

        try:
            image_data, _ = self.image_handler.load_image(image_path)
            response = self.vision_client.describe_image(image_data)

            descriptions = self.parser._parse_descriptions(response.get('descriptions', []))

            if descriptions:
                primary = descriptions[0]
                decision = self.confidence_engine.make_decision(
                    primary.confidence,
                    "description"
                )
            else:
                primary = None
                decision = None

            return {
                'descriptions': descriptions,
                'primary_description': primary,
                'decision': decision,
                'tags': response.get('tags', [])
            }

        except Exception as e:
            logger.error(f"Description generation failed: {str(e)}")
            raise AnalysisError(f"Description failed: {str(e)}")

    def _add_confidence_insights(self, result: AnalysisResult) -> None:
        """
        Add confidence-based insights to analysis result

        Args:
            result: AnalysisResult to enhance with insights
        """
        # This adds warnings and suggestions directly to the result
        # We'll access these in the UI
        pass

    def get_analysis_summary(self, result: AnalysisResult) -> Dict[str, Any]:
        """
        Generate analysis summary with confidence evaluation

        Args:
            result: AnalysisResult object

        Returns:
            Summary dictionary
        """
        reliability = self.confidence_engine.get_reliability_score(result)
        warnings = self.confidence_engine.generate_uncertainty_warnings(result)
        suggestions = self.confidence_engine.suggest_improvements(result)

        return {
            'reliability': reliability,
            'warnings': warnings,
            'suggestions': suggestions,
            'features_detected': {
                'descriptions': len(result.descriptions),
                'tags': len(result.tags),
                'objects': len(result.objects),
                'text_lines': len(result.ocr_text),
                'faces': len(result.faces)
            }
        }


def analyze_image(image_path: str, include_ocr: bool = True) -> AnalysisResult:
    """
    Convenience function to analyze image

    Args:
        image_path: Path to image
        include_ocr: Include OCR

    Returns:
        AnalysisResult object
    """
    analyzer = ImageAnalyzer()
    return analyzer.analyze_image(image_path, include_ocr)