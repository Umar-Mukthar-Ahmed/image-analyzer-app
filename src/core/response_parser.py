"""
Response Parser - JSON Parsing
Extracts insights from complex API responses
"""
import logging
from typing import Dict, Any, List, Optional
from src.models.analysis_result import (
    AnalysisResult,
    DetectedObject,
    ImageTag,
    ImageDescription,
    ImageCategory,
    OCRResult,
    FaceDetection,
    ImageMetadata
)

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parse Azure Vision API responses into structured data models"""

    @staticmethod
    def parse_full_analysis(response: Dict[str, Any]) -> AnalysisResult:
        """
        Parse complete analysis response

        Args:
            response: Raw API response dictionary

        Returns:
            AnalysisResult object with all parsed data
        """
        logger.info("Parsing full analysis response")

        result = AnalysisResult()

        # Parse descriptions
        if 'descriptions' in response:
            result.descriptions = ResponseParser._parse_descriptions(
                response['descriptions']
            )

        # Parse tags
        if 'tags' in response:
            result.tags = ResponseParser._parse_tags(response['tags'])

        # Parse objects
        if 'objects' in response:
            result.objects = ResponseParser._parse_objects(response['objects'])

        # Parse categories
        if 'categories' in response:
            result.categories = ResponseParser._parse_categories(response['categories'])

        # Parse adult content scores
        if 'adult' in response:
            ResponseParser._parse_adult_content(response['adult'], result)

        # Parse color information
        if 'color' in response:
            ResponseParser._parse_color_info(response['color'], result)

        # Parse faces
        if 'faces' in response:
            result.faces = ResponseParser._parse_faces(response['faces'])

        # Parse brands
        if 'brands' in response:
            result.brands = [brand['name'] for brand in response['brands']]

        # Parse metadata
        if 'metadata' in response:
            result.metadata = ResponseParser._parse_metadata(response['metadata'])

        # Store raw response
        result.raw_response = response

        logger.info(
            f"Parsed analysis: {len(result.descriptions)} descriptions, "
            f"{len(result.tags)} tags, {len(result.objects)} objects"
        )

        return result

    @staticmethod
    def _parse_descriptions(descriptions_data: List[Dict]) -> List[ImageDescription]:
        """Parse image descriptions"""
        descriptions = []

        for desc in descriptions_data:
            descriptions.append(ImageDescription(
                text=desc.get('text', ''),
                confidence=desc.get('confidence', 0.0)
            ))

        # Sort by confidence (highest first)
        descriptions.sort(key=lambda x: x.confidence, reverse=True)

        return descriptions

    @staticmethod
    def _parse_tags(tags_data: List[Dict]) -> List[ImageTag]:
        """Parse image tags"""
        tags = []

        for tag in tags_data:
            tags.append(ImageTag(
                name=tag.get('name', ''),
                confidence=tag.get('confidence', 0.0),
                hint=tag.get('hint')
            ))

        # Sort by confidence (highest first)
        tags.sort(key=lambda x: x.confidence, reverse=True)

        return tags

    @staticmethod
    def _parse_objects(objects_data: List[Dict]) -> List[DetectedObject]:
        """Parse detected objects"""
        objects = []

        for obj in objects_data:
            objects.append(DetectedObject(
                name=obj.get('object', ''),
                confidence=obj.get('confidence', 0.0),
                rectangle=obj.get('rectangle')
            ))

        # Sort by confidence (highest first)
        objects.sort(key=lambda x: x.confidence, reverse=True)

        return objects

    @staticmethod
    def _parse_categories(categories_data: List[Dict]) -> List[ImageCategory]:
        """Parse image categories"""
        categories = []

        for cat in categories_data:
            categories.append(ImageCategory(
                name=cat.get('name', ''),
                score=cat.get('score', 0.0),
                detail=cat.get('detail')
            ))

        # Sort by score (highest first)
        categories.sort(key=lambda x: x.score, reverse=True)

        return categories

    @staticmethod
    def _parse_adult_content(adult_data: Dict, result: AnalysisResult) -> None:
        """Parse adult content scores"""
        result.is_adult_content = adult_data.get('is_adult_content', False)
        result.adult_score = adult_data.get('adult_score', 0.0)
        result.is_racy_content = adult_data.get('is_racy_content', False)
        result.racy_score = adult_data.get('racy_score', 0.0)
        result.is_gory_content = adult_data.get('is_gory_content', False)
        result.gore_score = adult_data.get('gore_score', 0.0)

    @staticmethod
    def _parse_color_info(color_data: Dict, result: AnalysisResult) -> None:
        """Parse color information"""
        result.dominant_colors = color_data.get('dominant_colors', [])
        result.accent_color = color_data.get('accent_color')
        result.is_bw_image = color_data.get('is_bw_img', False)

    @staticmethod
    def _parse_faces(faces_data: List[Dict]) -> List[FaceDetection]:
        """Parse detected faces"""
        faces = []

        for face in faces_data:
            faces.append(FaceDetection(
                age=face.get('age'),
                gender=face.get('gender'),
                face_rectangle=face.get('face_rectangle')
            ))

        return faces

    @staticmethod
    def _parse_metadata(metadata: Dict) -> ImageMetadata:
        """Parse image metadata"""
        return ImageMetadata(
            width=metadata.get('width', 0),
            height=metadata.get('height', 0),
            format=metadata.get('format', 'unknown')
        )

    @staticmethod
    def parse_ocr_response(ocr_data: Dict) -> List[OCRResult]:
        """
        Parse OCR response

        Args:
            ocr_data: OCR response dictionary

        Returns:
            List of OCRResult objects
        """
        logger.info("Parsing OCR response")

        ocr_results = []

        text_results = ocr_data.get('text_results', [])
        language = ocr_data.get('language')

        for line in text_results:
            # Calculate average confidence from words
            words = line.get('words', [])
            if words:
                avg_confidence = sum(w.get('confidence', 0) for w in words) / len(words)
            else:
                avg_confidence = 0.8  # Default confidence if not available

            ocr_results.append(OCRResult(
                text=line.get('text', ''),
                confidence=avg_confidence,
                bounding_box=line.get('bounding_box'),
                language=language
            ))

        logger.info(f"Parsed {len(ocr_results)} OCR text lines")

        return ocr_results

    @staticmethod
    def extract_text_content(ocr_results: List[OCRResult]) -> str:
        """Extract all text content as single string"""
        return '\n'.join([ocr.text for ocr in ocr_results])

    @staticmethod
    def filter_by_confidence(
            items: List,
            min_confidence: float = 0.5,
            confidence_attr: str = 'confidence'
    ) -> List:
        """
        Filter items by minimum confidence threshold

        Args:
            items: List of items with confidence scores
            min_confidence: Minimum confidence threshold
            confidence_attr: Name of confidence attribute

        Returns:
            Filtered list
        """
        return [
            item for item in items
            if getattr(item, confidence_attr, 0) >= min_confidence
        ]


def parse_analysis_response(response: Dict[str, Any]) -> AnalysisResult:
    """
    Convenience function to parse analysis response

    Args:
        response: Raw API response

    Returns:
        AnalysisResult object
    """
    parser = ResponseParser()
    return parser.parse_full_analysis(response)