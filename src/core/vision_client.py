"""
Azure Computer Vision API Client
Handles communication with Azure Vision API
"""
import io
import time
import logging
from typing import Optional, Dict, Any
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from config.settings import settings

logger = logging.getLogger(__name__)


class VisionAPIError(Exception):
    """Custom exception for Vision API errors"""
    pass


class VisionClient:
    """Wrapper for Azure Computer Vision API"""

    def __init__(self):
        """Initialize Azure Computer Vision client"""
        settings.validate()

        try:
            credentials = CognitiveServicesCredentials(settings.AZURE_VISION_KEY)
            self._client = ComputerVisionClient(
                settings.AZURE_VISION_ENDPOINT,
                credentials
            )
            logger.info("Azure Vision client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Vision client: {str(e)}")
            raise VisionAPIError(f"Client initialization failed: {str(e)}")

    def analyze_image(
            self,
            image_data: bytes,
            visual_features: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Analyze image with Azure Computer Vision API

        Args:
            image_data: Binary image data
            visual_features: List of features to analyze

        Returns:
            Analysis results as dictionary
        """
        if visual_features is None:
            visual_features = [
                VisualFeatureTypes.categories,
                VisualFeatureTypes.description,
                VisualFeatureTypes.tags,
                VisualFeatureTypes.objects,
                VisualFeatureTypes.brands,
                VisualFeatureTypes.faces,
                VisualFeatureTypes.adult,
                VisualFeatureTypes.color,
                VisualFeatureTypes.image_type
            ]

        try:
            logger.info(f"Analyzing image with features: {[f.value for f in visual_features]}")

            # Convert bytes to stream
            image_stream = io.BytesIO(image_data)

            analysis = self._client.analyze_image_in_stream(
                image_stream,
                visual_features=visual_features,
                language=settings.LANGUAGE
            )

            logger.info("Image analysis completed successfully")

            return self._convert_to_dict(analysis)

        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            raise VisionAPIError(f"Analysis failed: {str(e)}")

    def detect_objects(self, image_data: bytes) -> Dict[str, Any]:
        """
        Detect objects in image

        Args:
            image_data: Binary image data

        Returns:
            Object detection results
        """
        try:
            logger.info("Detecting objects in image")

            # Convert bytes to stream
            image_stream = io.BytesIO(image_data)

            result = self._client.detect_objects_in_stream(image_stream)

            logger.info(f"Detected {len(result.objects)} objects")

            return {
                'objects': [
                    {
                        'object': obj.object_property,
                        'confidence': obj.confidence,
                        'rectangle': {
                            'x': obj.rectangle.x,
                            'y': obj.rectangle.y,
                            'w': obj.rectangle.w,
                            'h': obj.rectangle.h
                        } if obj.rectangle else None,
                        'parent': obj.parent.object_property if obj.parent else None
                    }
                    for obj in result.objects
                ],
                'metadata': {
                    'width': result.metadata.width,
                    'height': result.metadata.height,
                    'format': result.metadata.format
                } if result.metadata else None
            }

        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}")
            raise VisionAPIError(f"Object detection failed: {str(e)}")

    def read_text(self, image_data: bytes) -> Dict[str, Any]:
        """
        Extract text from image using OCR

        Args:
            image_data: Binary image data

        Returns:
            OCR results
        """
        try:
            logger.info("Extracting text from image (OCR)")

            # Convert bytes to stream
            image_stream = io.BytesIO(image_data)

            # Initiate read operation
            read_response = self._client.read_in_stream(image_stream, raw=True)

            # Get operation location
            operation_location = read_response.headers["Operation-Location"]
            operation_id = operation_location.split("/")[-1]

            # Wait for operation to complete
            max_retries = 30
            retry_count = 0
            while retry_count < max_retries:
                result = self._client.get_read_result(operation_id)
                if result.status.lower() not in ['notstarted', 'running']:
                    break
                time.sleep(1)
                retry_count += 1

            if result.status.lower() == 'failed':
                raise VisionAPIError("OCR operation failed")

            # Extract text
            text_results = []
            if result.analyze_result and result.analyze_result.read_results:
                for page in result.analyze_result.read_results:
                    for line in page.lines:
                        text_results.append({
                            'text': line.text,
                            'bounding_box': line.bounding_box,
                            'words': [
                                {
                                    'text': word.text,
                                    'confidence': word.confidence,
                                    'bounding_box': word.bounding_box
                                }
                                for word in line.words
                            ]
                        })

            logger.info(f"Extracted {len(text_results)} lines of text")

            return {
                'text_results': text_results,
                'language': result.analyze_result.read_results[
                    0].language if result.analyze_result.read_results else None
            }

        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            raise VisionAPIError(f"OCR failed: {str(e)}")

    def describe_image(self, image_data: bytes, max_descriptions: int = 3) -> Dict[str, Any]:
        """
        Generate natural language description of image

        Args:
            image_data: Binary image data
            max_descriptions: Maximum number of descriptions

        Returns:
            Image descriptions
        """
        try:
            logger.info("Generating image description")

            # Convert bytes to stream
            image_stream = io.BytesIO(image_data)

            result = self._client.describe_image_in_stream(
                image_stream,
                max_candidates=max_descriptions,
                language=settings.LANGUAGE
            )

            return {
                'descriptions': [
                    {'text': caption.text, 'confidence': caption.confidence}
                    for caption in result.captions
                ],
                'tags': result.tags,
                'metadata': {
                    'width': result.metadata.width,
                    'height': result.metadata.height,
                    'format': result.metadata.format
                } if result.metadata else None
            }

        except Exception as e:
            logger.error(f"Image description failed: {str(e)}")
            raise VisionAPIError(f"Description failed: {str(e)}")

    def tag_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Get tags for image

        Args:
            image_data: Binary image data

        Returns:
            Image tags with confidence scores
        """
        try:
            logger.info("Tagging image")

            # Convert bytes to stream
            image_stream = io.BytesIO(image_data)

            result = self._client.tag_image_in_stream(
                image_stream,
                language=settings.LANGUAGE
            )

            logger.info(f"Generated {len(result.tags)} tags")

            return {
                'tags': [
                    {
                        'name': tag.name,
                        'confidence': tag.confidence,
                        'hint': tag.hint
                    }
                    for tag in result.tags
                ]
            }

        except Exception as e:
            logger.error(f"Image tagging failed: {str(e)}")
            raise VisionAPIError(f"Tagging failed: {str(e)}")

    @staticmethod
    def _convert_to_dict(analysis_result) -> Dict[str, Any]:
        """Convert analysis result object to dictionary"""
        result = {}

        # Descriptions
        if hasattr(analysis_result, 'description') and analysis_result.description:
            result['descriptions'] = [
                {'text': caption.text, 'confidence': caption.confidence}
                for caption in analysis_result.description.captions
            ]
            result['description_tags'] = analysis_result.description.tags

        # Tags
        if hasattr(analysis_result, 'tags') and analysis_result.tags:
            result['tags'] = [
                {
                    'name': tag.name,
                    'confidence': tag.confidence,
                    'hint': tag.hint if hasattr(tag, 'hint') else None
                }
                for tag in analysis_result.tags
            ]

        # Objects
        if hasattr(analysis_result, 'objects') and analysis_result.objects:
            result['objects'] = [
                {
                    'object': obj.object_property,
                    'confidence': obj.confidence,
                    'rectangle': {
                        'x': obj.rectangle.x,
                        'y': obj.rectangle.y,
                        'w': obj.rectangle.w,
                        'h': obj.rectangle.h
                    } if obj.rectangle else None
                }
                for obj in analysis_result.objects
            ]

        # Categories
        if hasattr(analysis_result, 'categories') and analysis_result.categories:
            result['categories'] = [
                {
                    'name': cat.name,
                    'score': cat.score,
                    'detail': cat.detail.__dict__ if hasattr(cat, 'detail') and cat.detail else None
                }
                for cat in analysis_result.categories
            ]

        # Adult content
        if hasattr(analysis_result, 'adult') and analysis_result.adult:
            result['adult'] = {
                'is_adult_content': analysis_result.adult.is_adult_content,
                'adult_score': analysis_result.adult.adult_score,
                'is_racy_content': analysis_result.adult.is_racy_content,
                'racy_score': analysis_result.adult.racy_score,
                'is_gory_content': analysis_result.adult.is_gory_content,
                'gore_score': analysis_result.adult.gore_score
            }

        # Color
        if hasattr(analysis_result, 'color') and analysis_result.color:
            result['color'] = {
                'dominant_color_foreground': analysis_result.color.dominant_color_foreground,
                'dominant_color_background': analysis_result.color.dominant_color_background,
                'dominant_colors': analysis_result.color.dominant_colors,
                'accent_color': analysis_result.color.accent_color,
                'is_bw_img': analysis_result.color.is_bw_img
            }

        # Faces
        if hasattr(analysis_result, 'faces') and analysis_result.faces:
            result['faces'] = [
                {
                    'age': face.age,
                    'gender': face.gender,
                    'face_rectangle': {
                        'left': face.face_rectangle.left,
                        'top': face.face_rectangle.top,
                        'width': face.face_rectangle.width,
                        'height': face.face_rectangle.height
                    } if face.face_rectangle else None
                }
                for face in analysis_result.faces
            ]

        # Brands
        if hasattr(analysis_result, 'brands') and analysis_result.brands:
            result['brands'] = [
                {
                    'name': brand.name,
                    'confidence': brand.confidence,
                    'rectangle': {
                        'x': brand.rectangle.x,
                        'y': brand.rectangle.y,
                        'w': brand.rectangle.w,
                        'h': brand.rectangle.h
                    } if brand.rectangle else None
                }
                for brand in analysis_result.brands
            ]

        # Metadata
        if hasattr(analysis_result, 'metadata') and analysis_result.metadata:
            result['metadata'] = {
                'width': analysis_result.metadata.width,
                'height': analysis_result.metadata.height,
                'format': analysis_result.metadata.format
            }

        return result


# Singleton instance
_client_instance: Optional[VisionClient] = None


def get_vision_client() -> VisionClient:
    """Get or create Vision client singleton"""
    global _client_instance

    if _client_instance is None:
        _client_instance = VisionClient()

    return _client_instance