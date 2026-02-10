"""
Image Handler - Binary Data Handling
Manages image uploads, encoding, validation, and transmission
"""
import io
import base64
import logging
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
from config.settings import settings

logger = logging.getLogger(__name__)


class ImageValidationError(Exception):
    """Custom exception for image validation errors"""
    pass


class ImageHandler:
    """Handles image loading, validation, and encoding"""

    @staticmethod
    def validate_format(file_path: Path) -> bool:
        """Validate image file format"""
        extension = file_path.suffix.lower().lstrip('.')

        if extension not in settings.SUPPORTED_FORMATS:
            raise ImageValidationError(
                f"Unsupported format: {extension}. "
                f"Supported formats: {', '.join(settings.SUPPORTED_FORMATS)}"
            )

        return True

    @staticmethod
    def validate_size(file_path: Path) -> bool:
        """Validate image file size"""
        file_size = file_path.stat().st_size

        if file_size > settings.MAX_IMAGE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            raise ImageValidationError(
                f"File too large: {size_mb:.2f}MB. "
                f"Maximum allowed: {settings.MAX_IMAGE_SIZE_MB}MB"
            )

        return True

    @staticmethod
    def validate_image(file_path: Path) -> bool:
        """Validate that file is actually an image"""
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception as e:
            raise ImageValidationError(f"Invalid or corrupted image file: {str(e)}")

    @staticmethod
    def load_image(file_path: str) -> Tuple[bytes, Image.Image]:
        """
        Load and validate image from file path

        Args:
            file_path: Path to image file

        Returns:
            Tuple of (binary_data, PIL_Image)

        Raises:
            ImageValidationError: If validation fails
        """
        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            raise ImageValidationError(f"File not found: {file_path}")

        # Validate format
        ImageHandler.validate_format(path)

        # Validate size
        ImageHandler.validate_size(path)

        # Validate it's actually an image
        ImageHandler.validate_image(path)

        # Load image
        try:
            with open(path, 'rb') as f:
                image_data = f.read()

            pil_image = Image.open(path)

            logger.info(
                f"Image loaded successfully: {path.name} "
                f"({pil_image.width}x{pil_image.height}, "
                f"{len(image_data) / 1024:.1f}KB)"
            )

            return image_data, pil_image

        except Exception as e:
            raise ImageValidationError(f"Error loading image: {str(e)}")

    @staticmethod
    def encode_to_base64(image_data: bytes) -> str:
        """Encode image binary data to base64 string"""
        return base64.b64encode(image_data).decode('utf-8')

    @staticmethod
    def decode_from_base64(base64_string: str) -> bytes:
        """Decode base64 string to binary data"""
        return base64.b64decode(base64_string)

    @staticmethod
    def get_image_info(pil_image: Image.Image) -> dict:
        """Extract image metadata"""
        return {
            'width': pil_image.width,
            'height': pil_image.height,
            'format': pil_image.format,
            'mode': pil_image.mode,
            'size_kb': pil_image.size[0] * pil_image.size[1] * 3 / 1024  # Approximate
        }

    @staticmethod
    def resize_image(pil_image: Image.Image, max_dimension: int = 1024) -> Image.Image:
        """
        Resize image if it exceeds max dimension while maintaining aspect ratio
        """
        width, height = pil_image.size

        if width <= max_dimension and height <= max_dimension:
            return pil_image

        if width > height:
            new_width = max_dimension
            new_height = int((max_dimension / width) * height)
        else:
            new_height = max_dimension
            new_width = int((max_dimension / height) * width)

        logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")

        return pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    @staticmethod
    def image_to_bytes(pil_image: Image.Image, format: str = 'JPEG') -> bytes:
        """Convert PIL Image to bytes"""
        buffer = io.BytesIO()

        # Convert RGBA to RGB if saving as JPEG
        if format.upper() == 'JPEG' and pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')

        pil_image.save(buffer, format=format)
        return buffer.getvalue()

    @staticmethod
    def load_from_bytes(image_bytes: bytes) -> Image.Image:
        """Load PIL Image from bytes"""
        return Image.open(io.BytesIO(image_bytes))


def load_and_validate_image(file_path: str) -> Tuple[bytes, Image.Image]:
    """
    Convenience function to load and validate image

    Args:
        file_path: Path to image file

    Returns:
        Tuple of (binary_data, PIL_Image)
    """
    handler = ImageHandler()
    return handler.load_image(file_path)