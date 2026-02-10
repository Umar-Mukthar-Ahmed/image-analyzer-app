"""
Unit Tests for Image Handler
"""
import unittest
from pathlib import Path
from src.core.image_handler import ImageHandler, ImageValidationError


class TestImageHandler(unittest.TestCase):
    """Test cases for image handler"""

    def test_validate_format_valid(self):
        """Test validation of valid image formats"""
        valid_formats = ['jpg', 'jpeg', 'png', 'bmp', 'gif']

        for fmt in valid_formats:
            with self.subTest(format=fmt):
                path = Path(f"test.{fmt}")
                # Should not raise
                try:
                    ImageHandler.validate_format(path)
                except ImageValidationError:
                    self.fail(f"Valid format {fmt} raised validation error")

    def test_validate_format_invalid(self):
        """Test validation rejects invalid formats"""
        path = Path("test.txt")

        with self.assertRaises(ImageValidationError) as context:
            ImageHandler.validate_format(path)

        self.assertIn("Unsupported format", str(context.exception))

    def test_encode_decode_base64(self):
        """Test base64 encoding and decoding"""
        test_data = b"test image data"

        # Encode
        encoded = ImageHandler.encode_to_base64(test_data)
        self.assertIsInstance(encoded, str)

        # Decode
        decoded = ImageHandler.decode_from_base64(encoded)
        self.assertEqual(decoded, test_data)


if __name__ == '__main__':
    unittest.main()