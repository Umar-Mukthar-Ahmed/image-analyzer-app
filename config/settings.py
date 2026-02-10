"""
Configuration management for Image Analyzer App
"""
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Centralized configuration settings"""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    LOGS_DIR = PROJECT_ROOT / "logs"
    EXAMPLES_DIR = PROJECT_ROOT / "examples" / "sample_images"

    # Azure Computer Vision API Configuration
    AZURE_VISION_ENDPOINT: str = os.getenv("AZURE_VISION_ENDPOINT", "")
    AZURE_VISION_KEY: str = os.getenv("AZURE_VISION_KEY", "")

    # Application Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_IMAGE_SIZE_MB: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "4"))
    MAX_IMAGE_SIZE_BYTES: int = MAX_IMAGE_SIZE_MB * 1024 * 1024

    SUPPORTED_FORMATS: List[str] = os.getenv(
        "SUPPORTED_FORMATS",
        "jpg,jpeg,png,bmp,gif"
    ).split(",")

    # Confidence Thresholds
    HIGH_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.8")
    )
    MEDIUM_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("MEDIUM_CONFIDENCE_THRESHOLD", "0.5")
    )
    LOW_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.3")
    )

    # API Parameters
    LANGUAGE: str = "en"
    MAX_DESCRIPTIONS: int = 3
    MAX_TAGS: int = 10

    @classmethod
    def validate(cls) -> None:
        """Validate that all required settings are present"""
        required_settings = [
            ("AZURE_VISION_ENDPOINT", cls.AZURE_VISION_ENDPOINT),
            ("AZURE_VISION_KEY", cls.AZURE_VISION_KEY),
        ]

        missing = [name for name, value in required_settings if not value]

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Please copy .env.example to .env and fill in your credentials."
            )

    @classmethod
    def get_log_file_path(cls) -> Path:
        """Get the log file path"""
        cls.LOGS_DIR.mkdir(exist_ok=True)
        return cls.LOGS_DIR / "image_analyzer.log"


# Create a singleton instance
settings = Settings()