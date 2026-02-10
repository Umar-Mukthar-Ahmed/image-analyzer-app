"""
Data models for image analysis results
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class DetectedObject:
    """Represents a detected object in the image"""
    name: str
    confidence: float
    rectangle: Optional[Dict[str, int]] = None  # x, y, w, h

    @property
    def confidence_percentage(self) -> float:
        """Return confidence as percentage"""
        return round(self.confidence * 100, 2)

    @property
    def confidence_level(self) -> str:
        """Categorize confidence level"""
        if self.confidence >= 0.8:
            return "High"
        elif self.confidence >= 0.5:
            return "Medium"
        else:
            return "Low"


@dataclass
class ImageTag:
    """Represents a tag/label for the image"""
    name: str
    confidence: float
    hint: Optional[str] = None

    @property
    def confidence_percentage(self) -> float:
        """Return confidence as percentage"""
        return round(self.confidence * 100, 2)


@dataclass
class ImageDescription:
    """Represents a natural language description of the image"""
    text: str
    confidence: float

    @property
    def confidence_percentage(self) -> float:
        """Return confidence as percentage"""
        return round(self.confidence * 100, 2)


@dataclass
class ImageCategory:
    """Represents a category classification"""
    name: str
    score: float
    detail: Optional[str] = None


@dataclass
class OCRResult:
    """Represents text extracted from image (OCR)"""
    text: str
    confidence: float
    bounding_box: Optional[List[int]] = None
    language: Optional[str] = None


@dataclass
class FaceDetection:
    """Represents a detected face"""
    age: Optional[int] = None
    gender: Optional[str] = None
    face_rectangle: Optional[Dict[str, int]] = None


@dataclass
class ImageMetadata:
    """Image metadata"""
    width: int
    height: int
    format: str

    @property
    def dimensions(self) -> str:
        """Return formatted dimensions"""
        return f"{self.width}x{self.height}"


@dataclass
class AnalysisResult:
    """Complete image analysis result"""

    # Core results
    descriptions: List[ImageDescription] = field(default_factory=list)
    tags: List[ImageTag] = field(default_factory=list)
    objects: List[DetectedObject] = field(default_factory=list)
    categories: List[ImageCategory] = field(default_factory=list)

    # OCR results
    ocr_text: List[OCRResult] = field(default_factory=list)

    # Face detection
    faces: List[FaceDetection] = field(default_factory=list)

    # Metadata
    metadata: Optional[ImageMetadata] = None

    # Adult content scores
    is_adult_content: bool = False
    adult_score: float = 0.0
    is_racy_content: bool = False
    racy_score: float = 0.0
    is_gory_content: bool = False
    gore_score: float = 0.0

    # Brands
    brands: List[str] = field(default_factory=list)

    # Color information
    dominant_colors: List[str] = field(default_factory=list)
    accent_color: Optional[str] = None
    is_bw_image: bool = False

    # Raw response (for debugging)
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def has_descriptions(self) -> bool:
        """Check if descriptions are available"""
        return len(self.descriptions) > 0

    @property
    def has_objects(self) -> bool:
        """Check if objects are detected"""
        return len(self.objects) > 0

    @property
    def has_text(self) -> bool:
        """Check if text is detected"""
        return len(self.ocr_text) > 0

    @property
    def has_faces(self) -> bool:
        """Check if faces are detected"""
        return len(self.faces) > 0

    @property
    def primary_description(self) -> Optional[str]:
        """Get the highest confidence description"""
        if self.descriptions:
            return self.descriptions[0].text
        return None

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all detections"""
        confidences = []

        if self.descriptions:
            confidences.extend([d.confidence for d in self.descriptions])
        if self.tags:
            confidences.extend([t.confidence for t in self.tags])
        if self.objects:
            confidences.extend([o.confidence for o in self.objects])

        if confidences:
            return sum(confidences) / len(confidences)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "descriptions": [
                {"text": d.text, "confidence": d.confidence_percentage}
                for d in self.descriptions
            ],
            "tags": [
                {"name": t.name, "confidence": t.confidence_percentage}
                for t in self.tags
            ],
            "objects": [
                {"name": o.name, "confidence": o.confidence_percentage}
                for o in self.objects
            ],
            "categories": [
                {"name": c.name, "score": c.score}
                for c in self.categories
            ],
            "ocr_text": [
                {"text": ocr.text, "confidence": ocr.confidence}
                for ocr in self.ocr_text
            ],
            "metadata": {
                "dimensions": self.metadata.dimensions if self.metadata else None,
                "format": self.metadata.format if self.metadata else None
            },
            "content_flags": {
                "is_adult": self.is_adult_content,
                "is_racy": self.is_racy_content,
                "is_gory": self.is_gory_content
            },
            "colors": {
                "dominant": self.dominant_colors,
                "accent": self.accent_color,
                "is_black_and_white": self.is_bw_image
            },
            "average_confidence": round(self.average_confidence * 100, 2)
        }