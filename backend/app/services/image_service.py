from __future__ import annotations

import io

from krishi_core.preprocess import load_pil, pil_to_model_input, to_batch
from krishi_core.quality import QualityReport, assess_quality
from PIL import Image, UnidentifiedImageError


class InvalidImage(ValueError):
    pass


class ImageService:
    def __init__(self, settings):
        self.s = settings

    def validate_and_load(self, data: bytes, content_type: str | None) -> Image.Image:
        if len(data) > self.s.max_upload_mb * 1024 * 1024:
            raise InvalidImage(f"File too large (max {self.s.max_upload_mb} MB).")
        if content_type and content_type.lower() not in self.s.allowed_types:
            raise InvalidImage("Unsupported file type. Please upload a JPEG or PNG image.")
        if not data:
            raise InvalidImage("Empty file.")
        try:
            return load_pil(data)
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            raise InvalidImage("The image could not be read (corrupted or not an image).")

    def quality(self, img: Image.Image) -> QualityReport:
        s = self.s
        return assess_quality(img, s.min_side, s.blur_threshold, s.dark_threshold, s.bright_threshold)

    @staticmethod
    def to_batch(img: Image.Image, size: int):
        return to_batch(pil_to_model_input(img, (size, size)))

    @staticmethod
    def clean_jpeg_bytes(img: Image.Image) -> bytes:
        """Re-encode from pixels: strips EXIF/GPS metadata for privacy."""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
