from __future__ import annotations

from krishi_core.severity import estimate_severity


class SeverityService:
    def estimate(self, img, is_healthy: bool) -> dict:
        if is_healthy:
            return {"available": True, "percent": 0.0, "level": "None",
                    "note": "Classified healthy - no lesion area estimated.", "overlay": None}
        return estimate_severity(img)
