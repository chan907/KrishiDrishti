from __future__ import annotations

import json
from pathlib import Path

SUPPORTED_LANGS = ("en", "hi", "mr")


class AdvisoryService:
    """Curated knowledge base lookup. NO generative model, NO pesticide names/doses."""

    def __init__(self, kb_path: Path):
        self.kb = json.loads(Path(kb_path).read_text(encoding="utf-8"))

    @staticmethod
    def _pick(d, lang):
        if not isinstance(d, dict):
            return d
        return d.get(lang) or d.get("en")

    def crop_name(self, crop_id: str, lang="en") -> str:
        c = self.kb["crops"].get(crop_id)
        return self._pick(c["name"], lang) if c else crop_id.replace("_", " ")

    def disease_name(self, class_id: str, lang="en") -> str:
        d = self.kb["diseases"].get(class_id)
        if d:
            return self._pick(d["name"], lang)
        return class_id.split("___")[-1].replace("_", " ")

    def has(self, class_id: str) -> bool:
        return class_id in self.kb["diseases"]

    def get_advisory(self, class_id: str, severity_level: str | None = None, lang: str = "en") -> dict | None:
        lang = lang if lang in SUPPORTED_LANGS else "en"
        d = self.kb["diseases"].get(class_id)
        if not d:
            return None
        cat = self.kb["categories"][d["category"]]
        translated = lang in d.get("description", {}) and lang in cat["general_actions"]
        out = {
            "class_id": class_id,
            "disease": self._pick(d["name"], lang),
            "description": self._pick(d["description"], lang),
            "symptoms": self._pick(d["symptoms"], lang),
            "cause": d["cause"],
            "prevention": self._pick(cat["prevention"], lang),
            "general_actions": self._pick(cat["general_actions"], lang),
            "monitoring": self._pick(self.kb["monitoring"], lang),
            "consult_expert": self._pick(self.kb["consult_expert"], lang),
            "sources": d.get("sources", []),
            "language": lang,
            "translation_note": None if translated else "Showing English: reviewed translation not yet available.",
        }
        if severity_level in ("High", "Severe"):
            out["severity_note"] = self._pick(self.kb["high_severity_note"], lang)
        return out

    def crops(self):
        return [{"id": k, "name": v["name"]} for k, v in self.kb["crops"].items()]

    def diseases_for_crop(self, crop_id: str):
        return [{"class_id": k, "name": v["name"]} for k, v in self.kb["diseases"].items()
                if k.split("___")[0].lower() == crop_id.lower()]
