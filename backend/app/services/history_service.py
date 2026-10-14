from __future__ import annotations

import json
from pathlib import Path

from app.database.db import Prediction


class HistoryService:
    def __init__(self, session_factory, storage_dir: Path):
        self.sf = session_factory
        self.dir = Path(storage_dir)

    def save(self, **fields) -> Prediction:
        with self.sf() as s:
            row = Prediction(**fields)
            s.add(row); s.commit(); s.refresh(row)
            return row

    def update_result(self, pid: int, result: dict):
        with self.sf() as s:
            row = s.get(Prediction, pid)
            row.result_json = json.dumps(result, ensure_ascii=False)
            s.commit()

    def list(self, limit=50, offset=0):
        with self.sf() as s:
            return s.query(Prediction).order_by(Prediction.id.desc()).offset(offset).limit(limit).all()

    def get(self, pid: int):
        with self.sf() as s:
            return s.get(Prediction, pid)

    def delete(self, pid: int) -> bool:
        with self.sf() as s:
            row = s.get(Prediction, pid)
            if not row:
                return False
            for ref in (row.image_reference, row.gradcam_reference, row.severity_reference):
                if ref:
                    (self.dir / "media" / ref).unlink(missing_ok=True)
            s.delete(row); s.commit()
            return True
