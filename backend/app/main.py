from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config.settings import Settings
from app.database.db import make_session_factory
from app.services.advisory_service import AdvisoryService
from app.services.explain_service import ExplainService
from app.services.history_service import HistoryService
from app.services.image_service import ImageService
from app.services.model_service import ModelService
from app.services.severity_service import SeverityService

logging.basicConfig(level=logging.INFO)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    (settings.storage_dir / "media").mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model = ModelService(settings.model_dir, settings.default_thresholds)
        model.load_model()  # loaded ONCE, kept in memory
        sf = make_session_factory(settings.database_url)
        app.state.svc = {
            "settings": settings, "model": model, "image": ImageService(settings),
            "severity": SeverityService(), "explain": ExplainService(),
            "advisory": AdvisoryService(settings.kb_path), "history": HistoryService(sf, settings.storage_dir),
        }
        yield

    app = FastAPI(title="KrishiDrishti API", version="1.0.0",
                  description="AI-assisted crop disease detection. Not a guaranteed diagnosis.", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"])
    app.include_router(router)
    app.mount("/media", StaticFiles(directory=settings.storage_dir / "media"), name="media")
    return app


app = create_app()
