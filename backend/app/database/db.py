from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """Only non-sensitive scan metadata. No user identity, no GPS (EXIF is stripped on upload)."""
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_reference: Mapped[str] = mapped_column(String(255))
    gradcam_reference: Mapped[str | None] = mapped_column(String(255), nullable=True)
    severity_reference: Mapped[str | None] = mapped_column(String(255), nullable=True)
    selected_crop: Mapped[str | None] = mapped_column(String(64), nullable=True)
    class_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    crop: Mapped[str | None] = mapped_column(String(64), nullable=True)
    predicted_disease: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    reliable: Mapped[bool] = mapped_column(Boolean)
    severity_percentage: Mapped[float | None] = mapped_column(Float, nullable=True)
    severity_level: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_version: Mapped[str] = mapped_column(String(64))
    language: Mapped[str] = mapped_column(String(8), default="en")
    result_json: Mapped[str] = mapped_column(Text)  # full response snapshot for re-opening
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


def make_session_factory(url: str):
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
