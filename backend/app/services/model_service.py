from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from krishi_core.uncertainty import analyze_probs

log = logging.getLogger("krishidrishti.model")


class ModelUnavailable(RuntimeError):
    pass


class ModelService:
    """Loads the TensorFlow Lite model ONCE at startup and keeps it in memory."""

    def __init__(self, model_dir: Path, default_thresholds: dict):
        self.model_dir = Path(model_dir)
        self.default_thresholds = default_thresholds

        self.model = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        self.meta: dict = {}
        self.class_names: list[str] = []

    def load_model(self) -> bool:
        try:
            # Load metadata
            self.meta = json.loads(
                (self.model_dir / "model_meta.json").read_text(
                    encoding="utf-8"
                )
            )

            self.class_names = self.meta["class_names"]

            # Load TensorFlow Lite model
            import tensorflow as tf

            model_path = self.model_dir / "model_fp32.tflite"

            if not model_path.exists():
                raise FileNotFoundError(
                    f"TFLite model not found: {model_path}"
                )

            self.interpreter = tf.lite.Interpreter(
                model_path=str(model_path)
            )

            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]

            # Warm-up
            dummy_input = np.zeros(
                (1, self.input_size, self.input_size, 3),
                dtype=np.float32,
            )

            self.interpreter.set_tensor(
                self.input_details["index"],
                dummy_input,
            )

            self.interpreter.invoke()

            self.interpreter.get_tensor(
                self.output_details["index"]
            )

            log.info(
                "Loaded TFLite model %s (%d classes)",
                self.get_model_version(),
                len(self.class_names),
            )

            log.info(
                "TFLite input shape=%s dtype=%s",
                self.input_details["shape"],
                self.input_details["dtype"],
            )

            log.info(
                "TFLite output shape=%s dtype=%s",
                self.output_details["shape"],
                self.output_details["dtype"],
            )

            return True

        except Exception as e:
            # Backend must still start; /predict returns 503
            log.error(
                "Model could not be loaded from %s: %s",
                self.model_dir,
                e,
            )

            self.model = None
            self.interpreter = None
            self.input_details = None
            self.output_details = None

            return False

    @property
    def loaded(self) -> bool:
        return self.interpreter is not None

    @property
    def input_size(self) -> int:
        return int(self.meta.get("input_size", 224))

    @property
    def thresholds(self) -> dict:
        u = self.meta.get("uncertainty") or {}

        return {
            **self.default_thresholds,
            **{
                k: u[k]
                for k in ("confidence", "margin", "entropy")
                if k in u
            },
        }

    def get_model_version(self) -> str:
        return str(self.meta.get("model_version", "unknown"))

    def predict(self, batch: np.ndarray) -> np.ndarray:
        if not self.loaded:
            raise ModelUnavailable("model not loaded")

        # Ensure correct dtype for the TFLite model
        batch = np.asarray(batch, dtype=np.float32)

        self.interpreter.set_tensor(
            self.input_details["index"],
            batch,
        )

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details["index"]
        )

        return output[0]

    def analyze(self, probs) -> dict:
        return analyze_probs(
            probs,
            self.class_names,
            self.thresholds,
        )