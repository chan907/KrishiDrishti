from pathlib import Path

import numpy as np
import tensorflow as tf


RUN_NAME = "mobilenetv3s-v1"

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / RUN_NAME

MODEL_PATH = MODEL_DIR / "model_fp32.tflite"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = 224


class TFLitePredictor:
    def __init__(
        self,
        model_path=MODEL_PATH,
        class_names_path=CLASS_NAMES_PATH,
    ):
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)

        self.interpreter = tf.lite.Interpreter(
            model_path=str(self.model_path)
        )

        self.interpreter.allocate_tensors()

        self.input_details = (
            self.interpreter.get_input_details()[0]
        )

        self.output_details = (
            self.interpreter.get_output_details()[0]
        )

        import json

        with open(
            self.class_names_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.class_names = json.load(file)

        if len(self.class_names) != 17:
            raise ValueError(
                f"Expected 17 classes, "
                f"found {len(self.class_names)}"
            )

    def preprocess(self, image_path):
        """Prepare an image for the TFLite model."""

        image = tf.io.read_file(
            str(image_path)
        )

        image = tf.image.decode_image(
            image,
            channels=3,
            expand_animations=False,
        )

        image = tf.image.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE],
        )

        # The trained model expects float32 RGB 0..255.
        image = tf.cast(
            image,
            tf.float32,
        )

        image = tf.expand_dims(
            image,
            axis=0,
        )

        return image.numpy()

    def predict(self, image_path):
        """Run disease prediction."""

        image = self.preprocess(
            image_path
        )

        self.interpreter.set_tensor(
            self.input_details["index"],
            image,
        )

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details["index"]
        )[0]

        predicted_index = int(
            np.argmax(output)
        )

        confidence = float(
            output[predicted_index]
        )

        disease = self.class_names[
            predicted_index
        ]

        return {
            "class_index": predicted_index,
            "class_name": disease,
            "confidence": confidence,
        }


def main():
    print("Loading KrishiDrishti TFLite model...")

    predictor = TFLitePredictor()

    print("Model loaded successfully.")
    print()
    print(
        f"Input shape: "
        f"{predictor.input_details['shape']}"
    )

    print(
        f"Input dtype: "
        f"{predictor.input_details['dtype']}"
    )

    print(
        f"Number of classes: "
        f"{len(predictor.class_names)}"
    )

    print()
    print("Classes:")

    for index, name in enumerate(
        predictor.class_names
    ):
        print(f"{index}: {name}")


if __name__ == "__main__":
    main()