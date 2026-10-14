from pathlib import Path

import numpy as np
import tensorflow as tf


RUN_NAME = "mobilenetv3s-v1"

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / RUN_NAME
MODEL_PATH = MODEL_DIR / "model.keras"

OUTPUT_PATH = MODEL_DIR / "model_int8_balanced.tflite"

TRAIN_DIR = ROOT / "data" / "train"

IMAGE_SIZE = 224

IMAGES_PER_CLASS = 12


def get_calibration_images():
    """Select the same number of real images from every class."""

    class_dirs = sorted(
        path
        for path in TRAIN_DIR.iterdir()
        if path.is_dir()
    )

    if len(class_dirs) != 17:
        raise RuntimeError(
            f"Expected 17 classes, found {len(class_dirs)}"
        )

    selected_images = []

    rng = np.random.default_rng(42)

    print()
    print("Selecting balanced calibration images:")
    print("=" * 60)

    for class_dir in class_dirs:
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in {".jpg", ".jpeg", ".png"}
        )

        if len(image_paths) < IMAGES_PER_CLASS:
            raise RuntimeError(
                f"Class '{class_dir.name}' has only "
                f"{len(image_paths)} images."
            )

        selected = rng.choice(
            image_paths,
            size=IMAGES_PER_CLASS,
            replace=False,
        )

        selected_images.extend(selected)

        print(
            f"{class_dir.name}: "
            f"{IMAGES_PER_CLASS} images"
        )

    print("=" * 60)
    print(
        f"Total calibration images: "
        f"{len(selected_images)}"
    )

    return selected_images


def load_image(image_path):
    """Load image as float32 RGB in the 0..255 range."""

    image = tf.io.read_file(str(image_path))

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    image = tf.expand_dims(
        image,
        axis=0,
    )

    return image.numpy()


def representative_dataset(calibration_images):
    """Yield balanced real images for calibration."""

    for image_path in calibration_images:
        image = load_image(image_path)

        yield [image]


def main():
    print(
        f"Training data directory:\n{TRAIN_DIR}"
    )

    calibration_images = get_calibration_images()

    print()
    print(
        f"Loading model:\n{MODEL_PATH}"
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")
    print(
        f"Input shape : {model.input_shape}"
    )
    print(
        f"Output shape: {model.output_shape}"
    )

    converter = (
        tf.lite.TFLiteConverter.from_keras_model(
            model
        )
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = lambda: (
        representative_dataset(
            calibration_images
        )
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    # Keep application input/output as float32.
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    print()
    print(
        "Starting balanced INT8 quantization..."
    )
    print(
        "17 classes × 12 images = 204 "
        "calibration images."
    )

    tflite_model = converter.convert()

    OUTPUT_PATH.write_bytes(
        tflite_model
    )

    size_mb = (
        OUTPUT_PATH.stat().st_size
        / (1024 * 1024)
    )

    print()
    print("=" * 60)
    print(
        "BALANCED INT8 QUANTIZATION SUCCESSFUL"
    )
    print("=" * 60)

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        f"Size: {size_mb:.2f} MB"
    )

    print(
        "Calibration images: "
        f"{len(calibration_images)}"
    )


if __name__ == "__main__":
    main()