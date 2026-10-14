from pathlib import Path
import tensorflow as tf


RUN_NAME = "mobilenetv3s-v1"

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / RUN_NAME
MODEL_PATH = MODEL_DIR / "model.keras"
OUTPUT_PATH = MODEL_DIR / "model_fp32.tflite"


def main():
    print(f"Loading model: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    OUTPUT_PATH.write_bytes(tflite_model)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)

    print()
    print("TFLite conversion successful!")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()