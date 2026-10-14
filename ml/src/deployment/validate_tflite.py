from pathlib import Path

import numpy as np
import tensorflow as tf


RUN_NAME = "mobilenetv3s-v1"

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / RUN_NAME

KERAS_MODEL = MODEL_DIR / "model.keras"
TFLITE_MODEL = MODEL_DIR / "model_fp32.tflite"


def main():
    print("Loading Keras model...")
    keras_model = tf.keras.models.load_model(KERAS_MODEL)

    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print()
    print("Keras input shape:", keras_model.input_shape)
    print("Keras output shape:", keras_model.output_shape)

    print()
    print("TFLite input:")
    print(input_details[0])

    print()
    print("TFLite output:")
    print(output_details[0])

    # Create one deterministic test input.
    rng = np.random.default_rng(42)

    test_input = rng.uniform(
        0,
        255,
        size=(1, 224, 224, 3)
    ).astype(np.float32)

    # Keras prediction
    keras_output = keras_model.predict(
        test_input,
        verbose=0
    )

    # TFLite prediction
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    interpreter.set_tensor(input_index, test_input)
    interpreter.invoke()

    tflite_output = interpreter.get_tensor(output_index)

    # Compare
    max_difference = np.max(
        np.abs(keras_output - tflite_output)
    )

    mean_difference = np.mean(
        np.abs(keras_output - tflite_output)
    )

    keras_class = int(np.argmax(keras_output))
    tflite_class = int(np.argmax(tflite_output))

    print()
    print("=" * 50)
    print("TFLite Validation")
    print("=" * 50)

    print(f"Keras predicted class : {keras_class}")
    print(f"TFLite predicted class: {tflite_class}")

    print(f"Maximum difference   : {max_difference:.8f}")
    print(f"Mean difference      : {mean_difference:.8f}")

    if keras_class == tflite_class:
        print()
        print("✅ VALIDATION PASSED")
        print("Keras and TFLite predict the same class.")
    else:
        print()
        print("❌ VALIDATION FAILED")
        print("Keras and TFLite predictions differ.")


if __name__ == "__main__":
    main()