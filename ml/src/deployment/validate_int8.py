from pathlib import Path

import numpy as np
import tensorflow as tf


RUN_NAME = "mobilenetv3s-v1"

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / RUN_NAME

FP32_MODEL = MODEL_DIR / "model_fp32.tflite"
INT8_MODEL = MODEL_DIR / "model_int8.tflite"


def load_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details[0], output_details[0]


def run_inference(interpreter, input_detail, output_detail, image):
    interpreter.set_tensor(
        input_detail["index"],
        image
    )

    interpreter.invoke()

    return interpreter.get_tensor(
        output_detail["index"]
    )


def main():
    print("Loading FP32 TFLite model...")
    fp32_interpreter, fp32_input, fp32_output = load_interpreter(
        FP32_MODEL
    )

    print("Loading INT8 TFLite model...")
    int8_interpreter, int8_input, int8_output = load_interpreter(
        INT8_MODEL
    )

    print()
    print("FP32 input dtype :", fp32_input["dtype"])
    print("FP32 output dtype:", fp32_output["dtype"])

    print("INT8 input dtype :", int8_input["dtype"])
    print("INT8 output dtype:", int8_output["dtype"])

    # Deterministic test image.
    rng = np.random.default_rng(42)

    test_input = rng.uniform(
        0,
        255,
        size=(1, 224, 224, 3)
    ).astype(np.float32)

    # FP32 prediction
    fp32_prediction = run_inference(
        fp32_interpreter,
        fp32_input,
        fp32_output,
        test_input
    )

    # INT8 model currently expects float32 input/output.
    int8_prediction = run_inference(
        int8_interpreter,
        int8_input,
        int8_output,
        test_input
    )

    fp32_class = int(np.argmax(fp32_prediction))
    int8_class = int(np.argmax(int8_prediction))

    max_difference = np.max(
        np.abs(fp32_prediction - int8_prediction)
    )

    mean_difference = np.mean(
        np.abs(fp32_prediction - int8_prediction)
    )

    print()
    print("=" * 55)
    print("INT8 TFLite Validation")
    print("=" * 55)

    print(f"FP32 predicted class : {fp32_class}")
    print(f"INT8 predicted class : {int8_class}")

    print(f"Maximum difference  : {max_difference:.8f}")
    print(f"Mean difference     : {mean_difference:.8f}")

    if fp32_class == int8_class:
        print()
        print("✅ VALIDATION PASSED")
        print("FP32 and INT8 predict the same class.")
    else:
        print()
        print("⚠️ PREDICTION DIFFERENCE")
        print("FP32 and INT8 predict different classes.")

    print()
    print("NOTE:")
    print("This is a basic conversion check.")
    print("Real test-set evaluation will be performed separately.")


if __name__ == "__main__":
    main()