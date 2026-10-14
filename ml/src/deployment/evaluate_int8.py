from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


RUN_NAME = "mobilenetv3s-v1"

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / RUN_NAME
INT8_MODEL = MODEL_DIR / "model_int8_balanced.tflite"

TEST_DIR = ROOT / "data" / "test"

IMAGE_SIZE = 224


def load_interpreter():
    interpreter = tf.lite.Interpreter(
        model_path=str(INT8_MODEL)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    return interpreter, input_details, output_details


def load_image(image_path):
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

    image = tf.cast(image, tf.float32)

    image = tf.expand_dims(image, axis=0)

    return image.numpy()


def get_classes():
    classes = sorted(
        [
            path.name
            for path in TEST_DIR.iterdir()
            if path.is_dir()
        ]
    )

    return classes


def main():
    print(f"Loading INT8 model: {INT8_MODEL}")

    interpreter, input_details, output_details = load_interpreter()

    print("INT8 model loaded successfully.")

    print()
    print("Input:")
    print(input_details)

    print()
    print("Output:")
    print(output_details)

    classes = get_classes()

    print()
    print(f"Test classes: {len(classes)}")

    all_predictions = []
    all_labels = []

    total_images = 0

    for label_index, class_name in enumerate(classes):
        class_dir = TEST_DIR / class_name

        image_paths = sorted(
            [
                path
                for path in class_dir.iterdir()
                if path.is_file()
                and path.suffix.lower()
                in {".jpg", ".jpeg", ".png"}
            ]
        )

        print(
            f"[{label_index:02d}] "
            f"{class_name}: "
            f"{len(image_paths)} images"
        )

        for image_path in image_paths:
            image = load_image(image_path)

            interpreter.set_tensor(
                input_details["index"],
                image,
            )

            interpreter.invoke()

            prediction = interpreter.get_tensor(
                output_details["index"]
            )

            predicted_class = int(
                np.argmax(prediction[0])
            )

            all_predictions.append(predicted_class)
            all_labels.append(label_index)

            total_images += 1

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    errors = int(
        np.sum(all_labels != all_predictions)
    )

    print()
    print("=" * 70)
    print("REAL-DATA INT8 TEST EVALUATION")
    print("=" * 70)

    print(f"Total test images : {total_images}")
    print(f"Accuracy          : {accuracy:.6f}")
    print(f"Macro precision   : {macro_precision:.6f}")
    print(f"Macro recall      : {macro_recall:.6f}")
    print(f"Macro F1          : {macro_f1:.6f}")
    print(f"Errors            : {errors}")

    print()
    print("Classification Report")
    print("=" * 70)

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    print(report)

    print()
    print("Confusion Matrix")
    print("=" * 70)

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(classes))),
    )

    print(cm)


if __name__ == "__main__":
    main()