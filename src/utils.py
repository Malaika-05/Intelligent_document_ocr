import cv2
import json
import os


def draw_detections(
    image,
    detections,
    output_path
):

    annotated = image.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        class_name = detection["class"]

        confidence = detection["confidence"]

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    cv2.imwrite(
        output_path,
        annotated
    )


def draw_ocr_text(
    image,
    ocr_results,
    output_path
):

    annotated = image.copy()

    for item in ocr_results:

        x1, y1, x2, y2 = item["bbox"]

        text = item["text"]

        confidence = item["confidence"]

        label = f"{text} ({confidence:.2f})"

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            annotated,
            label[:60],
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    cv2.imwrite(
        output_path,
        annotated
    )


def save_json(
    data,
    output_path
):

    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


def save_text(
    data,
    output_path
):

    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        for item in data:

            file.write(
                item["text"] + "\n"
            )