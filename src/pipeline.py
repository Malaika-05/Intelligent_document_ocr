"""
Wraps the existing detection + OCR + handwriting modules into a single
callable so the Flask route can process one uploaded document at a time,
the same way main.py processes one file per loop iteration.

Nothing in detector.py, ocr.py, handwriting.py, or preprocess.py needs to
change -- this file just orchestrates them.
"""

from src.preprocess import load_image, preprocess_for_ocr


def process_document(image_path, detector, printed_ocr, handwriting_ocr):
    """
    Run the full pipeline on a single image and return a result dict,
    mirroring the `result` structure built in main.py.
    """

    image = load_image(image_path)

    # ------------------------------------------------------------
    # YOLO object detection
    # ------------------------------------------------------------
    detections = detector.detect(image)

    # ------------------------------------------------------------
    # Preprocess for OCR
    # ------------------------------------------------------------
    processed = preprocess_for_ocr(image)

    # ------------------------------------------------------------
    # Printed OCR
    # ------------------------------------------------------------
    printed_results = printed_ocr.extract_text(processed)

    # ------------------------------------------------------------
    # Handwriting OCR
    # ------------------------------------------------------------
    handwriting_text = handwriting_ocr.recognize(processed)

    result = {
        "objects": detections,
        "printed_text": printed_results,
        "handwritten_text": handwriting_text,
    }

    return result, image
