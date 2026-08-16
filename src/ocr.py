import os

# Must be set BEFORE importing Paddle
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

from paddleocr import PaddleOCR


class PrintedOCR:

    def __init__(self):

        print("Loading PaddleOCR...")

        self.ocr = PaddleOCR(
            lang="en",
            use_textline_orientation=True,
            device="cpu"
        )

        print("PaddleOCR loaded.")

    def extract_text(self, image):

        results = self.ocr.predict(image)

        extracted = []

        for result in results:

            rec_texts = result["rec_texts"]
            rec_scores = result["rec_scores"]
            rec_boxes = result["rec_boxes"]

            for text, score, box in zip(
                rec_texts,
                rec_scores,
                rec_boxes
            ):

                text = str(text).strip()

                if not text:
                    continue

                extracted.append({
                    "text": text,
                    "confidence": float(score),
                    "bbox": [
                        int(box[0]),
                        int(box[1]),
                        int(box[2]),
                        int(box[3])
                    ]
                })

        return extracted