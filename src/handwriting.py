import cv2
import torch

from PIL import Image

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)


class HandwritingOCR:

    def __init__(self):

        print("Loading TrOCR...")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"TrOCR device: {self.device}"
        )

        self.processor = (
            TrOCRProcessor.from_pretrained(
                "microsoft/trocr-base-handwritten"
            )
        )

        self.model = (
            VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-handwritten"
            ).to(self.device)
        )

        self.model.eval()

        print("TrOCR loaded.")

    def recognize(self, image):

        if image is None:
            return ""

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(rgb)

        pixel_values = self.processor(
            images=pil_image,
            return_tensors="pt"
        ).pixel_values

        pixel_values = pixel_values.to(
            self.device
        )

        with torch.no_grad():

            generated_ids = self.model.generate(
                pixel_values
            )

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return text.strip()