from ultralytics import YOLO


class DocumentDetector:

    def __init__(
        self,
        model_path="models/best.pt",
        confidence=0.30
    ):

        print("Loading YOLO model...")

        self.model = YOLO(model_path)
        self.confidence = confidence

        print("YOLO model loaded.")
        print("Classes:", self.model.names)

    def detect(self, image):

        results = self.model.predict(
            source=image,
            conf=self.confidence,
            verbose=False
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:

            cls_id = int(box.cls[0])

            class_name = self.model.names[cls_id]

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            detections.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            })

        return detections