import cv2


def load_image(image_path):

    image = cv2.imread(image_path)

    if image is None:

        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    return image


def preprocess_for_ocr(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        10,
        7,
        21
    )

    processed = cv2.cvtColor(
        denoised,
        cv2.COLOR_GRAY2BGR
    )

    return processed