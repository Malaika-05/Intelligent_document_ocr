import os
import traceback

from flask import Flask, render_template, request, jsonify

from src.detector import DocumentDetector
from src.ocr import PrintedOCR
from src.handwriting import HandwritingOCR
from src.pipeline import process_document
from src.indexer import document_index
from src.chatbot import answer_question


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/best.pt"
UPLOAD_DIR = "uploads"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)

# ============================================================
# LOAD MODELS ONCE AT STARTUP (not per-request)
# ============================================================

print("\nInitializing models...\n")

detector = DocumentDetector(model_path=MODEL_PATH, confidence=0.30)
printed_ocr = PrintedOCR()
handwriting_ocr = HandwritingOCR()

print("\nModels ready.\n")


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file was sent."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file was selected."}), 400

    if not file.filename.lower().endswith(VALID_EXTENSIONS):
        return jsonify({"error": "Unsupported file type."}), 400

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    try:
        result, _ = process_document(
            save_path, detector, printed_ocr, handwriting_ocr
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {exc}"}), 500

    indexed_count = document_index.build(result)
    print(f"Indexed chunks: {indexed_count}")

    return jsonify(
        {
            "filename": file.filename,
            "objects": result["objects"],
            "printed_text": result["printed_text"],
            "handwritten_text": result["handwritten_text"],
            "indexed_chunks": indexed_count,
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        result = answer_question(question)
    except RuntimeError as exc:
        # e.g. missing GROQ_API_KEY
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Chat failed: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
