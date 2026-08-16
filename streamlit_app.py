import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY from your existing .env file locally

from src.detector import DocumentDetector
from src.ocr import PrintedOCR
from src.handwriting import HandwritingOCR
from src.pipeline import process_document
from src.indexer import DocumentIndex
from src.chatbot import answer_question


st.set_page_config(page_title="Document Assistant", page_icon="📄", layout="wide")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================
# Load models ONCE and share across all visitors (read-only,
# safe to share -- unlike the document index below).
# ============================================================

@st.cache_resource(show_spinner="Loading models (first run only, ~1-2 min)...")
def load_models():
    detector = DocumentDetector(model_path="models/best.pt", confidence=0.30)
    printed_ocr = PrintedOCR()
    handwriting_ocr = HandwritingOCR()
    return detector, printed_ocr, handwriting_ocr


detector, printed_ocr, handwriting_ocr = load_models()


# ============================================================
# Per-session state -- each visitor gets their own document,
# index, and chat history. Nothing here is shared between users.
# ============================================================

if "index" not in st.session_state:
    st.session_state.index = DocumentIndex()
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.title("📄 Intelligent Document Assistant")
st.caption("Detection · OCR · Handwriting · Q&A")

col1, col2 = st.columns([1.1, 0.9])

# ------------------------------------------------------------
# LEFT: upload + results
# ------------------------------------------------------------
with col1:
    uploaded_file = st.file_uploader(
        "Upload a document image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    )

    if uploaded_file is not None:
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(save_path, caption=uploaded_file.name, use_container_width=True)

        if st.button("Process document", type="primary"):
            with st.spinner("Running detection + OCR..."):
                result, _ = process_document(
                    save_path, detector, printed_ocr, handwriting_ocr
                )
                indexed_count = st.session_state.index.build(result)
                result["indexed_chunks"] = indexed_count
                st.session_state.result = result
                st.session_state.chat_history = []

            st.success(f"Done — {indexed_count} chunk(s) indexed")

    if st.session_state.result:
        result = st.session_state.result

        st.subheader("Detected objects")
        if result["objects"]:
            st.write(
                ", ".join(
                    f"{o['class']} ({o['confidence']:.0%})"
                    for o in result["objects"]
                )
            )
        else:
            st.write("No objects detected.")

        st.subheader(f"Printed text ({len(result['printed_text'])})")
        st.code(
            "\n".join(item["text"] for item in result["printed_text"])
            or "No printed text detected."
        )

        st.subheader("Handwritten text")
        st.code(result["handwritten_text"] or "No handwritten text detected.")

# ------------------------------------------------------------
# RIGHT: chat
# ------------------------------------------------------------
with col2:
    st.subheader("Ask the document")

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)

    question = st.chat_input("Ask a question about the document...")

    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = answer_question(question, index=st.session_state.index)
                    answer = response["answer"]
                except RuntimeError as exc:
                    answer = f"⚠️ {exc}"
            st.write(answer)

        st.session_state.chat_history.append(("assistant", answer))