"""
Retrieval-augmented chatbot over the currently indexed document.

Retrieval: TF-IDF cosine similarity (src/indexer.py)
Generation: Groq LLaMA (llama-3.3-70b-versatile), same as your other
projects (Nova, PDF Chat Agent).

Requires GROQ_API_KEY to be set as an environment variable.
"""

import os

from groq import Groq

from src.indexer import document_index

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")

        # Fall back to Streamlit secrets when running on Streamlit Cloud.
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                api_key = None

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it (Flask) or add it to "
                "Streamlit secrets before starting the app."
            )
        _client = Groq(api_key=api_key)
    return _client


SYSTEM_PROMPT = (
    "You are a document assistant. Answer the user's question using ONLY "
    "the provided document excerpts. If the excerpts don't contain the "
    "answer, say you couldn't find it in the document -- do not guess."
)


MAX_FALLBACK_CHUNKS = 40  # cap so a large document doesn't blow the context window


def answer_question(question, index=None, top_k=4):
    """
    index: a DocumentIndex instance to query. Defaults to the shared
    global `document_index` (used by the Flask app, single doc overall).
    Pass a per-session DocumentIndex (e.g. from st.session_state) when
    calling this from Streamlit, so each visitor gets their own document.
    """
    idx = index if index is not None else document_index

    # No document indexed at all -> nothing we can do.
    if idx.chunk_count() == 0:
        return {
            "answer": (
                "I don't have any indexed content to search yet. "
                "Upload a document first."
            ),
            "sources": [],
        }

    retrieved = idx.query(question, top_k=top_k)

    used_fallback = False

    if not retrieved:
        # The document IS indexed, but TF-IDF found no word overlap with
        # this query (common for broad questions like "what is this
        # about?"). Fall back to sending the whole (small) document
        # instead of wrongly claiming nothing is indexed.
        all_chunks = idx.all_chunks()[:MAX_FALLBACK_CHUNKS]
        retrieved = [{"text": chunk, "score": 0.0} for chunk in all_chunks]
        used_fallback = True

    context = "\n\n".join(
        f"[Excerpt {i + 1}] {chunk['text']}"
        for i, chunk in enumerate(retrieved)
    )

    user_prompt = (
        f"Document excerpts:\n{context}\n\n"
        f"Question: {question}"
    )

    client = _get_client()

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    answer_text = completion.choices[0].message.content.strip()

    return {
        "answer": answer_text,
        "sources": retrieved,
        "used_fallback": used_fallback,
    }