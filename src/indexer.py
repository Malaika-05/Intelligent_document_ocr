"""
Thread-safe, single-document TF-IDF indexer.

Each time a new document is uploaded, the index is rebuilt from scratch
and replaces whatever was indexed before. All reads and writes go through
a single lock, so a chat request can never see a half-built index (this
is what caused the earlier "Indexed chunks: 0" / race-condition bug when
Flask handled requests on different threads).
"""

import threading
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentIndex:

    def __init__(self, min_chunk_words=6):
        self._lock = threading.Lock()
        self._vectorizer = None
        self._matrix = None
        self._chunks = []
        self.min_chunk_words = min_chunk_words

    # ------------------------------------------------------------
    # Build chunks from the pipeline result
    # ------------------------------------------------------------
    def _build_chunks(self, result):
        chunks = []

        # Each printed OCR region becomes a candidate chunk.
        for item in result.get("printed_text", []):
            text = (item.get("text") or "").strip()
            if text:
                chunks.append(text)

        # Merge short/adjacent OCR fragments into denser passages so
        # TF-IDF has enough context per chunk (raw OCR lines are often
        # only a few words each).
        merged = []
        buffer_words = []

        for text in chunks:
            buffer_words.extend(text.split())
            if len(buffer_words) >= self.min_chunk_words:
                merged.append(" ".join(buffer_words))
                buffer_words = []

        if buffer_words:
            merged.append(" ".join(buffer_words))

        # Handwritten text is its own chunk if present.
        handwritten = (result.get("handwritten_text") or "").strip()
        if handwritten:
            merged.append(handwritten)

        return merged

    # ------------------------------------------------------------
    # Rebuild the index for a newly uploaded document
    # ------------------------------------------------------------
    def build(self, result):
        chunks = self._build_chunks(result)

        with self._lock:
            self._chunks = chunks

            if not chunks:
                self._vectorizer = None
                self._matrix = None
                return 0

            self._vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
            )
            self._matrix = self._vectorizer.fit_transform(chunks)

            return len(chunks)

    # ------------------------------------------------------------
    # Retrieve top-k chunks for a query
    # ------------------------------------------------------------
    def query(self, question, top_k=4):
        with self._lock:
            if not self._chunks or self._vectorizer is None:
                return []

            query_vec = self._vectorizer.transform([question])
            scores = cosine_similarity(query_vec, self._matrix).flatten()

            ranked = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )

            results = []
            for idx in ranked[:top_k]:
                if scores[idx] <= 0:
                    continue
                results.append(
                    {"text": self._chunks[idx], "score": float(scores[idx])}
                )

            return results

    def chunk_count(self):
        with self._lock:
            return len(self._chunks)

    def all_chunks(self):
        """Return every indexed chunk, regardless of query relevance.
        Used as a fallback for broad questions ("what is this document
        about?") where TF-IDF term overlap is 0 for every chunk even
        though the document IS indexed."""
        with self._lock:
            return list(self._chunks)


# Single global index -- one document per session, as chosen. If you later
# need multi-document / multi-user support, key a dict of DocumentIndex
# instances by session id instead of using one shared instance.
document_index = DocumentIndex()