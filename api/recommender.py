import json
import os
import unicodedata
import re
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BookRecommender:
    # instanciar todo
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.books = []
        self.isbn_to_index = {}
        self.tfidf_matrix = None

        self.vectorizer = TfidfVectorizer(
            stop_words=self._spanish_stopwords(),
            ngram_range=(1, 2),
            min_df=2
        )

        self._load_data()
        self._build_matrix()

    # -------------------------
    # Stopwords simples
    # -------------------------
    def _spanish_stopwords(self):
        return [
            "de", "la", "que", "el", "en", "y", "a", "los", "del",
            "se", "las", "por", "un", "para", "con", "no", "una", "su"
        ]

    # -------------------------
    # Normalización
    # -------------------------
    def _normalize(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        return text

    # -------------------------
    # Carga
    # -------------------------
    def _load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Books file not found: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-16") as f:
            self.books = json.load(f)

        self.isbn_to_index = {
            book["ISBN"]: i for i, book in enumerate(self.books)
        }

    # -------------------------
    # TF-IDF
    # -------------------------
    def _build_matrix(self):
        texts = []

        for book in self.books:
            text = " ".join([
                self._normalize(book.get("TITLE", "")),
                self._normalize(book.get("SHORT_DESCRIPTION", "")),
                self._normalize(book.get("BIOGRAPHY", ""))
            ])
            texts.append(text)

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    # -------------------------
    # Explicación
    # -------------------------
    def _build_explanation(self, idx_a, idx_b, top_n=3):

        vec_a = self.tfidf_matrix[idx_a]
        vec_b = self.tfidf_matrix[idx_b]

        # intersección de términos importantes
        common = vec_a.multiply(vec_b)

        if common.nnz == 0:
            return "Similares en temática general"

        # obtener términos
        feature_names = self.vectorizer.get_feature_names_out()

        indices = common.indices
        scores = common.data

        # ordenar por importancia
        sorted_terms = sorted(
            zip(indices, scores),
            key=lambda x: x[1],
            reverse=True
        )

        keywords = [feature_names[i] for i, _ in sorted_terms[:top_n]]

        if not keywords:
            return "Similares en contenido"

        return f"Comparten temas como: {', '.join(keywords)}"

    # -------------------------
    # Recomendación
    # -------------------------
    def recommend(self, isbn: str, top_k: int = 5) -> List[Dict]:

        if isbn not in self.isbn_to_index:
            return []

        idx = self.isbn_to_index[isbn]

        similarities = cosine_similarity(
            self.tfidf_matrix[idx],
            self.tfidf_matrix
        ).flatten()

        similar_indices = similarities.argsort()[::-1]

        results = []

        for i in similar_indices[1: top_k + 1]:

            explanation = self._build_explanation(idx, i)

            book = self.books[i]

            results.append({
                "ISBN": book.get("ISBN"),
                "TITLE": book.get("TITLE"),
                "score": float(similarities[i]),
                "reason": explanation
            })

        return results