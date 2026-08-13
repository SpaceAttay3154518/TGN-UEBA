"""Incident-vector retrieval with an optional FAISS backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class IncidentIndex:
    def __init__(self, backend: str = "auto") -> None:
        self.backend_requested = backend
        self.backend = "sklearn"
        self.index = None
        self.vectors: np.ndarray | None = None
        self.metadata: list[dict] = []

    def fit(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        vectors = normalize(vectors)
        if len(vectors) != len(metadata):
            raise ValueError("Each incident vector requires one metadata record")
        self.metadata = list(metadata)
        use_faiss = self.backend_requested in {"auto", "faiss"}
        if use_faiss:
            try:
                import faiss  # type: ignore

                index = faiss.IndexFlatIP(vectors.shape[1])
                index.add(vectors)
                self.index = index
                self.backend = "faiss"
                self.vectors = vectors
                return
            except ImportError:
                if self.backend_requested == "faiss":
                    raise RuntimeError("FAISS requested but faiss-cpu is not installed")
        model = NearestNeighbors(metric="cosine", algorithm="brute")
        model.fit(vectors)
        self.index = model
        self.backend = "sklearn"
        self.vectors = vectors

    def search(self, query: np.ndarray, k: int = 5) -> list[dict]:
        if self.index is None or self.vectors is None:
            raise RuntimeError("Incident index has not been fitted")
        query = normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))
        k = min(int(k), len(self.metadata))
        if self.backend == "faiss":
            similarities, indices = self.index.search(query, k)
            pairs = zip(indices[0], similarities[0])
        else:
            distances, indices = self.index.kneighbors(query, n_neighbors=k)
            pairs = zip(indices[0], 1.0 - distances[0])
        return [
            {**self.metadata[int(index)], "cosine_similarity": float(similarity)}
            for index, similarity in pairs
        ]

    def save(self, directory: Path) -> None:
        if self.vectors is None:
            raise RuntimeError("Nothing to save")
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "incident_vectors.npy", self.vectors)
        (directory / "incident_metadata.json").write_text(json.dumps(self.metadata, indent=2))
        (directory / "backend.json").write_text(json.dumps({"backend": self.backend}, indent=2))

    @classmethod
    def load(cls, directory: Path, backend: str = "auto") -> "IncidentIndex":
        instance = cls(backend)
        vectors = np.load(directory / "incident_vectors.npy")
        metadata = json.loads((directory / "incident_metadata.json").read_text())
        instance.fit(vectors, metadata)
        return instance

