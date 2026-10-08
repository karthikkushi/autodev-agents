"""
Hybrid memory for agents:
- Vector memory (numpy cosine similarity + Gemini gemini-embedding-001) for semantic search
- Episodic memory (JSON) for tracking decisions and outcomes
No external vector DB needed — pure Python + numpy. Embeddings are the only
non-chat API call in this project; if GOOGLE_API_KEY isn't set, memory just
degrades to episodic-only (search() returns "") — never blocks the pipeline.
"""
import json
import os
import pickle
from datetime import datetime
import numpy as np
from rich.console import Console

console = Console()
# text-embedding-004 now returns 404 NOT_FOUND on the Gemini API (checked 2026-09-29).
EMBED_MODEL = "models/gemini-embedding-001"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class AgentMemory:
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.memory_dir = f"{project_path}/.memory"
        os.makedirs(self.memory_dir, exist_ok=True)

        self.episodic_path = f"{project_path}/docs/memory_log.json"
        self.vector_path = f"{self.memory_dir}/vectors.pkl"
        self.episodes = self._load_episodes()
        self.vectors: list[dict] = self._load_vectors()

        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            if not os.environ.get("GOOGLE_API_KEY"):
                raise RuntimeError("GOOGLE_API_KEY not set")
            self.embedder = GoogleGenerativeAIEmbeddings(
                model=EMBED_MODEL, google_api_key=os.environ.get("GOOGLE_API_KEY")
            )
            self._embed_available = True
        except Exception:
            self._embed_available = False

    def _load_episodes(self) -> list:
        if os.path.exists(self.episodic_path):
            try:
                with open(self.episodic_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_episodes(self):
        try:
            with open(self.episodic_path, "w") as f:
                json.dump(self.episodes, f, indent=2)
        except Exception:
            pass

    def _load_vectors(self) -> list:
        if os.path.exists(self.vector_path):
            try:
                with open(self.vector_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return []

    def _save_vectors(self):
        try:
            with open(self.vector_path, "wb") as f:
                pickle.dump(self.vectors, f)
        except Exception:
            pass

    def store(self, agent: str, content: str, metadata: dict = None):
        if self._embed_available:
            try:
                embedding = self.embedder.embed_query(content[:2000])
                self.vectors.append({
                    "embedding": np.array(embedding),
                    "content": content[:2000],
                    "agent": agent,
                    "time": datetime.now().isoformat(),
                    **(metadata or {})
                })
                self._save_vectors()
            except Exception as e:
                console.print(f"[dim yellow]Memory embed warning: {e}[/dim yellow]")

        self.episodes.append({
            "agent": agent,
            "time": datetime.now().isoformat(),
            "summary": content[:300],
            **(metadata or {})
        })
        self._save_episodes()

    def search(self, query: str, n: int = 3) -> str:
        if not self._embed_available or not self.vectors:
            return ""
        try:
            query_emb = np.array(self.embedder.embed_query(query))
            scored = [
                (_cosine_similarity(query_emb, v["embedding"]), v["content"])
                for v in self.vectors
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            top = [content for score, content in scored[:n] if score > 0.5]
            return "\n---\n".join(top) if top else ""
        except Exception:
            return ""

    def get_failures(self) -> list:
        return [e for e in self.episodes if e.get("outcome") == "failed"]

    def record_outcome(self, agent: str, outcome: str, detail: str = ""):
        self.episodes.append({
            "agent": agent,
            "time": datetime.now().isoformat(),
            "outcome": outcome,
            "detail": detail
        })
        self._save_episodes()
