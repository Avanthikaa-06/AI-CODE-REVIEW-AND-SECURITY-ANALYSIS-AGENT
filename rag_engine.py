
import os
import re
import pickle

import numpy as np
def _load_html(path: str) -> str:
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    html = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf is not installed. Run: pip install pypdf")

    text_parts = []
    reader = PdfReader(path)
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def _load_docx(path: str) -> str:
    try:
        import docx
    except ImportError:
        raise RuntimeError("python-docx is not installed. Run: pip install python-docx")

    document = docx.Document(path)
    return "\n".join(p.text for p in document.paragraphs)


def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


SUPPORTED_LOADERS = {
    ".pdf": _load_pdf,
    ".txt": _load_txt,
    ".docx": _load_docx,
    ".html": _load_html,
    ".htm": _load_html,
}


def load_documents(knowledge_base_dir: str = "knowledge_base") -> list:
    
    documents = []

    if not os.path.isdir(knowledge_base_dir):
        return documents

    for fname in sorted(os.listdir(knowledge_base_dir)):
        ext = os.path.splitext(fname)[1].lower()
        loader = SUPPORTED_LOADERS.get(ext)
        if loader is None:
            continue

        full_path = os.path.join(knowledge_base_dir, fname)
        try:
            text = loader(full_path)
            if text and text.strip():
                documents.append({"source": fname, "text": text})
        except Exception as e:
            print(f"[rag_engine] Skipped '{fname}': {e}")
            continue

    return documents

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
  
    if not text or not text.strip():
        return []

    text = text.strip()
    if chunk_size <= 0:
        return [text]

    step = max(chunk_size - overlap, 1)  # guard against overlap >= chunk_size

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start += step

    return chunks


def chunk_documents(documents: list) -> list:
 
    all_chunks = []

    for doc in documents:
        source = doc.get("source", "unknown")
        text = doc.get("text", "")
        pieces = chunk_text(text)

        for idx, piece in enumerate(pieces):
            all_chunks.append({
                "source": source,
                "chunk_id": idx,
                "text": piece,
            })

    return all_chunks

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None  # lazy-loaded singleton


def get_model():
    
    global _model

    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is not installed. Run: pip install sentence-transformers"
        )

    try:
        _model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Could not load embedding model '{MODEL_NAME}'. "
            f"Check your internet connection (first run downloads the model). Details: {e}"
        )

    return _model


def embed_texts(texts: list):
    
    if not texts:
        return []

    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings


def embed_query(query: str):
    
    if not query or not query.strip():
        return None

    model = get_model()
    return model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]

class VectorStore:
    

    def __init__(self, dimension: int = None):
        self.dimension = dimension
        self.index = None
        self.metadata = []  # list of {"source": ..., "chunk_id": ..., "text": ...}

    def build(self, embeddings, metadata: list):
        
        try:
            import faiss
        except ImportError:
            raise RuntimeError("faiss-cpu is not installed. Run: pip install faiss-cpu")

        if embeddings is None or len(embeddings) == 0:
            self.index = None
            self.metadata = []
            return

        embeddings = np.asarray(embeddings, dtype="float32")
        self.dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(self.dimension)
        index.add(embeddings)

        self.index = index
        self.metadata = metadata

    def search(self, query_embedding, top_k: int = 3) -> list:
      
        if self.index is None or query_embedding is None:
            return []

        query_vector = np.asarray([query_embedding], dtype="float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            record = dict(self.metadata[idx])
            record["score"] = float(dist)
            results.append(record)

        return results

    def is_ready(self) -> bool:
        return self.index is not None and len(self.metadata) > 0

    def save(self, directory: str):
        try:
            import faiss
        except ImportError:
            raise RuntimeError("faiss-cpu is not installed. Run: pip install faiss-cpu")

        os.makedirs(directory, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, directory: str) -> bool:
        index_path = os.path.join(directory, "index.faiss")
        meta_path = os.path.join(directory, "metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return False

        try:
            import faiss
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        except Exception:
            return False

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".rag_cache")


class RAGPipeline:
    
    def __init__(self, knowledge_base_dir: str = "knowledge_base"):
        self.knowledge_base_dir = knowledge_base_dir
        self.store = VectorStore()
        self.num_documents = 0
        self.num_chunks = 0
        self.error = None

    def build_index(self, use_cache: bool = True) -> dict:
        
        self.error = None

        if use_cache and self.store.load(CACHE_DIR):
            self.num_chunks = len(self.store.metadata)
            self.num_documents = len({m["source"] for m in self.store.metadata})
            return self._status(cached=True)

        try:
            documents = load_documents(self.knowledge_base_dir)
            self.num_documents = len(documents)

            if not documents:
                self.error = (
                    f"No documents found in '{self.knowledge_base_dir}/'. "
                    "Add .pdf, .txt, or .docx files and rebuild the index."
                )
                return self._status(cached=False)

            chunks = chunk_documents(documents)
            self.num_chunks = len(chunks)

            if not chunks:
                self.error = "Documents were found but no text could be extracted."
                return self._status(cached=False)

            texts = [c["text"] for c in chunks]
            embeddings = embed_texts(texts)

            self.store.build(embeddings, chunks)

            try:
                self.store.save(CACHE_DIR)
            except Exception:
                pass  # caching is a convenience, not a requirement

            return self._status(cached=False)

        except Exception as e:
            self.error = str(e)
            return self._status(cached=False)

    def _status(self, cached: bool) -> dict:
        return {
            "ready": self.store.is_ready(),
            "num_documents": self.num_documents,
            "num_chunks": self.num_chunks,
            "cached": cached,
            "error": self.error,
        }

    def query(self, question: str, top_k: int = 3) -> dict:
        
        if not question or not question.strip():
            return {"success": False, "error": "Please enter a question.", "results": []}

        if not self.store.is_ready():
            return {
                "success": False,
                "error": "The knowledge base index has not been built yet.",
                "results": [],
            }

        try:
            query_vector = embed_query(question)
            results = self.store.search(query_vector, top_k=top_k)
            return {"success": True, "error": None, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e), "results": []}
