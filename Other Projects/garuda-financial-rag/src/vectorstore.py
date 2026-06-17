import os
import sys
import shutil
from dotenv import load_dotenv
from langchain_core.documents import Document

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.parser import load_targeted_pages

load_dotenv()

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _build_embedding_model(device: str = "cpu"):
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device}
    )


def _split_documents(documents, chunk_size: int = 1400, overlap: int = 250):
    """Split dokumen agar tabel/angka panjang tidak hilang saat retrieval."""
    chunks = []
    for doc in documents:
        text = doc.page_content or ""
        metadata = dict(getattr(doc, "metadata", {}) or {})
        start = 0
        chunk_index = 1
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = dict(metadata)
                chunk_metadata["chunk_index"] = chunk_index
                chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
                chunk_index += 1
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return chunks


def create_vector_store():
    pages = load_targeted_pages()
    if not pages:
        raise RuntimeError("Tidak ada dokumen untuk diindeks. Pastikan parser berhasil mengekstrak halaman PDF.")
    chunks = _split_documents(pages)
    if not chunks:
        raise RuntimeError("Proses chunking gagal. Tidak ada potongan dokumen yang siap diindeks.")

    print("[*] Memuat model embedding lokal...")
    embedding_model = _build_embedding_model(device="cpu")

    if os.path.exists(CHROMA_PATH):
        print("[*] Menghapus database lama...")
        shutil.rmtree(CHROMA_PATH)

    print(f"[*] Mendaftarkan {len(chunks)} chunk laporan keuangan ke ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )
    print("[+] Database vektor berhasil diperbarui dengan struktur chunk + metadata halaman.")
    return vector_store


def get_retriever():
    if not os.path.isdir(CHROMA_PATH):
        raise FileNotFoundError(
            f"Vector store tidak ditemukan di '{CHROMA_PATH}'. Jalankan 'python src/vectorstore.py' untuk membuatnya."
        )

    vector_store = get_vector_store()
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5}
    )


def get_vector_store():
    """Expose vector store untuk kebutuhan retrieval lanjutan (hybrid)."""
    if not os.path.isdir(CHROMA_PATH):
        raise FileNotFoundError(
            f"Vector store tidak ditemukan di '{CHROMA_PATH}'. Jalankan 'python src/vectorstore.py' untuk membuatnya."
        )

    embedding_model = _build_embedding_model()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)


if __name__ == "__main__":
    create_vector_store()