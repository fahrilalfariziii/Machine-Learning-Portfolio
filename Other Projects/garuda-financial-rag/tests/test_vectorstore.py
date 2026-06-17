from langchain_core.documents import Document

from src.vectorstore import _split_documents


def test_split_documents_creates_chunks_with_metadata():
    docs = [
        Document(
            page_content="A" * 2000,
            metadata={"source": "test.pdf", "page": 540},
        )
    ]
    chunks = _split_documents(docs, chunk_size=500, overlap=100)

    assert len(chunks) >= 3
    assert chunks[0].metadata["page"] == 540
    assert chunks[0].metadata["chunk_index"] == 1
    assert chunks[1].metadata["chunk_index"] == 2


def test_split_documents_preserves_page_metadata():
    docs = [
        Document(
            page_content="laporan keuangan garuda " * 50,
            metadata={"source": "test.pdf", "page": 683},
        )
    ]
    chunks = _split_documents(docs, chunk_size=300, overlap=50)
    assert all(chunk.metadata["page"] == 683 for chunk in chunks)


def test_split_documents_skips_empty_content():
    docs = [Document(page_content="   ", metadata={"page": 1})]
    chunks = _split_documents(docs)
    assert chunks == []
