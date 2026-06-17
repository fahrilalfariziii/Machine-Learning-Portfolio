import pytest

from src.llm_chain import _compose_context, _retrieve_with_fallback


class _FakeRetriever:
    def __init__(self, docs_by_query):
        self.docs_by_query = docs_by_query

    def invoke(self, query):
        return self.docs_by_query.get(query, [])


@pytest.mark.integration
def test_retriever_returns_docs_for_financial_query(chroma_available, groq_available):
    if not chroma_available:
        pytest.skip("chroma_db belum dibuat. Jalankan: python src/vectorstore.py")
    if not groq_available:
        pytest.skip("GROQ_API_KEY tidak tersedia")

    from src.vectorstore import get_retriever

    retriever = get_retriever()
    docs = _retrieve_with_fallback(retriever, "aset tidak lancar 2025")
    assert len(docs) > 0
    assert any("aset" in (d.page_content or "").lower() for d in docs)


@pytest.mark.integration
def test_compose_context_includes_page_metadata(chroma_available):
    if not chroma_available:
        pytest.skip("chroma_db belum dibuat")

    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="pendapatan usaha 2025 3.216.604.484",
            metadata={"page": 540, "chunk_index": 1, "source": "data/garuda_annual_report_2025.pdf"},
        )
    ]
    context = _compose_context(docs, "berapa pendapatan usaha 2025")
    assert "[HALAMAN: 540]" in context
    assert "3.216.604.484" in context
