from langchain_core.documents import Document

from src.llm_chain import (
    _contains_compare_intent,
    _expand_query,
    _filter_documents_by_intent,
    _focused_queries_for_compare,
    _has_minimum_grounding,
    _keyword_candidates,
    _merge_retrieval_results,
    _normalize_text,
    _relevance_score,
)


def test_normalize_text():
    assert _normalize_text("Aset Tidak Lancar (2025)") == "aset tidak lancar 2025"


def test_contains_compare_intent():
    assert _contains_compare_intent("bandingkan pendapatan 2025 dan 2024") is True
    assert _contains_compare_intent("berapa aset 2025") is False


def test_expand_query_includes_synonyms():
    queries = _expand_query("berapa total liabilitas 2025")
    joined = " ".join(queries).lower()
    assert "liabilitas" in joined or "liability" in joined
    assert len(queries) >= 2


def test_keyword_candidates_enriches_financial_terms():
    keywords = _keyword_candidates("berapa pendapatan usaha 2025")
    joined = " ".join(keywords)
    assert "pendapatan" in joined
    assert "revenue" in joined or "income" in joined


def test_focused_queries_for_compare_revenue():
    queries = _focused_queries_for_compare("bandingkan pendapatan 2025 dan 2024")
    assert "pendapatan usaha tahun 2025" in queries
    assert "operating revenues 2024" in queries


def test_merge_retrieval_results_deduplicates():
    doc_a = Document(page_content="aset tidak lancar 2025", metadata={"page": 683, "chunk_index": 1})
    doc_b = Document(page_content="aset tidak lancar 2025", metadata={"page": 683, "chunk_index": 1})
    doc_c = Document(page_content="pendapatan usaha 2025", metadata={"page": 540, "chunk_index": 2})

    merged = _merge_retrieval_results([doc_a, doc_b], [doc_c], limit=10)
    assert len(merged) == 2


def test_filter_documents_by_intent_pendapatan():
    docs = [
        Document(page_content="arus kas operasi 2025", metadata={"page": 544}),
        Document(page_content="pendapatan usaha operating revenues 2025", metadata={"page": 540}),
    ]
    filtered = _filter_documents_by_intent(docs, "berapa pendapatan 2025")
    assert len(filtered) == 1
    assert filtered[0].metadata["page"] == 540


def test_relevance_score_positive_for_matching_docs():
    docs = [
        Document(
            page_content="Total aset tidak lancar 2025 Rp 4.743.900.635",
            metadata={"page": 683},
        )
    ]
    score = _relevance_score(docs, "berapa aset tidak lancar 2025")
    assert score > 0


def test_has_minimum_grounding_accepts_verified_answer():
    context = "Total aset tidak lancar 2025 Rp 4.743.900.635 halaman 683"
    answer = "Total aset tidak lancar 2025 adalah Rp 4.743.900.635. Sumber Data: halaman 683."
    assert _has_minimum_grounding(answer, context, "berapa aset tidak lancar 2025") is True


def test_has_minimum_grounding_rejects_hallucinated_numbers():
    context = "Total aset tidak lancar 2025 Rp 4.743.900.635"
    answer = "Total aset tidak lancar 2025 adalah Rp 9.999.999.999."
    assert _has_minimum_grounding(answer, context, "berapa aset tidak lancar 2025") is False


def test_has_minimum_grounding_blocks_off_topic_cash_flow_for_revenue():
    context = "arus kas dari aktivitas pendanaan 2025 4.033.751"
    answer = "Pendapatan 2025 naik, arus kas operasi meningkat."
    assert _has_minimum_grounding(answer, context, "bandingkan pendapatan 2025 dan 2024") is False
