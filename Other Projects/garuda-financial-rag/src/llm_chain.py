import os
import sys
import re
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from src.vectorstore import get_retriever, get_vector_store

load_dotenv()

SYSTEM_PROMPT = (
    "Anda adalah seorang pakar audit keuangan senior yang sangat teliti dan adaptif.\n"
    "Jawab pertanyaan user BERDASARKAN KONTEKS DOKUMEN YANG DIBERIKAN.\n\n"
    "PETUNJUK PENTING:\n"
    "1. HANYA gunakan angka, teks, tabel, dan informasi yang ada di dalam konteks dokumen.\n"
    "2. Jangan menebak, jangan menambah informasi dari luar dokumen.\n"
    "3. Jika pertanyaan meminta angka, cari baris/tabel yang memuat nama akun + nilai tahun yang diminta.\n"
    "4. Untuk perbandingan 2024 vs 2025: tampilkan kedua angka dan arahnya (naik/turun).\n"
    "5. Jika informasi tidak ada di konteks, jawab TEGAS: 'Informasi ini tidak tersedia di halaman yang dianalisis.'\n"
    "6. BOLEH mereferensikan sumber halaman jika ada di konteks.\n"
    "7. Format angka dengan unit yang sesuai (USD, persentase, dll).\n"
    "8. Setiap jawaban WAJIB menyertakan bagian 'Sumber Data' berisi halaman + kutipan singkat dari konteks.\n"
    "9. Jika angka pada jawaban tidak bisa ditemukan secara eksplisit di konteks, jangan tampilkan angka tersebut.\n\n"
    "10. Jika data untuk akun yang ditanya tidak ada, cukup katakan tidak tersedia. JANGAN membahas akun lain yang tidak diminta.\n\n"
    "KONTEKS DOKUMEN:\n"
    "{context}"
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])

FINANCIAL_SYNONYMS = {
    "aset": ["asset", "assets", "harta", "kekayaan"],
    "liabilitas": ["liability", "liabilities", "utang", "hutang", "kewajiban"],
    "ekuitas": ["equity", "modal"],
    "pendapatan": ["revenue", "revenues", "income", "penghasilan"],
    "beban": ["expense", "expenses", "biaya", "cost"],
    "laba": ["profit", "income for the year", "net income"],
    "rugi": ["loss"],
    "arus kas": ["cash flow", "kas"],
    "aset lancar": ["current assets"],
    "aset tidak lancar": ["non-current assets", "non current assets"],
}

def _expand_query(user_question: str) -> list:
    """Expand user query ke multiple queries untuk retrieval lebih luas."""
    queries = [user_question]  # Original query
    
    question_lower = user_question.lower()
    
    financial_keywords = {
        'aset': ['asset', 'kekayaan', 'harta', 'aset total'],
        'liabilitas': ['liability', 'hutang', 'kewajiban', 'utang jangka'],
        'ekuitas': ['equity', 'modal', 'kapital', 'kesejahteraan pemilik'],
        'pendapatan': ['revenue', 'income', 'penerimaan', 'hasil operasional', 'hasil penjualan'],
        'biaya': ['expense', 'beban', 'pengeluaran', 'cost', 'biaya operasional'],
        'neraca': ['balance sheet', 'laporan posisi keuangan', 'statement of financial position'],
        'laba rugi': ['income statement', 'laporan laba rugi', 'laporan hasil operasional'],
        'arus kas': ['cash flow', 'laporan arus kas', 'cash position'],
        'total': ['grand total', 'jumlah', 'keseluruhan', 'akumulasi']
    }
    
    for key, variations in financial_keywords.items():
        if key in question_lower:
            for var in variations:
                if var.lower() not in question_lower:
                    expanded = user_question.replace(key, var)
                    if expanded not in queries:
                        queries.append(expanded)
            break
    
    if len(user_question.split()) <= 3:
        queries.append("laporan keuangan konsolidasi " + user_question)
        queries.append("data finansial " + user_question)
    
    # Ekspansi generik 
    for key, synonyms in FINANCIAL_SYNONYMS.items():
        if key in question_lower:
            for synonym in synonyms:
                rewritten = re.sub(key, synonym, user_question, flags=re.IGNORECASE)
                if rewritten != user_question:
                    queries.append(rewritten)
                queries.append(f"{synonym} laporan keuangan konsolidasian")
                queries.append(f"{synonym} catatan atas laporan keuangan")

    deduped = []
    for q in queries:
        if q not in deduped:
            deduped.append(q)
    return deduped[:8]


def _retrieve_with_fallback(retriever, user_question: str, max_attempts: int = 3):
    """Retrieve documents dengan fallback ke expanded queries."""
    queries_to_try = _expand_query(user_question)
    all_docs = []
    seen_chunks = set()
    
    for query in queries_to_try[:max_attempts]:
        try:
            docs = retriever.invoke(query)
            for doc in docs:
                page = doc.metadata.get('page')
                chunk_index = doc.metadata.get('chunk_index', 0)
                dedupe_key = (page, chunk_index)
                if dedupe_key not in seen_chunks:
                    all_docs.append(doc)
                    seen_chunks.add(dedupe_key)
                if len(all_docs) >= 8: 
                    return all_docs
        except Exception as e:
            print(f"[Debug] Query '{query}' failed: {e}")
            continue
    
    return all_docs


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s%./-]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _keyword_candidates(user_question: str):
    stopwords = {
        "yang", "dan", "atau", "dari", "untuk", "dengan", "pada", "tahun", "berapa",
        "apa", "ada", "itu", "ini", "di", "ke", "dalam", "the", "of"
    }
    normalized = _normalize_text(user_question)
    tokens = [t for t in normalized.split() if len(t) > 2 and t not in stopwords]

    # sinonim otomatis untuk kata kunci utama.
    enriched = list(tokens)
    token_string = " ".join(tokens)
    for key, syns in FINANCIAL_SYNONYMS.items():
        if key in token_string:
            for syn in syns:
                enriched.extend(_normalize_text(syn).split())
    return list(dict.fromkeys(enriched))


def _contains_compare_intent(user_question: str) -> bool:
    q = user_question.lower()
    return (
        "banding" in q
        or "dibanding" in q
        or "vs" in q
        or ("2024" in q and "2025" in q)
    )


def _focused_queries_for_compare(user_question: str):
    base = re.sub(r"\b(2024|2025)\b", "", user_question, flags=re.IGNORECASE)
    base = re.sub(r"\b(dan|vs)\b", " ", base, flags=re.IGNORECASE)
    base = re.sub(r"\s+", " ", base).strip()
    queries = [
        f"{base} tahun 2025",
        f"{base} tahun 2024",
        user_question,
    ]
    base_norm = _normalize_text(base)
    if "pendapatan" in base_norm or "revenue" in base_norm:
        queries.extend([
            "pendapatan usaha tahun 2025",
            "pendapatan usaha tahun 2024",
            "operating revenues 2025",
            "operating revenues 2024",
        ])
    return queries


def _lexical_retrieve_from_store(vector_store, user_question: str, limit: int = 8):
    """Fallback lexical retrieval langsung ke semua chunk Chroma."""
    payload = vector_store.get(include=["documents", "metadatas"])
    documents = payload.get("documents", []) or []
    metadatas = payload.get("metadatas", []) or []
    if not documents:
        return []

    normalized_question = _normalize_text(user_question)
    keywords = _keyword_candidates(user_question)
    if not keywords:
        return []

    years_in_question = set(re.findall(r"\b(20\d{2})\b", normalized_question))
    scored = []
    for idx, content in enumerate(documents):
        text = _normalize_text(content)
        if not text:
            continue
        hit_count = sum(1 for kw in keywords if kw in text)
        phrase_bonus = 3 if normalized_question and normalized_question in text else 0
        numeric_bonus = 2 if re.search(r"\b\d[\d.,]*\b", text) else 0
        year_bonus = sum(2 for year in years_in_question if year in text)
        score = hit_count + phrase_bonus + numeric_bonus + year_bonus
        if score > 0:
            metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
            scored.append((score, content, metadata))

    scored.sort(key=lambda item: item[0], reverse=True)
    lexical_docs = []
    for score, content, metadata in scored[:limit]:
        # simpan skor lexical 
        merged_metadata = dict(metadata)
        merged_metadata["lexical_score"] = score
        lexical_docs.append(Document(page_content=content, metadata=merged_metadata))
    return lexical_docs


def _merge_retrieval_results(semantic_docs, lexical_docs, limit: int = 10):
    merged = []
    seen_keys = set()

    for doc in list(semantic_docs) + list(lexical_docs):
        page = doc.metadata.get("page") if isinstance(doc.metadata, dict) else None
        chunk = doc.metadata.get("chunk_index", 0) if isinstance(doc.metadata, dict) else 0
        key = (page, chunk, _normalize_text(doc.page_content)[:120])
        if key in seen_keys:
            continue
        merged.append(doc)
        seen_keys.add(key)
        if len(merged) >= limit:
            break

    return merged


def _relevance_score(documents, user_question: str) -> float:
    if not documents:
        return 0.0
    keywords = _keyword_candidates(user_question)
    if not keywords:
        return 0.0

    top_docs = documents[:5]
    total = 0.0
    for doc in top_docs:
        text = _normalize_text(doc.page_content)
        if not text:
            continue
        matched = sum(1 for kw in keywords if kw in text)
        total += matched / max(len(keywords), 1)
    return total / max(len(top_docs), 1)


def _intent_terms(user_question: str):
    q = _normalize_text(user_question)
    terms = []
    if "pendapatan" in q or "revenue" in q:
        terms.extend(["pendapatan", "revenue", "operating revenues", "pendapatan usaha"])
    if "liabilitas" in q or "liability" in q or "utang" in q or "kewajiban" in q:
        terms.extend(["liabilitas", "liability", "kewajiban", "utang"])
    if "aset" in q or "asset" in q:
        terms.extend(["aset", "asset"])
    if "ekuitas" in q or "equity" in q:
        terms.extend(["ekuitas", "equity"])
    if "arus kas" in q or "cash flow" in q:
        terms.extend(["arus kas", "cash flow"])
    return list(dict.fromkeys(terms))


def _filter_documents_by_intent(documents, user_question: str):
    terms = _intent_terms(user_question)
    if not documents or not terms:
        return documents
    filtered = []
    for doc in documents:
        text = _normalize_text(doc.page_content)
        if any(term in text for term in terms):
            filtered.append(doc)
    return filtered or documents


def _smart_truncate_content(content: str, max_chars: int = 2500, target_tokens: int = 4500) -> str:
    """Truncate content smartly based on token budget."""
    # Rough estimate: 1 token ≈ 4 chars
    estimated_tokens = len(content) // 4
    
    if estimated_tokens > target_tokens:
        # Aggressive truncation
        truncated = content[:max_chars]
    else:
        truncated = content[:min(len(content), max_chars)]
    
    if len(truncated) < len(content):
        truncated += "\n[... teks terpotong untuk efisiensi token ...]"
    
    return truncated


def _simple_keyword_rerank(documents, user_question: str):
    question_tokens = set(_keyword_candidates(user_question))
    if not question_tokens:
        return documents

    scored_docs = []
    for doc in documents:
        text = _normalize_text(doc.page_content or "")
        score = sum(1 for token in question_tokens if token in text)
        if re.search(r"\b(2024|2025)\b", user_question) and re.search(r"\b(2024|2025)\b", text):
            score += 2
        if re.search(r"\b\d[\d.,]*\b", text):
            score += 1
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_docs]


def _compose_context(documents, user_question: str) -> str:
    """Compose context dari retrieved documents dengan smart truncation."""
    if not documents:
        return ""

    ranked_documents = _simple_keyword_rerank(documents, user_question)
    page_summaries = []
    total_chars = 0
    max_total_chars = 12000  # Hard limit 
    
    for doc in ranked_documents:
        if total_chars >= max_total_chars:
            break
            
        metadata = getattr(doc, 'metadata', {})
        if isinstance(metadata, dict):
            page = metadata.get('page')
            source = metadata.get('source')
            chunk_index = metadata.get('chunk_index')
        else:
            page = getattr(metadata, 'page', None)
            source = getattr(metadata, 'source', None)
            chunk_index = getattr(metadata, 'chunk_index', None)

        header = f"[HALAMAN: {page}]" if page else "[HALAMAN: tidak diketahui]"
        if chunk_index:
            header += f" [CHUNK: {chunk_index}]"
        if source:
            header += f" [SOURCE: {source}]"

        content = _smart_truncate_content(doc.page_content, max_chars=2200)
        section = f"{header}\n{content}"
        page_summaries.append(section)
        total_chars += len(section)

    return "\n\n" + "="*60 + "\n\n".join(page_summaries)


def _extract_answer_from_result(result):
    """Extract answer from LLM generation result."""
    if result is None:
        return ""

    answer = ""
    if hasattr(result, 'generations'):
        generations = result.generations
        if generations and len(generations) > 0 and len(generations[0]) > 0:
            first = generations[0][0]
            if hasattr(first, 'text') and first.text:
                answer = first.text
            elif hasattr(first, 'message') and hasattr(first.message, 'content'):
                answer = first.message.content

    return answer.strip()


def _extract_numbers(text: str):
    return re.findall(r"\b\d[\d.,]{1,}\b", text or "")


def _has_minimum_grounding(answer: str, context: str, user_question: str) -> bool:
    answer_norm = _normalize_text(answer)
    context_norm = _normalize_text(context)
    if not answer_norm or not context_norm:
        return False

    answer_numbers = _extract_numbers(answer)
    if answer_numbers:
        matched = sum(1 for num in answer_numbers if num in context)
        if matched / max(len(answer_numbers), 1) < 0.6:
            return False

    if _contains_compare_intent(user_question):
        if ("2024" not in answer and "2025" not in answer) or len(answer_numbers) < 2:
            return False

    q_norm = _normalize_text(user_question)
    a_norm = _normalize_text(answer)
    if ("pendapatan" in q_norm or "revenue" in q_norm):
        if ("arus kas" in a_norm or "cash flow" in a_norm) and ("pendapatan usaha" not in a_norm and "revenue" not in a_norm):
            return False
    if ("arus kas" in q_norm or "cash flow" in q_norm):
        if ("pendapatan" in a_norm or "revenue" in a_norm) and ("arus kas" not in a_norm and "cash flow" not in a_norm):
            return False

    query_tokens = set(_keyword_candidates(user_question))
    if query_tokens:
        overlap = sum(1 for token in query_tokens if token in answer_norm and token in context_norm)
        if overlap == 0:
            return False

    return True


def get_financial_rag_response(user_question: str):
    """Main RAG response pipeline dengan multi-query retrieval."""
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY belum disetel. Atur variabel lingkungan ini sebelum menjalankan aplikasi.")

    retriever = get_retriever()
    vector_store = get_vector_store()
    
    retrieval_query = user_question
    normalized_question = _normalize_text(user_question)
    if _contains_compare_intent(user_question) and ("pendapatan" in normalized_question or "revenue" in normalized_question):
        retrieval_query = "pendapatan usaha operating revenues 2025 2024"

    # Retrieve fallback ke expanded queries
    print(f"[*] Mencari dokumen untuk query: '{user_question}'")
    semantic_docs = _retrieve_with_fallback(retriever, retrieval_query)
    lexical_docs = _lexical_retrieve_from_store(vector_store, retrieval_query)

    # pertanyaan perbandingan
    if _contains_compare_intent(user_question):
        for q in _focused_queries_for_compare(user_question):
            semantic_docs.extend(_retrieve_with_fallback(retriever, q, max_attempts=2))
            lexical_docs.extend(_lexical_retrieve_from_store(vector_store, q, limit=4))

    documents = _merge_retrieval_results(semantic_docs, lexical_docs, limit=10)
    documents = _filter_documents_by_intent(documents, user_question)
    
    if not documents:
        print(f"[-] Tidak ada dokumen yang cocok ditemukan.")
        return "Informasi ini tidak tersedia di halaman yang dianalisis. Coba pertanyaan yang lebih spesifik tentang laporan keuangan Garuda Indonesia 2025."

    relevance = _relevance_score(documents, retrieval_query)
    print(f"[*] Skor relevansi retrieval: {relevance:.2f}")
    if relevance < 0.10:
        return "Informasi ini tidak tersedia di halaman yang dianalisis. Query belum cukup cocok dengan konteks PDF yang terindeks."

    print(f"[+] Ditemukan {len(documents)} chunk relevan dari halaman: {[d.metadata.get('page') for d in documents]}")
    
    context = _compose_context(documents, user_question)
    prompt_messages = PROMPT_TEMPLATE.format_messages(context=context, input=user_question)

    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    result = llm.generate(messages=[prompt_messages])
    answer = _extract_answer_from_result(result)

    if answer and "informasi ini tidak tersedia" in answer.lower() and documents:
        retry_input = (
            f"{user_question}\n\n"
            "Jika konteks memuat nilai atau tabel yang relevan, berikan jawaban berdasarkan data tersebut "
            "dan sebutkan halaman. Gunakan jawaban 'tidak tersedia' hanya jika benar-benar tidak ada."
        )
        retry_messages = PROMPT_TEMPLATE.format_messages(context=context, input=retry_input)
        retry_result = llm.generate(messages=[retry_messages])
        retry_answer = _extract_answer_from_result(retry_result)
        if retry_answer:
            answer = retry_answer

    if answer and not _has_minimum_grounding(answer, context, user_question):
        # verifikasi ketat untuk menekan halusinasi.
        strict_input = (
            f"{user_question}\n\n"
            "Jawab hanya jika setiap angka/fakta ada secara eksplisit di konteks. "
            "Wajib sertakan bagian 'Sumber Data' dengan kutipan singkat."
        )
        strict_messages = PROMPT_TEMPLATE.format_messages(context=context, input=strict_input)
        strict_result = llm.generate(messages=[strict_messages])
        strict_answer = _extract_answer_from_result(strict_result)
        if strict_answer and _has_minimum_grounding(strict_answer, context, user_question):
            answer = strict_answer
        else:
            return (
                "Informasi ini belum dapat diverifikasi secara kuat dari konteks PDF yang terambil. "
                "Silakan perjelas akun/tahun yang ditanya agar sistem mengambil kutipan tabel yang lebih spesifik."
            )
    
    if not answer:
        return "Informasi ini tidak tersedia di halaman yang dianalisis."

    return answer


def ask_financial_ai(user_question):
    """Public interface untuk RAG."""
    try:
        return get_financial_rag_response(user_question)
    except Exception as e:
        print(f"[-] RAG Error: {e}")
        raise RuntimeError(f"Gagal mengeksekusi RAG: {e}") from e


if __name__ == "__main__":
    try:
        # Test queries
        queries = [
            "total aset 2025",
            "berapa ekuitas tahun ini",
            "revenue operasional"
        ]
        for q in queries:
            print(f"\n{'='*60}")
            print(f"Query: {q}")
            print(f"{'='*60}")
            response = get_financial_rag_response(q)
            print(f"Jawaban: {response[:500]}...\n")
    except Exception as e:
        print(f"[-] Error: {e}")
