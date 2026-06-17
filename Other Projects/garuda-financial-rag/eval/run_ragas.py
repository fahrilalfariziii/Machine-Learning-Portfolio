"""
Evaluasi kualitas RAG dengan Ragas.

Prasyarat:
  pip install -r requirements-dev.txt
  python src/vectorstore.py
  .env berisi GROQ_API_KEY

Jalankan:
  python eval/run_ragas.py
  python eval/run_ragas.py --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def _patch_ragas_langchain_compat():
    """Ragas 0.4.x masih mengimpor VertexAI dari langchain-community yang sudah dipindah."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    from types import ModuleType
    from unittest.mock import MagicMock

    stub = ModuleType(module_name)
    stub.ChatVertexAI = MagicMock(name="ChatVertexAI")
    sys.modules[module_name] = stub


_patch_ragas_langchain_compat()

from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.llm_chain import (
    _filter_documents_by_intent,
    _lexical_retrieve_from_store,
    _merge_retrieval_results,
    _retrieve_with_fallback,
    ask_financial_ai,
    get_retriever,
    get_vector_store,
)
from src.vectorstore import EMBEDDING_MODEL


def load_golden_dataset(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_rag_outputs(questions: list[str]):
    retriever = get_retriever()
    vector_store = get_vector_store()

    answers = []
    contexts = []
    for question in questions:
        semantic_docs = _retrieve_with_fallback(retriever, question)
        lexical_docs = _lexical_retrieve_from_store(vector_store, question)
        documents = _merge_retrieval_results(semantic_docs, lexical_docs, limit=10)
        documents = _filter_documents_by_intent(documents, question)
        context_list = [doc.page_content for doc in documents if doc.page_content]
        contexts.append(context_list)
        answers.append(ask_financial_ai(question))

    return answers, contexts


def extract_ragas_scores(result) -> dict:
    """Ambil skor agregat dari EvaluationResult Ragas 0.4.x."""
    if hasattr(result, "_repr_dict"):
        scores = {}
        for metric, value in result._repr_dict.items():
            if value is None:
                scores[metric] = None
            else:
                scores[metric] = float(value)
        return scores

    if getattr(result, "scores", None):
        aggregated = {}
        keys = result.scores[0].keys()
        for key in keys:
            values = [row[key] for row in result.scores if row.get(key) is not None]
            aggregated[key] = float(sum(values) / len(values)) if values else None
        return aggregated

    return {}


def extract_per_sample_scores(result) -> list[dict]:
    if not getattr(result, "scores", None):
        return []

    serialized = []
    for row in result.scores:
        item = {}
        for key, value in row.items():
            item[key] = None if value is None else float(value)
        serialized.append(item)
    return serialized


def run_ragas_evaluation(dataset_dict: dict):
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError("GROQ_API_KEY belum disetel untuk evaluasi Ragas.")

    llm = LangchainLLMWrapper(
        ChatGroq(
            temperature=0,
            model_name="llama-3.1-8b-instant",
            groq_api_key=groq_api_key,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    )

    eval_dataset = Dataset.from_dict(dataset_dict)
    return evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        column_map={
            "user_input": "question",
            "response": "answer",
            "retrieved_contexts": "contexts",
            "reference": "ground_truth",
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluasi RAG Garuda Financial dengan Ragas")
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "eval" / "golden_dataset.json"),
        help="Path ke dataset evaluasi",
    )
    parser.add_argument("--limit", type=int, default=0, help="Batasi jumlah pertanyaan (0 = semua)")
    parser.add_argument(
        "--output",
        default=str(ROOT / "eval" / "results"),
        help="Folder output hasil evaluasi",
    )
    args = parser.parse_args()

    if not os.path.isdir(ROOT / "chroma_db"):
        raise FileNotFoundError("chroma_db tidak ditemukan. Jalankan: python src/vectorstore.py")

    golden = load_golden_dataset(Path(args.dataset))
    if args.limit > 0:
        golden = golden[: args.limit]

    questions = [item["question"] for item in golden]
    ground_truths = [item["ground_truth"] for item in golden]

    print(f"[*] Menjalankan RAG untuk {len(questions)} pertanyaan...")
    answers, contexts = collect_rag_outputs(questions)

    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    print("[*] Menjalankan metrik Ragas...")
    result = run_ragas_evaluation(dataset_dict)
    scores = extract_ragas_scores(result)
    per_sample_scores = extract_per_sample_scores(result)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ragas_report_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "num_questions": len(questions),
        "scores": scores,
        "per_sample_scores": per_sample_scores,
        "samples": [
            {
                "question": q,
                "ground_truth": gt,
                "answer": ans,
                "num_contexts": len(ctx),
            }
            for q, gt, ans, ctx in zip(questions, ground_truths, answers, contexts)
        ],
    }

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[+] Evaluasi selesai. Laporan disimpan ke: {output_file}")
    print("[+] Skor metrik:")
    for metric, value in scores.items():
        print(f"    - {metric}: {value}")


if __name__ == "__main__":
    main()
