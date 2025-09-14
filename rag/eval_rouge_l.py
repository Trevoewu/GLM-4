#!/usr/bin/env python3
"""
Minimal ROUGE-L evaluator for the RAG system.

Reads question-answer pairs from rag/output/qa_pairs.json, queries the
RAG system (which calls the local GLM4V server on port 8001), computes
ROUGE-L (F1) per sample, prints the average, and saves detailed results.

KISS: no frameworks, minimal dependencies (rouge-score, tqdm).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from rouge_score import rouge_scorer
from tqdm import tqdm

from rag_system import RAGSystem


def load_qa_pairs(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Ensure minimal schema
    cleaned: List[Dict[str, str]] = []
    for item in data:
        if "question" in item and "answer" in item:
            cleaned.append({
                "question": str(item["question"]),
                "answer": str(item["answer"]),
            })
    return cleaned


def compute_rouge_l(reference: str, prediction: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return float(scores["rougeL"].fmeasure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ROUGE-L on RAG outputs")
    parser.add_argument(
        "--qa",
        type=Path,
        default=Path("/data/glm4/rag/output/qa_pairs.json"),
        help="Path to QA pairs JSON (list of {question, answer})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/data/glm4/rag/output/rouge_l_results.json"),
        help="Where to save detailed results JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of QA pairs to evaluate",
    )

    args = parser.parse_args()

    qa_pairs = load_qa_pairs(args.qa)
    if args.limit is not None:
        qa_pairs = qa_pairs[: max(0, args.limit)]

    # Ensure vector store path is resolved relative to this script directory
    script_dir = Path(__file__).resolve().parent
    vector_store_path = script_dir / "output" / "vector_store"

    rag = RAGSystem(vector_store_path=str(vector_store_path))  # Uses local GLM4V server per config

    results: List[Dict[str, Any]] = []
    rouge_l_sum = 0.0

    for pair in tqdm(qa_pairs, desc="Evaluating", unit="qa"):
        question = pair["question"].strip()
        reference = pair["answer"].strip()

        try:
            answer_result = rag.answer_question(question)
            prediction = str(answer_result.get("answer", "")).strip()
        except Exception as e:
            prediction = f"ERROR: {e}"

        score_l = compute_rouge_l(reference, prediction)
        rouge_l_sum += score_l

        results.append(
            {
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "rougeL": score_l,
            }
        )

    n = len(results)
    avg_rouge_l = (rouge_l_sum / n) if n > 0 else 0.0

    summary = {
        "num_samples": n,
        "average_rougeL": avg_rouge_l,
    }

    print("\n==============================")
    print(f"Samples: {n}")
    print(f"Average ROUGE-L (F1): {avg_rouge_l:.4f}")
    print("==============================\n")

    output = {"summary": summary, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

