#!/usr/bin/env python3
"""
Minimal BERTScore evaluator for the RAG system.

Reads QA pairs from rag/output/qa_pairs.json, queries the RAG system
(GLM4V server on port 8001), computes BERTScore (F1) per sample, prints
the average, and saves detailed results.

KISS: minimal args, single pass scoring.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from bert_score import score as bertscore
from tqdm import tqdm

from rag_system import RAGSystem


def load_qa_pairs(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned: List[Dict[str, str]] = []
    for item in data:
        if "question" in item and "answer" in item:
            cleaned.append({
                "question": str(item["question"]),
                "answer": str(item["answer"]),
            })
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BERTScore on RAG outputs")
    parser.add_argument(
        "--qa",
        type=Path,
        default=Path("/data/glm4/rag/output/qa_pairs.json"),
        help="Path to QA pairs JSON (list of {question, answer})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/data/glm4/rag/output/bertscore_results.json"),
        help="Where to save detailed results JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of QA pairs to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-chinese",
        help="HF model name for BERTScore (default: bert-base-chinese)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for BERTScore computation",
    )

    args = parser.parse_args()

    qa_pairs = load_qa_pairs(args.qa)
    if args.limit is not None:
        qa_pairs = qa_pairs[: max(0, args.limit)]

    # Ensure vector store path is resolved relative to this script directory
    script_dir = Path(__file__).resolve().parent
    vector_store_path = script_dir / "output" / "vector_store"
    rag = RAGSystem(vector_store_path=str(vector_store_path))

    predictions: List[str] = []
    references: List[str] = []
    q_texts: List[str] = []

    for pair in tqdm(qa_pairs, desc="Querying RAG", unit="qa"):
        question = pair["question"].strip()
        reference = pair["answer"].strip()
        answer_result = rag.answer_question(question)
        prediction = str(answer_result.get("answer", "")).strip()

        q_texts.append(question)
        references.append(reference)
        predictions.append(prediction)

    # Compute BERTScore over all pairs at once
    _, _, f1 = bertscore(
        cands=predictions,
        refs=references,
        lang="zh",
        model_type=args.model,
        batch_size=args.batch_size,
        verbose=False,
        rescale_with_baseline=False,
    )

    f1_list = [float(x) for x in f1.tolist()]
    avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    results: List[Dict[str, Any]] = []
    for q, ref, pred, s in zip(q_texts, references, predictions, f1_list):
        results.append(
            {
                "question": q,
                "reference": ref,
                "prediction": pred,
                "bertscore_f1": s,
            }
        )

    summary = {
        "num_samples": len(results),
        "average_bertscore_f1": avg_f1,
        "model": args.model,
    }

    print("\n==============================")
    print(f"Samples: {summary['num_samples']}")
    print(f"Average BERTScore (F1): {avg_f1:.4f}")
    print("==============================\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

