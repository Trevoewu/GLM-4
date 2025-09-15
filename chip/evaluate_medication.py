#!/usr/bin/env python3
"""
Evaluate a fine-tuned GLM-4 medication prediction model on a dev set.

Metrics:
- Example-averaged Jaccard (IoU) over medication sets
- Micro F1 over the entire label space
- Final score = 0.5 * (Jaccard + F1)

Model:
- Base: THUDM/GLM-4-9B-0414
- Finetuned adapters: finetune/output/medication_prediction_model (default)

Data format:
- JSON array of conversations in GLM-4 chat format with messages: [system, user, assistant]
- Ground-truth medications are in assistant JSON field "出院带药列表"
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM


def safe_json_extract_medications(text: str) -> List[str]:
    """Extract medication list from model/gt JSON-ish text.

    Expected format:
    {
      "出院带药列表": ["药物1", "药物2", ...]
    }
    """
    if not text:
        return []

    # Try to find a JSON object first
    json_obj = None
    # Greedy match the outermost braces
    candidates = re.findall(r"\{[\s\S]*\}", text)
    for cand in candidates[::-1]:  # prefer the last (often the final block)
        try:
            json_obj = json.loads(cand)
            break
        except Exception:
            continue

    if isinstance(json_obj, dict):
        meds = json_obj.get("出院带药列表")
        if isinstance(meds, list):
            return [str(m).strip() for m in meds if isinstance(m, (str, int, float))]

    # Fallback: regex to capture array after the key
    m = re.search(r"出院带药列表\"?\s*[:：]\s*\[(.*?)\]", text, flags=re.S)
    if m:
        inner = m.group(1)
        # Split by commas and strip quotes/spaces
        items = [x.strip().strip('"\'\'') for x in inner.split(',') if x.strip()]
        return [x for x in items if x]

    return []


def jaccard_index(pred: Set[str], gt: Set[str]) -> float:
    if not pred and not gt:
        return 1.0
    if not pred and gt:
        return 0.0
    if pred and not gt:
        return 0.0
    inter = len(pred & gt)
    union = len(pred | gt)
    return inter / union if union > 0 else 0.0


def compute_micro_f1(all_preds: List[Set[str]], all_gts: List[Set[str]]) -> float:
    # Build a vocabulary over all seen labels
    vocab: Dict[str, int] = {}
    for s in all_preds + all_gts:
        for label in s:
            if label not in vocab:
                vocab[label] = len(vocab)

    if not vocab:
        return 1.0  # if no labels anywhere, define as perfect

    num_samples = len(all_preds)
    num_labels = len(vocab)
    y_true = np.zeros((num_samples, num_labels), dtype=np.int32)
    y_pred = np.zeros((num_samples, num_labels), dtype=np.int32)

    for i, (pred_set, gt_set) in enumerate(zip(all_preds, all_gts)):
        for lab in gt_set:
            y_true[i, vocab[lab]] = 1
        for lab in pred_set:
            y_pred[i, vocab[lab]] = 1

    return f1_score(y_true.reshape(-1), y_pred.reshape(-1), average='binary', zero_division=0)


class MedicationEvaluator:
    def __init__(self,
                 base_model_path: str,
                 finetuned_path: str,
                 data_file: str,
                 output_dir: str = "eval_medication_output",
                 use_4bit: bool = True):
        self.base_model_path = base_model_path
        self.finetuned_path = finetuned_path
        self.data_file = data_file
        self.output_dir = output_dir
        self.use_4bit = use_4bit

        os.makedirs(self.output_dir, exist_ok=True)

        self.tokenizer = None
        self.model = None

    def _resolve_adapter_dir(self, path: str) -> str:
        """Return a directory containing adapter_config.json.

        If the provided path is a parent folder (e.g. output dir) without
        adapter_config.json, pick the latest checkpoint-* subdir that has it.
        """
        cfg = os.path.join(path, "adapter_config.json")
        if os.path.isfile(cfg):
            return path

        # scan checkpoints
        if os.path.isdir(path):
            candidates: List[Tuple[int, str]] = []
            for name in os.listdir(path):
                sub = os.path.join(path, name)
                if os.path.isdir(sub) and name.startswith("checkpoint-"):
                    if os.path.isfile(os.path.join(sub, "adapter_config.json")):
                        # extract step number
                        m = re.match(r"checkpoint-(\d+)$", name)
                        step = int(m.group(1)) if m else -1
                        candidates.append((step, sub))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                return candidates[-1][1]

        # fall back to original path; PEFT will raise a clear error
        return path

    def load_model(self):
        adapter_dir = self._resolve_adapter_dir(self.finetuned_path)
        print(f"Loading model adapters from: {adapter_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)

        model_kwargs = {
            "use_cache": False,
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = bnb
            model_kwargs["device_map"] = "auto"

        base = AutoModelForCausalLM.from_pretrained(self.base_model_path, **model_kwargs)
        self.model = PeftModelForCausalLM.from_pretrained(base, adapter_dir)

        if self.use_4bit:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()

        self.model.eval()
        print("Model loaded.")

    def load_data(self) -> List[Dict]:
        print(f"Loading dev data from: {self.data_file}")
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples")
        return data

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        model_inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                do_sample=True,
                top_p=0.8,
                temperature=0.6,
                repetition_penalty=1.15,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_ids = outputs[:, model_inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

    def evaluate(self, samples: List[Dict]) -> Dict:
        preds: List[Set[str]] = []
        gts: List[Set[str]] = []
        details: List[Dict] = []

        for i, ex in enumerate(tqdm(samples, desc="Evaluating")):
            # Expect messages: [system, user, assistant]
            msgs = ex.get("messages", [])
            if len(msgs) < 2:
                continue

            system_prompt = msgs[0].get("content", "")
            user_prompt = msgs[1].get("content", "")
            gt_text = msgs[2].get("content", "") if len(msgs) > 2 else ""

            # Ground truth extraction
            gt_list = safe_json_extract_medications(gt_text)

            # Generate prediction
            pred_text = self.generate(system_prompt, user_prompt)
            pred_list = safe_json_extract_medications(pred_text)

            # Normalize items by stripping spaces
            pred_set = {p.strip() for p in pred_list if p and str(p).strip()}
            gt_set = {g.strip() for g in gt_list if g and str(g).strip()}

            preds.append(pred_set)
            gts.append(gt_set)

            details.append({
                "index": i,
                "pred_text": pred_text,
                "pred_list": sorted(list(pred_set)),
                "gt_list": sorted(list(gt_set)),
                "jaccard": jaccard_index(pred_set, gt_set),
            })

        # Metrics
        per_example_j = [jaccard_index(p, t) for p, t in zip(preds, gts)]
        jaccard = float(np.mean(per_example_j)) if per_example_j else 0.0
        micro_f1 = float(compute_micro_f1(preds, gts)) if preds else 0.0
        final_score = 0.5 * (jaccard + micro_f1)

        return {
            "num_samples": len(preds),
            "jaccard": jaccard,
            "micro_f1": micro_f1,
            "final_score": final_score,
            "details": details,
        }

    def save_results(self, results: Dict):
        path = os.path.join(self.output_dir, "results.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved results to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate medication prediction model on dev set")
    parser.add_argument("--data-file", type=str, default="../data/CDrugRed-A-v1/train.json",
                        help="Path to dev JSON file with ground-truth assistant outputs")
    parser.add_argument("--model-path", type=str, default="../finetune/output/medication_prediction_model",
                        help="Path to finetuned adapter directory")
    parser.add_argument("--output-dir", type=str, default="./output_medication_eval",
                        help="Directory to save evaluation outputs")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 = all)")
    args = parser.parse_args()

    base_model = "THUDM/GLM-4-9B-0414"

    evaluator = MedicationEvaluator(
        base_model_path=base_model,
        finetuned_path=args.model_path,
        data_file=args.data_file,
        output_dir=args.output_dir,
        use_4bit=(not args.no_4bit),
    )

    evaluator.load_model()
    data = evaluator.load_data()
    if args.limit and args.limit > 0:
        data = data[:args.limit]
        print(f"Limiting to {len(data)} samples")

    results = evaluator.evaluate(data)

    print("\n================ Medication Dev Evaluation ================")
    print(f"Samples: {results['num_samples']}")
    print(f"Jaccard: {results['jaccard']:.4f}")
    print(f"Micro F1: {results['micro_f1']:.4f}")
    print(f"Final Score (0.5*(J+F1)): {results['final_score']:.4f}")

    evaluator.save_results(results)


if __name__ == "__main__":
    main()


