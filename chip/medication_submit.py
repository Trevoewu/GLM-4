#!/usr/bin/env python3
"""
Generate submission JSON for the medication prediction task on the test set.

Inputs:
- Finetuned adapters: /data/long/glm4/finetune/output/medication_prediction_model
- Test set (no labels): /data/long/glm4/data/CDrugRed-A-v1/test.json

Output:
- UTF-8 encoded JSON file matching sample format submit_pred_ex.json
  [
    {"ID": "1-1", "prediction": ["药物A", ...]},
    ...
  ]

Notes:
- IDs are constructed as "{患者序号}-{序号}" where 序号 is the per-患者序号 running index
- Model base: THUDM/GLM-4-9B-0414; LoRA/QLoRA adapters loaded from latest checkpoint-*
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM


def resolve_latest_checkpoint(adapter_root: str) -> str:
    if not os.path.isdir(adapter_root):
        raise FileNotFoundError(f"Adapter root not found: {adapter_root}")
    candidates = []
    for name in os.listdir(adapter_root):
        path = os.path.join(adapter_root, name)
        if os.path.isdir(path) and name.startswith("checkpoint-"):
            try:
                step = int(name.split("-", 1)[1])
            except Exception:
                continue
            candidates.append((step, path))
    if not candidates:
        return adapter_root
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_model(base_model_path: str, finetuned_path: str, use_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    model_kwargs = {
        "use_cache": False,
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        # Ensure the model is placed onto available GPUs by default
        "device_map": "auto",
    }
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = bnb
        # device_map is already set to auto above

    base = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    model = PeftModelForCausalLM.from_pretrained(base, finetuned_path)

    if use_4bit:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.eval()
    return tokenizer, model


def safe_json_extract_medications(text: str) -> List[str]:
    if not text:
        return []
    # Try JSON object extraction
    obj = None
    for cand in re.findall(r"\{[\s\S]*\}", text)[::-1]:
        try:
            obj = json.loads(cand)
            break
        except Exception:
            continue
    if isinstance(obj, dict):
        meds = obj.get("出院带药列表")
        if isinstance(meds, list):
            return [str(m).strip() for m in meds if isinstance(m, (str, int, float))]
    # Fallback: regex list after key
    m = re.search(r"出院带药列表\"?\s*[:：]\s*\[(.*?)\]", text, flags=re.S)
    if m:
        inner = m.group(1)
        items = [x.strip().strip("\"'\"") for x in inner.split(',') if x.strip()]
        return [x for x in items if x]
    return []


def build_ids_for_test(samples: List[Dict]) -> List[str]:
    counters: Dict[str, int] = {}
    ids: List[str] = []
    for ex in samples:
        msgs = ex.get("messages", [])
        user = msgs[1]["content"] if len(msgs) > 1 and isinstance(msgs[1], dict) else ""
        m = re.search(r"患者序号\s*[:：]\s*(\d+)", user)
        patient_no = m.group(1) if m else "0"
        counters[patient_no] = counters.get(patient_no, 0) + 1
        ids.append(f"{patient_no}-{counters[patient_no]}")
    return ids


@torch.inference_mode()
def generate_once(tokenizer, model, system_prompt: str, user_prompt: str, max_new_tokens: int = 128) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )

    # Resolve the actual device for model parameters (works with device_map="auto")
    try:
        param_device = next(model.parameters()).device
    except StopIteration:
        param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_inputs = {k: v.to(param_device) for k, v in model_inputs.items()}

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.8,
        temperature=0.6,
        repetition_penalty=1.1,
        eos_token_id=getattr(model, "config", None) and getattr(model.config, "eos_token_id", None),
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = outputs[:, model_inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Generate medication prediction submission JSON")
    parser.add_argument("--test-file", type=str, default="/data/long/glm4/data/CDrugRed-A-v1/test.json")
    parser.add_argument("--model-path", type=str, default="/data/long/glm4/finetune/output/medication_prediction_model")
    parser.add_argument("--base-model", type=str, default="THUDM/GLM-4-9B-0414")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output", type=str, default="/data/long/glm4/chip/submission_pred.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data: List[Dict] = json.load(f)
    if args.limit and args.limit > 0:
        test_data = test_data[: args.limit]

    adapter_dir = resolve_latest_checkpoint(args.model_path)
    tokenizer, model = load_model(args.base_model, adapter_dir, use_4bit=(not args.no_4bit))

    ids = build_ids_for_test(test_data)
    results: List[Dict[str, object]] = []

    for ex_id, ex in tqdm(list(zip(ids, test_data)), desc="Predicting", total=len(test_data)):
        msgs = ex.get("messages", [])
        if len(msgs) < 2:
            results.append({"ID": ex_id, "prediction": []})
            continue
        system_prompt = msgs[0].get("content", "")
        user_prompt = msgs[1].get("content", "")

        pred_text = generate_once(tokenizer, model, system_prompt, user_prompt)
        pred_list = safe_json_extract_medications(pred_text)

        # Normalize and de-duplicate while preserving order
        seen = set()
        normed: List[str] = []
        for item in pred_list:
            name = str(item).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            normed.append(name)

        results.append({"ID": ex_id, "prediction": normed})

    # Ensure UTF-8 JSON output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved submission to: {args.output}")


if __name__ == "__main__":
    main()


