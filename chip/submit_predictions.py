#!/usr/bin/env python3
"""
Generate medication predictions for the test set in submission format.

Output format matches data/CDrugRed-A-v1/submit_pred_ex.json:
[
  {"ID": "1-1", "prediction": ["药物A", "药物B", ...]},
  ...
]
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM


def load_candidates(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def safe_json_extract_medications(text: str) -> List[str]:
    if not text:
        return []
    # Try parse last JSON block
    candidates = re.findall(r"\{[\s\S]*\}", text)
    json_obj = None
    for cand in candidates[::-1]:
        try:
            json_obj = json.loads(cand)
            break
        except Exception:
            continue
    if isinstance(json_obj, dict):
        meds = json_obj.get("出院带药列表")
        if isinstance(meds, list):
            return [str(m).strip() for m in meds if isinstance(m, (str, int, float))]
    # Fallback regex
    m = re.search(r"出院带药列表\"?\s*[:：]\s*\[(.*?)\]", text, flags=re.S)
    if m:
        inner = m.group(1)
        items = [x.strip().strip('"\'\'') for x in inner.split(',') if x.strip()]
        return [x for x in items if x]
    return []


def build_user_prompt(record: Dict) -> str:
    # Construct a concise user prompt using available fields
    fields = []
    def add(title: str, key: str):
        if key in record and record[key] not in (None, ""):
            fields.append(f"**{title}:** {record[key]}")

    fields.append(f"**患者基本信息:**\n- 患者序号: {record.get('患者序号', '')}\n- 性别: {record.get('性别', '')}\n- 出生日期: {record.get('出生日期', '')}\n- BMI: {record.get('BMI', '')}")
    add("主诉", "主诉")
    add("既往史", "既往史")
    add("现病史", "现病史")
    add("入院情况", "入院情况")
    if isinstance(record.get("出院诊断"), list):
        fields.append(f"**出院诊断:** {record['出院诊断']}")

    task = (
        "**任务:**\n请从候选药物列表中选择最适合的药物，预测该患者的出院用药清单.\n\n"
        "**输出格式:**\n```json\n{\n \"出院带药列表\": [\"药物1\", \"药物2\", \"...\"]\n}\n```\n"
    )

    return "\n\n".join(["请根据以下中文病历信息，为患者推荐合适的出院用药方案："] + fields + [task])


class SubmitPredictor:
    def __init__(self,
                 base_model_path: str,
                 finetuned_path: str,
                 system_prompt: str,
                 candidate_list: List[str],
                 use_4bit: bool = True):
        self.base_model_path = base_model_path
        self.finetuned_path = finetuned_path
        self.system_prompt = system_prompt
        self.candidate_set = set(candidate_list)
        self.use_4bit = use_4bit
        self.tokenizer = None
        self.model = None

    def _resolve_adapter_dir(self, path: str) -> str:
        cfg = os.path.join(path, "adapter_config.json")
        if os.path.isfile(cfg):
            return path
        if os.path.isdir(path):
            best: Tuple[int, str] = (-1, path)
            for name in os.listdir(path):
                sub = os.path.join(path, name)
                if os.path.isdir(sub) and name.startswith("checkpoint-") and os.path.isfile(os.path.join(sub, "adapter_config.json")):
                    m = re.match(r"checkpoint-(\d+)$", name)
                    step = int(m.group(1)) if m else -1
                    if step > best[0]:
                        best = (step, sub)
            if best[0] >= 0:
                return best[1]
        return path

    def load_model(self):
        adapter_dir = self._resolve_adapter_dir(self.finetuned_path)
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

    def _truncate_to_context(self, system_text: str, user_text: str, max_input_tokens: int = 7000) -> Tuple[str, str]:
        sys_ids = self.tokenizer(system_text, add_special_tokens=False).input_ids
        usr_ids = self.tokenizer(user_text, add_special_tokens=False).input_ids
        budget = max(256, max_input_tokens - len(sys_ids) - 64)  # leave small buffer
        if len(usr_ids) > budget:
            usr_ids = usr_ids[-budget:]  # keep tail (often contains summary/diagnosis)
        return system_text, self.tokenizer.decode(usr_ids, skip_special_tokens=True)

    def generate(self, user_prompt: str, max_new_tokens: int = 256) -> str:
        sys_text, usr_text = self._truncate_to_context(self.system_prompt, user_prompt)
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": usr_text},
        ]
        model_inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic for submission
                repetition_penalty=1.1,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_ids = outputs[:, model_inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

    def predict_record(self, record: Dict) -> List[str]:
        prompt = build_user_prompt(record)
        text = self.generate(prompt)
        meds = safe_json_extract_medications(text)
        # Normalize and filter to candidate list
        norm = []
        for m in meds:
            s = str(m).strip()
            if s and s in self.candidate_set and s not in norm:
                norm.append(s)
        return norm


def main():
    parser = argparse.ArgumentParser(description="Generate medication predictions submission")
    parser.add_argument("--test-file", type=str, default="../data/CDrugRed-A-v1/CDrugRed_test-A.jsonl",
                        help="Path to test jsonl (no labels)")
    parser.add_argument("--output", type=str, default="./submission.json",
                        help="Path to write submission JSON")
    parser.add_argument("--model-path", type=str, default="../finetune/output/medication_prediction_model",
                        help="Path to finetuned adapter dir or checkpoint")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of test samples (0=all)")
    args = parser.parse_args()

    # Prompts and candidates
    from pathlib import Path
    from prompt import SYSTEM_PROMPT
    candidates = load_candidates(str(Path(__file__).parent / "候选药物列表.json"))

    predictor = SubmitPredictor(
        base_model_path="THUDM/GLM-4-9B-0414",
        finetuned_path=args.model_path,
        system_prompt=SYSTEM_PROMPT,
        candidate_list=candidates,
        use_4bit=(not args.no_4bit),
    )
    predictor.load_model()

    # Load test jsonl
    test_records: List[Dict] = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            test_records.append(json.loads(line))

    if args.limit and args.limit > 0:
        test_records = test_records[:args.limit]

    # Predict
    results: List[Dict] = []
    for rec in test_records:
        rec_id = rec.get("就诊标识") or rec.get("ID") or rec.get("id")
        pred_list = predictor.predict_record(rec)
        results.append({"ID": rec_id, "prediction": pred_list})

    # Save
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved submission to: {args.output}  (samples={len(results)})")


if __name__ == "__main__":
    main()




