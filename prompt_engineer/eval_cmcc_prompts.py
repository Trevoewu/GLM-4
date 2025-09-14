#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import requests
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# -------------------------
# Basic configuration
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("prompt_eval")


ALLOWED_LABELS = ["办理", "投诉", "咨询"]
LABEL_LIST_STR = "、".join(ALLOWED_LABELS)

# Prefer 127.0.0.1 to avoid IPv6 resolution pitfalls
GLM4_BASE_URL = os.environ.get("GLM4_BASE_URL", "http://127.0.0.1:8001")
GLM4_MODEL_NAME = os.environ.get("GLM4_MODEL_NAME", "glm4-9b")
RESULTS_WEBHOOK_URL = os.environ.get("RESULTS_WEBHOOK_URL")

DEFAULT_DATASET = os.environ.get(
    "CMCC_TEST_PATH", 
    "/data/glm4/data/cmcc-34/test_new.csv"
)


# -------------------------
# Prompt templates
# -------------------------
def prompt_zero_shot(sentence: str) -> str:
    return (
        "你是一个专业的中国移动业务分类助手。下面是一段客户与客服的对话，请仔细阅读对话内容，并从以下给出的业务类别中，选择一个最合适的。\n"
        f"可选业务类别：{LABEL_LIST_STR}\n"
        f"对话文本：{sentence}\n"
        "请注意：你只需要返回最准确的业务类别名称，不要包含任何其他解释或说明。"
    )


def prompt_few_shot(sentence: str) -> str:
    return (
        "你是一个专业的中国移动业务分类助手。你的任务是根据提供的对话内容，准确地判断其所属的业务类别。\n"
        "[示例1]\n"
        "对话内容: \"您好，我想查一下我这个月的话费和流量用了多少啊？\"\n"
        "业务类别: \"咨询\"\n"
        "[示例2]\n"
        "对话内容: \"喂，你好，帮我把那个来电显示功能打开吧，每个月扣钱的那个。\"\n"
        "业务类别: \"办理\"\n"
        "现在，请根据以下对话内容，从给出的可选类别中选择最合适的一项。\n"
        f"可选业务类别：{LABEL_LIST_STR}\n"
        f"对话文本：{sentence}"
    )


def prompt_cot(sentence: str) -> str:
    return (
        "你是一位经验丰富的中国移动金牌客服主管。你的工作是分析客服与客户的对话，并将其精确地归类到相应的业务范畴。\n"
        "现在，请分析以下这段对话。首先，在你的思考过程中，总结出客户的核心诉求是什么。"
        "然后，基于你的分析，从下列的业务类别中选择最匹配的一项作为最终答案。\n"
        f"可选业务类别：{LABEL_LIST_STR}\n"
        f"对话文本：{sentence}\n"
        "请只在最后一行输出最终类别名称，且不包含其他解释。"
    )


TEMPLATE_BUILDERS: Dict[str, Callable[[str], str]] = {
    "zero_shot": prompt_zero_shot,
    "few_shot": prompt_few_shot,
    "cot": prompt_cot,
}


# -------------------------
# Client for local GLM4
# -------------------------
def call_glm4_chat(prompt: str, temperature: float = 0.0, max_tokens: int = 256, timeout_s: int = 60) -> str:
    url = f"{GLM4_BASE_URL}/v1/chat/completions"
    payload = {
        "model": GLM4_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        if resp.status_code != 200:
            logger.warning("HTTP %s from LLM: %s", resp.status_code, resp.text[:200])
            return ""
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.exception("Error calling GLM4: %s", e)
        return ""


# -------------------------
# Output parsing
# -------------------------
_punct_re = re.compile(r"[\s\t\n\r\f\v\-—~·,，.。!！?？:：;；\[\]()（）\"'“”‘’<>《》]+")


def normalize_text(text: str) -> str:
    return _punct_re.sub("", str(text or "")).strip()


KEYWORD_TO_LABEL: List[Tuple[re.Pattern, str]] = [
    # 办理
    (re.compile(r"办理|开通|变更|取消|销户|补卡|复机|停机|缴费|移机|装机|拆机|修改|重置"), "办理"),
    # 投诉
    (re.compile(r"投诉|抱怨|不满|服务差|乱扣费|网络差|骚扰|欺诈|欺骗"), "投诉"),
    # 咨询
    (re.compile(r"咨询|查询|问下|了解一下|想知道|怎么用|如何办理|怎么查|请问"), "咨询"),
]


def parse_label(raw_output: str) -> str:
    if not raw_output:
        return ""
    text = str(raw_output).strip()

    # Try exact match variants first
    variants = [
        text,
        text.strip('"\'\n\r '),
        normalize_text(text),
    ]
    for v in variants:
        if v in ALLOWED_LABELS:
            return v

    # If any allowed label name appears in the output, pick the first occurrence order by ALLOWED_LABELS
    for label in ALLOWED_LABELS:
        if label in text:
            return label

    # Keyword heuristics
    text_simple = normalize_text(text)
    for pattern, label in KEYWORD_TO_LABEL:
        if pattern.search(text) or pattern.search(text_simple):
            return label

    return ""  # unknown


# -------------------------
# Ground-truth mapping
# -------------------------
def map_ground_truth_to_major(row: pd.Series) -> str:
    # Prefer num_cnum if present
    if "num_cnum" in row and pd.notna(row["num_cnum"]):
        try:
            code = int(row["num_cnum"])
            return {0: "咨询", 1: "投诉", 2: "办理"}.get(code, "")
        except Exception:
            pass

    # Fallback via label_raw prefix keywords
    label_raw = str(row.get("label_raw", ""))
    if label_raw.startswith("咨询"):
        return "咨询"
    if label_raw.startswith("投诉"):
        return "投诉"
    if label_raw.startswith("办理"):
        return "办理"
    return ""


# -------------------------
# Evaluation core
# -------------------------
@dataclass
class EvalResult:
    template_name: str
    accuracy: float
    macro_f1: float
    total: int
    invalid_pred: int


def evaluate_template(
    template_name: str,
    builder: Callable[[str], str],
    df: pd.DataFrame,
    sleep_seconds: float = 0.0,
) -> EvalResult:
    y_true: List[str] = []
    y_pred: List[str] = []
    invalid = 0

    for idx, row in df.iterrows():
        sentence = str(row.get("sentence_sep", "")).strip()
        if not sentence:
            continue

        gt = map_ground_truth_to_major(row)
        if gt not in ALLOWED_LABELS:
            # Skip if cannot determine ground truth major class
            continue

        prompt = builder(sentence)
        output = call_glm4_chat(prompt, temperature=0.0, max_tokens=32, timeout_s=60)
        pred = parse_label(output)
        if pred not in ALLOWED_LABELS:
            invalid += 1
            # Optionally, try a second pass with a forced instruction
            retry_prompt = (
                f"只回答以下三个词之一：{LABEL_LIST_STR}。不要添加其他内容。对话：{sentence}"
            )
            output_retry = call_glm4_chat(retry_prompt, temperature=0.0, max_tokens=8, timeout_s=30)
            pred = parse_label(output_retry)

        if pred not in ALLOWED_LABELS:
            # Last fallback: default to 咨询 for neutral wording
            pred = "咨询"

        y_true.append(gt)
        y_pred.append(pred)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        if (len(y_true) % 100) == 0:
            logger.info(
                "%s processed %d samples...", template_name, len(y_true)
            )

    if not y_true:
        return EvalResult(template_name, 0.0, 0.0, 0, invalid)

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    return EvalResult(template_name, float(acc), float(macro), len(y_true), invalid)


def load_dataset(csv_path: str) -> pd.DataFrame:
    usecols = None  # read all and select later to be safe
    df = pd.read_csv(csv_path)
    needed = ["sentence_sep", "label_raw", "num_cnum"]
    # Ensure columns exist; if not, try renaming alternatives
    missing = [c for c in needed if c not in df.columns]
    if missing:
        logger.warning("Missing columns %s in %s; available: %s", missing, csv_path, list(df.columns))
    return df


def maybe_sync_results(payload: dict) -> None:
    if not RESULTS_WEBHOOK_URL:
        return
    try:
        resp = requests.post(RESULTS_WEBHOOK_URL, json=payload, timeout=10)
        if resp.status_code >= 300:
            logger.warning("Webhook sync failed: %s %s", resp.status_code, resp.text[:200])
        else:
            logger.info("Results synced to webhook successfully.")
    except Exception as e:
        logger.warning("Webhook sync error: %s", e)


def main():
    csv_path = DEFAULT_DATASET
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    logger.info("Loading dataset: %s", csv_path)
    df = load_dataset(csv_path)
    logger.info("Loaded %d rows", len(df))

    # Run three templates
    results: List[EvalResult] = []
    for name, builder in (
        ("zero_shot", TEMPLATE_BUILDERS["zero_shot"]),
        ("few_shot", TEMPLATE_BUILDERS["few_shot"]),
        ("cot", TEMPLATE_BUILDERS["cot"]),
    ):
        logger.info("Evaluating template: %s", name)
        res = evaluate_template(name, builder, df)
        logger.info(
            "%s -> Accuracy: %.4f, Macro F1: %.4f (N=%d, invalid_pred=%d)",
            name, res.accuracy, res.macro_f1, res.total, res.invalid_pred,
        )
        results.append(res)

    # Assemble final report
    report = {
        "model": GLM4_MODEL_NAME,
        "base_url": GLM4_BASE_URL,
        "dataset": os.path.abspath(csv_path),
        "labels": ALLOWED_LABELS,
        "metrics": {
            r.template_name: {
                "accuracy": r.accuracy,
                "macro_f1": r.macro_f1,
                "total": r.total,
                "invalid_pred": r.invalid_pred,
            }
            for r in results
        },
        "timestamp": int(time.time()),
    }

    # Persist locally
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prompt_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Saved results to %s", out_path)

    # Optional sync
    maybe_sync_results(report)

    # Pretty print to stdout
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

