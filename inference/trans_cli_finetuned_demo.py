"""
This script creates a CLI demo for the fine-tuned GLM-4-9B model with QLoRA adapters,
allowing users to interact with the model through a command-line interface.

Usage:
- Run the script to start the CLI demo.
- Interact with the model by typing questions and receiving responses.

This script is specifically designed for loading fine-tuned models with LoRA/QLoRA adapters.
"""

from threading import Thread
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from peft import PeftModelForCausalLM

# Fix for PyTorch 2.6 weights_only loading issue
import torch.serialization
import numpy.core.multiarray
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])


# Configuration - Update these paths as needed
BASE_MODEL_PATH = "THUDM/GLM-4-9B-0414"  # Base model path
# Your fine-tuned checkpoint
FINETUNED_MODEL_PATH = "../finetune/output/cmcc34_qlora_system_prompt/checkpoint-5000"
USE_4BIT = True  # Set to True if you used QLoRA with 4-bit quantization


def load_finetuned_model(base_model_path: str, finetuned_path: str, use_4bit: bool = True):
    """
    Load the fine-tuned model with LoRA/QLoRA adapters.
    """
    print(f"Loading base model from: {base_model_path}")
    print(f"Loading fine-tuned adapters from: {finetuned_path}")

    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True)

    # Prepare model loading kwargs
    model_kwargs = {
        "use_cache": False,
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }

    # Add quantization config if using 4-bit
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, **model_kwargs)

    # Load LoRA/QLoRA adapters
    model = PeftModelForCausalLM.from_pretrained(model, finetuned_path)

    # Enable gradient checkpointing for QLoRA if using 4-bit
    if use_4bit:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.config.eos_token_id
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


if __name__ == "__main__":
    # Load the fine-tuned model
    tokenizer, model = load_finetuned_model(
        BASE_MODEL_PATH, FINETUNED_MODEL_PATH, USE_4BIT)

    history = []
    max_length = 2048  # Reduced for memory efficiency
    top_p = 0.8
    temperature = 0.6
    stop = StopOnTokens()

    print("Welcome to the Fine-tuned GLM-4-9B CLI chat!")
    print("This model has been fine-tuned on the CMCC-34 dataset.")
    print("Type your messages below. Type 'exit' or 'quit' to end the conversation.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        history.append([user_input, ""])

        messages = []
        for idx, (user_msg, model_msg) in enumerate(history):
            if idx == len(history) - 1 and not model_msg:
                messages.append({"role": "user", "content": user_msg})
                break
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})

        model_inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer=tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "streamer": streamer,
            "max_new_tokens": max_length,
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
            "stopping_criteria": StoppingCriteriaList([stop]),
            "repetition_penalty": 1.2,
            "eos_token_id": model.config.eos_token_id,
        }

        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        print("GLM-4 (Fine-tuned):", end="", flush=True)
        for new_token in streamer:
            if new_token:
                print(new_token, end="", flush=True)
                history[-1][1] += new_token

        history[-1][1] = history[-1][1].strip()
