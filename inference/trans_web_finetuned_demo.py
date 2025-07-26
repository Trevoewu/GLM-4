"""
Gradio web interface for the fine-tuned GLM-4-9B model with QLoRA adapters.
This provides a user-friendly web interface to interact with your fine-tuned model.
"""

import os
from threading import Thread

# Simple fix for PyTorch 2.6 weights_only issue
import torch
import os

# Set environment variable to disable weights_only globally
os.environ['PYTORCH_DISABLE_WEIGHTS_ONLY_LOAD'] = '1'

# Simple global patch
_original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from peft import PeftModelForCausalLM






# Configuration
BASE_MODEL_PATH = "THUDM/GLM-4-9B-0414"
FINETUNED_MODEL_PATH = "../finetune/output/cmcc34_qlora/checkpoint-5000"
USE_4BIT = True

# Global variables for model and tokenizer
tokenizer = None
model = None


def load_finetuned_model():
    """Load the fine-tuned model with QLoRA adapters."""
    global tokenizer, model

    print(f"Loading base model: {BASE_MODEL_PATH}")
    print(f"Loading fine-tuned adapters: {FINETUNED_MODEL_PATH}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True)

    # Model loading kwargs
    model_kwargs = {
        "use_cache": False,
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }

    # Add quantization for QLoRA
    if USE_4BIT:
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
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, **model_kwargs)

    # Load LoRA adapters
    print("Loading LoRA adapters...")
    model = PeftModelForCausalLM.from_pretrained(model, FINETUNED_MODEL_PATH)

    if USE_4BIT:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.eval()
    print("Model loaded successfully!")


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.config.eos_token_id
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def preprocess_messages(history, system_prompt):
    """Preprocess messages for the model."""
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

    return messages


def generate_response(history, system_prompt, max_new_tokens, temperature, top_p):
    """Generate response using the fine-tuned model."""
    if model is None or tokenizer is None:
        return "Model not loaded. Please wait for the model to load."

    messages = preprocess_messages(history, system_prompt)

    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList([StopOnTokens()]),
        "repetition_penalty": 1.2,
        "eos_token_id": model.config.eos_token_id,
    }

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    response = ""
    for new_token in streamer:
        if new_token:
            response += new_token
            yield response.strip()


def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(
        title="Fine-tuned GLM-4-9B Chat"
    ) as demo:
        gr.Markdown(
            """
            # Fine-tuned GLM-4-9B Chat Interface
            
            This interface allows you to interact with your fine-tuned GLM-4-9B model that has been trained on the CMCC-34 dataset using QLoRA.
            
            **Model Info:**
            - Base Model: GLM-4-9B-0414
            - Fine-tuning Method: QLoRA (4-bit quantization)
            - Dataset: CMCC-34
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=600,
                    show_copy_button=True,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Input",
                        placeholder="Type your message here...",
                        lines=2,
                        scale=4
                    )
                    send = gr.Button("Send", scale=1)

                with gr.Row():
                    clear = gr.Button("Clear Chat")
                    regenerate = gr.Button("Regenerate")

            with gr.Column(scale=1):
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="You are a helpful AI assistant...",
                    lines=3
                )

                with gr.Accordion("Generation Parameters", open=False):
                    max_new_tokens = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        value=512,
                        step=64,
                        label="Max New Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.6,
                        step=0.1,
                        label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.8,
                        step=0.1,
                        label="Top-p"
                    )

        # Event handlers
        def user_input(user_message, history, system_prompt, max_new_tokens, temperature, top_p):
            if not user_message.strip():
                return "", history

            history.append([user_message, ""])
            return "", history

        def bot_response(history, system_prompt, max_new_tokens, temperature, top_p):
            if not history:
                return history

            # Generate response
            response_generator = generate_response(
                history, system_prompt, max_new_tokens, temperature, top_p)
            response = ""
            for partial_response in response_generator:
                response = partial_response
                history[-1][1] = response
                yield history

        # Connect events
        msg.submit(
            user_input,
            [msg, chatbot, system_prompt, max_new_tokens, temperature, top_p],
            [msg, chatbot],
            queue=False
        ).then(
            bot_response,
            [chatbot, system_prompt, max_new_tokens, temperature, top_p],
            [chatbot]
        )

        send.click(
            user_input,
            [msg, chatbot, system_prompt, max_new_tokens, temperature, top_p],
            [msg, chatbot],
            queue=False
        ).then(
            bot_response,
            [chatbot, system_prompt, max_new_tokens, temperature, top_p],
            [chatbot]
        )

        clear.click(lambda: [], outputs=[chatbot])

        regenerate.click(
            bot_response,
            [chatbot, system_prompt, max_new_tokens, temperature, top_p],
            [chatbot]
        )

    return demo


if __name__ == "__main__":
    # Load the model first
    print("Loading fine-tuned model...")
    load_finetuned_model()

    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
