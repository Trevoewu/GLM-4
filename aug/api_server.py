#!/usr/bin/env python3
"""
简化的GLM-4 API服务器，用于数据增强
"""

import json
import time
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

# 全局变量
model = None
tokenizer = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "glm-4"
    messages: List[ChatMessage]
    temperature: float = 0.8
    max_tokens: int = 1000

def load_model():
    global model, tokenizer
    print("🔄 正在加载GLM-4模型...")
    model_path = "THUDM/GLM-4-9B-0414"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            encode_special_tokens=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto", 
            trust_remote_code=True
        ).eval()
        
        print("✅ GLM-4模型加载成功！")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    success = load_model()
    if not success:
        print("⚠️ 模型加载失败")

@app.get("/health")
async def health_check():
    return {"status": "ok" if model is not None else "error", "model_loaded": model is not None}

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "glm-4", "object": "model"}]}

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        conversation = []
        for msg in request.messages:
            conversation.append({"role": msg.role, "content": msg.content})
        
        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(request.max_tokens, 1024),
                temperature=request.temperature,
                do_sample=True if request.temperature > 0 else False,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = full_response[len(prompt):].strip()
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "glm-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

if __name__ == "__main__":
    print("🚀 启动GLM-4数据增强API服务...")
    print("📡 端口: 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)
