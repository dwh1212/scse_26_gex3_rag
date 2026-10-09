## THIS FILE COMPLETES THE RAG PIPELINE.
## It retrieves the most relevant policy excerpts and answers a question with
## the local Qwen3-0.6B model, grounded ONLY in those excerpts.

import re

import torch
from rag_tool import retrieve
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = r"D:\学校作业\SCSE_26_GEX3\models\qwen3-0.6b"

print("Loading Qwen3-0.6B ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)

SYSTEM_PROMPT = (
    "You are a university IT support assistant. Answer the user's question using ONLY "
    "the provided policy excerpts. If the exact information requested (such as a phone "
    "number) is NOT present in the excerpts, say that it is not available rather than "
    "inventing it. Keep the answer concise."
)


def build_prompt(query, docs):
    context = "\n\n".join(
        f"[{d.metadata['policy_id']} - topic: {d.metadata['topic']}]\n{d.page_content}"
        for d in docs
    )
    return f"{SYSTEM_PROMPT}\n\nRelevant policy excerpts:\n{context}\n\nUser: {query}\n\nAssistant:"


def _clean_generated(text):
    """Remove any thinking block and cut the model's auto-continuation loop."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for marker in ("\nThe answer", "\nAnswer:", "\nThe assistant", "\nThe user's", "\nUser:"):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx].strip()
            break
    return text


def answer(query, top_k=3, max_new_tokens=100):
    """Retrieve relevant excerpts and generate a grounded answer."""
    docs = retrieve(query, top_k=top_k)
    prompt = build_prompt(query, docs)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tail = full[len(prompt):].strip() if full.startswith(prompt) else full
    return _clean_generated(tail)
