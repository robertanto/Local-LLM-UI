import logging

# Configure logger
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger('tag_api')
logger.setLevel(logging.INFO)

from fastapi import FastAPI
from pydantic import BaseModel
from lc_hf_hub import HuggingFaceInferenceLLM
from typing import List
from transformers import AutoTokenizer


model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model_args = {"max_new_tokens":1000}
llm = HuggingFaceInferenceLLM(model_id=model_id, model_args=model_args)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

app = FastAPI()

class Input(BaseModel):
    message: str
    history: List[List[str]]

@app.post("/chat")  
async def answer(input: Input):
    try:
        messages = []
        for usr,assist in input.history:
            messages.append({"role": "user", "content": usr})
            messages.append({"role": "assistant", "content": assist})
        
        messages.append({"role": "user", "content": input.message})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        answer = llm(prompt)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Failed processing of {input}.",exc_info=e)
        return {"answer": ""}


