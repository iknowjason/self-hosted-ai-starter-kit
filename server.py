from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Model ID
MODEL_ID = "fdtn-ai/Foundation-Sec-8B-Instruct"

# Load API key from env var
API_KEY = os.environ.get("FOUNDATION_API_KEY", "changeme")

print("Loading model. This could take a bit of time for first time running of script.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16, 
    device_map="auto"
)

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.7

def verify_api_key(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.post("/generate")
async def generate_text(req: GenerateRequest, auth: None = Depends(verify_api_key)):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}
