"""
scripts/launch_kaggle_judge.py
==============================
Standalone script to launch a FastAPI + vLLM judge on a Kaggle notebook.
Designed to be run on a secondary Kaggle T4 instance to offload judging
during end-to-end evaluation.

Setup in your Kaggle Notebook:
-----------------------------
1. Turn ON Internet and GPU (T4 x1 or T4 x2).
2. Install dependencies:
   !pip install -q vllm pyngrok fastapi uvicorn
3. Set your Ngrok Auth Token in Kaggle Secrets as "NGROK_TOKEN"
   (Create a free account at ngrok.com to get a token)
4. Run this script!

Usage:
------
Copy the Ngrok URL printed in the output and run your eval:
python scripts/run_e2e_eval.py --judge-url <NGROK_URL>
"""

import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pyngrok import ngrok

try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    NGROK_TOKEN = secrets.get_secret("NGROK_TOKEN")
    ngrok.set_auth_token(NGROK_TOKEN)
except Exception:
    print("WARNING: Could not load NGROK_TOKEN from Kaggle Secrets.")
    print("If ngrok fails, please set it manually using: ngrok.set_auth_token('...')")

app = FastAPI()

# ---------------------------------------------------------
# Configuration
# For 1x T4: Use a 3B model (fp16) or 7B AWQ/GPTQ
# For 2x T4: Use a 7B/8B model with tensor_parallel_size=2
# ---------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  
TENSOR_PARALLEL_SIZE = 1  # Set to 2 if using T4x2

print(f"Loading {MODEL_ID} in vLLM...")
from vllm import LLM, SamplingParams
llm = LLM(
    model=MODEL_ID, 
    max_model_len=4096, 
    dtype="half", 
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    enforce_eager=True # Recommended for T4 to save memory
)

class JudgeRequest(BaseModel):
    topics: List[str]
    answer: str

@app.post("/judge")
def judge(req: JudgeRequest):
    readable_topics = [t.replace("_", " ").title() for t in req.topics]
    
    prompt = (
        "You are an evaluator grading an AI's response.\n"
        f"Goal: Does the answer address all of these topics: {readable_topics}?\n"
        "Example:\n"
        "Topics: ['Disease Pest', 'Market Price']\n"
        "Answer: The leaf shows rust. You should spray fungicide. I cannot help with market prices.\n"
        "Decision: yes\n\n"
        f"Topics: {readable_topics}\n"
        f"Answer: {req.answer}\n"
        "Decision:"
    )
    
    sampling_params = SamplingParams(temperature=0.1, max_tokens=10)
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    res = outputs[0].outputs[0].text.strip().lower()
    
    verdict = "yes" if "yes" in res else "no"
    return {"verdict": verdict, "raw_response": res}

if __name__ == "__main__":
    port = 8000
    try:
        public_url = ngrok.connect(port).public_url
        print("\n" + "="*70)
        print(f"🚀 JUDGE API IS LIVE AT: {public_url}")
        print(f"👉 Use this in your eval: --judge-url {public_url}")
        print("="*70 + "\n")
    except Exception as e:
        print(f"Ngrok connection failed: {e}")
        print("Falling back to local-only mode.")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
