"""
HTTP wrapper around run_kev.py so the pipeline can call Kev per earnings call, the same way it calls Qwen on Bedrock.
Runs on the RunPod GPU pod; it is NOT part of the pipeline image's runtime (nothing in the pipeline imports it).

Setup on the pod (from the folder holding run_kev.py, after `bash setup_runpod.sh`):
    copy this file next to run_kev.py
    pip install fastapi "uvicorn[standard]"
    export KEV_API_KEY=<long random string, same as in the KEV_RUNPOD_ENDPOINT secret>
    export HF_HOME=/workspace/hf-cache
    nohup uvicorn kev_server:app --host 0.0.0.0 --port 8000 > kev_server.log 2>&1 &

Endpoints:
    GET  /health      -> {"status": "ok", ...}        (no key needed)
    POST /v1/score    -> body: one request or a list of requests in the requests.json format
                         header: Authorization: Bearer <KEV_API_KEY>
                         returns a list of {"id", "answers", "state_tokens", "seconds"}

Settings (environment variables): KEV_API_KEY (required), KEV_CHECKPOINT (default jaredpalmer/kev-9b),
KEV_DEVICE (default cuda), KEV_DTYPE (bf16 | fp32, default bf16).
"""
import os
import sys
import time
import threading
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import Body, FastAPI, Header, HTTPException

import run_kev

API_KEY = os.environ.get("KEV_API_KEY")
CHECKPOINT = os.environ.get("KEV_CHECKPOINT", "jaredpalmer/kev-9b")
DEVICE = run_kev.pick_device(os.environ.get("KEV_DEVICE", "cuda"))
DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("KEV_DTYPE", "bf16")]

if not API_KEY:
    sys.exit("Set KEV_API_KEY before starting the server: the RunPod URL is public.")

# Load once at start-up; every request reuses the model
t0 = time.perf_counter()
tok, lm, head = run_kev.load_kev(CHECKPOINT, DEVICE, DTYPE)
print(f"Kev loaded in {time.perf_counter() - t0:.0f} s", file=sys.stderr)

# One GPU, one model: requests from parallel pipeline tasks queue here and run one at a time
gpu_lock = threading.Lock()
app = FastAPI(title="Kev tone server")


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": CHECKPOINT, "device": DEVICE}


@app.post("/v1/score")
def score(
        payload: Union[Dict[str, Any], List[Dict[str, Any]]] = Body(...),
        authorization: Optional[str] = Header(None)
):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    kev_requests = payload if isinstance(payload, list) else [payload]
    try:
        records = [run_kev.to_record(req) for req in kev_requests]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    results = []
    with gpu_lock:
        for req, (rec, meta) in zip(kev_requests, records):
            t1 = time.perf_counter()
            try:
                probs, n_state = run_kev.score(tok, lm, head, rec, DEVICE)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise HTTPException(status_code=500, detail=f"GPU out of memory on {req.get('id')}: state too long")
            results.append({
                "id": req.get("id"),
                "answers": run_kev.to_answers(probs, meta),
                "state_tokens": n_state,
                "seconds": round(time.perf_counter() - t1, 2)})
            print(f"{req.get('id')}: {n_state} state tokens, {results[-1]['seconds']} s", file=sys.stderr)
    return results
