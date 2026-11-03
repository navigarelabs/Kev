"""
RunPod Serverless worker for Kev (replaces kev_server.py in production).

Each worker loads Kev once when it starts (the cold start), then answers jobs until RunPod shuts it down after the
endpoint's idle timeout. One job at a time per worker; RunPod queues the rest or starts another worker.

Job input (sent by kev_tone.py to https://api.runpod.ai/v2/<endpoint_id>/run):
    {"input": {"requests": [<request in the requests.json format>, ...]}}
Job output:
    [{"id", "answers", "state_tokens", "seconds"}, ...]       or {"error": "..."} (RunPod marks the job FAILED)

Where the model is read from (first that applies):
    HF_HOME env var  >  /runpod-volume/hf-cache (network volume attached)  >  /models/hf-cache (baked into the image)

Settings (endpoint environment variables): KEV_CHECKPOINT (default jaredpalmer/kev-9b), KEV_DTYPE (bf16 | fp32).
"""
import os
import sys
import time

# HF_HOME must be set before transformers / huggingface_hub are imported (run_kev imports them inside load_kev)
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/runpod-volume/hf-cache" if os.path.isdir("/runpod-volume") else "/models/hf-cache"

import torch
import runpod

import run_kev

CHECKPOINT = os.environ.get("KEV_CHECKPOINT", "jaredpalmer/kev-9b")
DEVICE = "cuda"
DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("KEV_DTYPE", "bf16")]

# Cold start: load once per worker, reused by every job this worker handles
print(f"HF_HOME: {os.environ['HF_HOME']}", file=sys.stderr)
t0 = time.perf_counter()
tok, lm, head = run_kev.load_kev(CHECKPOINT, DEVICE, DTYPE)
print(f"Kev loaded in {time.perf_counter() - t0:.0f} s on {torch.cuda.get_device_name(0)}", file=sys.stderr)


def handler(job):
    kev_requests = (job.get("input") or {}).get("requests")
    if not isinstance(kev_requests, list) or not kev_requests:
        return {"error": "input.requests must be a non-empty list of Kev requests"}

    try:
        records = [run_kev.to_record(req) for req in kev_requests]
    except ValueError as e:
        return {"error": f"Invalid request: {e}"}

    results = []
    for req, (rec, meta) in zip(kev_requests, records):
        t1 = time.perf_counter()
        try:
            probs, n_state = run_kev.score(tok, lm, head, rec, DEVICE)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {"error": f"GPU out of memory on {req.get('id')}: state too long for this GPU"}
        results.append({
            "id": req.get("id"),
            "answers": run_kev.to_answers(probs, meta),
            "state_tokens": n_state,
            "seconds": round(time.perf_counter() - t1, 2)})
        print(f"{req.get('id')}: {n_state} state tokens, {results[-1]['seconds']} s", file=sys.stderr)
    return results


runpod.serverless.start({"handler": handler})
