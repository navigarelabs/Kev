"""
Download Kev (LoRA + head) and the base model it is pinned to into HF_HOME, without loading anything onto a GPU.

Used two ways:
  - Network volume: run once on any pod with the volume attached
        HF_HOME=/runpod-volume/hf-cache python download_model.py      (on a Serverless volume)
        HF_HOME=/workspace/hf-cache     python download_model.py      (the same volume seen from a normal pod)
  - Baked into the image: the Dockerfile runs it at build time when BAKE_MODEL=1

Settings: KEV_CHECKPOINT (default jaredpalmer/kev-9b).
"""
import os
import sys

import torch
from huggingface_hub import snapshot_download

CHECKPOINT = os.environ.get("KEV_CHECKPOINT", "jaredpalmer/kev-9b")

repo, _, revision = CHECKPOINT.partition("@")
print(f"HF_HOME    : {os.environ.get('HF_HOME', '(default)')}", file=sys.stderr)
print(f"Adapter    : {CHECKPOINT}", file=sys.stderr)
path = snapshot_download(repo, revision=revision or None,
                         allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"])

# head.pt pins the base model and its revision (same as run_kev.load_kev)
meta = torch.load(os.path.join(path, "head.pt"), map_location="cpu", weights_only=True)
print(f"Base model : {meta['base']} @ {meta.get('base_revision')}", file=sys.stderr)
snapshot_download(meta["base"], revision=meta.get("base_revision"),
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
print("Done.", file=sys.stderr)
