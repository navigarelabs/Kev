# Kev RunPod Serverless worker. Build context = this folder (kev_service/), NOT the repo root.
#   Model on a network volume (default):  docker build -t <registry>/kev-worker:v1 .
#   Model baked into the image (~20 GB):  docker build --build-arg BAKE_MODEL=1 -t <registry>/kev-worker:v1-baked .
# Same base image as the test pod, so the torch / triton / flash-linear-attention combination is the one that worked.
FROM runpod/pytorch:1.5.0-cu1281-torch280-ubuntu2404

WORKDIR /app
ENV PYTHONUNBUFFERED=1
 # The base image ships an Ubuntu-installed cryptography (41.x) that pip cannot uninstall; the runpod SDK needs >=50.                                                                
 # Install the newer one over it first (only this package, so torch etc. are untouched).                                                                                             
RUN pip install --no-cache-dir --ignore-installed "cryptography>=50.0.2"   

# Keep the image's own CUDA build of torch + triton (reinstalling from PyPI can break the CUDA match),
# then install what run_kev.py needs plus the RunPod Serverless SDK
RUN python -c "from importlib.metadata import version; print('\n'.join(f'{p}=={version(p)}' for p in ('torch', 'triton')))" > /tmp/constraints.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt -U "transformers>=5.17,<6" "peft>=0.21" einops huggingface_hub runpod

# flash-linear-attention: GPU kernels for Kev-9B's DeltaNet layers (big speed-up). Same version search as setup_runpod.sh:
# take the newest release that imports with this image's Triton.
RUN for SPEC in "" "<0.5" "<0.4.2" "<0.4" "<0.3.2"; do \
      pip uninstall -y -q flash-linear-attention fla-core >/dev/null 2>&1 || true; \
      if pip install --no-cache-dir -q -c /tmp/constraints.txt "flash-linear-attention${SPEC}" \
         && python -c "import fla, fla.layers, fla.ops"; then echo "flash-linear-attention OK (${SPEC:-latest})"; exit 0; fi; \
    done; \
    echo "No flash-linear-attention release works with this image's Triton" >&2; exit 1

COPY run_kev.py handler.py download_model.py ./

# Optional: bake the model into the image (no network volume needed, but a ~20 GB larger image)
ARG BAKE_MODEL=0
RUN if [ "$BAKE_MODEL" = "1" ]; then HF_HOME=/models/hf-cache python download_model.py; fi

CMD ["python", "-u", "handler.py"]
