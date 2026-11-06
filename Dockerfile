# Kev RunPod Serverless worker. Build context = this folder (kev_service/ here; the repo root in the Kev worker repo).
#   Model baked into the image (default, ~30 GB image): no network volume, so RunPod can use GPUs in ANY data centre
#     (a network volume ties the endpoint to one data centre; EU-RO-1 was often out of 24/32 GB GPUs).
#   Model on a network volume instead:  docker build --build-arg BAKE_MODEL=0 ...   (and attach the volume)
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

# Bake the model into the image (~20 GB). Done BEFORE copying the worker code, so a later change to handler.py or
# run_kev.py reuses this cached layer instead of downloading the model again.
COPY download_model.py ./
ARG BAKE_MODEL=1
RUN if [ "$BAKE_MODEL" = "1" ]; then HF_HOME=/models/hf-cache python download_model.py; fi

COPY run_kev.py handler.py ./

CMD ["python", "-u", "handler.py"]
