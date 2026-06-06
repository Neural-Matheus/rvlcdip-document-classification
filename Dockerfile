FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/workspace/.hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    MPLBACKEND=Agg

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-c", "import torch; print('container OK, cuda', torch.cuda.is_available())"]
