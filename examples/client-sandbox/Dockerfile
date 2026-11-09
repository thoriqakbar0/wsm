FROM python:3.12-slim
RUN pip install --no-cache-dir torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers==5.22.2 safetensors==0.8.0
COPY packages/python/src/wsm/binary64.py packages/python/src/wsm/target_worker.py /opt/weightless/wsm/
RUN touch /opt/weightless/wsm/__init__.py
ENV PYTHONPATH=/opt/weightless PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HOME=/tmp
WORKDIR /tmp
