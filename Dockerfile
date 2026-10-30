# GPU-capable JAX installation uses CUDA 12 wheels; the host provides the NVIDIA driver.
FROM python:3.11-slim-bookworm
WORKDIR /app
ARG SOURCE_REVISION=unknown
ENV SOURCE_REVISION=${SOURCE_REVISION}
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
RUN pip install --no-cache-dir 'jax[cuda12]==0.9.0.1' 'numpy==2.3.5'
COPY src/ src/
COPY train.py evaluate.py sweep.py campaign.py ./
COPY experiments/ experiments/
COPY tests/ tests/
COPY deploy/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
EXPOSE 8080
CMD ["/app/entrypoint.sh"]
