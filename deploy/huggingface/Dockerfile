# DriftGuard on Hugging Face Spaces -- API and dashboard in one container.
#
# A Space exposes a single public port (7860). The dashboard serves it; the
# API listens on 127.0.0.1:8000 inside the container only, so visitors can use
# the dashboard but cannot call the API directly.
#
# Assembled into a Space by scripts/make_space.py -- see README "Deployment".
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8

# Install CPU-only torch first: the default wheel pulls ~2GB of CUDA libraries
# that this project never uses.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Spaces run the container as uid 1000, so everything the app writes to
# (the SQLite data dir, the model cache) must belong to that user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    DRIFTGUARD_DATA_DIR=/home/user/data \
    API_BASE_URL=http://127.0.0.1:8000 \
    LLM_PROVIDER=mock
WORKDIR /home/user/app

# Bake the sentence-transformer into the image so the first probe run does not
# have to download it.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=user driftguard ./driftguard
COPY --chown=user dashboard ./dashboard
COPY --chown=user start.sh ./start.sh
RUN mkdir -p /home/user/data

EXPOSE 7860
CMD ["bash", "start.sh"]
