FROM python:3.12-slim-bookworm

WORKDIR /app

# System deps for OpenCV, moviepy, etc.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6 /uv /uvx /bin/

# Copy project files
COPY pyproject.toml uv.lock ./
COPY api_server.py ./
COPY configs/ ./configs/
COPY agents/ ./agents/
COPY interfaces/ ./interfaces/
COPY pipelines/ ./pipelines/
COPY tools/ ./tools/
COPY utils/ ./utils/
COPY main_idea2video.py main_script2video.py ./

# Install deps
RUN uv sync --no-dev

# Output directory
RUN mkdir -p /app/output

ENV PORT=8000
ENV VIMAX_OUTPUT_DIR=/app/output
ENV VIMAX_JOBS_FILE=/app/jobs.json
ENV VIMAX_CONFIG=/app/configs/idea2video.yaml

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
