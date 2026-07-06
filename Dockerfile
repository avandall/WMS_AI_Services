FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Copy the AI Service package definition
COPY pyproject.toml ./

# Copy shared-utils from nested submodule
COPY Libraries/shared-utils ./Libraries/shared-utils

# Copy source code files so hatchling can register package paths
COPY src ./src

# Install virtualenv and package dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv && uv pip install -e .

# Copy training models and scripts
COPY training ./training

CMD ["ai-grpc"]
