FROM us-central1-docker.pkg.dev/rembrand-ai/rembrand-repo/research-cuda12:latest

# Install repo and dependencies
WORKDIR /workspace
COPY . /workspace/glue-factory
RUN pip install --no-cache-dir -e /workspace/glue-factory

WORKDIR /workspace/glue-factory
