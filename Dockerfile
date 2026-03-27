FROM us-central1-docker.pkg.dev/rembrand-ai/rembrand-repo/research-cuda12:latest

# Install repo and dependencies
WORKDIR /workspace
COPY . /workspace/glue-factory
RUN pip install --no-cache-dir -e /workspace/glue-factory

# Pre-download pretrained weights so the job has no external dependency at runtime
RUN mkdir -p /root/.cache/torch/hub/checkpoints
# SuperPoint weights
RUN wget -q -O /root/.cache/torch/hub/checkpoints/superpoint_v6_from_tf.pth \
    https://github.com/rpautrat/SuperPoint/raw/master/weights/superpoint_v6_from_tf.pth
# LightGlue pretrained on SuperPoint (v0.1_arxiv)
RUN wget -q -O /root/.cache/torch/hub/checkpoints/superpoint_lightglue_v0-1_arxiv.pth \
    https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth

WORKDIR /workspace/glue-factory
