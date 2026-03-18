FROM gcr.io/deeplearning-platform-release/pytorch-gpu

RUN mkdir /workdir
WORKDIR /workdir

# Install Python dependencies
COPY xcloud/requirements.txt /workdir/requirements.txt
RUN python3 -m pip --no-cache-dir install -r requirements.txt

# Copy source code
COPY clsplats /workdir/clsplats
COPY configs /workdir/configs
COPY simple_knn /workdir/simple_knn
COPY pyproject.toml /workdir/pyproject.toml
COPY README.md /workdir/README.md

# Install package
RUN python3 -m pip install -e .

# Pre-download DINOv2 model weights into the image so XCloud (no internet) can load them.
# TORCH_HOME controls where torch.hub caches models.
ENV TORCH_HOME=/workdir/torch_cache
RUN python3 -c "\
import torch; \
torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True); \
print('DINOv2 downloaded successfully.')"

# Pre-download Depth-Anything V2 from HuggingFace.
# HF_HOME controls where transformers caches models.
ENV HF_HOME=/workdir/hf_cache
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('depth-anything/Depth-Anything-V2-Small-hf'); \
print('Depth-Anything downloaded successfully.')"

# Copy entrypoint
COPY xcloud/entrypoint.py /workdir/entrypoint.py
RUN chmod -R 755 /workdir
RUN chmod +x /workdir/entrypoint.py

ENTRYPOINT ["python3", "/workdir/entrypoint.py"]
