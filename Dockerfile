# Use PyTorch devel image with CUDA support (includes compiler for mmcv build)
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

WORKDIR /app

# Install system dependencies (ninja for faster build)
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install mmengine first
RUN pip install --no-cache-dir openmim mmengine

# Build mmcv from source with CUDA ops for sm_120 (RTX 5090)
RUN git clone --depth 1 --branch v2.1.0 https://github.com/open-mmlab/mmcv.git /tmp/mmcv && \
    cd /tmp/mmcv && \
    MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="12.0" pip install -e . && \
    rm -rf /tmp/mmcv/.git

# Install mmdet and mmpose
RUN pip install --no-cache-dir mmdet==3.2.0 mmpose==1.3.2

# Rebuild xtcocotools from source to ensure numpy compatibility
RUN pip uninstall -y xtcocotools && \
    pip install --no-cache-dir --no-binary :all: xtcocotools

# Copy project files
COPY pyproject.toml README.md demo_gradio.py ./
COPY src/anime_face_detector ./anime_face_detector/

# Install the package (without OpenMMLab deps, already installed)
RUN pip install --no-cache-dir -e . --no-deps && \
    pip install --no-cache-dir opencv-python-headless

# Install Gradio for web demo
RUN pip install --no-cache-dir gradio

# Set default command
CMD ["python", "-c", "from anime_face_detector import create_detector; print('Import successful')"]
