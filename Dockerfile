# Use PyTorch devel image with CUDA support (includes compiler for mmcv build)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install OpenMMLab packages with pre-built wheels
RUN pip install --no-cache-dir openmim mmengine && \
    mim install mmcv==2.1.0 && \
    pip install --no-cache-dir mmdet==3.2.0 mmpose==1.3.2

# Copy project files
COPY pyproject.toml ./
COPY anime_face_detector ./anime_face_detector/

# Install the package (without OpenMMLab deps, already installed)
RUN pip install --no-cache-dir -e . --no-deps && \
    pip install --no-cache-dir numpy opencv-python-headless

# Set default command
CMD ["python", "-c", "from anime_face_detector import create_detector; print('Import successful')"]
