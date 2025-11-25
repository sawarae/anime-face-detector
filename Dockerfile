# Use PyTorch image with CUDA support
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install OpenMMLab packages with pre-built wheels for CUDA 12.1
RUN pip install --no-cache-dir openmim mmengine && \
    pip install --no-cache-dir mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html && \
    pip install --no-cache-dir mmdet==3.2.0 mmpose==1.3.2

# Copy project files
COPY pyproject.toml ./
COPY anime_face_detector ./anime_face_detector/

# Install the package (without OpenMMLab deps, already installed)
RUN pip install --no-cache-dir -e . --no-deps && \
    pip install --no-cache-dir numpy opencv-python-headless

# Set default command
CMD ["python", "-c", "from anime_face_detector import create_detector; print('Import successful')"]
