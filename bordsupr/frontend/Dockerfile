FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
    git \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    libxcb1 libxcb-xfixes0 libxcb-shape0 libxcb-xinerama0 libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN python -m pip install huggingface_hub

# Install Real-ESRGAN after torch/torchvision are already present so basicsr's
# setup.py can resolve dependencies correctly.
RUN python -m pip install --no-cache-dir realesrgan basicsr
# Patch basicsr for torchvision 0.15+ compatibility (functional_tensor was removed)
RUN find /usr/local/lib/python*/site-packages/basicsr -name "*.py" -exec sed -i 's/torchvision.transforms.functional_tensor/torchvision.transforms.functional/g' {} +

# Install FastReID from source (no setup.py at root; copy package directly)
# Fallback: if git clone fails, the layer continues so the build does not break.
RUN python -m pip install --no-cache-dir yacs termcolor && \
    (cd /tmp && git clone --depth 1 https://github.com/JDAI-CV/fast-reid.git && \
     cp -r fast-reid/fastreid /usr/local/lib/python*/site-packages/ && \
     rm -rf fast-reid) || echo "FastReID git clone failed; skipping"

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
