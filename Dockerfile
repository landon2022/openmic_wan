# Stage 1: Base image with common dependencies
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    ninja-build \
    aria2 \
    git \
    git-lfs \
    wget \
    vim \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    build-essential \
    gcc \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv (latest) using official installer and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

#Install pytorch and cuda wheel
# RUN uv pip install --no-cache torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --no-cache torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install comfy-cli + dependencies needed by it to install ComfyUI
RUN uv pip install comfy-cli pip packaging setuptools wheel pyyaml gdown triton

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --version 0.3.43

# Install sageattn
RUN pip install https://huggingface.co/landon2022/sageattn_wheel/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl

# Stage 2
FROM base AS final
# Make sure to use the virtual environment here too
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install opencv-python
# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
RUN uv pip install runpod==1.7.10 requests websocket-client

# Add script to install custom nodes
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

# Prevent pip from asking for confirmation during uninstall steps in custom nodes
ENV PIP_NO_INPUT=1
# install custom nodes using comfy-cli
# RUN comfy-node-install ComfyUI-WanVideoWrapper ComfyUI-VideoHelperSuite cg-use-everywhere ComfyUI_JPS-Nodes ComfyUI-Frame-Interpolation ComfyUI-Easy-Use ComfyLiterals

# Copy helper script to switch Manager network mode at container start
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# Download required models
WORKDIR /comfyui/models

# Create model directories
# RUN mkdir -p checkpoints clip clip_vision vae unet diffusion_models text_encoders upscale_models

# Declare build argument for HuggingFace token (must be before any conditional downloads)
# ARG HUGGINGFACE_ACCESS_TOKEN=Your_Huggingface_Token

# Download CLIP and text_encoder models
RUN wget -O clip_vision/clip_vision_h.safetensors "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"
RUN wget -O text_encoders/umt5-xxl-enc-bf16.safetensors "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"

# Download upscale models
RUN wget -O upscale_models/4xLSDIR.pth "https://github.com/Phhofm/models/raw/main/4xLSDIR/4xLSDIR.pth"

# Download Lora models
RUN wget -O loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"
RUN wget -O loras/BouncyWalkV01.safetensors "https://huggingface.co/KinkSociety1/KS-Wan-BouncyWalk-LoRA/resolve/main/BouncyWalkV01.safetensors"
RUN wget -O loras/Su_Bounce_Ep50.safetensors "https://huggingface.co/yeqiu168182/Su_Bounce_Ep50/resolve/main/Su_Bounce_Ep50.safetensors"
RUN wget -O loras/sh4rpn3ss_e18.safetensors "https://huggingface.co/minaiosu/Alissonerdx/resolve/main/sh4rpn3ss_e18.safetensors"

# Download Wan VAE model
RUN wget -O vae/Wan2_1_VAE_bf16.safetensors "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"

# Download Wan2.1 14b t2v base model
RUN wget -O diffusion_models/wan2.1_t2v_14B_bf16.safetensors "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors"

# # Download Wan2.1 14b 480p i2v base model
# RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors"

# Clone and install custom nodes
WORKDIR /comfyui/custom_nodes


RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && \
    pip install -r requirements.txt


RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt


RUN git clone https://github.com/chrisgoringe/cg-use-everywhere.git cg-use-everywhere


RUN git clone https://github.com/JPS-GER/ComfyUI_JPS-Nodes.git ComfyUI_JPS-Nodes


RUN git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git ComfyUI-Frame-Interpolation && \
    cd ComfyUI-Frame-Interpolation && \
    pip install -r requirements-no-cupy.txt


RUN git clone https://github.com/yolain/ComfyUI-Easy-Use.git ComfyUI-Easy-Use && \
    cd ComfyUI-Easy-Use && \
    pip install -r requirements.txt

RUN git clone https://github.com/M1kep/ComfyLiterals.git ComfyLiterals


# Download Frame_Interpolation model
WORKDIR /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife
RUN wget -O rife49.pth "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/rife49.pth"

# Go back to the root
WORKDIR /

RUN pip list --format=freeze
# Add application code and scripts
ADD src/start.sh handler.py test_input.json ./
RUN chmod +x /start.sh

# Set the default command to run when starting the container
CMD ["/start.sh"]

# # Stage 2: Download models
# FROM base AS downloader

# ARG HUGGINGFACE_ACCESS_TOKEN
# # Set default model type if none is provided
# ARG MODEL_TYPE=flux1-dev-fp8

# # Change working directory to ComfyUI
# WORKDIR /comfyui

# # Create necessary directories upfront
# RUN mkdir -p models/checkpoints models/vae models/unet models/clip

# # Download checkpoints/vae/unet/clip models to include in image based on model type
# RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
#       wget -q -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
#       wget -q -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
#       wget -q -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
#     fi

# RUN if [ "$MODEL_TYPE" = "sd3" ]; then \
#       wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
#     fi

# RUN if [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
#       wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
#       wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
#       wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
#       wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
#     fi

# RUN if [ "$MODEL_TYPE" = "flux1-dev" ]; then \
#       wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
#       wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
#       wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
#       wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
#     fi

# RUN if [ "$MODEL_TYPE" = "flux1-dev-fp8" ]; then \
#       wget -q -O models/checkpoints/flux1-dev-fp8.safetensors https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors; \
#     fi

# # Stage 3: Final image
# FROM base AS final

# # Copy models from stage 2 to the final image
# COPY --from=downloader /comfyui/models /comfyui/models