# Stage 1: Base image with common dependencies
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

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
    git \
    git-lfs \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
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

# Install comfy-cli + dependencies needed by it to install ComfyUI
RUN uv pip install comfy-cli pip setuptools wheel

#Install pytorch and cuda wheel
RUN uv pip install --no-cache torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --version 0.3.43

# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
RUN uv pip install runpod requests websocket-client

# Add script to install custom nodes
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

# Prevent pip from asking for confirmation during uninstall steps in custom nodes
ENV PIP_NO_INPUT=1
# install custom nodes using comfy-cli
RUN comfy-node-install ComfyUI-WanVideoWrapper ComfyUI-VideoHelperSuite cg-use-everywhere ComfyUI_JPS-Nodes ComfyUI-Frame-Interpolation ComfyUI-Easy-Use

# Copy helper script to switch Manager network mode at container start
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# Download required models
WORKDIR /comfyui/models

# Create model directories
# RUN mkdir -p checkpoints clip clip_vision vae unet diffusion_models text_encoders upscale_models

# Declare build argument for HuggingFace token (must be before any conditional downloads)
ARG HUGGINGFACE_ACCESS_TOKEN=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

# Download CLIP and text_encoder models
RUN wget -O clip_vision/clip_vision_h.safetensors "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"
RUN wget -O text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors"

# Download upscale models
RUN wget -O upscale_models/4xLSDIR.pth "https://github.com/Phhofm/models/raw/main/4xLSDIR/4xLSDIR.pth"

# Download Lora models
RUN wget -O loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"
RUN wget -O loras/BouncyWalkV01.safetensors "https://huggingface.co/KinkSociety1/KS-Wan-BouncyWalk-LoRA/resolve/main/BouncyWalkV01.safetensors"
RUN wget -O loras/Su_Bounce_Ep50.safetensors "https://huggingface.co/yeqiu168182/Su_Bounce_Ep50/resolve/main/Su_Bounce_Ep50.safetensors"
RUN wget -O loras/sh4rpn3ss_e18.safetensors "https://huggingface.co/minaiosu/Alissonerdx/resolve/main/sh4rpn3ss_e18.safetensors"

# Download Wan VAE model
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O vae/Wan2_1_VAE_bf16.safetensors "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"

# Download Wan2.1 14b t2v base model
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O diffusion_models/wan2.1_t2v_14B_bf16.safetensors "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors"

# # Download Wan2.1 14b 480p i2v base model
# RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors"

# Go back to the root
WORKDIR /

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