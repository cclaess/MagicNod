FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get update --fix-missing -y

RUN apt-get install -y ffmpeg libsm6 libxrender1 libxtst6 zip 

RUN apt-get install -y p7zip-full

# Library components for av
RUN apt-get install -y \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev


RUN apt-get install -y python3 python3-pip git python3-dev pkg-config htop wget

# VS code
RUN apt-get install wget gpg -y
RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
RUN install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
RUN sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
RUN rm -f packages.microsoft.gpg

RUN apt install apt-transport-https -y
RUN apt update
RUN apt install code -y # or code-insiders

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio
RUN pip3 install torch-uncertainty
RUN pip3 install pytorch-lightning lightly
RUN pip3 install numpy pandas scikit-image scikit-learn scipy opencv-python
RUN pip3 install matplotlib seaborn
RUN pip3 install nibabel SimpleITK monai wandb
RUN pip3 install pytz python-dateutil h5py openpyxl
RUN pip3 install transformers huggingface_hub[cli] datasets

# dependencies for caetas/GenerativeZoo
RUN pip3 install einops Flask gitdb idna loguru mdurl medmnist ml_collections monai-generative ninja oauthlib python-dotenv s3fs sympy toml tqdm waitress pydicom lpips accelerate bitsandbytes
RUN pip3 install git+https://github.com/huggingface/diffusers
RUN pip3 install git+https://github.com/jonbarron/robust_loss_pytorch

WORKDIR /app/script