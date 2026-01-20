# MediaPipe Pose Estimator Docker Container for Jetson
FROM nvcr.io/nvidia/l4t-base:r36.2.0

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    pkg-config \
    libopencv-dev \
    python3-opencv \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libprotobuf-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install Python dependencies (excluding opencv to keep system opencv with GStreamer)
# Use numpy<2 for compatibility with system OpenCV
# Use protobuf 4.x for mediapipe 0.10.x compatibility
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir "numpy<2" "protobuf>=4.21,<5" absl-py attrs flatbuffers matplotlib sounddevice && \
    pip3 install --no-cache-dir mediapipe --no-deps

# Set working directory
WORKDIR /app

# Copy application files
COPY mediapipe_pose.py .
COPY mediapipe_pose_gstreamer.py .
COPY mediapipe_pose_data.py .

# Set display environment variable
ENV DISPLAY=:0

# Default command
CMD ["python3", "mediapipe_pose.py"]
