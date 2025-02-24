# Use an official Python runtime as a parent image.
FROM --platform=linux/amd64 python:3.11-slim

# Install system dependencies required by some packages.
RUN apt-get update && apt-get install -y \
    libglew-dev \
    libgl1-mesa-dev \
    libglfw3 \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container.
WORKDIR /usr/src/app

# Set required environment variables for MuJoCo and PyOpenGL.
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Install pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy the package files and install dependencies via pyproject.toml
COPY pyproject.toml .
COPY pi_incubator pi_incubator
RUN pip install .[dev]  # Install package along with dev dependencies

# Copy the rest of your application code (Metaflow, scripts, configs)
COPY . .

# # Default command (adjust as needed)
# CMD ["python", "ppo_metaflow.py", "run", "--with", "kubernetes"]
