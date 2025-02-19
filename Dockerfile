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

# # Copy the environment file (if you use it for local settings)
# COPY .env .env

# Set required environment variables for MuJoCo and PyOpenGL.
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Copy requirements.txt and install pip dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code.
COPY . .

# # Set the default command.
# # Adjust this entrypoint as needed for your use case.
# CMD ["python", "ppo_metaflow.py", "run", "--with", "kubernetes"]
