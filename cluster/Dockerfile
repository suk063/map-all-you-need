FROM python:3.12-slim-bookworm
WORKDIR /app
COPY requirements.txt ./
COPY cluster/image-setup.sh /tmp/image-setup.sh
RUN bash /tmp/image-setup.sh && rm /tmp/image-setup.sh
COPY benchmark ./benchmark
COPY cluster ./cluster
COPY tests ./tests
COPY pytest.ini ./
ENV MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics DINO_SOURCE=/opt/dinov3
ARG CODE_REVISION=unknown
LABEL org.opencontainers.image.revision=$CODE_REVISION
RUN printf '%s\n' "$CODE_REVISION" > /app/code-revision.txt
CMD ["python", "-m", "cluster.worker"]
