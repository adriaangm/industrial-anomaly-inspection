# syntax=docker/dockerfile:1
FROM ros:jazzy-ros-base

# --- Dependencias de sistema: solo lo que cv_bridge y el runtime necesitan ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    ros-jazzy-vision-opencv \
    ros-jazzy-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# --- Venv aislado (system-site-packages para heredar rclpy y cv_bridge, ver fase 4) ---
RUN python3 -m venv --system-site-packages /opt/venv_inspection
ENV PATH="/opt/venv_inspection/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/src/industrial-anomaly-inspection

# Solo el grupo base (sin torch/anomalib): pyproject.toml y el paquete primero, para cachear la capa de pip
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e . "numpy>=1.26,<2"

# Paquetes ROS
WORKDIR /workspace
COPY ros2/anomaly_inspection_msgs ./src/anomaly_inspection_msgs
COPY ros2/anomaly_inspection_ros ./src/anomaly_inspection_ros

# python3 -m colcon: apt's colcon has a /usr/bin/python3 shebang, which would make the generated
# entry points ignore the venv (onnxruntime and anomaly_inspection live only there).
RUN . /opt/ros/jazzy/setup.sh && \
    python3 -m colcon build --packages-select anomaly_inspection_msgs anomaly_inspection_ros --symlink-install && \
    head -1 install/anomaly_inspection_ros/lib/anomaly_inspection_ros/inspection_node | grep -q "/opt/venv_inspection/bin/python3" \
    || (echo "ERROR: entry point shebang does not point to the venv" && exit 1)

# --- Modelo exportado: se monta como volumen en tiempo de ejecución, no se hornea en la imagen ---
VOLUME ["/models"]

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "anomaly_inspection_ros", "inspection_only.launch.py", \
     "model_path:=/models/patchcore_metal_nut.onnx", "device:=cuda"]
