FROM python:3.12-slim-bookworm

# PyGObject + gstreamer stack for the reachy_mini SDK's remote media path.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ pkg-config python3-dev curl cmake \
    libgirepository1.0-dev gobject-introspection libcairo2-dev \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-nice \
    gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 gir1.2-gst-plugins-bad-1.0 \
    && rm -rf /var/lib/apt/lists/*

# sphn publishes no linux-aarch64 wheels; build it from source once, here,
# where we control the toolchain (never on the robot). The cmake policy pin
# works around audiopus_sys's ancient CMakeLists.
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}" \
    CMAKE_POLICY_VERSION_MINIMUM=3.5

WORKDIR /app
COPY pyproject.toml README.md ./
COPY kuroko/ kuroko/
COPY probe/ probe/
RUN pip install --no-cache-dir -e .

# zenoh scouting + webrtc want host networking:
#   docker run --network host --name kuroko kuroko:latest
CMD ["python", "-m", "kuroko"]
