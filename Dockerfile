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

# The reachy_mini SDK's remote media path uses webrtcsrc from gst-plugins-rs,
# which no distro packages. Build it once here; the robot never compiles anything.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev libglib2.0-dev libssl-dev nasm \
    && rm -rf /var/lib/apt/lists/*
RUN cargo install cargo-c --locked
RUN git clone --depth 1 -b 0.12.11 \
      https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git /tmp/gpr \
    && cd /tmp/gpr \
    && cargo cbuild -p gst-plugin-webrtc -p gst-plugin-rtp --release \
    && cargo cinstall -p gst-plugin-webrtc -p gst-plugin-rtp --release --prefix /usr \
    && rm -rf /tmp/gpr /root/.cargo/registry

WORKDIR /app
COPY pyproject.toml README.md ./
COPY kuroko/ kuroko/
COPY probe/ probe/
RUN pip install --no-cache-dir -e .

# zenoh scouting + webrtc want host networking:
#   docker run --network host --name kuroko kuroko:latest
CMD ["python", "-m", "kuroko"]
