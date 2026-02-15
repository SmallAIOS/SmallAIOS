# SmallAIOS Container Build (multi-arch: linux/amd64, linux/arm64)
# Usage: docker buildx build --platform linux/amd64,linux/arm64 -t smallaios .
# GPU:   docker buildx build --platform linux/amd64 --build-arg ENABLE_GPU=1 -t smallaios-gpu .

FROM rustlang/rust:nightly-slim AS builder

ARG TARGETARCH

RUN case "$TARGETARCH" in \
      arm64) RUST_TARGET=aarch64-unknown-linux-musl ;; \
      *)     RUST_TARGET=x86_64-unknown-linux-musl ;; \
    esac && \
    echo "$RUST_TARGET" > /rust_target

RUN apt-get update && apt-get install -y musl-tools && rm -rf /var/lib/apt/lists/*
RUN rustup target add "$(cat /rust_target)"

WORKDIR /app
COPY . .

ARG ENABLE_GPU=0
RUN RUST_TARGET=$(cat /rust_target) && \
    if [ "$ENABLE_GPU" = "1" ]; then \
      cargo build --release --target "$RUST_TARGET" \
        -p smallaios-container --features nvidia_gpu; \
    else \
      cargo build --release --target "$RUST_TARGET" \
        -p smallaios-container; \
    fi && \
    cp "/app/target/${RUST_TARGET}/release/smallaios-container" /app/smallaios

FROM scratch
COPY --from=builder /app/smallaios /smallaios
ENTRYPOINT ["/smallaios"]
