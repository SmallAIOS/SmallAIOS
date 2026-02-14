# SmallAIOS Container Build
# Usage: docker build -t smallaios .
# GPU:   docker build --build-arg ENABLE_GPU=1 -t smallaios-gpu .

FROM rustlang/rust:nightly-slim AS builder

RUN apt-get update && apt-get install -y musl-tools && rm -rf /var/lib/apt/lists/*
RUN rustup target add x86_64-unknown-linux-musl

WORKDIR /app
COPY . .

ARG ENABLE_GPU=0
RUN if [ "$ENABLE_GPU" = "1" ]; then \
      cargo build --release --target x86_64-unknown-linux-musl \
        -p smallaios-container --features nvidia_gpu; \
    else \
      cargo build --release --target x86_64-unknown-linux-musl \
        -p smallaios-container; \
    fi

FROM scratch
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/smallaios-container /smallaios
ENTRYPOINT ["/smallaios"]
