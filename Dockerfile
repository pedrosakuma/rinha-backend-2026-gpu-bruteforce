FROM rust:1.95-slim-bookworm AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM debian:bookworm-slim

LABEL org.opencontainers.image.source="https://github.com/pedrosakuma/rinha-backend-2026-gpu-bruteforce" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /src/target/release/rinha-gpu-bruteforce /app/rinha-gpu-bruteforce
COPY resources ./resources

RUN test -s /app/resources/references.json.gz \
    && test -s /app/resources/mcc_risk.json \
    && test -s /app/resources/normalization.json

ENV XDG_RUNTIME_DIR=/tmp

ENTRYPOINT ["/app/rinha-gpu-bruteforce"]
