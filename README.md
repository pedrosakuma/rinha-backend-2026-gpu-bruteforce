# Rinha Backend 2026 GPU brute force sandbox

Offline microbench for issue #1: test whether exact brute-force k-NN over
14-dimensional vectors can be competitive when the reference set stays resident
in GPU memory.

The first implementation uses Rust + `wgpu` with WGSL compute. It generates a
deterministic reference set shaped like the official one, quantizes reference
vectors to signed 16-bit values with 16 physical dimensions (14 real dimensions
plus padding), keeps them in persistent GPU buffers, copies one quantized query
vector per iteration, computes distances in parallel, reduces each workgroup to a
local top-5 on the GPU, copies only those block candidates back, then merges the
final exact quantized top-5 and looks up bit-packed labels on CPU.

## Competition surface and rules

The official contact surface is intentionally small:

- `GET /ready` returns any `2xx` when the API instance is ready.
- `POST /fraud-score` receives one transaction payload and returns
  `{"approved":bool,"fraud_score":number}`.
- The public port is `9999`, exposed by a load balancer.

The official architecture constraints matter for this GPU experiment:

- at least one load balancer plus two API instances;
- load balancer must be round-robin only, with no fraud logic;
- `docker-compose.yml` submission, public `linux-amd64` images;
- total declared resources across all services: at most `1 CPU` and `350 MB`;
- network mode must be `bridge`; `host` and `privileged` are not allowed.

The benchmark prints a `competition_fit` line estimating the minimum persistent
buffers for two API instances plus a load-balancer memory budget. References are
stored as packed `i16` and labels are bit-packed, which makes the generated 3M
reference set fit the memory budget in this prototype before runtime/container
overhead.

## Eval topology

This repository now includes a submission-shaped topology:

- `Dockerfile` builds a `linux-amd64` Rust binary and a Debian runtime image with
  `libvulkan1` and Mesa Vulkan drivers.
- The API image embeds `resources/references.json.gz`, `resources/mcc_risk.json`
  and `resources/normalization.json`; there is no runtime volume for official
  resources.
- `docker-compose.yml` runs two API instances plus an Nginx round-robin load
  balancer on port `9999`.
- `Dockerfile.lb` embeds `nginx.conf` in the load-balancer image; there is no
  runtime config volume.
- API limits are `0.40 CPU / 160M` each; LB limit is `0.20 CPU / 30M`, totaling
  `1.00 CPU / 350M`.
- The API containers map `/dev/dri:/dev/dri` without `privileged` or `host`
  networking so the Mac Mini Intel/Mesa Vulkan adapter can be tested.

Before building the API image, put the official resource files in the build
context:

```text
resources/references.json.gz
resources/mcc_risk.json
resources/normalization.json
```

Then run:

```bash
docker build -t pedrosakuma/rinha-gpu-bruteforce:latest .
docker build -t pedrosakuma/rinha-gpu-bruteforce-lb:latest -f Dockerfile.lb .
docker compose up
curl -s http://localhost:9999/ready
```

For actual submission, publish the image referenced by `docker-compose.yml`
(`pedrosakuma/rinha-gpu-bruteforce:latest`) and the LB image
(`pedrosakuma/rinha-gpu-bruteforce-lb:latest`) as public `linux-amd64` images,
or change the compose file to the image names you publish. The compose file
intentionally contains no `build:` and no resource/config volumes.

## Official participants entry

The official Rinha repository already has `participants/pedrosakuma.json` with
the CPU/.NET submission. Add this backend as a second entry with:

```bash
python3 scripts/add_official_entry.py ../rinha-de-backend-2026/participants/pedrosakuma.json
```

The script is idempotent and inserts:

```json
{
  "id": "pedrosakuma-gpu-bruteforce",
  "repo": "https://github.com/pedrosakuma/rinha-backend-2026-gpu-bruteforce"
}
```

Then open a PR against `zanfranceschi/rinha-de-backend-2026` with that updated
participants file. This repository also includes the required `info.json` and
MIT `LICENSE`.

## Important environment caveat

Local results are only valid for the local machine, driver, and adapter printed
by the benchmark. The target Rinha environment may be different, may not expose
`/dev/dri`, and may fall back to a software Vulkan adapter. Treat every reported
latency together with the printed adapter name, backend, device type, and driver.

Current target host for this investigation: Mac Mini Late 2014 running Ubuntu
24.04. On that machine, prefer testing the Vulkan backend first and confirm that
the printed adapter is the Intel iGPU/Mesa driver rather than a software fallback
such as `llvmpipe`.

## Run

Install Rust locally if needed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
```

On Ubuntu, native build tools are also required:

```bash
sudo apt-get update && sudo apt-get install -y build-essential pkg-config
```

Quick smoke run:

```bash
cargo run --release -- --references 65536 --queries 8 --warmup 2 --iterations 10
```

Target-host smoke run:

```bash
cargo run --release -- --backend vulkan --references 65536 --queries 8 --warmup 2 --iterations 10
```

HTTP surface smoke run:

```bash
cargo run --release -- --serve --backend vulkan --references 65536 --skip-validate
```

Then in another shell:

```bash
curl -s http://localhost:9999/ready
curl -s -X POST http://localhost:9999/fraud-score \
  -H 'content-type: application/json' \
  -d '{
    "id":"tx-123",
    "transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},
    "customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009","MERC-001"]},
    "merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},
    "terminal":{"is_online":false,"card_present":true,"km_from_home":13.7},
    "last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.8}
  }'
```

Full-shape run:

```bash
cargo run --release -- --backend vulkan --references 3000000 --queries 32 --warmup 5 --iterations 50
```

Official resource microbench:

```bash
cargo run --release -- \
  --backend vulkan \
  --references-path resources/references.json.gz \
  --mcc-risk-path resources/mcc_risk.json \
  --normalization-path resources/normalization.json \
  --require-resource-files \
  --skip-validate \
  --warmup 5 \
  --iterations 50
```

Useful flags:

```text
--backend auto|vulkan|metal|dx12|gl
--force-fallback-adapter
--skip-validate
--validate-queries N
--seed N
--serve
--listen 0.0.0.0:9999
--references-path resources/references.json.gz
--mcc-risk-path resources/mcc_risk.json
--normalization-path resources/normalization.json
--require-resource-files
--force-generate
--api-instances 2
--memory-limit-mb 350
--lb-memory-mb 30
```

## Output

The benchmark prints:

- local environment caveat and run configuration;
- selected adapter/backend/device type/driver;
- quantized layout (`i16_packed`), physical dimension count, and scale;
- workgroup count, number of candidates copied back, packed label size, and
  memory fit estimate for the competition topology;
- reference chunk count, useful because some Vulkan adapters cap a single
  storage-buffer binding at 128 MB;
- CPU/GPU correctness validation result unless skipped;
- `min`, `p50`, `p95`, `p99`, and `max` latency in microseconds;
- last query fraud decision using `fraud_score = frauds_in_top5 / 5` and
  `approved = fraud_score < 0.6`.

## Current scope

Implemented:

- reproducible generated dataset with 14 logical dimensions, packed `i16`
  references with 16 physical dimensions, and `fraud`/`legit` labels;
- streaming loader for official `resources/references.json.gz`;
- loading official `resources/mcc_risk.json` and `resources/normalization.json`;
- query generation that includes `-1` sentinels in dimensions 5 and 6 for a
  subset of queries;
- exact quantized CPU baseline for GPU validation plus f32 top-5/decision
  divergence counters;
- portable `wgpu` compute path with block-level GPU top-5 reduction;
- reference buffer chunking for adapters that cannot bind the full 3M x 14 f32
  dataset as a single storage buffer;
- final exact top-5 merge from block candidates;
- minimal HTTP mode for the official `GET /ready` and `POST /fraud-score`
  surface using the 14 official normalization dimensions;
- Docker/Nginx topology shaped like the official requirements.
- Docker images embed the official resources and Nginx config at build time.

Not implemented yet:

- public image publishing;
- concurrent/high-performance HTTP serving;
- target-host measurement with real `/dev/dri`/Intel adapter.

The `--serve` mode is still intentionally simple: it is single-process TCP and
will queue concurrent requests. The compose topology validates the official
surface and resource shape, but performance work is still needed before treating
it as a competitive submission.

## Current local analog result

On this local environment the selected adapter was `llvmpipe`, i.e. Vulkan over
CPU/software, not the Mac Mini Intel iGPU. With generated 3M references and the
current `i16_packed` layout:

```text
microbench: p50 ~= 178 ms, p95/p99 ~= 195 ms, errors=0
HTTP sequential: p50 ~= 180 ms, p95 ~= 222 ms, p99 ~= 228 ms, errors=0
HTTP concurrency=4: p50 ~= 727 ms, p95 ~= 761 ms, p99 ~= 776 ms, errors=0
competition_fit: 2 APIs + 30 MB LB ~= 217.4 MiB minimum persistent buffers, fits_limit=true
validation: quantized CPU/GPU matched; sampled f32 decision mismatches=0
```

This is not an official score because it does not use the official resources,
does not run the final two-API/LB container topology, and runs on a software
adapter locally. Quantization fixes the first memory blocker for this prototype,
but latency is still far above the 1 ms saturation point on the local software
adapter. The next meaningful measurement needs to happen on the Mac Mini target
host with the real Intel/Mesa adapter instead of `llvmpipe`.

## Infra reference notes

The `jairoblatt/rinha-2026-rust` repository is useful as an infra reference:
two API instances behind a tiny round-robin load balancer, resource split around
`0.4 CPU / 160 MB` per API plus `0.2 CPU / 30 MB` for the LB, and Unix sockets
to reduce loopback overhead. For this GPU sandbox, copying that shape directly is
not enough: the key open question is whether two API instances can each keep the
reference buffer resident without exceeding the 350 MB total limit or losing GPU
access under the official container restrictions.

If the full path cannot get close to the target p99 range on the target
environment, or if it needs to copy all 3M distances back per query, this GPU
approach should be considered non-competitive for the Rinha constraints.
