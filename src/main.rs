use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{Read, Write};
use std::mem::size_of;
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use bytemuck::{Pod, Zeroable};
use clap::{Parser, ValueEnum};
use flate2::read::GzDecoder;
use serde::de::{DeserializeSeed, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use wgpu::util::DeviceExt;

const DIMS: usize = 14;
const PADDED_DIMS: usize = 16;
const PACKED_DIMS: usize = PADDED_DIMS / 2;
const TOP_K: usize = 5;
const WORKGROUP_SIZE: u32 = 256;
const MAX_REFERENCE_BUFFERS: usize = 2;
const QUANT_SCALE: f32 = 8192.0;

#[derive(Debug, Parser)]
#[command(
    version,
    about = "GPU brute-force exact k-NN microbench for Rinha Backend 2026"
)]
struct Args {
    /// Number of reference vectors to generate.
    #[arg(long, default_value_t = 3_000_000)]
    references: usize,

    /// Official references.json.gz path. Used when it exists unless --force-generate is set.
    #[arg(long, default_value = "resources/references.json.gz")]
    references_path: PathBuf,

    /// Force deterministic generated references even if --references-path exists.
    #[arg(long)]
    force_generate: bool,

    /// Fail if official resource files are missing instead of falling back to built-in defaults.
    #[arg(long)]
    require_resource_files: bool,

    /// Official mcc_risk.json path.
    #[arg(long, default_value = "resources/mcc_risk.json")]
    mcc_risk_path: PathBuf,

    /// Official normalization.json path.
    #[arg(long, default_value = "resources/normalization.json")]
    normalization_path: PathBuf,

    /// Number of distinct query vectors to generate and rotate through.
    #[arg(long, default_value_t = 32)]
    queries: usize,

    /// Warmup GPU queries before measurement.
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// Measured GPU query iterations.
    #[arg(long, default_value_t = 50)]
    iterations: usize,

    /// CPU/GPU correctness checks before the benchmark.
    #[arg(long, default_value_t = 1)]
    validate_queries: usize,

    /// Skip exact CPU validation.
    #[arg(long)]
    skip_validate: bool,

    /// Seed for deterministic generated references and queries.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// wgpu backend to request.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    backend: Backend,

    /// Allow software/fallback adapters when no hardware adapter is available.
    #[arg(long)]
    force_fallback_adapter: bool,

    /// Limit references only when wgpu selects a CPU adapter; useful for CI smoke tests without /dev/dri.
    #[arg(long, default_value_t = 0)]
    cpu_adapter_reference_limit: usize,

    /// Serve the competition HTTP surface instead of running the microbench.
    #[arg(long)]
    serve: bool,

    /// TCP address used by --serve. The official public port is exposed by the load balancer.
    #[arg(long, default_value = "0.0.0.0:9999")]
    listen: String,

    /// API instance count required by the competition architecture.
    #[arg(long, default_value_t = 2)]
    api_instances: usize,

    /// Load balancer memory budget used in the competition fit estimate.
    #[arg(long, default_value_t = 30.0)]
    lb_memory_mb: f64,

    /// Total docker-compose memory limit from the competition rules.
    #[arg(long, default_value_t = 350.0)]
    memory_limit_mb: f64,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Auto,
    Vulkan,
    Metal,
    Dx12,
    Gl,
}

impl Backend {
    fn to_wgpu(self) -> wgpu::Backends {
        match self {
            Self::Auto => wgpu::Backends::PRIMARY,
            Self::Vulkan => wgpu::Backends::VULKAN,
            Self::Metal => wgpu::Backends::METAL,
            Self::Dx12 => wgpu::Backends::DX12,
            Self::Gl => wgpu::Backends::GL,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Params {
    ref_count: u32,
    refs_per_chunk: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuCandidate {
    distance: f32,
    index: u32,
    label: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct QueryUniform {
    values: [i32; PADDED_DIMS],
}

impl QueryUniform {
    fn from_query(query: &[f32; DIMS]) -> Self {
        let mut values = [0; PADDED_DIMS];
        for dim in 0..DIMS {
            values[dim] = quantize_value(query[dim]) as i32;
        }
        Self { values }
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    distance: f32,
    index: u32,
    label: u32,
}

impl Candidate {
    fn worst() -> Self {
        Self {
            distance: f32::INFINITY,
            index: u32::MAX,
            label: 0,
        }
    }

    fn from_gpu(candidate: GpuCandidate) -> Option<Self> {
        (candidate.index != u32::MAX).then_some(Self {
            distance: candidate.distance,
            index: candidate.index,
            label: candidate.label,
        })
    }
}

#[derive(Debug)]
struct ReferenceData {
    vectors_f32: Vec<f32>,
    packed_vectors: Vec<u32>,
    labels: PackedLabels,
}

impl ReferenceData {
    fn len(&self) -> usize {
        self.labels.len
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Debug)]
struct PackedLabels {
    words: Vec<u32>,
    len: usize,
}

#[derive(Debug)]
struct Decision {
    fraud_score: f32,
    approved: bool,
}

#[derive(Debug)]
struct GpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    query_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    labels: PackedLabels,
    output_size: u64,
    reference_count: usize,
    workgroup_count: u32,
    reference_chunk_count: usize,
    refs_per_chunk: u32,
    adapter_info: wgpu::AdapterInfo,
}

#[derive(Debug)]
struct MemoryPlan {
    reference_bytes: u64,
    packed_label_bytes: u64,
    candidate_buffer_bytes: u64,
    persistent_gpu_bytes_per_api: u64,
}

#[derive(Debug)]
struct ValidationStats {
    checked: usize,
    f32_top5_mismatches: usize,
    f32_decision_mismatches: usize,
}

#[derive(Debug)]
struct RuntimeConfig {
    normalization: Normalization,
    mcc_risk: HashMap<String, f32>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct Normalization {
    max_amount: f32,
    max_installments: f32,
    amount_vs_avg_ratio: f32,
    max_minutes: f32,
    max_km: f32,
    max_tx_count_24h: f32,
    max_merchant_avg_amount: f32,
}

impl Default for Normalization {
    fn default() -> Self {
        Self {
            max_amount: 10_000.0,
            max_installments: 12.0,
            amount_vs_avg_ratio: 10.0,
            max_minutes: 1_440.0,
            max_km: 1_000.0,
            max_tx_count_24h: 20.0,
            max_merchant_avg_amount: 10_000.0,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ReferenceRecord {
    vector: [f32; DIMS],
    label: ReferenceLabel,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ReferenceLabel {
    Fraud,
    Legit,
}

#[derive(Debug, Deserialize)]
struct FraudPayload {
    #[serde(rename = "id")]
    _id: String,
    transaction: TransactionPayload,
    customer: CustomerPayload,
    merchant: MerchantPayload,
    terminal: TerminalPayload,
    last_transaction: Option<LastTransactionPayload>,
}

#[derive(Debug, Deserialize)]
struct TransactionPayload {
    amount: f32,
    installments: u32,
    requested_at: String,
}

#[derive(Debug, Deserialize)]
struct CustomerPayload {
    avg_amount: f32,
    tx_count_24h: u32,
    known_merchants: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct MerchantPayload {
    id: String,
    mcc: String,
    avg_amount: f32,
}

#[derive(Debug, Deserialize)]
struct TerminalPayload {
    is_online: bool,
    card_present: bool,
    km_from_home: f32,
}

#[derive(Debug, Deserialize)]
struct LastTransactionPayload {
    timestamp: String,
    km_from_current: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;

    let runtime_config = load_runtime_config(&args)?;
    let keep_f32_baseline = !args.skip_validate && args.validate_queries > 0;
    let references = load_or_generate_references(&args, keep_f32_baseline)?;
    if references.is_empty() {
        bail!("reference dataset is empty");
    }
    let reference_count = references.len();
    let queries = generate_queries(args.queries, args.seed ^ 0xa076_1d64_78bd_642f);
    let engine = pollster::block_on(GpuEngine::new(&args, &references))?;
    let memory_plan = MemoryPlan::new(engine.reference_count);

    println!(
        "environment_note=local benchmark results describe this machine/driver only; the target Rinha environment may differ"
    );
    println!(
        "config references={} logical_dims={} physical_dims={} layout=i16_packed scale={} queries={} warmup={} iterations={} validate_queries={} seed={}",
        reference_count,
        DIMS,
        PADDED_DIMS,
        QUANT_SCALE,
        args.queries,
        args.warmup,
        args.iterations,
        if args.skip_validate {
            0
        } else {
            args.validate_queries
        },
        args.seed
    );

    println!(
        "adapter name={:?} backend={:?} type={:?} driver={:?}",
        engine.adapter_info.name,
        engine.adapter_info.backend,
        engine.adapter_info.device_type,
        engine.adapter_info.driver
    );
    println!(
        "gpu_plan workgroups={} candidates_returned={} reference_buffer_mb={:.1} packed_labels_mb={:.2} block_candidate_buffers_mb={:.2}",
        engine.workgroup_count,
        engine.workgroup_count as usize * TOP_K,
        bytes_to_mib(memory_plan.reference_bytes),
        bytes_to_mib(memory_plan.packed_label_bytes),
        bytes_to_mib(memory_plan.candidate_buffer_bytes)
    );
    println!(
        "gpu_chunks reference_chunks={} refs_per_chunk={} max_reference_buffers={}",
        engine.reference_chunk_count, engine.refs_per_chunk, MAX_REFERENCE_BUFFERS
    );
    println!(
        "competition_fit api_instances={} lb_memory_mb={:.1} limit_mb={:.1} min_persistent_gpu_buffers_mb={:.1} fits_limit={}",
        args.api_instances,
        args.lb_memory_mb,
        args.memory_limit_mb,
        memory_plan.total_with_lb_mib(args.api_instances, args.lb_memory_mb),
        memory_plan.fits_limit(args.api_instances, args.lb_memory_mb, args.memory_limit_mb)
    );

    if !args.skip_validate && args.validate_queries > 0 {
        let validation = validate_cpu_gpu(&references, &queries, &engine, args.validate_queries)?;
        println!(
            "validation=passed checked_queries={} f32_top5_mismatches={} f32_decision_mismatches={}",
            validation.checked, validation.f32_top5_mismatches, validation.f32_decision_mismatches
        );
    }

    drop(references);

    if args.serve {
        serve_http(&args.listen, &engine, &runtime_config)?;
        return Ok(());
    }

    for i in 0..args.warmup {
        let _ = engine.run_query(&queries[i % queries.len()])?;
    }

    let mut durations = Vec::with_capacity(args.iterations);
    let mut last_top = [Candidate::worst(); TOP_K];
    for i in 0..args.iterations {
        let query = &queries[i % queries.len()];
        let started = Instant::now();
        last_top = engine.run_query(query)?;
        durations.push(started.elapsed());
    }

    let decision = decision_from_top5(&last_top);
    let stats = LatencyStats::new(&durations);
    println!(
        "latency_us min={:.1} p50={:.1} p95={:.1} p99={:.1} max={:.1}",
        stats.min_us, stats.p50_us, stats.p95_us, stats.p99_us, stats.max_us
    );
    println!(
        "last_decision fraud_score={:.1} approved={} top5={}",
        decision.fraud_score,
        decision.approved,
        format_top5(&last_top)
    );

    Ok(())
}

fn validate_args(args: &Args) -> Result<()> {
    if args.references < TOP_K {
        bail!("--references must be at least {TOP_K}");
    }
    if args.references > u32::MAX as usize {
        bail!("--references cannot exceed u32::MAX because GPU indices are u32");
    }
    if args.queries == 0 {
        bail!("--queries must be greater than zero");
    }
    if args.iterations == 0 {
        bail!("--iterations must be greater than zero");
    }
    if args.api_instances < 2 {
        bail!("--api-instances must be at least 2 to match the competition rules");
    }
    if args.memory_limit_mb <= 0.0 {
        bail!("--memory-limit-mb must be greater than zero");
    }
    Ok(())
}

fn load_runtime_config(args: &Args) -> Result<RuntimeConfig> {
    let normalization = if args.normalization_path.exists() {
        let file = File::open(&args.normalization_path).with_context(|| {
            format!(
                "failed to open normalization file {}",
                args.normalization_path.display()
            )
        })?;
        serde_json::from_reader(file).with_context(|| {
            format!(
                "failed to parse normalization file {}",
                args.normalization_path.display()
            )
        })?
    } else if args.require_resource_files {
        bail!(
            "normalization file {} does not exist",
            args.normalization_path.display()
        );
    } else {
        Normalization::default()
    };

    let mcc_risk = if args.mcc_risk_path.exists() {
        let file = File::open(&args.mcc_risk_path).with_context(|| {
            format!(
                "failed to open MCC risk file {}",
                args.mcc_risk_path.display()
            )
        })?;
        serde_json::from_reader(file).with_context(|| {
            format!(
                "failed to parse MCC risk file {}",
                args.mcc_risk_path.display()
            )
        })?
    } else if args.require_resource_files {
        bail!(
            "MCC risk file {} does not exist",
            args.mcc_risk_path.display()
        );
    } else {
        default_mcc_risk()
    };

    Ok(RuntimeConfig {
        normalization,
        mcc_risk,
    })
}

fn default_mcc_risk() -> HashMap<String, f32> {
    [
        ("5411", 0.15),
        ("5812", 0.30),
        ("5912", 0.20),
        ("5944", 0.45),
        ("7801", 0.80),
        ("7802", 0.75),
        ("7995", 0.85),
        ("4511", 0.35),
        ("5311", 0.25),
        ("5999", 0.50),
    ]
    .into_iter()
    .map(|(mcc, risk)| (mcc.to_string(), risk))
    .collect()
}

fn load_or_generate_references(args: &Args, keep_f32_baseline: bool) -> Result<ReferenceData> {
    if !args.force_generate && args.references_path.exists() {
        println!(
            "reference_source=file path={} keep_f32_baseline={}",
            args.references_path.display(),
            keep_f32_baseline
        );
        return load_references_json_gz(&args.references_path, keep_f32_baseline).with_context(
            || {
                format!(
                    "failed to load references from {}",
                    args.references_path.display()
                )
            },
        );
    }

    if args.require_resource_files {
        bail!(
            "reference file {} does not exist",
            args.references_path.display()
        );
    }

    println!(
        "reference_source=generated count={} keep_f32_baseline={}",
        args.references, keep_f32_baseline
    );
    Ok(generate_references(
        args.references,
        args.seed,
        keep_f32_baseline,
    ))
}

fn load_references_json_gz(path: &Path, keep_f32_baseline: bool) -> Result<ReferenceData> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let decoder = GzDecoder::new(file);
    let reader = std::io::BufReader::new(decoder);
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    let references = ReferenceDataSeed { keep_f32_baseline }.deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(references)
}

struct ReferenceDataSeed {
    keep_f32_baseline: bool,
}

impl<'de> DeserializeSeed<'de> for ReferenceDataSeed {
    type Value = ReferenceData;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(ReferenceDataVisitor {
            keep_f32_baseline: self.keep_f32_baseline,
        })
    }
}

struct ReferenceDataVisitor {
    keep_f32_baseline: bool,
}

impl<'de> Visitor<'de> for ReferenceDataVisitor {
    type Value = ReferenceData;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an array of reference records")
    }

    fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let capacity = seq.size_hint().unwrap_or(3_000_000);
        let mut vectors_f32 = if self.keep_f32_baseline {
            Vec::with_capacity(capacity.saturating_mul(DIMS))
        } else {
            Vec::new()
        };
        let mut packed_vectors = Vec::with_capacity(capacity.saturating_mul(PACKED_DIMS));
        let mut label_words = Vec::with_capacity(capacity.div_ceil(32));
        let mut count = 0usize;

        while let Some(record) = seq.next_element::<ReferenceRecord>()? {
            append_reference_vector(
                &record.vector,
                self.keep_f32_baseline,
                &mut vectors_f32,
                &mut packed_vectors,
            );

            if count % 32 == 0 {
                label_words.push(0);
            }
            if matches!(record.label, ReferenceLabel::Fraud) {
                let word_index = count / 32;
                label_words[word_index] |= 1u32 << (count % 32);
            }
            count += 1;
        }

        Ok(ReferenceData {
            vectors_f32,
            packed_vectors,
            labels: PackedLabels {
                words: label_words,
                len: count,
            },
        })
    }
}

fn serve_http(listen: &str, engine: &GpuEngine, runtime_config: &RuntimeConfig) -> Result<()> {
    let listener = TcpListener::bind(listen).with_context(|| format!("failed to bind {listen}"))?;
    println!("server=listening addr={listen} endpoints=/ready,/fraud-score");

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(error) = handle_connection(&mut stream, engine, runtime_config) {
                    eprintln!("connection_error={error:#}");
                }
            }
            Err(error) => eprintln!("accept_error={error}"),
        }
    }

    Ok(())
}

fn handle_connection(
    stream: &mut TcpStream,
    engine: &GpuEngine,
    runtime_config: &RuntimeConfig,
) -> Result<()> {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .context("failed to set read timeout")?;

    let request = read_http_request(stream)?;
    let header_end = find_header_end(&request).ok_or_else(|| anyhow!("missing header end"))?;
    let headers = &request[..header_end];
    let first_line_end = headers
        .iter()
        .position(|byte| *byte == b'\r')
        .ok_or_else(|| anyhow!("missing request line"))?;
    let request_line = &headers[..first_line_end];

    if method_path_matches(request_line, b"GET", b"/ready") {
        write_http_response(stream, "200 OK", "text/plain", b"ready\n")?;
        return Ok(());
    }

    if method_path_matches(request_line, b"POST", b"/fraud-score") {
        let body = &request[header_end + 4..];
        match process_fraud_request(body, engine, runtime_config) {
            Ok(response) => {
                write_http_response(stream, "200 OK", "application/json", response.as_bytes())?;
            }
            Err(error) => {
                eprintln!("bad_fraud_request={error:#}");
                write_http_response(stream, "400 Bad Request", "text/plain", b"bad request\n")?;
            }
        }
        return Ok(());
    }

    write_http_response(stream, "404 Not Found", "text/plain", b"not found\n")?;
    Ok(())
}

fn read_http_request(stream: &mut TcpStream) -> Result<Vec<u8>> {
    const MAX_REQUEST_BYTES: usize = 128 * 1024;

    let mut request = Vec::with_capacity(4096);
    let mut scratch = [0u8; 4096];

    loop {
        let read = stream
            .read(&mut scratch)
            .context("failed to read request")?;
        if read == 0 {
            bail!("connection closed before request was complete");
        }
        request.extend_from_slice(&scratch[..read]);

        if request.len() > MAX_REQUEST_BYTES {
            bail!("request exceeded {MAX_REQUEST_BYTES} bytes");
        }

        if let Some(header_end) = find_header_end(&request) {
            let content_length = parse_content_length(&request[..header_end]).unwrap_or(0);
            let total_len = header_end + 4 + content_length;
            if request.len() >= total_len {
                request.truncate(total_len);
                return Ok(request);
            }
        }
    }
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_content_length(headers: &[u8]) -> Option<usize> {
    for line in headers.split(|byte| *byte == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.len() < "content-length:".len() {
            continue;
        }
        let (name, value) = line.split_at("content-length:".len());
        if name.eq_ignore_ascii_case(b"content-length:") {
            return std::str::from_utf8(value)
                .ok()?
                .trim()
                .parse::<usize>()
                .ok();
        }
    }
    None
}

fn method_path_matches(request_line: &[u8], method: &[u8], path: &[u8]) -> bool {
    if !request_line.starts_with(method) || request_line.get(method.len()) != Some(&b' ') {
        return false;
    }

    let rest = &request_line[method.len() + 1..];
    if !rest.starts_with(path) {
        return false;
    }

    matches!(rest.get(path.len()), Some(b' ' | b'?'))
}

fn write_http_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    let headers = format!(
        "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(headers.as_bytes())
        .context("failed to write response headers")?;
    stream
        .write_all(body)
        .context("failed to write response body")?;
    Ok(())
}

fn process_fraud_request(
    body: &[u8],
    engine: &GpuEngine,
    runtime_config: &RuntimeConfig,
) -> Result<String> {
    let payload: FraudPayload =
        serde_json::from_slice(body).context("failed to parse fraud-score JSON payload")?;
    let query = vectorize_payload(&payload, runtime_config)?;
    let top = engine.run_query(&query)?;
    let decision = decision_from_top5(&top);

    Ok(format!(
        r#"{{"approved":{},"fraud_score":{:.1}}}"#,
        decision.approved, decision.fraud_score
    ))
}

fn vectorize_payload(
    payload: &FraudPayload,
    runtime_config: &RuntimeConfig,
) -> Result<[f32; DIMS]> {
    let requested = parse_timestamp(&payload.transaction.requested_at).with_context(|| {
        format!(
            "invalid requested_at {:?}",
            payload.transaction.requested_at
        )
    })?;
    let mut vector = [0.0f32; DIMS];
    let normalization = runtime_config.normalization;

    vector[0] = clamp01(payload.transaction.amount / normalization.max_amount);
    vector[1] = clamp01(payload.transaction.installments as f32 / normalization.max_installments);
    vector[2] = if payload.customer.avg_amount > 0.0 {
        clamp01(
            (payload.transaction.amount / payload.customer.avg_amount)
                / normalization.amount_vs_avg_ratio,
        )
    } else {
        1.0
    };
    vector[3] = round4(requested.hour as f32 / 23.0);
    vector[4] = round4(requested.day_of_week as f32 / 6.0);

    if let Some(last_transaction) = &payload.last_transaction {
        let previous = parse_timestamp(&last_transaction.timestamp).with_context(|| {
            format!(
                "invalid last_transaction.timestamp {:?}",
                last_transaction.timestamp
            )
        })?;
        let minutes = (requested.total_minutes - previous.total_minutes).max(0) as f32;
        vector[5] = clamp01(minutes / normalization.max_minutes);
        vector[6] = clamp01(last_transaction.km_from_current / normalization.max_km);
    } else {
        vector[5] = -1.0;
        vector[6] = -1.0;
    }

    vector[7] = clamp01(payload.terminal.km_from_home / normalization.max_km);
    vector[8] = clamp01(payload.customer.tx_count_24h as f32 / normalization.max_tx_count_24h);
    vector[9] = payload.terminal.is_online as u8 as f32;
    vector[10] = payload.terminal.card_present as u8 as f32;
    vector[11] = (!payload
        .customer
        .known_merchants
        .iter()
        .any(|known| known == &payload.merchant.id)) as u8 as f32;
    vector[12] = mcc_risk(&payload.merchant.mcc, runtime_config);
    vector[13] = clamp01(payload.merchant.avg_amount / normalization.max_merchant_avg_amount);

    Ok(vector)
}

#[derive(Clone, Copy, Debug)]
struct ParsedTimestamp {
    total_minutes: i64,
    hour: u32,
    day_of_week: u32,
}

fn parse_timestamp(value: &str) -> Option<ParsedTimestamp> {
    let bytes = value.as_bytes();
    if bytes.len() < 20
        || bytes.get(4) != Some(&b'-')
        || bytes.get(7) != Some(&b'-')
        || bytes.get(10) != Some(&b'T')
        || bytes.get(13) != Some(&b':')
        || bytes.get(16) != Some(&b':')
    {
        return None;
    }

    let year = parse_digits(&bytes[0..4])? as i32;
    let month = parse_digits(&bytes[5..7])? as u32;
    let day = parse_digits(&bytes[8..10])? as u32;
    let hour = parse_digits(&bytes[11..13])? as u32;
    let minute = parse_digits(&bytes[14..16])? as u32;

    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || minute > 59 {
        return None;
    }

    let days = days_from_civil(year, month, day);
    Some(ParsedTimestamp {
        total_minutes: days * 1_440 + hour as i64 * 60 + minute as i64,
        hour,
        day_of_week: ((days + 3).rem_euclid(7)) as u32,
    })
}

fn parse_digits(bytes: &[u8]) -> Option<u32> {
    let mut value = 0u32;
    for byte in bytes {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value * 10 + (byte - b'0') as u32;
    }
    Some(value)
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let year = year - (month <= 2) as i32;
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let month = month as i32;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day as i32 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    (era * 146_097 + day_of_era - 719_468) as i64
}

fn clamp01(value: f32) -> f32 {
    round4(value.clamp(0.0, 1.0))
}

fn round4(value: f32) -> f32 {
    (value * 10_000.0).round() * 0.0001
}

fn mcc_risk(mcc: &str, runtime_config: &RuntimeConfig) -> f32 {
    runtime_config.mcc_risk.get(mcc).copied().unwrap_or(0.5)
}

impl GpuEngine {
    async fn new(args: &Args, references: &ReferenceData) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: args.backend.to_wgpu(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: args.force_fallback_adapter,
            })
            .await
            .ok_or_else(|| anyhow!("no compatible wgpu adapter found"))?;

        let adapter_info = adapter.get_info();
        let loaded_reference_count = references.len();
        let reference_count =
            effective_reference_count(args, loaded_reference_count, adapter_info.device_type);
        if reference_count != loaded_reference_count {
            println!(
                "cpu_adapter_reference_limit loaded_references={} effective_references={}",
                loaded_reference_count, reference_count
            );
        }
        if reference_count > u32::MAX as usize {
            bail!("reference count cannot exceed u32::MAX because GPU indices are u32");
        }
        let workgroup_count = div_ceil(reference_count as u32, WORKGROUP_SIZE);
        let candidate_count = workgroup_count as usize * TOP_K;
        let bytes_per_reference = PACKED_DIMS * size_of::<u32>();
        let adapter_limits = adapter.limits();
        let refs_per_chunk =
            (adapter_limits.max_storage_buffer_binding_size as usize / bytes_per_reference).max(1);
        let reference_chunk_count = reference_count.div_ceil(refs_per_chunk);
        if reference_chunk_count > MAX_REFERENCE_BUFFERS {
            bail!(
                "reference dataset needs {reference_chunk_count} storage buffers, but this prototype supports {MAX_REFERENCE_BUFFERS}; refs_per_chunk={refs_per_chunk}"
            );
        }

        let max_reference_chunk_size =
            (refs_per_chunk.min(reference_count) * bytes_per_reference) as u64;
        let query_size = size_of::<QueryUniform>() as u64;
        let output_size = (candidate_count * size_of::<GpuCandidate>()) as u64;

        let limits = required_limits(
            &adapter,
            max_reference_chunk_size.max(output_size),
            max_reference_chunk_size.max(output_size),
        )?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rinha-gpu-bruteforce-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                },
                None,
            )
            .await
            .context("failed to create wgpu device")?;

        let mut reference_buffers = Vec::with_capacity(MAX_REFERENCE_BUFFERS);
        let packed_reference_len = reference_count * PACKED_DIMS;
        for (chunk_index, chunk) in references.packed_vectors[..packed_reference_len]
            .chunks(refs_per_chunk * PACKED_DIMS)
            .enumerate()
        {
            reference_buffers.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("reference-vectors-{chunk_index}")),
                    contents: bytemuck::cast_slice(chunk),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            );
        }
        while reference_buffers.len() < MAX_REFERENCE_BUFFERS {
            reference_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("reference-vectors-unused"),
                size: bytes_per_reference as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        }

        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("query-vector"),
            size: query_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = Params {
            ref_count: reference_count as u32,
            refs_per_chunk: refs_per_chunk as u32,
            _pad: [0; 2],
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block-top5-output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block-top5-readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("knn-bind-group-layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(4, false),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("knn-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                buffer_entry(0, &reference_buffers[0]),
                buffer_entry(1, &reference_buffers[1]),
                buffer_entry(2, &query_buffer),
                buffer_entry(3, &params_buffer),
                buffer_entry(4, &output_buffer),
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("knn-block-top5-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("knn-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("knn-block-top5-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group,
            query_buffer,
            output_buffer,
            readback_buffer,
            labels: references.labels.clone(),
            output_size,
            reference_count,
            workgroup_count,
            reference_chunk_count,
            refs_per_chunk: refs_per_chunk as u32,
            adapter_info,
        })
    }

    fn run_query(&self, query: &[f32; DIMS]) -> Result<[Candidate; TOP_K]> {
        let query_uniform = QueryUniform::from_query(query);
        self.queue
            .write_buffer(&self.query_buffer, 0, bytemuck::bytes_of(&query_uniform));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("knn-command-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("knn-compute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.readback_buffer,
            0,
            self.output_size,
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.readback_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .context("failed to receive GPU readback map result")?
            .context("failed to map GPU readback buffer")?;

        let top = {
            let mapped = slice.get_mapped_range();
            let candidates: &[GpuCandidate] = bytemuck::cast_slice(&mapped);
            let mut top = [Candidate::worst(); TOP_K];
            for mut candidate in candidates.iter().filter_map(|c| Candidate::from_gpu(*c)) {
                candidate.label = self.labels.get(candidate.index as usize);
                insert_top5(&mut top, candidate);
            }
            top
        };
        self.readback_buffer.unmap();

        Ok(top)
    }
}

fn effective_reference_count(
    args: &Args,
    loaded_reference_count: usize,
    device_type: wgpu::DeviceType,
) -> usize {
    if args.cpu_adapter_reference_limit > 0 && device_type == wgpu::DeviceType::Cpu {
        args.cpu_adapter_reference_limit.min(loaded_reference_count)
    } else {
        loaded_reference_count
    }
}

fn required_limits(
    adapter: &wgpu::Adapter,
    max_storage_binding_size: u64,
    max_buffer_size: u64,
) -> Result<wgpu::Limits> {
    let adapter_limits = adapter.limits();
    if max_storage_binding_size > adapter_limits.max_storage_buffer_binding_size as u64 {
        bail!(
            "adapter max_storage_buffer_binding_size={} bytes is smaller than required {} bytes",
            adapter_limits.max_storage_buffer_binding_size,
            max_storage_binding_size
        );
    }
    if max_buffer_size > adapter_limits.max_buffer_size {
        bail!(
            "adapter max_buffer_size={} bytes is smaller than required {} bytes",
            adapter_limits.max_buffer_size,
            max_buffer_size
        );
    }
    if adapter_limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE {
        bail!(
            "adapter max_compute_invocations_per_workgroup={} is smaller than required {}",
            adapter_limits.max_compute_invocations_per_workgroup,
            WORKGROUP_SIZE
        );
    }
    if adapter_limits.max_compute_workgroup_size_x < WORKGROUP_SIZE {
        bail!(
            "adapter max_compute_workgroup_size_x={} is smaller than required {}",
            adapter_limits.max_compute_workgroup_size_x,
            WORKGROUP_SIZE
        );
    }

    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffer_binding_size = limits
        .max_storage_buffer_binding_size
        .max(max_storage_binding_size as u32);
    limits.max_buffer_size = limits.max_buffer_size.max(max_buffer_size);
    limits.max_compute_invocations_per_workgroup = limits
        .max_compute_invocations_per_workgroup
        .max(WORKGROUP_SIZE);
    limits.max_compute_workgroup_size_x = limits.max_compute_workgroup_size_x.max(WORKGROUP_SIZE);
    Ok(limits)
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn validate_cpu_gpu(
    references: &ReferenceData,
    queries: &[[f32; DIMS]],
    engine: &GpuEngine,
    validate_queries: usize,
) -> Result<ValidationStats> {
    let checked = validate_queries.min(queries.len());
    let mut f32_top5_mismatches = 0;
    let mut f32_decision_mismatches = 0;

    for (query_index, query) in queries.iter().take(checked).enumerate() {
        let cpu_quantized = cpu_quantized_top5(references, query);
        let gpu = engine.run_query(query)?;
        if !same_top5(&cpu_quantized, &gpu) {
            bail!(
                "quantized CPU/GPU top5 mismatch on query {query_index}: cpu={} gpu={}",
                format_top5(&cpu_quantized),
                format_top5(&gpu)
            );
        }

        let cpu_f32 = cpu_f32_top5(references, query);
        if !same_top5(&cpu_f32, &gpu) {
            f32_top5_mismatches += 1;
        }
        if decision_from_top5(&cpu_f32).approved != decision_from_top5(&gpu).approved {
            f32_decision_mismatches += 1;
        }
    }

    Ok(ValidationStats {
        checked,
        f32_top5_mismatches,
        f32_decision_mismatches,
    })
}

fn same_top5(left: &[Candidate; TOP_K], right: &[Candidate; TOP_K]) -> bool {
    left.iter()
        .zip(right.iter())
        .all(|(a, b)| a.index == b.index && a.label == b.label)
}

fn cpu_f32_top5(references: &ReferenceData, query: &[f32; DIMS]) -> [Candidate; TOP_K] {
    let mut top = [Candidate::worst(); TOP_K];
    for (index, vector) in references.vectors_f32.chunks_exact(DIMS).enumerate() {
        let mut distance = 0.0f32;
        for dim in 0..DIMS {
            let diff = vector[dim] - query[dim];
            distance += diff * diff;
        }
        insert_top5(
            &mut top,
            Candidate {
                distance,
                index: index as u32,
                label: references.labels.get(index),
            },
        );
    }
    top
}

fn cpu_quantized_top5(references: &ReferenceData, query: &[f32; DIMS]) -> [Candidate; TOP_K] {
    let mut top = [Candidate::worst(); TOP_K];
    let query = quantize_query(query);

    for (index, vector) in references
        .packed_vectors
        .chunks_exact(PACKED_DIMS)
        .enumerate()
    {
        let mut distance = 0.0f32;
        for dim in 0..DIMS {
            let diff = unpack_i16(vector[dim / 2], dim % 2 == 1) as i32 - query[dim] as i32;
            distance += (diff * diff) as f32;
        }
        insert_top5(
            &mut top,
            Candidate {
                distance,
                index: index as u32,
                label: references.labels.get(index),
            },
        );
    }
    top
}

fn insert_top5(top: &mut [Candidate; TOP_K], candidate: Candidate) {
    for position in 0..TOP_K {
        if candidate_order(candidate, top[position]) == Ordering::Less {
            for shift in (position + 1..TOP_K).rev() {
                top[shift] = top[shift - 1];
            }
            top[position] = candidate;
            break;
        }
    }
}

fn candidate_order(left: Candidate, right: Candidate) -> Ordering {
    match left.distance.total_cmp(&right.distance) {
        Ordering::Equal => left.index.cmp(&right.index),
        order => order,
    }
}

fn decision_from_top5(top: &[Candidate; TOP_K]) -> Decision {
    let frauds = top.iter().filter(|candidate| candidate.label == 1).count();
    let fraud_score = frauds as f32 / TOP_K as f32;
    Decision {
        fraud_score,
        approved: fraud_score < 0.6,
    }
}

fn generate_references(count: usize, seed: u64, keep_f32_baseline: bool) -> ReferenceData {
    let mut rng = SplitMix64::new(seed);
    let mut vectors_f32 = if keep_f32_baseline {
        Vec::with_capacity(count * DIMS)
    } else {
        Vec::new()
    };
    let mut packed_vectors = Vec::with_capacity(count * PACKED_DIMS);
    let mut labels = PackedLabels::new(count);

    for index in 0..count {
        let mut weighted_sum = 0.0f32;
        let mut vector = [0.0f32; DIMS];
        for dim in 0..DIMS {
            let value = rng.next_f32();
            weighted_sum += value * (dim as f32 + 1.0);
            vector[dim] = value;
        }
        append_reference_vector(
            &vector,
            keep_f32_baseline,
            &mut vectors_f32,
            &mut packed_vectors,
        );
        let normalized = weighted_sum / ((DIMS * (DIMS + 1) / 2) as f32);
        let noisy_score = normalized + (rng.next_f32() - 0.5) * 0.2;
        labels.set(index, noisy_score >= 0.52);
    }

    ReferenceData {
        vectors_f32,
        packed_vectors,
        labels,
    }
}

fn append_reference_vector(
    vector: &[f32; DIMS],
    keep_f32_baseline: bool,
    vectors_f32: &mut Vec<f32>,
    packed_vectors: &mut Vec<u32>,
) {
    let mut quantized = [0i16; PADDED_DIMS];
    for dim in 0..DIMS {
        if keep_f32_baseline {
            vectors_f32.push(vector[dim]);
        }
        quantized[dim] = quantize_value(vector[dim]);
    }
    for pair in 0..PACKED_DIMS {
        packed_vectors.push(pack_i16_pair(quantized[pair * 2], quantized[pair * 2 + 1]));
    }
}

fn quantize_query(query: &[f32; DIMS]) -> [i16; PADDED_DIMS] {
    let mut quantized = [0; PADDED_DIMS];
    for dim in 0..DIMS {
        quantized[dim] = quantize_value(query[dim]);
    }
    quantized
}

fn quantize_value(value: f32) -> i16 {
    (value * QUANT_SCALE)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn pack_i16_pair(low: i16, high: i16) -> u32 {
    (low as u16 as u32) | ((high as u16 as u32) << 16)
}

fn unpack_i16(word: u32, high: bool) -> i16 {
    if high {
        (word >> 16) as u16 as i16
    } else {
        (word & 0xffff) as u16 as i16
    }
}

impl PackedLabels {
    fn new(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(32)],
            len,
        }
    }

    fn set(&mut self, index: usize, fraud: bool) {
        debug_assert!(index < self.len);
        if fraud {
            self.words[index / 32] |= 1u32 << (index % 32);
        }
    }

    fn get(&self, index: usize) -> u32 {
        debug_assert!(index < self.len);
        (self.words[index / 32] >> (index % 32)) & 1
    }
}

impl MemoryPlan {
    fn new(reference_count: usize) -> Self {
        let workgroup_count = div_ceil(reference_count as u32, WORKGROUP_SIZE);
        let candidate_count = workgroup_count as usize * TOP_K;
        let reference_bytes = (reference_count * PACKED_DIMS * size_of::<u32>()) as u64;
        let packed_label_bytes = (reference_count.div_ceil(32) * size_of::<u32>()) as u64;
        let candidate_buffer_bytes = (candidate_count * size_of::<GpuCandidate>()) as u64;
        let persistent_gpu_bytes_per_api = reference_bytes
            + packed_label_bytes
            + (2 * candidate_buffer_bytes)
            + size_of::<QueryUniform>() as u64
            + size_of::<Params>() as u64;

        Self {
            reference_bytes,
            packed_label_bytes,
            candidate_buffer_bytes,
            persistent_gpu_bytes_per_api,
        }
    }

    fn total_with_lb_mib(&self, api_instances: usize, lb_memory_mb: f64) -> f64 {
        bytes_to_mib(self.persistent_gpu_bytes_per_api * api_instances as u64) + lb_memory_mb
    }

    fn fits_limit(&self, api_instances: usize, lb_memory_mb: f64, memory_limit_mb: f64) -> bool {
        self.total_with_lb_mib(api_instances, lb_memory_mb) <= memory_limit_mb
    }
}

fn generate_queries(count: usize, seed: u64) -> Vec<[f32; DIMS]> {
    let mut rng = SplitMix64::new(seed);
    let mut queries = Vec::with_capacity(count);

    for index in 0..count {
        let mut query = [0.0f32; DIMS];
        for value in query.iter_mut() {
            *value = rng.next_f32();
        }
        if index % 4 == 0 {
            query[5] = -1.0;
            query[6] = -1.0;
        }
        queries.push(query);
    }

    queries
}

#[derive(Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / (1u32 << 24) as f32
    }
}

#[derive(Debug)]
struct LatencyStats {
    min_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    max_us: f64,
}

impl LatencyStats {
    fn new(durations: &[Duration]) -> Self {
        let mut micros: Vec<f64> = durations
            .iter()
            .map(|duration| duration.as_secs_f64() * 1_000_000.0)
            .collect();
        micros.sort_by(f64::total_cmp);

        Self {
            min_us: micros[0],
            p50_us: percentile(&micros, 50.0),
            p95_us: percentile(&micros, 95.0),
            p99_us: percentile(&micros, 99.0),
            max_us: micros[micros.len() - 1],
        }
    }
}

fn percentile(sorted_values: &[f64], percentile: f64) -> f64 {
    let rank = ((percentile / 100.0) * (sorted_values.len() - 1) as f64).ceil() as usize;
    sorted_values[rank]
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

fn format_top5(top: &[Candidate; TOP_K]) -> String {
    let parts: Vec<String> = top
        .iter()
        .map(|candidate| {
            format!(
                "{{idx:{},dist:{:.6},label:{}}}",
                candidate.index, candidate.distance, candidate.label
            )
        })
        .collect();
    format!("[{}]", parts.join(","))
}
