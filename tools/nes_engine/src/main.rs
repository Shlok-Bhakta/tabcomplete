mod history;
mod index;
mod model;
mod retrieval;
mod service;
mod sweep;

use anyhow::{Context as _, Result};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::{Value, json};
use service::{ObserveRequest, PredictRequest, Service, Settings};
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "tabcomplete-nes", about = "CPU next-edit engine for Sweep next-edit models", args_override_self = true)]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    /// Recorded model hash, reported to the editor for telemetry.
    #[arg(long, default_value = "")]
    model_sha256: String,
    #[arg(long, default_value = "sweep-next-edit-0.5b-q8_0")]
    model_name: String,
    /// Context size of each file slot (each slot has its own small context:
    /// cache shifts and attention cost scale with context size).
    #[arg(long, default_value_t = 4096)]
    ctx_size: u32,
    /// Layers offloaded to a GPU (CUDA builds only).
    #[arg(long, default_value_t = 0)]
    gpu_layers: u32,
    #[arg(long, default_value_t = 512)]
    batch_size: u32,
    /// Decode threads.
    #[arg(long, default_value_t = 4)]
    threads: i32,
    /// Prompt-processing threads.
    #[arg(long, default_value_t = 6)]
    threads_batch: i32,
    #[arg(long, default_value_t = 2)]
    retrieval_chunks: usize,
    /// Recent edits shown in the prompt's history section (3 was as fast as
    /// 6 with small slot contexts and slightly more accurate on real edits).
    #[arg(long, default_value_t = 3)]
    max_hunks: usize,
    /// sweep | volatile-late | retrieval-late: where retrieved context sits
    /// relative to the edit history. volatile-late keeps diagnostics and the
    /// clipboard (which change while typing) from invalidating the cached
    /// history.
    #[arg(long, default_value = "volatile-late")]
    context_order: String,
    /// Lines of the most recently visited other file (0 disables).
    #[arg(long, default_value_t = 30)]
    file_chunk_lines: usize,
    /// Whole-file context up to this many lines, else a chunk around the
    /// cursor. Every cached token is attended to on each keystroke, so this
    /// bounds per-keystroke cost for large files (120 matched 240 on real edits).
    #[arg(long, default_value_t = 120)]
    whole_file_lines: usize,
    #[arg(long, default_value_t = 120)]
    chunk_lines: usize,
    /// absolute | sweep-local | none (Sweep's reference server uses sweep-local)
    #[arg(long, default_value = "sweep-local")]
    line_numbers: String,
    /// server | card
    #[arg(long, default_value = "server")]
    diff_format: String,
    /// none | first-line | before-cursor-line
    #[arg(long, default_value = "first-line")]
    prefill: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cursor_marker: bool,
    /// Keep the in-progress edit (shown by the window) out of the history.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    hide_active_edit: bool,
    #[arg(long, default_value_t = 2)]
    lines_before: usize,
    /// Lines below the cursor in the rewrite window (3 beat Sweep's 5 on
    /// real edits: faster, and fewer wrong rewrites).
    #[arg(long, default_value_t = 3)]
    lines_after: usize,
    #[arg(long, default_value_t = 60)]
    index_refresh_secs: u64,
    /// Matching lines after an edit before the rest of the window is spliced.
    #[arg(long, default_value_t = 1)]
    splice_anchors: usize,
    /// Reuse the unchanged prompt suffix's KV (shifted) instead of recomputing.
    /// Keystroke replay: prefill 459 -> 153 ms, p50 1.41 -> 1.09 s, same quality.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    reuse_suffix: bool,
    /// Files whose prompt caches are kept at once (KV slots).
    #[arg(long, default_value_t = 4)]
    slots: u32,
    /// KV cache type: f32, f16 or q8_0.
    #[arg(long, default_value = "f16")]
    cache_type: String,
    /// Memory-map the model file (false reads it into memory).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    mmap: bool,
    /// Prompt tokens per prefill decode call (cancellation granularity).
    #[arg(long, default_value_t = 512)]
    prefill_chunk: usize,
    /// Flash attention: -1 auto, 0 off, 1 on.
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    flash_attn: i32,
    /// Draft from the previous completion for the same file first.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    draft_previous: bool,
    /// Largest speculative draft verified in one batch.
    #[arg(long, default_value_t = 16)]
    draft_max: usize,
    /// Output-tail tokens that must match a draft source to start a new
    /// draft. Weaker matches are rejected so often that verifying them costs
    /// more than decoding one token at a time (keystroke replay: 16.0 -> 14.4
    /// ms per generated token).
    #[arg(long, default_value_t = 3)]
    draft_min_match: usize,
    /// Stop decoding once the window rewrite rejoins the original text.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    early_stop: bool,
    /// Minimum log-probability of the weakest changed token to show an edit.
    #[arg(long, default_value_t = -1.0, allow_hyphen_values = true)]
    min_confidence: f32,
    /// Stricter gate for edits that rewrite more than one existing line.
    #[arg(long, default_value_t = -0.5, allow_hyphen_values = true)]
    multiline_min_confidence: f32,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the HTTP API.
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 19094)]
        port: u16,
    },
    /// Run JSON lines of {"observe": {...}} / {"predict": {...}} from stdin.
    Replay,
    /// Time engine primitives on a prompt read from stdin.
    Bench {
        #[arg(long, default_value_t = 900)]
        depth: usize,
        #[arg(long, default_value_t = 10)]
        reps: usize,
    },
    /// Greedy completion of a raw prompt read from stdin (format experiments).
    Complete {
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReplayLine {
    Observe(ObserveRequest),
    Predict(PredictRequest),
    /// Forget editor state between independent cases.
    Reset {},
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let line_numbers = match cli.line_numbers.as_str() {
        "absolute" => sweep::LineNumbers::Absolute,
        "sweep-local" | "sweep_local" => sweep::LineNumbers::SweepLocal,
        "none" => sweep::LineNumbers::None,
        other => anyhow::bail!("unknown --line-numbers {other}"),
    };
    let diff_format = match cli.diff_format.as_str() {
        "server" => sweep::DiffFormat::Server,
        "card" => sweep::DiffFormat::Card,
        other => anyhow::bail!("unknown --diff-format {other}"),
    };
    let prefill = match cli.prefill.as_str() {
        "none" => sweep::Prefill::None,
        "first-line" => sweep::Prefill::FirstLine,
        "before-cursor-line" => sweep::Prefill::BeforeCursorLine,
        "through-cursor" => sweep::Prefill::ThroughCursor,
        other => anyhow::bail!("unknown --prefill {other}"),
    };
    let context_order = match cli.context_order.as_str() {
        "sweep" => sweep::ContextOrder::Sweep,
        "volatile-late" => sweep::ContextOrder::VolatileLate,
        "retrieval-late" => sweep::ContextOrder::RetrievalLate,
        other => anyhow::bail!("unknown --context-order {other}"),
    };
    let settings = Settings {
        model_name: cli.model_name.clone(),
        model_sha256: cli.model_sha256.clone(),
        prompt: sweep::PromptOptions {
            line_numbers,
            diff_format,
            prefill,
            cursor_marker: cli.cursor_marker,
            hide_active_edit: cli.hide_active_edit,
            lines_before: cli.lines_before,
            lines_after: cli.lines_after,
            whole_file_lines: cli.whole_file_lines,
            max_hunks: cli.max_hunks,
            context_order,
            chunk_lines: cli.chunk_lines,
            ..Default::default()
        },
        max_retrieval_chunks: cli.retrieval_chunks,
        file_chunk_lines: cli.file_chunk_lines,
        index_refresh_secs: cli.index_refresh_secs,
        early_stop: cli.early_stop,
        draft_max: cli.draft_max,
        draft_min_match: cli.draft_min_match,
        splice_anchors: cli.splice_anchors,
        draft_previous: cli.draft_previous,
        min_confidence: cli.min_confidence,
        multiline_min_confidence: cli.multiline_min_confidence,
    };
    let config = model::ModelConfig {
        gpu_layers: cli.gpu_layers,
        flash_attn: cli.flash_attn,
        prefill_chunk: cli.prefill_chunk,
        slots: cli.slots,
        reuse_suffix: cli.reuse_suffix,
        cache_type: cli.cache_type.clone(),
        use_mmap: cli.mmap,
        n_ctx: cli.ctx_size,
        n_batch: cli.batch_size,
        threads: cli.threads,
        threads_batch: cli.threads_batch,
    };
    if let Command::Bench { depth, reps } = cli.command {
        use std::io::Read;
        let mut prompt = String::new();
        std::io::stdin().read_to_string(&mut prompt)?;
        let mut engine = model::Engine::load(&cli.model, &config)?;
        let tokens = engine.tokenize(&prompt);
        for (name, ms) in engine.bench(&tokens, depth, reps)? {
            println!("{ms:9.2} ms  {name}");
        }
        println!("{}", engine.perf_report()?);
        return Ok(());
    }
    if let Command::Complete { max_tokens } = cli.command {
        return complete(&cli.model, &config, max_tokens);
    }
    let svc = Service::start(&cli.model, config, settings).context("start model")?;
    match cli.command {
        Command::Replay => replay(&svc),
        Command::Serve { host, port } => serve(svc, &host, port),
        Command::Complete { .. } | Command::Bench { .. } => unreachable!(),
    }
}

fn complete(model_path: &std::path::Path, config: &model::ModelConfig, max_tokens: usize) -> Result<()> {
    use std::io::Read;
    let mut prompt = String::new();
    std::io::stdin().read_to_string(&mut prompt)?;
    let mut engine = model::Engine::load(model_path, config)?;
    let tokens = engine.tokenize(&prompt);
    let latest = std::sync::atomic::AtomicU64::new(1);
    match engine.generate(model::GenRequest {
        prompt: &tokens,
        draft_source: &[],
        draft_seed: &[],
        max_tokens,
        max_newlines: 200,
        draft_max: 0,
        draft_min_match: 3,
        draft_fallback: &[],
        draft_previous: &[],
        generation: 1,
        latest: &latest,
        slot: "complete",
        novelty: None,
        early_stop: None,
    })? {
        model::Outcome::Done(g) => {
            println!("{}", serde_json::to_string(&json!({"text": g.text, "finish": g.finish_reason, "stats": g.stats}))?);
        }
        model::Outcome::Cancelled(_) => unreachable!(),
    }
    Ok(())
}

fn replay(svc: &Arc<Service>) -> Result<()> {
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<ReplayLine>(&line).context("replay line")? {
            ReplayLine::Observe(o) => svc.observe(&o),
            ReplayLine::Reset {} => svc.reset(),
            ReplayLine::Predict(p) => {
                let r = svc.predict(p)?;
                println!("{}", serde_json::to_string(&r)?);
            }
        }
    }
    Ok(())
}

fn serve(svc: Arc<Service>, host: &str, port: u16) -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let app = Router::new()
            .route("/health", get(health))
            .route("/v1/predict", post(predict))
            .route("/v1/observe", post(observe))
            .route("/v1/prefetch", post(prefetch))
            .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
            .with_state(svc);
        let listener = tokio::net::TcpListener::bind((host, port)).await?;
        eprintln!("nes: listening on {host}:{port}");
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        Ok(())
    })
}

async fn health(State(svc): State<Arc<Service>>) -> Json<Value> {
    let index: Vec<Value> = svc
        .registry
        .stats()
        .into_iter()
        .map(|(root, s)| json!({ "root": root, "stats": s }))
        .collect();
    Json(json!({
        "status": "ok",
        "protocol": service::PROTOCOL,
        "model": svc.settings.model_name,
        "model_sha256": svc.settings.model_sha256,
        "n_ctx": svc.n_ctx(),
        "prompt": svc.settings.prompt,
        "retrieval_chunks": svc.settings.max_retrieval_chunks,
        "index": index,
    }))
}

async fn predict(
    State(svc): State<Arc<Service>>,
    Json(req): Json<PredictRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let request_id = req.request_id.clone();
    let result = tokio::task::spawn_blocking(move || svc.predict(req)).await;
    match result {
        Ok(Ok(r)) => {
            let stages: Vec<String> = r
                .timings
                .stages
                .iter()
                .map(|s| {
                    format!(
                        "prompt={} cached={} prefill={:.0}ms decode={:.0}ms gen={} draft={}/{}",
                        s.prompt_tokens, s.cached_tokens, s.prefill_ms, s.decode_ms, s.generated_tokens, s.draft_accepted, s.draft_proposed
                    )
                })
                .collect();
            eprintln!(
                "nes: {} {} {} kind={} edits={} total={:.0}ms [{}]",
                r.request_id.as_deref().unwrap_or("-"),
                r.status,
                r.exit_reason,
                r.kind,
                r.edits.len(),
                r.timings.total_ms,
                stages.join("; ")
            );
            Ok(Json(serde_json::to_value(r).unwrap()))
        }
        Ok(Err(e)) => {
            eprintln!("nes: predict error: {e:#}");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "status": "error", "request_id": request_id, "error": format!("{e:#}") })),
            ))
        }
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "status": "error", "error": e.to_string() })))),
    }
}

async fn prefetch(State(svc): State<Arc<Service>>, Json(req): Json<PredictRequest>) -> Json<Value> {
    let result = tokio::task::spawn_blocking(move || svc.prefetch(req)).await;
    match result {
        Ok(Ok(())) => Json(json!({ "status": "ok" })),
        Ok(Err(e)) => Json(json!({ "status": "error", "error": format!("{e:#}") })),
        Err(e) => Json(json!({ "status": "error", "error": e.to_string() })),
    }
}

async fn observe(State(svc): State<Arc<Service>>, Json(req): Json<ObserveRequest>) -> Json<Value> {
    let _ = tokio::task::spawn_blocking(move || svc.observe(&req)).await;
    Json(json!({ "status": "ok" }))
}
