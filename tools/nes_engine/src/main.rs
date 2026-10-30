mod history;
mod index;
mod model;
mod retrieval;
mod service;
mod sweep;

use anyhow::{Context as _, Result};
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::{Value, json};
use service::{EditOut, ObserveRequest, PredictRequest, PredictResponse, Service, Settings};
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
    /// Threads for decode calls of at most 9 tokens (draft verification);
    /// 0 uses --threads-batch.
    #[arg(long, default_value_t = 0)]
    threads_small_batch: i32,
    /// Persistent llama.cpp threadpools shared by all file slots: decode
    /// threads pinned to physical cores, batch threads adding SMT siblings.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    threadpool: bool,
    /// Threadpool polling level between graphs: 0 sleeps at once, 50
    /// (llama.cpp's default) spins ~0.3 s. Pools are paused between requests.
    #[arg(long, default_value_t = 1)]
    threadpool_poll: u32,
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
    /// none | first-line | before-cursor-line | through-cursor.
    /// before-cursor-line puts the window lines above the cursor in the
    /// prompt, where they stay cached while typing on the cursor line:
    /// keystroke p50 494 -> 370 ms, equal accuracy on 761 cases.
    #[arg(long, default_value = "before-cursor-line")]
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
    /// Splice once the start of the next line (this many non-blank
    /// characters) uniquely matches a window line below the change, instead
    /// of waiting for a complete anchor line. 0 disables.
    #[arg(long, default_value_t = 0)]
    splice_partial_chars: usize,
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
    /// Flash attention: -1 auto, 0 off, 1 on. Off by default: for the 2-63
    /// token batches of draft verification and short prefills, llama.cpp's
    /// CPU flash-attention path is a scalar per-row loop, while the plain
    /// path uses GEMMs (17-token verify 170 -> 124 ms on the i7-8650U).
    #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
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
    /// Stricter gate for edits that rewrite more than one existing line
    /// (-0.4 vs -0.5 on 761 cases: helpful -1, harmful -9).
    #[arg(long, default_value_t = -0.4, allow_hyphen_values = true)]
    multiline_min_confidence: f32,
    /// Looser gate for insertions and appends that keep the existing text
    /// (only lowers --min-confidence; -1.3 vs -1.0 on 761 cases: helpful
    /// +14, harmful +7, typing saved +2.6).
    #[arg(long, default_value_t = -1.3, allow_hyphen_values = true)]
    insert_min_confidence: f32,
    /// Keep the history section stable while typing: hide hunks the window
    /// shows before capping the history, keep the live edit group out of the
    /// inverse/duplicate/whitespace filters, and revert hunks for `original/`
    /// by line position (false: substring reverts, cap before hiding).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    stable_history: bool,
    /// Map the cursor into the original file by line (not byte offset) and
    /// keep a long file's shown chunk while the cursor stays 10 lines inside it.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    stable_window: bool,
    /// Warm requests (observe with warm) prefill the full local prompt
    /// rather than only the file context.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    warm_full_prompt: bool,
    /// Do not split the live edit group on pauses (up to 60 s) and show it
    /// only in the window, reverted to its base in `original/`.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    session_base: bool,
    /// Retrieve definitions for files the index cannot parse (YAML etc.).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    retrieval_unindexed: bool,
    /// Advance a file's original snapshot (the prompt's file section) past
    /// edits the history no longer shows, once it is clearly stale (12 lines
    /// of drift, or under half the current lines), and warm the new prompt
    /// while idle. Without it a file first seen empty stays empty there
    /// (dekit.yaml replay: helpful 7 -> 13, harmful 8 -> 6; 761 cases:
    /// helpful +4, harmful -6; keystroke latency unchanged).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    refresh_original: bool,
    /// Trim edits that would rewrite text typed in the live edit (keeping
    /// lines they add next to it); withhold them when nothing is left.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    protect_live_edit: bool,
    /// Run the jump stage in the file slot's scratch sequence, branching
    /// from the cached local prompt, so the local prompt's cache survives
    /// and no re-warm follows a jump request (761 cases: local answers as
    /// with jumps off; the next request evaluates 192 -> 107 tokens, p50).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    jump_scratch: bool,
    /// The jump prompt repeats the local prompt's edit history and appends
    /// the hunks the local window hid, so it shares the local prompt's KV
    /// through the history (jump prefill tokens -16%, same jump edits
    /// helpful/harmful on 761 cases).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    jump_shared_history: bool,
    /// Prompt tokens per prefill decode call in the jump stage (0: as
    /// --prefill-chunk). The next keystroke can only cancel between calls;
    /// 64 costs no measurable prefill throughput.
    #[arg(long, default_value_t = 64)]
    jump_prefill_chunk: usize,
    /// Comma-separated jump reasons that may run the jump stage
    /// (deleted_line_match, diagnostic, identifier_match,
    /// block_after_cursor); empty allows all. On 761 cases
    /// block_after_cursor (70 runs) and identifier_match (42) showed 0
    /// helpful and 12 harmful jumps at ~2.1 s each; deleted_line_match
    /// (27 runs) showed 2 helpful, 4 harmful. No case has diagnostics.
    #[arg(long, default_value = "deleted_line_match,diagnostic")]
    jump_reasons: String,
    /// Skip jump targets more than this many lines from the cursor (0: no limit).
    #[arg(long, default_value_t = 0)]
    jump_max_distance: usize,
    /// When a jump request (leaving insert mode) has the content and cursor
    /// line of a just-answered local request that found nothing, skip the
    /// local stage and run only the jump stage (leave-insert replay: p50
    /// 133 -> 2 ms, same answers).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    jump_reuse_local: bool,
    /// Stop a generation that repeats the same 1-3 lines (digits masked)
    /// past the window's line count, ending it as the full generation would
    /// have ("length", discarded, or "max_lines" with the copies filled in).
    /// All 12 runaway generations on 761 cases were such loops, 186-266
    /// tokens at 2-5 s each; slowest 10% mean 2.78 -> 2.55 s, p90 1.88 ->
    /// 1.84 s, helpful 29.2% -> 29.6%, harmful 16.8% both.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    stop_repetition: bool,
    /// With --refresh-original, keep prompting from the old snapshot while
    /// the prompt with the new one is warmed in a second sequence of the
    /// file's cache slot, and switch once that costs no more than the old
    /// prompt (or after 30 s), instead of re-evaluating the whole prompt at
    /// the next request. Keystroke replay: slowest request 3.9 -> 2.4 s,
    /// slowest 10% mean -11%; but on 761 cases (no idle time between
    /// requests, so refreshes wait the full 30 s) helpful 29.6% -> 28.8%,
    /// harmful 16.8% -> 17.3%, so it is off.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    defer_refresh: bool,
    /// Keep retrieval chunks that stay selected first in the prompt, in
    /// their previous order, with newly selected chunks after them (best
    /// last), so replacing one chunk does not re-evaluate the others.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    retrieval_stable_order: bool,
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
            stable_history: cli.stable_history,
            session_base: cli.session_base,
            ..Default::default()
        },
        max_retrieval_chunks: cli.retrieval_chunks,
        file_chunk_lines: cli.file_chunk_lines,
        index_refresh_secs: cli.index_refresh_secs,
        early_stop: cli.early_stop,
        draft_max: cli.draft_max,
        draft_min_match: cli.draft_min_match,
        splice_anchors: cli.splice_anchors,
        splice_partial_chars: cli.splice_partial_chars,
        draft_previous: cli.draft_previous,
        min_confidence: cli.min_confidence,
        multiline_min_confidence: cli.multiline_min_confidence,
        insert_min_confidence: cli.insert_min_confidence,
        stable_window: cli.stable_window,
        warm_full_prompt: cli.warm_full_prompt,
        retrieval_unindexed: cli.retrieval_unindexed,
        refresh_original: cli.refresh_original,
        protect_live_edit: cli.protect_live_edit,
        jump_scratch: cli.jump_scratch,
        jump_shared_history: cli.jump_shared_history,
        jump_prefill_chunk: cli.jump_prefill_chunk,
        jump_reasons: cli.jump_reasons.split(',').map(str::trim).filter(|r| !r.is_empty()).map(str::to_string).collect(),
        jump_max_distance: cli.jump_max_distance,
        jump_reuse_local: cli.jump_reuse_local,
        stop_repetition: cli.stop_repetition,
        defer_refresh: cli.defer_refresh,
        retrieval_stable_order: cli.retrieval_stable_order,
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
        threads_small_batch: cli.threads_small_batch,
        threadpool: cli.threadpool,
        threadpool_poll: cli.threadpool_poll,
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
        alt_prompt: &[],
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
        early_stop_every_batch: false,
        progress: None,
        scratch: false,
        prefill_chunk: 0,
        repeat_after_lines: 0,
    })? {
        model::Outcome::Done(g) => {
            println!("{}", serde_json::to_string(&json!({"text": g.text, "finish": g.finish_reason, "stats": g.stats}))?);
        }
        model::Outcome::Cancelled(_) => unreachable!(),
    }
    Ok(())
}

/// Longest idle time given to a warm-up between replayed events.
const REPLAY_MAX_IDLE: std::time::Duration = std::time::Duration::from_secs(10);

fn replay(svc: &Arc<Service>) -> Result<()> {
    let mut lines = Vec::new();
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if !line.trim().is_empty() {
            lines.push(serde_json::from_str::<ReplayLine>(&line).context("replay line")?);
        }
    }
    let stamps: Vec<Option<u64>> = lines
        .iter()
        .map(|l| match l {
            ReplayLine::Observe(o) => o.ts_ms,
            ReplayLine::Predict(p) => p.ts_ms,
            ReplayLine::Reset {} => None,
        })
        .collect();
    for (i, line) in lines.into_iter().enumerate() {
        match line {
            ReplayLine::Observe(o) => {
                svc.observe(&o);
                // Replay has no idle time of its own: give a warm-up the time
                // the recording had before the next event.
                let next = stamps[i + 1..].iter().flatten().next();
                if let (true, Some(ts), Some(next)) = (o.warm, o.ts_ms, next) {
                    let gap = std::time::Duration::from_millis(next.saturating_sub(ts));
                    svc.wait_idle(gap.min(REPLAY_MAX_IDLE));
                }
            }
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

#[derive(Deserialize)]
struct PredictQuery {
    /// `stream=1`: newline-delimited JSON events, provisional edits first.
    #[serde(default)]
    stream: Option<String>,
}

async fn predict(State(svc): State<Arc<Service>>, Query(q): Query<PredictQuery>, Json(req): Json<PredictRequest>) -> Response {
    if q.stream.as_deref().is_some_and(|v| v != "0" && v != "false") {
        return predict_stream(svc, req);
    }
    let request_id = req.request_id.clone();
    let result = tokio::task::spawn_blocking(move || svc.predict(req)).await;
    match result {
        Ok(Ok(r)) => {
            log_response(&r);
            Json(serde_json::to_value(r).unwrap()).into_response()
        }
        Ok(Err(e)) => {
            eprintln!("nes: predict error: {e:#}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "status": "error", "request_id": request_id, "error": format!("{e:#}") })),
            )
                .into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "status": "error", "error": e.to_string() }))).into_response(),
    }
}

/// Streaming predict: `{"event":"provisional","edits":[...]}` lines while
/// decoding, then one `{"event":"final", ...}` line with the normal response
/// body (or `{"event":"error", ...}`).
fn predict_stream(svc: Arc<Service>, req: PredictRequest) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    tokio::task::spawn_blocking(move || {
        let started = std::time::Instant::now();
        let request_id = req.request_id.clone();
        let sink = |e: &EditOut| {
            let ev = json!({
                "event": "provisional",
                "request_id": request_id,
                "ms": started.elapsed().as_secs_f64() * 1000.0,
                "edits": [e],
            });
            let _ = tx.send(format!("{ev}\n"));
        };
        let ev = match svc.predict_streaming(req, Some(&sink as &dyn Fn(&EditOut))) {
            Ok(r) => {
                log_response(&r);
                let mut v = serde_json::to_value(r).unwrap();
                v["event"] = json!("final");
                v
            }
            Err(e) => {
                eprintln!("nes: predict error: {e:#}");
                json!({ "event": "error", "status": "error", "request_id": request_id, "error": format!("{e:#}") })
            }
        };
        let _ = tx.send(format!("{ev}\n"));
    });
    let events = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|line| (Ok::<_, std::convert::Infallible>(line), rx))
    });
    ([(header::CONTENT_TYPE, "application/x-ndjson")], Body::from_stream(events)).into_response()
}

fn log_response(r: &PredictResponse) {
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
    let first = r.timings.first_edit_ms.map_or("-".to_string(), |m| format!("{m:.0}ms"));
    eprintln!(
        "nes: {} {} {} kind={} edits={} first={} total={:.0}ms [{}]",
        r.request_id.as_deref().unwrap_or("-"),
        r.status,
        r.exit_reason,
        r.kind,
        r.edits.len(),
        first,
        r.timings.total_ms,
        stages.join("; ")
    );
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
