//! Prediction service: editor state in, edit proposals out.
//!
//! One worker thread owns the llama context. Requests increment a generation
//! counter; the worker abandons any job whose generation is no longer the
//! latest, so a burst of keystrokes costs one prediction, and the KV prefix
//! evaluated by an abandoned job is kept for the next one.

use crate::history::{self, DiffHunk, Tracker};
use crate::index::{self, Registry};
use crate::model::{Engine, GenRequest, GenStats, ModelConfig, Outcome, Tokenizer};
use crate::retrieval::{self, RetrievalInput};
use crate::sweep::{self, Block, Chunk, Diagnostic, Edit, PromptInput, PromptOptions};
use anyhow::{Context as _, Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const PROTOCOL: &str = "sweep-next-edit-v1";

#[derive(Clone, Debug, Deserialize)]
pub struct Cursor {
    /// 0-based line.
    pub line: usize,
    /// Byte column in the line.
    pub col: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DiagnosticIn {
    pub line: usize,
    #[serde(default)]
    pub col: usize,
    #[serde(default)]
    pub severity: String,
    #[serde(default)]
    pub message: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PredictRequest {
    #[serde(default)]
    pub request_id: Option<String>,
    pub path: String,
    #[serde(default)]
    pub root: Option<String>,
    pub content: String,
    pub cursor: Cursor,
    #[serde(default)]
    pub diagnostics: Vec<DiagnosticIn>,
    #[serde(default)]
    pub clipboard: Option<String>,
    /// Allow a second, jump-to-edit prediction when the local window has none.
    #[serde(default = "yes")]
    pub jump: bool,
    /// Replay-only synthetic timestamp in milliseconds.
    #[serde(default)]
    pub ts_ms: Option<u64>,
}

fn yes() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize)]
pub struct ObserveRequest {
    pub path: String,
    #[serde(default)]
    pub root: Option<String>,
    pub content: String,
    #[serde(default)]
    pub cursor: Option<Cursor>,
    #[serde(default)]
    pub saved: bool,
    /// Prefill the stable prompt prefix for this file while idle.
    #[serde(default)]
    pub warm: bool,
    /// Replay-only synthetic timestamp in milliseconds.
    #[serde(default)]
    pub ts_ms: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct EditOut {
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
    pub text: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ContextSummary {
    pub root: String,
    pub path: String,
    pub prompt_chars: usize,
    pub prompt_tokens: usize,
    pub stable_prefix_chars: usize,
    pub hunks: usize,
    pub retrieval: Vec<ChunkSummary>,
    pub file_chunks: Vec<ChunkSummary>,
    pub initial_file_lines: usize,
    pub block_start_line: usize,
    pub block_lines: usize,
    pub trimmed: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChunkSummary {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub source: &'static str,
    pub score: f32,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Timings {
    pub total_ms: f64,
    pub context_ms: f64,
    pub stages: Vec<GenStats>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PredictResponse {
    pub request_id: Option<String>,
    pub status: &'static str,
    pub exit_reason: String,
    pub kind: &'static str,
    pub jump_reason: Option<&'static str>,
    pub edits: Vec<EditOut>,
    /// Edits withheld by the confidence gate (telemetry only).
    pub withheld: Vec<EditOut>,
    pub protocol: &'static str,
    pub model: String,
    pub model_sha256: String,
    pub prompt: String,
    pub completion: String,
    pub prompt_sha256: String,
    pub finish_reason: String,
    /// Served from a prefetch computed before the request arrived.
    pub prefetched: bool,
    pub confidence: Option<sweep::Confidence>,
    pub context: ContextSummary,
    pub timings: Timings,
}

pub struct Settings {
    pub model_name: String,
    pub model_sha256: String,
    pub prompt: PromptOptions,
    pub max_retrieval_chunks: usize,
    pub file_chunk_lines: usize,
    pub index_refresh_secs: u64,
    pub early_stop: bool,
    pub draft_max: usize,
    pub draft_min_match: usize,
    pub splice_anchors: usize,
    pub draft_previous: bool,
    /// Withhold edits whose weakest changed token is below this log-probability.
    pub min_confidence: f32,
    pub multiline_min_confidence: f32,
    /// Map the cursor into the original file by line and keep the shown
    /// chunk of a long file while the cursor stays well inside it.
    pub stable_window: bool,
    /// Warm requests prefill the whole local prompt, not only its stable head.
    pub warm_full_prompt: bool,
}

/// Extra finalized hunks fetched for `stable_history`, so hunks hidden by
/// the window do not shrink the visible history.
const HIDDEN_HEADROOM: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Purpose {
    /// A real request: records the state and updates sticky choices.
    Predict,
    /// Prefetch: works on a copy of the edit history.
    Speculative,
    /// Idle warm-up of a state the tracker has already observed.
    Warm,
}

/// Everything a prompt is assembled from, for one buffer state.
struct Gathered {
    display: String,
    cursor: usize,
    hunks: Vec<DiffHunk>,
    /// Trailing `hunks` from the live edit group.
    live: usize,
    original: String,
    initial_start: Option<usize>,
    block: Block,
    file_chunks: Vec<Chunk>,
    retrieval: Vec<Chunk>,
    diagnostics: Vec<Diagnostic>,
}

/// A prompt that fits the context budget, and what it was built from.
struct StagePrompt {
    built: sweep::Built,
    tokens: Vec<llama_cpp_2::token::LlamaToken>,
    block_tokens: Vec<llama_cpp_2::token::LlamaToken>,
    max_tokens: usize,
    max_newlines: usize,
    trimmed: Vec<&'static str>,
    file_chunks_used: bool,
    /// Index of the first retrieval chunk kept.
    retrieval_from: usize,
}

enum Job {
    Generate {
        generation: u64,
        prompt: Vec<llama_cpp_2::token::LlamaToken>,
        draft_source: Vec<llama_cpp_2::token::LlamaToken>,
        draft_seed: Vec<llama_cpp_2::token::LlamaToken>,
        max_tokens: usize,
        max_newlines: usize,
        /// Cache slot key (the file).
        slot: String,
        /// Tokens of the previous completion for this file (draft source).
        previous: Vec<llama_cpp_2::token::LlamaToken>,
        /// (prefill, window, anchor lines) for splicing the unchanged rest.
        splice: Option<(String, String, usize)>,
        reply: mpsc::Sender<Result<Outcome>>,
    },
    Warm {
        generation: u64,
        slot: String,
        prefix: Vec<llama_cpp_2::token::LlamaToken>,
    },
    /// Answered once every earlier job has finished (replay idle time).
    Ping(mpsc::Sender<()>),
}

pub struct Service {
    pub settings: Settings,
    tokenizer: Tokenizer,
    n_ctx: usize,
    jobs: Mutex<mpsc::Sender<Job>>,
    latest: Arc<AtomicU64>,
    tracker: Mutex<Tracker>,
    pub registry: Arc<Registry>,
    sticky: Mutex<HashMap<PathBuf, Vec<String>>>,
    clock: Mutex<Option<(u64, Instant)>>,
    /// Recent prefetched responses keyed by (path, content hash, cursor, jump).
    prefetched: Mutex<Vec<(String, PredictResponse)>>,
    sticky_file_chunks: Mutex<HashMap<PathBuf, (Instant, Vec<Chunk>)>>,
    /// Recent real answers keyed by (path, content, cursor line, jump).
    answers: Mutex<Vec<(String, (bool, PredictResponse))>>,
    /// Last completion (prefill + generated) per file, as draft tokens.
    previous: Mutex<HashMap<String, Vec<llama_cpp_2::token::LlamaToken>>>,
    /// First line of the original-file chunk last shown per file.
    chunk_starts: Mutex<HashMap<PathBuf, usize>>,
}

fn sha256_hex(text: &str) -> String {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

impl Service {
    pub fn start(model_path: &Path, config: ModelConfig, settings: Settings) -> Result<Arc<Self>> {
        let (tx, rx) = mpsc::channel::<Job>();
        let draft_max = settings.draft_max;
        let draft_min_match = settings.draft_min_match;
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(Tokenizer, usize)>>();
        let latest = Arc::new(AtomicU64::new(0));
        let worker_latest = Arc::clone(&latest);
        let path = model_path.to_path_buf();
        std::thread::Builder::new()
            .name("nes-model".into())
            .spawn(move || {
                let mut engine = match Engine::load(&path, &config) {
                    Ok(e) => {
                        let _ = ready_tx.send(Ok((e.tokenizer(), e.n_ctx())));
                        e
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                while let Ok(job) = rx.recv() {
                    match job {
                        Job::Generate { generation, prompt, draft_source, draft_seed, max_tokens, max_newlines, slot, previous, splice, reply } => {
                            if worker_latest.load(Ordering::SeqCst) != generation {
                                let _ = reply.send(Ok(Outcome::Cancelled(GenStats::default())));
                                continue;
                            }
                            let splice_text = splice.as_ref().map(|(p, w, _)| (p.clone(), w.clone()));
                            let splicer = splice.map(|(prefill, window, anchors)| {
                                move |generated: &str| sweep::splice_rest_n(&prefill, generated, &window, anchors)
                            });
                            let result = engine.generate(GenRequest {
                                prompt: &prompt,
                                draft_source: &draft_source,
                                draft_seed: &draft_seed,
                                max_tokens,
                                max_newlines,
                                draft_max,
                                draft_min_match,
                                draft_fallback: &prompt[prompt.len().saturating_sub(1024)..],
                                draft_previous: &previous,
                                generation,
                                latest: &worker_latest,
                                slot: &slot,
                                novelty: splice_text.as_ref().map(|(p, w)| (p.as_str(), w.as_str())),
                                early_stop: splicer.as_ref().map(|f| f as &dyn Fn(&str) -> Option<String>),
                            });
                            let _ = reply.send(result);
                        }
                        Job::Warm { generation, slot, prefix } => {
                            if worker_latest.load(Ordering::SeqCst) != generation {
                                continue;
                            }
                            if let Err(e) = engine.warm(&slot, &prefix, generation, &worker_latest) {
                                eprintln!("nes: warm failed: {e:#}");
                            }
                        }
                        Job::Ping(reply) => {
                            let _ = reply.send(());
                        }
                    }
                }
            })?;
        let (tokenizer, n_ctx) = ready_rx.recv().context("model worker exited")??;
        Ok(Arc::new(Self {
            tokenizer,
            n_ctx,
            jobs: Mutex::new(tx),
            latest,
            tracker: Mutex::new(Tracker::default()),
            registry: Arc::new(Registry::default()),
            sticky: Mutex::new(HashMap::new()),
            clock: Mutex::new(None),
            prefetched: Mutex::new(Vec::new()),
            previous: Mutex::new(HashMap::new()),
            sticky_file_chunks: Mutex::new(HashMap::new()),
            answers: Mutex::new(Vec::new()),
            chunk_starts: Mutex::new(HashMap::new()),
            settings,
        }))
    }

    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    pub fn reset(&self) {
        *self.tracker.lock().unwrap() = Tracker::default();
        self.chunk_starts.lock().unwrap().clear();
        self.sticky.lock().unwrap().clear();
        self.sticky_file_chunks.lock().unwrap().clear();
        self.answers.lock().unwrap().clear();
        self.prefetched.lock().unwrap().clear();
        *self.clock.lock().unwrap() = None;
    }

    /// Map a replay timestamp onto the monotonic clock.
    fn instant(&self, ts_ms: Option<u64>) -> Instant {
        let Some(ts) = ts_ms else {
            return Instant::now();
        };
        let mut clock = self.clock.lock().unwrap();
        let (origin_ts, origin) = *clock.get_or_insert((ts, Instant::now()));
        origin + std::time::Duration::from_millis(ts.saturating_sub(origin_ts))
    }

    fn resolve(&self, path: &str, root: Option<&str>) -> (PathBuf, PathBuf, String) {
        let path = PathBuf::from(path);
        let root = root
            .map(PathBuf::from)
            .filter(|r| path.starts_with(r))
            .or_else(|| index::find_root(&path))
            .unwrap_or_else(|| path.parent().map(Path::to_path_buf).unwrap_or_default());
        let display = index::rel_path(&root, &path);
        (path, root, display)
    }

    pub fn observe(&self, req: &ObserveRequest) {
        let (path, root, display) = self.resolve(&req.path, req.root.as_deref());
        let line = req.cursor.as_ref().map_or(0, |c| c.line);
        let now = self.instant(req.ts_ms);
        self.tracker.lock().unwrap().observe_at(&path, &display, &req.content, line, req.saved, now);
        self.registry.ensure_fresh(&root, self.settings.index_refresh_secs);
        self.registry.get(&root).lock().unwrap().update_buffer(&path, &req.content);
        if !req.warm {
            return;
        }
        if self.settings.warm_full_prompt {
            let cursor = req.cursor.clone().unwrap_or(Cursor { line: 0, col: 0 });
            let state = PredictRequest {
                request_id: None,
                path: req.path.clone(),
                root: req.root.clone(),
                content: req.content.clone(),
                cursor,
                diagnostics: Vec::new(),
                clipboard: None,
                jump: false,
                ts_ms: req.ts_ms,
            };
            if let Some(tokens) = self.build_tokens(&state) {
                self.warm(&display, tokens);
            }
            return;
        }
        let original = self.tracker.lock().unwrap().original(&path).map(str::to_string);
        let original = original.unwrap_or_else(|| req.content.clone());
        let cursor = req.cursor.as_ref().map_or(0, |c| byte_offset(&req.content, c.line, c.col));
        let file_chunks = self.file_chunks(&path);
        let mut prefix: String = file_chunks.iter().map(Chunk::render).collect();
        let initial = match self.initial_start(&path, &req.content, &original, line, true) {
            Some(start) => sweep::lines_from(&original, start, &self.settings.prompt),
            None => sweep::lines_around_cursor(&original, cursor.min(original.len()), &self.settings.prompt),
        };
        prefix.push_str(&format!("<|file_sep|>{display}\n{initial}"));
        self.warm(&display, self.tokenizer.tokenize(&prefix));
    }

    /// Prefill `tokens` into the file's cache in the background. Warm jobs
    /// do not bump the generation; any prediction cancels them.
    fn warm(&self, slot: &str, tokens: Vec<llama_cpp_2::token::LlamaToken>) {
        let generation = self.latest.load(Ordering::SeqCst);
        let _ = self.jobs.lock().unwrap().send(Job::Warm { generation, slot: slot.to_string(), prefix: tokens });
    }

    /// Wait until queued work (warms) has finished, at most `timeout`.
    pub fn wait_idle(&self, timeout: Duration) {
        let (tx, rx) = mpsc::channel();
        if self.jobs.lock().unwrap().send(Job::Ping(tx)).is_ok() {
            let _ = rx.recv_timeout(timeout);
        }
    }

    /// Full local prompt for an observed buffer state, as a prediction
    /// would build it.
    fn build_tokens(&self, req: &PredictRequest) -> Option<Vec<llama_cpp_2::token::LlamaToken>> {
        let resolved = self.resolve(&req.path, req.root.as_deref());
        let g = self.gather(req, &resolved, Purpose::Warm).ok()?;
        let p = self.stage_prompt(&g, &req.content, g.cursor, &g.block, &g.retrieval).ok()?;
        Some(p.tokens)
    }

    /// First line of the original-file chunk to show, or None for the old
    /// byte-offset mapping. Long files keep their chunk while the cursor
    /// stays inside it (`remember` records the choice).
    fn initial_start(&self, path: &Path, content: &str, original: &str, line: usize, remember: bool) -> Option<usize> {
        if !self.settings.stable_window {
            return None;
        }
        let opts = &self.settings.prompt;
        let total = original.lines().count();
        if total <= opts.whole_file_lines {
            return Some(0);
        }
        let mapped = history::map_line(content, original, line);
        let mut starts = self.chunk_starts.lock().unwrap();
        let start = sweep::chunk_start(total, mapped, opts, starts.get(path).copied());
        if remember {
            starts.insert(path.to_path_buf(), start);
        }
        Some(start)
    }

    /// The recent-file snippet opens the prompt, so a new one invalidates the
    /// whole cached prefix. Each file keeps the snippet it was first given for
    /// a while, so switching between files reuses their caches.
    fn file_chunks(&self, path: &Path) -> Vec<Chunk> {
        const STICKY: std::time::Duration = std::time::Duration::from_secs(600);
        if let Some((at, chunks)) = self.sticky_file_chunks.lock().unwrap().get(path) {
            if at.elapsed() < STICKY {
                return chunks.clone();
            }
        }
        let chunks = self.fresh_file_chunks(path);
        self.sticky_file_chunks
            .lock()
            .unwrap()
            .insert(path.to_path_buf(), (Instant::now(), chunks.clone()));
        chunks
    }

    fn fresh_file_chunks(&self, path: &Path) -> Vec<Chunk> {
        let tracker = self.tracker.lock().unwrap();
        let n = self.settings.file_chunk_lines;
        if n == 0 {
            return Vec::new();
        }
        tracker
            .recent_visits(path, 1)
            .into_iter()
            .filter_map(|v| {
                let lines: Vec<&str> = v.content.lines().collect();
                if lines.is_empty() {
                    return None;
                }
                let start = v.cursor_line.saturating_sub(n / 3).min(lines.len() - 1);
                let end = (start + n).min(lines.len());
                Some(Chunk {
                    path: v.display.clone(),
                    start_line: start + 1,
                    end_line: end,
                    content: lines[start..end].join("\n"),
                    source: "recent_file",
                    score: 0.0,
                })
            })
            .collect()
    }

    fn cache_key(req: &PredictRequest) -> String {
        format!("{}\u{0}{}\u{0}{}:{}\u{0}{}", req.path, sha256_hex(&req.content), req.cursor.line, req.cursor.col, req.jump)
    }

    pub fn predict(&self, req: PredictRequest) -> Result<PredictResponse> {
        let key = Self::cache_key(&req);
        let hit = {
            let mut cache = self.prefetched.lock().unwrap();
            cache.iter().position(|(k, _)| *k == key).map(|i| cache.remove(i).1)
        };
        // Same content and line as a recent answer (leaving insert mode,
        // moving along the line): the window and edits are unchanged.
        let line_key = Self::line_key(&req);
        let hit = hit.or_else(|| {
            let cache = self.answers.lock().unwrap();
            cache
                .iter()
                .rev()
                .find(|(k, (jumped, r))| {
                    // A stored edit answers either kind of request; an empty
                    // answer without the jump stage cannot answer a jump request.
                    *k == line_key && (!r.edits.is_empty() || *jumped || !req.jump)
                })
                .map(|(_, (_, r))| r.clone())
        });
        if let Some(mut cached) = hit {
            // Record the state as usual and cancel any background work.
            self.latest.fetch_add(1, Ordering::SeqCst);
            let (path, _root, display) = self.resolve(&req.path, req.root.as_deref());
            let now = self.instant(req.ts_ms);
            self.tracker.lock().unwrap().observe_at(&path, &display, &req.content, req.cursor.line, false, now);
            cached.request_id = req.request_id.clone();
            cached.prefetched = true;
            cached.timings.total_ms = 0.0;
            return Ok(cached);
        }
        let jump = req.jump;
        let resp = self.predict_inner(req, false)?;
        if resp.status == "ok" || resp.status == "empty" {
            let mut cache = self.answers.lock().unwrap();
            cache.retain(|(k, _)| *k != line_key);
            cache.push((line_key, (jump, resp.clone())));
            let excess = cache.len().saturating_sub(16);
            cache.drain(..excess);
        }
        Ok(resp)
    }

    fn line_key(req: &PredictRequest) -> String {
        format!("{}\u{0}{}\u{0}{}", req.path, sha256_hex(&req.content), req.cursor.line)
    }

    /// Compute a prediction for a likely next state (after the shown edit is
    /// accepted) without touching the real edit history or cancelling work.
    pub fn prefetch(&self, req: PredictRequest) -> Result<()> {
        let key = Self::cache_key(&req);
        if self.prefetched.lock().unwrap().iter().any(|(k, _)| *k == key) {
            return Ok(());
        }
        let resp = self.predict_inner(req, true)?;
        if resp.status == "ok" || resp.status == "empty" {
            let mut cache = self.prefetched.lock().unwrap();
            cache.push((key, resp));
            let excess = cache.len().saturating_sub(4);
            cache.drain(..excess);
        }
        Ok(())
    }

    fn predict_inner(&self, req: PredictRequest, speculative: bool) -> Result<PredictResponse> {
        let started = Instant::now();
        let mut resp = PredictResponse {
            request_id: req.request_id.clone(),
            protocol: PROTOCOL,
            model: self.settings.model_name.clone(),
            model_sha256: self.settings.model_sha256.clone(),
            kind: "local",
            ..Default::default()
        };
        // Real requests claim a generation first so older queued work is
        // abandoned at once; speculative work runs at the current generation
        // and is abandoned by the next real request.
        let generation = if speculative {
            self.latest.load(Ordering::SeqCst)
        } else {
            self.latest.fetch_add(1, Ordering::SeqCst) + 1
        };
        let resolved = self.resolve(&req.path, req.root.as_deref());
        resp.context.root = resolved.1.to_string_lossy().into_owned();
        resp.context.path = resolved.2.clone();
        let purpose = if speculative { Purpose::Speculative } else { Purpose::Predict };
        let g = match self.gather(&req, &resolved, purpose) {
            Ok(g) => g,
            Err(reason) => {
                resp.status = "disabled";
                resp.exit_reason = reason.into();
                return Ok(resp);
            }
        };
        let content = req.content.as_str();
        let cursor = g.cursor;
        resp.timings.context_ms = ms(started);

        let (stage, local_tokens) = self.run_stage(generation, &g, content, cursor, &g.block, &g.retrieval, &mut resp)?;
        let mut outcome = match stage {
            None => {
                resp.status = "cancelled";
                resp.exit_reason = "superseded".into();
                resp.timings.total_ms = ms(started);
                return Ok(resp);
            }
            Some(p) => p,
        };
        let local_empty = outcome.edits.is_empty();
        if local_empty && req.jump && !g.hunks.is_empty() {
            if let Some(j) = sweep::find_jump_block(content, &g.hunks, &g.display, cursor, &g.diagnostics, self.settings.prompt.lines_before, self.settings.prompt.lines_after) {
                if !sweep::should_disable_for_block(&j.block.text) && j.cursor >= j.block.start {
                    let mut jump_retrieval = g.retrieval.clone();
                    if let Some(msg) = &j.diagnostic {
                        jump_retrieval.insert(0, Chunk { path: "diagnostics".into(), start_line: 1, end_line: 2, content: msg.clone(), source: "diagnostics", score: 0.0 });
                    }
                    let local_prompt = std::mem::take(&mut resp.prompt);
                    let local_completion = std::mem::take(&mut resp.completion);
                    let (jumped, _) = self.run_stage(generation, &g, content, j.cursor, &j.block, &jump_retrieval, &mut resp)?;
                    match jumped {
                        Some(p) if !p.edits.is_empty() && p.edits[0].text.trim().len() + (p.edits[0].end - p.edits[0].start) > 0 => {
                            resp.kind = "jump";
                            resp.jump_reason = Some(j.reason);
                            outcome = p;
                        }
                        Some(_) => {
                            resp.prompt = local_prompt;
                            resp.completion = local_completion;
                        }
                        None => {
                            resp.status = "cancelled";
                            resp.exit_reason = "superseded".into();
                            resp.timings.total_ms = ms(started);
                            return Ok(resp);
                        }
                    }
                    // The jump prompt shares the file's cache slot and
                    // replaced the local prompt in it; restore that while
                    // idle so the next keystroke here stays incremental.
                    self.warm(&g.display, local_tokens);
                }
            }
        }
        resp.exit_reason = outcome.exit_reason.to_string();
        resp.edits = outcome.edits.iter().map(|e| edit_out(content, e)).collect();
        // Multi-line rewrites are wrong more often than single-line edits, so
        // they need more confidence to be shown.
        let multiline = outcome.edits.first().is_some_and(|e| {
            let old = &content[e.start..e.end];
            old.matches('\n').count() > 1 || (!old.is_empty() && e.text.matches('\n').count() > 1)
        });
        let threshold = if multiline {
            self.settings.min_confidence.max(self.settings.multiline_min_confidence)
        } else {
            self.settings.min_confidence
        };
        let weak = resp.confidence.as_ref().is_some_and(|c| c.changed_tokens > 0 && c.min_logprob < threshold);
        if weak && !resp.edits.is_empty() {
            resp.withheld = std::mem::take(&mut resp.edits);
            resp.exit_reason = "low_confidence".into();
        }
        resp.status = if resp.edits.is_empty() { "empty" } else { "ok" };
        resp.prompt_sha256 = sha256_hex(&resp.prompt);
        resp.timings.total_ms = ms(started);
        Ok(resp)
    }

    /// Assemble prompt inputs for a buffer state: edit history, original
    /// file, window and retrieval. Err holds the reason the file or window
    /// is disabled.
    fn gather(&self, req: &PredictRequest, resolved: &(PathBuf, PathBuf, String), purpose: Purpose) -> Result<Gathered, &'static str> {
        let (path, root, display) = resolved;
        let opts = &self.settings.prompt;
        let content = req.content.as_str();
        let cursor = byte_offset(content, req.cursor.line, req.cursor.col);
        if let Some(reason) = sweep::should_disable_file(content) {
            return Err(reason);
        }
        let (hunks, live, original) = {
            let now = self.instant(req.ts_ms);
            let mut real = self.tracker.lock().unwrap();
            let mut copy;
            let t: &mut Tracker = if purpose == Purpose::Speculative {
                copy = real.clone();
                &mut copy
            } else {
                &mut real
            };
            if purpose != Purpose::Warm {
                t.observe_at(path, display, content, req.cursor.line, false, now);
            }
            let (h, live) = if opts.stable_history {
                t.recent_split_at(opts.max_hunks + HIDDEN_HEADROOM, now)
            } else {
                (t.recent_hunks_at(opts.max_hunks, now), 0)
            };
            let o = t.original(path).unwrap_or(content).to_string();
            (h, live, o)
        };
        self.registry.ensure_fresh(root, self.settings.index_refresh_secs);
        let ws = self.registry.get(root);
        ws.lock().unwrap().update_buffer(path, content);
        let diagnostics: Vec<Diagnostic> = req
            .diagnostics
            .iter()
            .map(|d| Diagnostic { line: d.line, col: d.col, severity: d.severity.clone(), message: d.message.clone() })
            .collect();
        let block = sweep::block_at_cursor(content, cursor, opts.lines_before, opts.lines_after);
        if sweep::should_disable_for_block(&block.text) {
            return Err("long_lines_code_block");
        }
        let file_chunks = self.file_chunks(path);
        let recent_files: Vec<PathBuf> = self.tracker.lock().unwrap().recent_visits(path, 5).into_iter().map(|v| v.path).collect();
        let initial_start = self.initial_start(path, content, &original, req.cursor.line, purpose != Purpose::Speculative);
        let visible = match initial_start {
            Some(start) => {
                let total = original.lines().count();
                if total <= opts.whole_file_lines { (0, total) } else { (start, (start + opts.chunk_lines.max(2)).min(total)) }
            }
            None => visible_range(&original, cursor.min(original.len()), opts),
        };
        let previous = self.sticky.lock().unwrap().get(path).cloned().unwrap_or_default();
        let mut retrieval = {
            let ws = ws.lock().unwrap();
            retrieval::definition_chunks(
                &ws,
                &RetrievalInput {
                    path,
                    content,
                    cursor_line: req.cursor.line,
                    visible,
                    hunks: &hunks,
                    recent_files: &recent_files,
                    previous: &previous,
                    max_chunks: self.settings.max_retrieval_chunks,
                },
            )
        };
        if purpose != Purpose::Warm {
            self.sticky
                .lock()
                .unwrap()
                .insert(path.clone(), retrieval.iter().map(retrieval::chunk_key).collect());
        }
        // Best definition last; diagnostics and clipboard sit closest to the edit.
        retrieval.reverse();
        if let Some(c) = req.clipboard.as_deref().and_then(retrieval::clipboard_chunk) {
            retrieval.push(c);
        }
        if let Some(c) = retrieval::diagnostics_chunk(display, &diagnostics, req.cursor.line) {
            retrieval.push(c);
        }
        Ok(Gathered {
            display: display.clone(),
            cursor,
            hunks,
            live,
            original,
            initial_start,
            block,
            file_chunks,
            retrieval,
            diagnostics,
        })
    }

    /// Build the prompt for a window, dropping context until it fits.
    fn stage_prompt(&self, g: &Gathered, content: &str, cursor: usize, block: &Block, retrieval: &[Chunk]) -> Result<StagePrompt> {
        let block_tokens = self.tokenizer.tokenize_plain(&block.text);
        let max_tokens = (block_tokens.len() * 2 + 160).min(sweep::OUTPUT_MAX_TOKENS);
        let max_newlines = block.text.lines().count() + 20;
        let budget = self.n_ctx.saturating_sub(max_tokens + self.settings.draft_max + 8);
        let mut opts = self.settings.prompt.clone();
        let mut fc: &[Chunk] = &g.file_chunks;
        let mut rc: &[Chunk] = retrieval;
        let mut trimmed = Vec::new();
        loop {
            let built = sweep::build_prompt(&PromptInput {
                path: &g.display,
                content,
                original: &g.original,
                cursor,
                block,
                hunks: &g.hunks,
                file_chunks: fc,
                retrieval: rc,
                opts: &opts,
                live: g.live,
                initial_start: g.initial_start,
            });
            let tokens = self.tokenizer.tokenize(&built.prompt);
            if tokens.len() <= budget {
                return Ok(StagePrompt {
                    built,
                    tokens,
                    block_tokens,
                    max_tokens,
                    max_newlines,
                    trimmed,
                    file_chunks_used: !fc.is_empty(),
                    retrieval_from: retrieval.len() - rc.len(),
                });
            }
            if !fc.is_empty() {
                fc = &[];
                trimmed.push("file_chunks");
            } else if !rc.is_empty() {
                rc = &rc[rc.len() / 2 + rc.len() % 2..];
                trimmed.push("retrieval");
            } else if opts.whole_file_lines > 40 || opts.chunk_lines > 40 {
                opts.whole_file_lines /= 2;
                opts.chunk_lines = (opts.chunk_lines / 2).max(40);
                trimmed.push("initial_file");
            } else if opts.max_hunks > 1 {
                opts.max_hunks /= 2;
                trimmed.push("hunks");
            } else {
                return Err(anyhow!("prompt of {} tokens exceeds budget {budget}", tokens.len()));
            }
        }
    }

    /// Run one prediction stage. Returns the processed outcome (None when
    /// superseded) and the prompt tokens.
    #[allow(clippy::too_many_arguments)]
    fn run_stage(
        &self,
        generation: u64,
        g: &Gathered,
        content: &str,
        cursor: usize,
        block: &Block,
        retrieval: &[Chunk],
        resp: &mut PredictResponse,
    ) -> Result<(Option<sweep::Processed>, Vec<llama_cpp_2::token::LlamaToken>)> {
        let StagePrompt { built, tokens, block_tokens, max_tokens, max_newlines, trimmed, file_chunks_used, retrieval_from } =
            self.stage_prompt(g, content, cursor, block, retrieval)?;
        let rc = &retrieval[retrieval_from..];
        let display = g.display.as_str();
        resp.context.prompt_chars = built.prompt.len();
        resp.context.prompt_tokens = tokens.len();
        resp.context.stable_prefix_chars = built.stable_prefix;
        resp.context.hunks = built.hunks_used;
        resp.context.initial_file_lines = built.initial_file_lines;
        resp.context.block_start_line = block.start_line;
        resp.context.block_lines = block.text.lines().count();
        resp.context.trimmed = trimmed;
        resp.context.retrieval = rc[rc.len() - built.retrieval_used..].iter().map(summary).collect();
        resp.context.file_chunks = if file_chunks_used { g.file_chunks.iter().map(summary).collect() } else { Vec::new() };
        let draft_seed = self.tokenizer.tokenize_plain(&built.prefill);
        let early = self.settings.early_stop;
        let (reply_tx, reply_rx) = mpsc::channel();
        self.jobs
            .lock()
            .unwrap()
            .send(Job::Generate {
                generation,
                prompt: tokens.clone(),
                draft_source: block_tokens,
                draft_seed,
                max_tokens,
                max_newlines,
                slot: display.to_string(),
                previous: if self.settings.draft_previous {
                    self.previous.lock().unwrap().get(display).cloned().unwrap_or_default()
                } else {
                    Vec::new()
                },
                splice: early.then(|| (built.prefill.clone(), block.text.clone(), self.settings.splice_anchors)),
                reply: reply_tx,
            })
            .map_err(|_| anyhow!("model worker stopped"))?;
        let outcome = reply_rx.recv().map_err(|_| anyhow!("model worker dropped request"))??;
        resp.prompt = built.prompt.clone();
        match outcome {
            Outcome::Cancelled(stats) => {
                resp.timings.stages.push(stats);
                Ok((None, tokens))
            }
            Outcome::Done(gen_out) => {
                resp.timings.stages.push(gen_out.stats.clone());
                resp.completion = gen_out.text.clone();
                resp.finish_reason = gen_out.finish_reason.to_string();
                let prev = self.tokenizer.tokenize_plain(&format!("{}{}", built.prefill, gen_out.text));
                self.previous.lock().unwrap().insert(display.to_string(), prev);
                resp.confidence = Some(sweep::change_confidence(
                    &built.prefill,
                    &gen_out.text,
                    &block.text,
                    &gen_out.token_logprobs,
                    &gen_out.token_offsets,
                ));
                let processed = sweep::process_completion(&gen_out.text, gen_out.finish_reason, &built, block, content, cursor);
                Ok((Some(processed), tokens))
            }
        }
    }
}

fn summary(c: &Chunk) -> ChunkSummary {
    ChunkSummary { path: c.path.clone(), start_line: c.start_line, end_line: c.end_line, source: c.source, score: c.score }
}

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

/// Byte offset for a 0-based line and byte column, clamped to the line.
pub fn byte_offset(content: &str, line: usize, col: usize) -> usize {
    let mut offset = 0;
    for (i, l) in content.split_inclusive('\n').enumerate() {
        if i == line {
            let body = l.strip_suffix('\n').unwrap_or(l);
            let mut c = col.min(body.len());
            while !body.is_char_boundary(c) {
                c -= 1;
            }
            return offset + c;
        }
        offset += l.len();
    }
    content.len()
}

fn position(content: &str, offset: usize) -> (usize, usize) {
    let before = &content[..offset.min(content.len())];
    let line = before.matches('\n').count();
    let col = before.len() - before.rfind('\n').map_or(0, |i| i + 1);
    (line, col)
}

fn edit_out(content: &str, e: &Edit) -> EditOut {
    let (sl, sc) = position(content, e.start);
    let (el, ec) = position(content, e.end);
    EditOut { start_byte: e.start, end_byte: e.end, start_line: sl, start_col: sc, end_line: el, end_col: ec, text: e.text.clone() }
}

/// Lines (0-based, inclusive) of the current file shown as initial context.
fn visible_range(original: &str, cursor: usize, opts: &PromptOptions) -> (usize, usize) {
    let total = original.lines().count();
    if total <= opts.whole_file_lines {
        return (0, total);
    }
    let chunk = opts.chunk_lines.max(2);
    let stride = chunk / 2;
    let cursor_line = sweep::line_of_offset(original, cursor) as i64;
    let ideal = cursor_line - (chunk / 2) as i64;
    let index = ((ideal as f64) / stride as f64).round().max(0.0) as usize;
    let start = (index * stride).min(total.saturating_sub(1));
    (start, (start + chunk).min(total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_round_trip() {
        let s = "ab\ncdé\nf";
        assert_eq!(byte_offset(s, 1, 2), 5);
        assert_eq!(position(s, 5), (1, 2));
        assert_eq!(byte_offset(s, 1, 3), 5); // inside é clamps back
        assert_eq!(byte_offset(s, 9, 0), s.len());
    }
}
