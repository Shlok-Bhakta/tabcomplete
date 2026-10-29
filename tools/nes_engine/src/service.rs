//! Prediction service: editor state in, edit proposals out.
//!
//! One worker thread owns the llama context. Requests increment a generation
//! counter; the worker abandons any job whose generation is no longer the
//! latest, so a burst of keystrokes costs one prediction, and the KV prefix
//! evaluated by an abandoned job is kept for the next one.

use crate::history::{self, DiffHunk, Tracker};
use crate::index::{self, Lang, Registry};
use crate::model::{Engine, GenRequest, GenStats, ModelConfig, Outcome, Tokenizer};
use crate::retrieval::{self, RetrievalInput};
use crate::sweep::{self, Block, Chunk, Diagnostic, Edit, PromptInput, PromptOptions, Provisional};
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
    /// The original-file snapshot was advanced after this request.
    pub original_refreshed: bool,
    /// The edit was trimmed to leave text typed in the live edit unchanged.
    pub live_edit_trimmed: bool,
    /// The jump-stage candidate, when the jump search found one.
    pub jump: Option<JumpSummary>,
}

#[derive(Clone, Debug, Serialize)]
pub struct JumpSummary {
    pub reason: &'static str,
    /// First line (0-based) of the jump window.
    pub block_start_line: usize,
    /// Lines between the cursor and the jump target.
    pub distance: usize,
    /// edit (the jump stage proposed one), none, skipped (`--jump-reasons`,
    /// `--jump-max-distance`), disabled (long lines) or cancelled.
    pub outcome: &'static str,
    /// The local stage was not run again: a request for the same content and
    /// line had just found nothing (`--jump-reuse-local`).
    pub local_reused: bool,
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
    /// When the shown edit was first displayable: the first provisional
    /// edit the final agrees with, else `total_ms`. None when nothing is shown.
    pub first_edit_ms: Option<f64>,
    /// First provisional edit, whether or not the final kept it.
    pub provisional_ms: Option<f64>,
    /// A provisional edit was emitted and the final edit does not extend it.
    pub provisional_retracted: bool,
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
    /// Characters of a partially generated line that may serve as the
    /// splice anchor (0: anchors are complete lines only).
    pub splice_partial_chars: usize,
    pub draft_previous: bool,
    /// Withhold edits whose weakest changed token is below this log-probability.
    pub min_confidence: f32,
    pub multiline_min_confidence: f32,
    /// Gate for insertions and appends that keep the existing text.
    pub insert_min_confidence: f32,
    /// Map the cursor into the original file by line and keep the shown
    /// chunk of a long file while the cursor stays well inside it.
    pub stable_window: bool,
    /// Warm requests prefill the whole local prompt, not only its stable head.
    pub warm_full_prompt: bool,
    /// Retrieve definitions for files of languages the index does not parse.
    pub retrieval_unindexed: bool,
    /// Advance a file's original snapshot past edits the history no longer
    /// shows, when it is clearly stale (`Tracker::refresh_original_at`).
    pub refresh_original: bool,
    /// Trim edits that would rewrite text typed in the live edit group.
    pub protect_live_edit: bool,
    /// Run the jump stage in the slot's scratch sequence, keeping the local
    /// prompt's cache (no re-warm afterwards).
    pub jump_scratch: bool,
    /// The jump prompt repeats the local prompt's history and adds the hunks
    /// the local window hid, sharing its cached prefix through the history.
    pub jump_shared_history: bool,
    /// Prefill chunk of the jump stage (0: `--prefill-chunk`); smaller
    /// chunks let the next keystroke cancel it sooner.
    pub jump_prefill_chunk: usize,
    /// Jump reasons that may run the jump stage (empty: all).
    pub jump_reasons: Vec<String>,
    /// Skip jump targets farther than this many lines (0: no limit).
    pub jump_max_distance: usize,
    /// A jump request for the content and line of a just-answered local
    /// request that found nothing runs only the jump stage.
    pub jump_reuse_local: bool,
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
    session_base: Option<String>,
    original: String,
    initial_start: Option<usize>,
    block: Block,
    file_chunks: Vec<Chunk>,
    retrieval: Vec<Chunk>,
    diagnostics: Vec<Diagnostic>,
    /// Byte ranges typed in the live edit group (`protect_live_edit`).
    typed: Vec<(usize, usize)>,
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
        splice: Option<(String, String, usize, usize)>,
        /// Send the output after every decode batch.
        progress: bool,
        /// Evaluate in the scratch sequence, leaving the file cache intact.
        scratch: bool,
        prefill_chunk: usize,
        reply: mpsc::Sender<Reply>,
    },
    Warm {
        generation: u64,
        slot: String,
        prefix: Vec<llama_cpp_2::token::LlamaToken>,
    },
    /// Answered once every earlier job has finished (replay idle time).
    Ping(mpsc::Sender<()>),
}

/// Messages from the model worker for one generation.
enum Reply {
    /// Output so far (longest valid UTF-8 prefix), token log-probabilities
    /// and token start offsets.
    Progress { text: String, logprobs: Vec<f32>, offsets: Vec<usize> },
    Done(Result<Outcome>),
}

/// Provisional edits of one request: what was emitted and when.
struct Early<'a> {
    started: Instant,
    min_confidence: f32,
    sink: Option<&'a dyn Fn(&EditOut)>,
    shown: Option<Edit>,
    first_ms: Option<f64>,
    done: bool,
}

impl Early<'_> {
    #[allow(clippy::too_many_arguments)]
    fn update(&mut self, generated: &str, logprobs: &[f32], offsets: &[usize], built: &sweep::Built, block: &Block, content: &str, cursor: usize) {
        if self.done {
            return;
        }
        match sweep::provisional_ghost(generated, logprobs, offsets, built, block, content, cursor, self.min_confidence) {
            Provisional::Pending => {}
            Provisional::Never => self.done = true,
            Provisional::Ghost { edit, complete, .. } => {
                if self.shown.as_ref() != Some(&edit) {
                    self.first_ms.get_or_insert_with(|| ms(self.started));
                    if let Some(sink) = self.sink {
                        sink(&edit_out(content, &edit));
                    }
                    self.shown = Some(edit);
                }
                self.done = complete;
            }
        }
    }

    /// Fill the first-edit timings once the final edits are known.
    fn settle(&self, resp: &mut PredictResponse) {
        let kept = match (&self.shown, resp.edits.first()) {
            (Some(p), Some(f)) => f.start_byte == p.start && f.end_byte == p.end && f.text.starts_with(&p.text),
            _ => false,
        };
        resp.timings.provisional_ms = self.first_ms;
        resp.timings.provisional_retracted = self.first_ms.is_some() && !kept;
        resp.timings.first_edit_ms = if resp.edits.is_empty() {
            None
        } else if kept {
            self.first_ms
        } else {
            Some(resp.timings.total_ms)
        };
    }
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
                        Job::Generate {
                            generation,
                            prompt,
                            draft_source,
                            draft_seed,
                            max_tokens,
                            max_newlines,
                            slot,
                            previous,
                            splice,
                            progress,
                            scratch,
                            prefill_chunk,
                            reply,
                        } => {
                            if worker_latest.load(Ordering::SeqCst) != generation {
                                let _ = reply.send(Reply::Done(Ok(Outcome::Cancelled(GenStats::default()))));
                                continue;
                            }
                            let on_progress = |text: &[u8], logprobs: &[f32], offsets: &[usize]| {
                                // A token may end inside a multi-byte character.
                                let valid = match std::str::from_utf8(text) {
                                    Ok(t) => t,
                                    Err(e) => std::str::from_utf8(&text[..e.valid_up_to()]).unwrap_or(""),
                                };
                                let _ = reply.send(Reply::Progress { text: valid.to_string(), logprobs: logprobs.to_vec(), offsets: offsets.to_vec() });
                            };
                            let splice_text = splice.as_ref().map(|(p, w, _, _)| (p.clone(), w.clone()));
                            let every_batch = splice.as_ref().is_some_and(|s| s.3 > 0);
                            let splicer = splice.map(|(prefill, window, anchors, partial)| {
                                move |generated: &str| sweep::splice_rest_n(&prefill, generated, &window, anchors, partial)
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
                                early_stop_every_batch: every_batch,
                                progress: progress.then_some(&on_progress as &dyn Fn(&[u8], &[f32], &[usize])),
                                scratch,
                                prefill_chunk,
                            });
                            let _ = reply.send(Reply::Done(result));
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
            tracker: Mutex::new(Tracker::new(settings.prompt.session_base)),
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
        *self.tracker.lock().unwrap() = Tracker::new(self.settings.prompt.session_base);
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
            if let Some((mut tokens, shown_from)) = self.build_tokens(&state) {
                if self.refresh_original(&path, shown_from, now) {
                    if let Some((fresh, _)) = self.build_tokens(&state) {
                        tokens = fresh;
                    }
                }
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
    /// would build it, and the oldest edit group it shows.
    fn build_tokens(&self, req: &PredictRequest) -> Option<(Vec<llama_cpp_2::token::LlamaToken>, Option<u64>)> {
        let resolved = self.resolve(&req.path, req.root.as_deref());
        let g = self.gather(req, &resolved, Purpose::Warm).ok()?;
        let p = self.stage_prompt(&g, &req.content, g.cursor, &g.block, &g.retrieval, None).ok()?;
        Some((p.tokens, p.built.shown_from))
    }

    /// Advance the original snapshot of `path` when a prompt showing edit
    /// groups from `shown_from` on leaves it clearly stale. Returns true when
    /// it changed; the caller re-warms the prompt.
    fn refresh_original(&self, path: &Path, shown_from: Option<u64>, now: Instant) -> bool {
        self.settings.refresh_original && self.tracker.lock().unwrap().refresh_original_at(path, shown_from, now)
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
        self.predict_streaming(req, None)
    }

    /// `predict`, calling `sink` with each provisional edit (ghost text at
    /// the cursor that grows as the cursor line is decoded) before the
    /// final response is ready.
    pub fn predict_streaming(&self, req: PredictRequest, sink: Option<&dyn Fn(&EditOut)>) -> Result<PredictResponse> {
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
            cached.timings.first_edit_ms = (!cached.edits.is_empty()).then_some(0.0);
            cached.timings.provisional_ms = None;
            cached.timings.provisional_retracted = false;
            return Ok(cached);
        }
        // Leaving insert mode asks again, with jumps, for the content and
        // line the last keystroke was answered for.
        let local_known_empty = req.jump && self.settings.jump_reuse_local && {
            let cache = self.answers.lock().unwrap();
            cache.iter().rev().find(|(k, _)| *k == line_key).is_some_and(|(_, (jumped, r))| {
                !*jumped && r.kind == "local" && r.status == "empty" && r.edits.is_empty() && r.withheld.is_empty()
            })
        };
        let jump = req.jump;
        let resp = self.predict_inner(req, false, sink, local_known_empty)?;
        if resp.status == "ok" || resp.status == "empty" {
            let mut cache = self.answers.lock().unwrap();
            cache.retain(|(k, _)| *k != line_key);
            cache.push((line_key, (jump, resp.clone())));
            let excess = cache.len().saturating_sub(16);
            cache.drain(..excess);
        }
        Ok(resp)
    }

    fn jump_allowed(&self, reason: &str, distance: usize) -> bool {
        let s = &self.settings;
        (s.jump_reasons.is_empty() || s.jump_reasons.iter().any(|r| r == reason)) && (s.jump_max_distance == 0 || distance <= s.jump_max_distance)
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
        let resp = self.predict_inner(req, true, None, false)?;
        if resp.status == "ok" || resp.status == "empty" {
            let mut cache = self.prefetched.lock().unwrap();
            cache.push((key, resp));
            let excess = cache.len().saturating_sub(4);
            cache.drain(..excess);
        }
        Ok(())
    }

    /// `local_known_empty`: the local stage is known to find nothing for this
    /// state (`jump_reuse_local`) and is skipped.
    fn predict_inner(
        &self,
        req: PredictRequest,
        speculative: bool,
        sink: Option<&dyn Fn(&EditOut)>,
        local_known_empty: bool,
    ) -> Result<PredictResponse> {
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

        // Provisional edits come from the local stage of real requests; they
        // are computed without a sink too, for the first-edit timing.
        // Provisional edits are insertions at the cursor.
        let mut early = (!speculative).then(|| Early {
            started,
            min_confidence: self.settings.min_confidence.min(self.settings.insert_min_confidence),
            sink,
            shown: None,
            first_ms: None,
            done: false,
        });
        let (stage, local_tokens, shown_from) = if local_known_empty {
            let p = self.stage_prompt(&g, content, cursor, &g.block, &g.retrieval, None)?;
            self.describe_stage(&g, &g.block, &g.retrieval, &p, &mut resp);
            resp.prompt = p.built.prompt.clone();
            let empty = sweep::Processed { edits: Vec::new(), exit_reason: "local_reused", completion: String::new() };
            (Some(empty), p.tokens, p.built.shown_from)
        } else {
            self.run_stage(generation, &g, content, cursor, &g.block, &g.retrieval, &mut resp, early.as_mut(), None)?
        };
        let mut outcome = match stage {
            None => {
                resp.status = "cancelled";
                resp.exit_reason = "superseded".into();
                resp.timings.total_ms = ms(started);
                return Ok(resp);
            }
            Some(p) => p,
        };
        // A stale snapshot is advanced after the answer and the prompt built
        // from it warmed while idle, so the next request finds it cached.
        let refreshed = !speculative && self.refresh_original(&resolved.0, shown_from, self.instant(req.ts_ms));
        let local_empty = outcome.edits.is_empty();
        if local_empty && req.jump && !g.hunks.is_empty() {
            if let Some(j) = sweep::find_jump_block(content, &g.hunks, &g.display, cursor, &g.diagnostics, self.settings.prompt.lines_before, self.settings.prompt.lines_after) {
                let distance = sweep::line_of_offset(content, j.cursor).abs_diff(sweep::line_of_offset(content, cursor));
                let mut summary = JumpSummary {
                    reason: j.reason,
                    block_start_line: j.block.start_line,
                    distance,
                    outcome: "none",
                    local_reused: local_known_empty,
                };
                if sweep::should_disable_for_block(&j.block.text) || j.cursor < j.block.start {
                    summary.outcome = "disabled";
                } else if !self.jump_allowed(j.reason, distance) {
                    summary.outcome = "skipped";
                } else {
                    let mut jump_retrieval = g.retrieval.clone();
                    if let Some(msg) = &j.diagnostic {
                        jump_retrieval.insert(0, Chunk { path: "diagnostics".into(), start_line: 1, end_line: 2, content: msg.clone(), source: "diagnostics", score: 0.0 });
                    }
                    let local_prompt = std::mem::take(&mut resp.prompt);
                    let local_completion = std::mem::take(&mut resp.completion);
                    let local_context = resp.context.clone();
                    let local_confidence = resp.confidence.take();
                    let local_finish = std::mem::take(&mut resp.finish_reason);
                    let (jumped, _, _) = self.run_stage(generation, &g, content, j.cursor, &j.block, &jump_retrieval, &mut resp, None, Some(&g.block))?;
                    match jumped {
                        Some(p) if !p.edits.is_empty() && p.edits[0].text.trim().len() + (p.edits[0].end - p.edits[0].start) > 0 => {
                            resp.kind = "jump";
                            resp.jump_reason = Some(j.reason);
                            summary.outcome = "edit";
                            outcome = p;
                        }
                        Some(_) => {
                            resp.prompt = local_prompt;
                            resp.completion = local_completion;
                            resp.context = local_context;
                            resp.confidence = local_confidence;
                            resp.finish_reason = local_finish;
                        }
                        None => {
                            summary.outcome = "cancelled";
                            resp.context.jump = Some(summary);
                            resp.status = "cancelled";
                            resp.exit_reason = "superseded".into();
                            resp.timings.total_ms = ms(started);
                            return Ok(resp);
                        }
                    }
                    // Unless it ran in the scratch sequence, the jump prompt
                    // replaced the local prompt in the file's cache slot;
                    // restore that while idle so the next keystroke here
                    // stays incremental.
                    let scratch = resp.timings.stages.last().is_some_and(|s| s.scratch);
                    if !refreshed && !scratch {
                        self.warm(&g.display, local_tokens);
                    }
                }
                resp.context.jump = Some(summary);
            }
        }
        resp.exit_reason = outcome.exit_reason.to_string();
        let mut edits = std::mem::take(&mut outcome.edits);
        let untrimmed = edits.first().cloned();
        if self.settings.protect_live_edit && !g.typed.is_empty() && !edits.is_empty() {
            match sweep::protect_typed(content, &edits[0], &g.typed) {
                Some(first) => {
                    let rest: Vec<Edit> = edits[1..].iter().filter_map(|e| sweep::protect_typed(content, e, &g.typed)).collect();
                    resp.context.live_edit_trimmed = first != edits[0] || rest.len() + 1 != edits.len();
                    edits = std::iter::once(first).chain(rest).collect();
                }
                None => {
                    resp.withheld = edits.iter().map(|e| edit_out(content, e)).collect();
                    resp.exit_reason = "live_edit_conflict".into();
                    edits.clear();
                }
            }
        }
        resp.edits = edits.iter().map(|e| edit_out(content, e)).collect();
        let s = &self.settings;
        let gate = |e: &Edit| edit_threshold(content, e, s.min_confidence, s.multiline_min_confidence, s.insert_min_confidence);
        // A trimmed edit keeps the gate of the edit the model wrote: the
        // confidence covers the dropped part too.
        let threshold = match (edits.first(), &untrimmed) {
            (Some(e), Some(u)) => gate(e).max(gate(u)),
            (Some(e), None) => gate(e),
            (None, _) => s.min_confidence,
        };
        let weak = resp.confidence.as_ref().is_some_and(|c| c.changed_tokens > 0 && c.min_logprob < threshold);
        if weak && !resp.edits.is_empty() {
            resp.withheld = std::mem::take(&mut resp.edits);
            resp.exit_reason = "low_confidence".into();
        }
        resp.status = if resp.edits.is_empty() { "empty" } else { "ok" };
        resp.prompt_sha256 = sha256_hex(&resp.prompt);
        resp.context.original_refreshed = refreshed;
        if refreshed {
            if let Some((tokens, _)) = self.build_tokens(&req) {
                self.warm(&g.display, tokens);
            }
        }
        resp.timings.total_ms = ms(started);
        if let Some(e) = &early {
            e.settle(&mut resp);
        }
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
        let (hunks, live, session_base, original, typed) = {
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
            let (h, live) = if opts.stable_history || opts.session_base {
                t.recent_split_at(opts.max_hunks + HIDDEN_HEADROOM, now)
            } else {
                (t.recent_hunks_at(opts.max_hunks, now), 0)
            };
            let base = if opts.session_base { t.session_base(path).map(str::to_string) } else { None };
            let o = t.original(path).unwrap_or(content).to_string();
            let typed = if self.settings.protect_live_edit && purpose != Purpose::Warm { t.live_typed(path, content) } else { Vec::new() };
            (h, live, base, o, typed)
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
        // Identifiers of a file the index cannot parse (YAML, Markdown...)
        // only match unrelated definitions elsewhere.
        let retrieve = self.settings.retrieval_unindexed || Lang::from_path(path).is_some();
        let mut retrieval = if retrieve {
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
        } else {
            Vec::new()
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
            session_base,
            original,
            initial_start,
            block,
            file_chunks,
            retrieval,
            diagnostics,
            typed,
        })
    }

    /// Build the prompt for a window, dropping context until it fits.
    /// `history_from`: the local window, for a jump prompt sharing the
    /// local prompt's history (`PromptInput::history_from`).
    fn stage_prompt(
        &self,
        g: &Gathered,
        content: &str,
        cursor: usize,
        block: &Block,
        retrieval: &[Chunk],
        history_from: Option<&Block>,
    ) -> Result<StagePrompt> {
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
                session_base: g.session_base.as_deref(),
                initial_start: g.initial_start,
                history_from,
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

    /// Record what a stage prompt was built from in `resp.context`.
    fn describe_stage(&self, g: &Gathered, block: &Block, retrieval: &[Chunk], p: &StagePrompt, resp: &mut PredictResponse) {
        let rc = &retrieval[p.retrieval_from..];
        let built = &p.built;
        resp.context.prompt_chars = built.prompt.len();
        resp.context.prompt_tokens = p.tokens.len();
        resp.context.stable_prefix_chars = built.stable_prefix;
        resp.context.hunks = built.hunks_used;
        resp.context.initial_file_lines = built.initial_file_lines;
        resp.context.block_start_line = block.start_line;
        resp.context.block_lines = block.text.lines().count();
        resp.context.trimmed = p.trimmed.clone();
        resp.context.retrieval = rc[rc.len() - built.retrieval_used..].iter().map(summary).collect();
        resp.context.file_chunks = if p.file_chunks_used { g.file_chunks.iter().map(summary).collect() } else { Vec::new() };
    }

    /// Run one prediction stage. Returns the processed outcome (None when
    /// superseded), the prompt tokens and the oldest edit group shown.
    /// `jump_from` is the local window when this is the jump stage.
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
        mut early: Option<&mut Early<'_>>,
        jump_from: Option<&Block>,
    ) -> Result<(Option<sweep::Processed>, Vec<llama_cpp_2::token::LlamaToken>, Option<u64>)> {
        let s = &self.settings;
        let history_from = jump_from.filter(|_| s.jump_shared_history);
        let p = self.stage_prompt(g, content, cursor, block, retrieval, history_from)?;
        self.describe_stage(g, block, retrieval, &p, resp);
        let StagePrompt { built, tokens, block_tokens, max_tokens, max_newlines, .. } = p;
        let display = g.display.as_str();
        // The jump stage in the scratch sequence keeps the local stage's
        // completion as the next keystroke's draft source.
        let scratch = jump_from.is_some() && s.jump_scratch;
        let draft_seed = self.tokenizer.tokenize_plain(&built.prefill);
        let splice = self.settings.early_stop;
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
                splice: splice.then(|| {
                    (built.prefill.clone(), block.text.clone(), self.settings.splice_anchors, self.settings.splice_partial_chars)
                }),
                progress: early.is_some(),
                scratch,
                prefill_chunk: if jump_from.is_some() { s.jump_prefill_chunk } else { 0 },
                reply: reply_tx,
            })
            .map_err(|_| anyhow!("model worker stopped"))?;
        let outcome = loop {
            match reply_rx.recv().map_err(|_| anyhow!("model worker dropped request"))? {
                Reply::Progress { text, logprobs, offsets } => {
                    if let Some(e) = early.as_deref_mut() {
                        e.update(&text, &logprobs, &offsets, &built, block, content, cursor);
                    }
                }
                Reply::Done(result) => break result?,
            }
        };
        resp.prompt = built.prompt.clone();
        let shown_from = built.shown_from;
        match outcome {
            Outcome::Cancelled(stats) => {
                resp.timings.stages.push(stats);
                Ok((None, tokens, shown_from))
            }
            Outcome::Done(gen_out) => {
                resp.timings.stages.push(gen_out.stats.clone());
                resp.completion = gen_out.text.clone();
                resp.finish_reason = gen_out.finish_reason.to_string();
                if !scratch {
                    let prev = self.tokenizer.tokenize_plain(&format!("{}{}", built.prefill, gen_out.text));
                    self.previous.lock().unwrap().insert(display.to_string(), prev);
                }
                resp.confidence = Some(sweep::change_confidence(
                    &built.prefill,
                    &gen_out.text,
                    &block.text,
                    &gen_out.token_logprobs,
                    &gen_out.token_offsets,
                ));
                let processed = sweep::process_completion(&gen_out.text, gen_out.finish_reason, &built, block, content, cursor);
                Ok((Some(processed), tokens, shown_from))
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

/// Confidence the weakest changed token of `e` needs for the edit to be
/// shown. Multi-line rewrites are wrong more often than single-line edits,
/// so they need more (`multiline` only raises `min`); insertions and appends
/// that keep the existing text need less (`insert` only lowers it).
fn edit_threshold(content: &str, e: &Edit, min: f32, multiline: f32, insert: f32) -> f32 {
    let old = &content[e.start..e.end];
    if old.matches('\n').count() > 1 || (!old.is_empty() && e.text.matches('\n').count() > 1) {
        return min.max(multiline);
    }
    let line = old.strip_suffix('\n').unwrap_or(old);
    let appended = e.text.starts_with(line) && e.text.ends_with('\n') == old.ends_with('\n');
    if old.is_empty() || appended { min.min(insert) } else { min }
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

    #[test]
    fn gate_depends_on_the_kind_of_edit() {
        let file = "a: 1\nb: 2\nc: 3\n";
        let t = |start: usize, end: usize, text: &str| edit_threshold(file, &Edit { start, end, text: text.into() }, -1.0, -0.4, -1.3);
        // Ghost text, a new line, a block of new lines, an append.
        assert_eq!(t(4, 4, "0"), -1.3);
        assert_eq!(t(5, 5, "x: 9\n"), -1.3);
        assert_eq!(t(5, 5, "x: 9\ny: 8\n"), -1.3);
        assert_eq!(t(5, 10, "b: 2, 3\n"), -1.3);
        // Single-line rewrites keep the base gate.
        assert_eq!(t(5, 10, "b: 4\n"), -1.0);
        assert_eq!(t(5, 10, "b: 2"), -1.0);
        // Rewrites of more than one line.
        assert_eq!(t(0, 10, "a: 1\nb: 3\n"), -0.4);
        assert_eq!(t(5, 10, "b: 2\nx\ny\n"), -0.4);
        // The kind gates only move away from the base gate.
        let e = Edit { start: 4, end: 4, text: "0".into() };
        assert_eq!(edit_threshold(file, &e, -2.0, -0.4, -1.3), -2.0);
        let e = Edit { start: 0, end: 10, text: "z\n".into() };
        assert_eq!(edit_threshold(file, &e, -0.2, -0.4, -1.3), -0.2);
    }
}
