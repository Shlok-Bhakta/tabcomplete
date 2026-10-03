mod context;
mod embedded;
mod syntax_guard;
use anyhow::{Result, ensure};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use clap::Parser;
use llama_cpp_2::{
    context::params::{KvCacheType, LlamaContextParams},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    convert::Infallible,
    fs::File,
    io::Read,
    net::IpAddr,
    num::NonZeroU32,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc as channel,
    },
    time::Instant,
};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

#[derive(Parser, Clone, Serialize)]
struct Args {
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    model_sha256: Option<String>,
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,
    #[arg(long, default_value_t = 19094)]
    port: u16,
    #[arg(long, default_value_t = 4)]
    threads: i32,
    #[arg(long, default_value_t = 4)]
    prompt_threads: i32,
    #[arg(long, default_value_t = 2304)]
    context_size: u32,
    #[arg(long, default_value_t = 256)]
    batch_size: u32,
    #[arg(long, default_value_t = 64)]
    microbatch_size: u32,
    #[arg(long, default_value_t = 1024)]
    input_tokens: usize,
    #[arg(long, default_value_t = 64)]
    output_tokens: usize,
    #[arg(long, default_value = "single-line-edit-v1")]
    protocol: String,
    #[arg(long, default_value = "cursor-last-v1")]
    context_layout: String,
    #[arg(long, default_value = "f16")]
    cache_type: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    syntax_validation: bool,
    #[arg(long)]
    model_registry: Option<PathBuf>,
    #[arg(long)]
    state_file: Option<PathBuf>,
}
#[derive(Clone, Deserialize, Serialize)]
struct Profile {
    path: PathBuf,
    sha256: String,
    protocol: String,
    #[serde(skip)]
    embedded: Option<embedded::Payload>,
}
#[derive(Clone)]
struct App {
    tx: channel::SyncSender<Job>,
    busy: Arc<AtomicBool>,
    identity: Arc<Mutex<Value>>,
    registry: Arc<BTreeMap<String, Profile>>,
    embedded_mode: bool,
}
type Reply = oneshot::Sender<std::result::Result<Value, String>>;
enum Job {
    Context(context::ContextRequest, Reply),
    Tokenize(String, bool, Reply),
    Generate(
        Generate,
        mpsc::Sender<std::result::Result<Event, Infallible>>,
    ),
    Switch(String, Reply),
}
#[derive(Deserialize)]
struct Generate {
    prompt: String,
    #[serde(default)]
    n_predict: Option<usize>,
    #[serde(default)]
    window: Option<context::Window>,
    #[serde(default)]
    repository_identity: String,
    #[serde(default = "default_cache")]
    cache_prompt: bool,
}
fn default_cache() -> bool {
    true
}
struct Busy(Arc<AtomicBool>);
impl Drop for Busy {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn verify_model(profile: &Profile) -> Result<()> {
    if profile.embedded.is_some() {
        // The payload was bounded and streaming-hash verified before worker startup.
        return Ok(());
    }
    ensure!(
        profile.sha256.len() == 64 && profile.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid weight digest"
    );
    let mut file = File::open(&profile.path)?;
    let mut hash = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    ensure!(
        format!("{:x}", hash.finalize()) == profile.sha256,
        "model identity mismatch"
    );
    ensure!(
        matches!(
            profile.protocol.as_str(),
            "single-line-edit-v1" | "sweep-full-file-v1"
        ),
        "unsupported protocol"
    );
    Ok(())
}
fn output_limit(args: &Args, p: &Profile) -> usize {
    if let Some(payload) = &p.embedded {
        return payload.output_tokens;
    }
    if p.protocol == "sweep-full-file-v1" {
        args.output_tokens.clamp(192, 512)
    } else {
        args.output_tokens.min(64)
    }
}
fn worker(
    args: Args,
    app: App,
    initial: String,
    ready: oneshot::Sender<bool>,
    rx: channel::Receiver<Job>,
) {
    let mut ready = Some(ready);
    let outcome = (|| -> Result<()> {
        let mut backend = LlamaBackend::init()?;
        backend.void_logs();
        let mut alias = initial;
        let mut switch_reply: Option<Reply> = None;
        let mut fallback_alias: Option<String> = None;
        let mut switch_failed = false;
        loop {
            let profile = app
                .registry
                .get(&alias)
                .ok_or_else(|| anyhow::anyhow!("unknown model"))?
                .clone();
            verify_model(&profile)?;
            let load = Instant::now();
            // Retain the borrowed FILE until after model/context destruction.
            let embedded_file = profile
                .embedded
                .as_ref()
                .map(embedded::Payload::open_stream)
                .transpose()?;
            let model_params = LlamaModelParams::default()
                .with_n_gpu_layers(0)
                .with_use_mmap(true);
            let loaded = if let Some(file) = &embedded_file {
                // SAFETY: owned seekable FILE at the verified aligned GGUF;
                // no other thread accesses it, and it outlives the model.
                unsafe { LlamaModel::load_from_file_ptr(&backend, file.as_ptr(), &model_params) }
            } else {
                LlamaModel::load_from_file(&backend, &profile.path, &model_params)
            };
            let model = match loaded {
                Ok(model) => model,
                Err(error) => {
                    if let Some(previous) = fallback_alias.take() {
                        alias = previous;
                        switch_failed = true;
                        continue;
                    }
                    return Err(error.into());
                }
            };
            let cache_type = match args.cache_type.as_str() {
                "f16" => KvCacheType::F16,
                "q8" => KvCacheType::Q8_0,
                _ => anyhow::bail!("unsupported cache precision"),
            };
            let params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(args.context_size))
                .with_n_batch(args.batch_size)
                .with_n_ubatch(args.microbatch_size)
                .with_n_seq_max(1)
                .with_n_threads(args.threads)
                .with_n_threads_batch(args.prompt_threads)
                .with_type_k(cache_type)
                .with_type_v(cache_type)
                .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO)
                .with_no_perf(false);
            let mut ctx = match model.new_context(&backend, params) {
                Ok(ctx) => ctx,
                Err(error) => {
                    if let Some(previous) = fallback_alias.take() {
                        alias = previous;
                        switch_failed = true;
                        continue;
                    }
                    return Err(error.into());
                }
            };
            if switch_reply.is_some()
                && !switch_failed
                && persist_alias(args.state_file.as_ref(), &alias).is_err()
            {
                drop(ctx);
                drop(model);
                alias = fallback_alias
                    .take()
                    .ok_or_else(|| anyhow::anyhow!("missing switch rollback"))?;
                switch_failed = true;
                continue;
            }
            let context_layout =
                context::effective_layout(&profile.protocol, &args.context_layout)?;
            let mut runtime_args = args.clone();
            runtime_args.context_layout = context_layout.into();
            let identity = json!({"status":"ok","alias":alias,"model_sha256":profile.sha256,"model_protocol":profile.protocol,
                "context_layout":context_layout,
                "model_embedded":app.embedded_mode,"model_switch_supported":!app.embedded_mode,
                "model_selection":if app.embedded_mode{"declarative"}else{"research-registry"},
                "model_storage":if app.embedded_mode{"executable-mmap"}else{"external-file-mmap"},
                "backend":"llama.cpp CPU via Rust","llama_cpp_2":"0.1.157","llama_cpp_sys_2":"0.1.158",
                "runtime_config_hash":digest(&serde_json::to_vec(&(&runtime_args,&profile))?),"threads":args.threads,"prompt_threads":args.prompt_threads,
                "context_size":args.context_size,"batch_size":args.batch_size,"microbatch_size":args.microbatch_size,"input_tokens":args.input_tokens,"output_tokens":output_limit(&args,&profile),
                "cache_type":args.cache_type,"syntax_validation":args.syntax_validation,
                "saved_contexts":0,"active_slots":1,"load_ms":load.elapsed().as_secs_f64()*1000.});
            *app.identity.lock().unwrap() = identity.clone();
            if let Some(reply) = ready.take() {
                let _ = reply.send(true);
            }
            if let Some(reply) = switch_reply.take() {
                let _ = reply.send(if switch_failed {
                    Err("model selection could not be committed".into())
                } else {
                    Ok(identity.clone())
                });
                switch_failed = false;
                app.busy.store(false, Ordering::Release);
            }
            let mut cached: Vec<LlamaToken> = Vec::new();
            let mut cache_repo = String::new();
            let mut prepared_context: Option<(String, context::EditorState)> = None;
            let next = loop {
                let Ok(job) = rx.recv() else {
                    return Ok(());
                };
                if let Job::Switch(next, reply) = job {
                    if next == alias {
                        let result = persist_alias(args.state_file.as_ref(), &alias)
                            .map(|_| identity.clone())
                            .map_err(|_| "model selection could not be committed".into());
                        let _ = reply.send(result);
                        app.busy.store(false, Ordering::Release);
                        continue;
                    }
                    if app
                        .registry
                        .get(&next)
                        .is_none_or(|p| verify_model(p).is_err())
                    {
                        let _ = reply.send(Err("candidate identity unavailable".into()));
                        app.busy.store(false, Ordering::Release);
                        continue;
                    }
                    fallback_alias = Some(alias.clone());
                    switch_reply = Some(reply);
                    break next;
                }
                let _busy = Busy(app.busy.clone());
                match job {
                    Job::Context(request, reply) => {
                        if reply.is_closed() {
                            continue;
                        }
                        let result = context::prepare_layout(
                            &request,
                            &profile.protocol,
                            &args.context_layout,
                            args.input_tokens,
                            |p| Ok(model.str_to_token(p, AddBos::Always)?.len()),
                        )
                        .and_then(|p| {
                            prepared_context = Some((p.prompt.clone(), request.state.clone()));
                            Ok(serde_json::to_value(p)?)
                        });
                        if result.is_err() {
                            prepared_context = None;
                        }
                        let result = result.map(|mut p| {
                            p["model_identity"] = identity.clone();
                            p["context_hash"] =
                                json!(digest(p["prompt"].as_str().unwrap().as_bytes()));
                            p
                        });
                        let _ = reply
                            .send(result.map_err(|_| "editor context outside contract".into()));
                    }
                    Job::Tokenize(prompt, special, reply) => {
                        let result=model.str_to_token(&prompt,if special{AddBos::Always}else{AddBos::Never})
                            .map(|tokens|json!({"tokens":tokens.iter().map(|t|t.0).collect::<Vec<_>>()}))
                            .map_err(|_|"tokenization failed".into());
                        let _ = reply.send(result);
                    }
                    Job::Generate(request, reply) => {
                        let result = (|| -> Result<()> {
                            let started = Instant::now();
                            let tokens = model.str_to_token(&request.prompt, AddBos::Always)?;
                            let limit = request
                                .n_predict
                                .unwrap_or(output_limit(&args, &profile))
                                .min(output_limit(&args, &profile));
                            ensure!(
                                limit > 0 && request.repository_identity.len() <= 4096,
                                "invalid generation bounds"
                            );
                            if profile.protocol == "sweep-full-file-v1" {
                                context::validate_window(
                                    request
                                        .window
                                        .as_ref()
                                        .ok_or_else(|| anyhow::anyhow!("missing Sweep window"))?,
                                )?;
                            }
                            ensure!(
                                !tokens.is_empty()
                                    && tokens.len() <= args.input_tokens
                                    && tokens.len() + limit <= args.context_size as usize,
                                "request exceeds explicit context budget"
                            );
                            if !request.cache_prompt || cache_repo != request.repository_identity {
                                ctx.clear_kv_cache();
                                cached.clear();
                            }
                            cache_repo = request.repository_identity.clone();
                            let common = tokens
                                .iter()
                                .zip(&cached)
                                .take_while(|(a, b)| a == b)
                                .count()
                                .min(tokens.len() - 1);
                            if !ctx.clear_kv_cache_seq(Some(0), Some(common as u32), None)? {
                                ctx.clear_kv_cache();
                                cached.clear();
                            }
                            let common = if cached.is_empty() { 0 } else { common };
                            cached.truncate(common);
                            let mut batch = LlamaBatch::new(args.batch_size as usize, 1);
                            let prefill = Instant::now();
                            for start in
                                (common..tokens.len()).step_by(args.microbatch_size as usize)
                            {
                                if reply.is_closed() {
                                    return Ok(());
                                }
                                batch.clear();
                                let end = (start + args.microbatch_size as usize).min(tokens.len());
                                for (position, token) in
                                    tokens.iter().enumerate().take(end).skip(start)
                                {
                                    batch.add(
                                        *token,
                                        position as i32,
                                        &[0],
                                        position + 1 == tokens.len(),
                                    )?;
                                }
                                ctx.decode(&mut batch)?;
                                cached.extend_from_slice(&tokens[start..end]);
                            }
                            let prompt_ms = prefill.elapsed().as_secs_f64() * 1000.;
                            let generated = Instant::now();
                            let mut sampler = LlamaSampler::greedy();
                            let mut bytes = Vec::new();
                            let mut sent = 0;
                            let mut predicted = 0;
                            let mut eos = false;
                            for _ in 0..limit {
                                if reply.is_closed() {
                                    return Ok(());
                                }
                                let token = sampler.sample(&ctx, -1);
                                if model.is_eog_token(token) {
                                    eos = true;
                                    break;
                                }
                                sampler.accept(token);
                                predicted += 1;
                                let piece = match model
                                    .token_to_piece_bytes(token, 128, false, None)
                                {
                                    Err(
                                        llama_cpp_2::TokenToStringError::InsufficientBufferSpace(n),
                                    ) => model.token_to_piece_bytes(
                                        token,
                                        n.unsigned_abs() as usize,
                                        false,
                                        None,
                                    ),
                                    result => result,
                                }?;
                                bytes.extend(piece);
                                let valid = match std::str::from_utf8(&bytes[sent..]) {
                                    Ok(s) => s.len(),
                                    Err(e) if e.error_len().is_none() => e.valid_up_to(),
                                    Err(_) => anyhow::bail!("invalid model UTF-8"),
                                };
                                {
                                    let text = std::str::from_utf8(&bytes[sent..sent + valid])?;
                                    if reply
                                        .blocking_send(Ok(Event::default().json_data(
                                            json!({"content":text,"tokens":[token.0],"stop":false}),
                                        )?))
                                        .is_err()
                                    {
                                        return Ok(());
                                    }
                                    sent += valid;
                                }
                                batch.clear();
                                batch.add(token, cached.len() as i32, &[0], true)?;
                                ctx.decode(&mut batch)?;
                                cached.push(token);
                            }
                            let mut action = if eos {
                                let raw = std::str::from_utf8(&bytes)?;
                                if profile.protocol == "single-line-edit-v1" {
                                    context::decode_action(raw).ok()
                                } else {
                                    request
                                        .window
                                        .as_ref()
                                        .and_then(|w| context::map_sweep(raw, w).ok())
                                }
                            } else {
                                None
                            };
                            let action_validation = validate_action(
                                args.syntax_validation,
                                &request.prompt,
                                &prepared_context,
                                &mut action,
                            );
                            let terminal = json!({"content":"","stop":true,"stop_type":if eos{"eos"}else{"limit"},"tokens_predicted":predicted,
                                "canonical_action":action,"action_validation":action_validation,"model_protocol":profile.protocol,"model_sha256":profile.sha256,
                                "context_layout":context_layout,
                                "timings":{"cache_n":common,"prompt_n":tokens.len()-common,"prompt_ms":prompt_ms,"predicted_n":predicted,
                                    "predicted_ms":generated.elapsed().as_secs_f64()*1000.,"total_ms":started.elapsed().as_secs_f64()*1000.}});
                            let _ = reply.blocking_send(Ok(Event::default().json_data(terminal)?));
                            Ok(())
                        })();
                        if result.is_err() {
                            ctx.clear_kv_cache();
                            cached.clear();
                            let _ = reply.blocking_send(Ok(Event::default()
                                .json_data(json!({"content":"","stop":true,"stop_type":"error"}))
                                .unwrap()));
                        }
                    }
                    Job::Switch(_, _) => unreachable!(),
                }
            };
            drop(ctx);
            drop(model);
            alias = next;
            *app.identity.lock().unwrap() = json!({"status":"loading","alias":alias});
        }
    })();
    if outcome.is_err() {
        *app.identity.lock().unwrap() = json!({"status":"failed","error":"model worker failed"});
        app.busy.store(false, Ordering::Release);
        if let Some(reply) = ready.take() {
            let _ = reply.send(false);
        }
    }
}
fn validate_action(
    enabled: bool,
    prompt: &str,
    prepared: &Option<(String, context::EditorState)>,
    action: &mut Option<Value>,
) -> Value {
    if !enabled {
        return json!({"policy":"rust-syntax-v1","status":"disabled"});
    }
    if let Some((prepared_prompt, state)) = prepared
        && prepared_prompt == prompt
        && let Some(candidate) = action
    {
        return match syntax_guard::check(state, candidate) {
            Ok(()) => json!({"policy":"rust-syntax-v1","status":"passed",
                "reason":if state.filetype == "rust" {"syntax_preserved"} else {"not_applicable"}}),
            Err(error) => {
                // Guard errors are stable labels, never source snippets.
                *action = None;
                json!({"policy":"rust-syntax-v1","status":"rejected","reason":error.to_string()})
            }
        };
    }
    json!({"policy":"rust-syntax-v1","status":"unavailable"})
}

fn persist_alias(path: Option<&PathBuf>, alias: &str) -> Result<()> {
    if let Some(path) = path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(&json!({"alias":alias}))?)?;
        std::fs::rename(tmp, path)?;
    }
    Ok(())
}
fn claim(app: &App) -> std::result::Result<(), StatusCode> {
    app.busy
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .map(|_| ())
        .map_err(|_| StatusCode::CONFLICT)
}
async fn rpc(app: App, make: impl FnOnce(Reply) -> Job) -> Response {
    if let Err(code) = claim(&app) {
        return (code, Json(json!({"error":"busy","retry_after_ms":100}))).into_response();
    }
    let (tx, rx) = oneshot::channel();
    if app.tx.try_send(make(tx)).is_err() {
        app.busy.store(false, Ordering::Release);
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    match rx.await {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(_)) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({"error":"request outside contract"})),
        )
            .into_response(),
        Err(_) => StatusCode::SERVICE_UNAVAILABLE.into_response(),
    }
}
async fn health(State(app): State<App>) -> Response {
    let identity = app.identity.lock().unwrap().clone();
    let status = if identity["status"] == "ok" {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(identity)).into_response()
}
async fn slots(State(app): State<App>) -> Json<Value> {
    Json(json!([{"id":0,"is_processing":app.busy.load(Ordering::Acquire)}]))
}
async fn models(State(app): State<App>) -> Json<Value> {
    Json(json!({"current":app.identity.lock().unwrap().clone(),"models":&*app.registry}))
}
#[derive(Deserialize)]
struct Switch {
    alias: String,
}
async fn switch(State(app): State<App>, Json(request): Json<Switch>) -> Response {
    if app.embedded_mode {
        return (
            StatusCode::METHOD_NOT_ALLOWED,
            Json(json!({"error":"model_selection_is_declarative"})),
        )
            .into_response();
    }
    if !app.registry.contains_key(&request.alias) {
        return StatusCode::NOT_FOUND.into_response();
    }
    rpc(app, |reply| Job::Switch(request.alias, reply)).await
}
async fn editor_context(
    State(app): State<App>,
    Json(request): Json<context::ContextRequest>,
) -> Response {
    rpc(app, |reply| Job::Context(request, reply)).await
}
#[derive(Deserialize)]
struct Tokenize {
    content: String,
    #[serde(default)]
    add_special: bool,
}
async fn tokenize(State(app): State<App>, Json(request): Json<Tokenize>) -> Response {
    rpc(app, |reply| {
        Job::Tokenize(request.content, request.add_special, reply)
    })
    .await
}
async fn completion(State(app): State<App>, Json(request): Json<Generate>) -> Response {
    if let Err(code) = claim(&app) {
        return (code, Json(json!({"error":"busy","retry_after_ms":100}))).into_response();
    }
    let (tx, rx) = mpsc::channel(8);
    if app.tx.try_send(Job::Generate(request, tx)).is_err() {
        app.busy.store(false, Ordering::Release);
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    Sse::new(ReceiverStream::new(rx)).into_response()
}
#[tokio::main]
async fn main() -> Result<()> {
    let mut args = Args::parse();
    let payload = if args.model.is_none() {
        ensure!(
            args.model_sha256.is_none()
                && args.model_registry.is_none()
                && args.state_file.is_none(),
            "embedded runtime has no external model or registry"
        );
        let payload = embedded::Payload::inspect_executable()?;
        args.protocol = payload.protocol.clone();
        args.output_tokens = payload.output_tokens;
        Some(payload)
    } else {
        None
    };
    ensure!(
        args.host.is_loopback(),
        "inference must remain loopback-only"
    );
    ensure!(
        (1..=8).contains(&args.threads) && (1..=8).contains(&args.prompt_threads),
        "invalid thread count"
    );
    ensure!(
        (64..=512).contains(&args.batch_size) && (256..=1984).contains(&args.input_tokens),
        "invalid batch/input budget"
    );
    ensure!(
        (1024..=4096).contains(&args.context_size) && (1..=512).contains(&args.output_tokens),
        "invalid context/output budget"
    );
    ensure!(
        matches!(
            args.context_layout.as_str(),
            "trained-v2" | "cursor-last-v1"
        ),
        "unsupported context layout"
    );
    ensure!(
        (1..=args.batch_size).contains(&args.microbatch_size),
        "invalid microbatch budget"
    );
    let default = Profile {
        path: args
            .model
            .clone()
            .unwrap_or_else(|| "/proc/self/exe".into()),
        sha256: if let Some(p) = &payload {
            p.sha256.clone()
        } else {
            args.model_sha256
                .clone()
                .ok_or_else(|| anyhow::anyhow!("research model requires --model-sha256"))?
        },
        protocol: args.protocol.clone(),
        embedded: payload.clone(),
    };
    let mut registry: BTreeMap<String, Profile> = if let Some(path) = &args.model_registry {
        serde_json::from_slice(&std::fs::read(path)?)?
    } else {
        BTreeMap::new()
    };
    if registry.is_empty() {
        registry.insert(
            payload
                .as_ref()
                .map_or("default", |p| p.alias.as_str())
                .into(),
            default.clone(),
        );
    }
    ensure!(registry.len() <= 2, "at most two registered models");
    ensure!(
        registry
            .values()
            .all(|p| args.input_tokens + output_limit(&args, p) <= args.context_size as usize),
        "input and output exceed context budget"
    );
    let mut initial = registry
        .iter()
        .find(|(_, p)| p.sha256 == default.sha256)
        .map(|(key, _)| key.clone())
        .ok_or_else(|| anyhow::anyhow!("initial model not registered"))?;
    if let Some(path) = &args.state_file
        && let Ok(bytes) = std::fs::read(path)
        && let Ok(value) = serde_json::from_slice::<Value>(&bytes)
        && let Some(alias) = value["alias"]
            .as_str()
            .filter(|alias| registry.contains_key(*alias))
    {
        initial = alias.into();
    }
    let (tx, rx) = channel::sync_channel(1);
    let app = App {
        tx,
        busy: Arc::new(AtomicBool::new(false)),
        identity: Arc::new(Mutex::new(json!({"status":"loading"}))),
        registry: Arc::new(registry),
        embedded_mode: payload.is_some(),
    };
    let (ready_tx, ready_rx) = oneshot::channel();
    let worker_app = app.clone();
    let worker_args = args.clone();
    std::thread::spawn(move || worker(worker_args, worker_app, initial, ready_tx, rx));
    ensure!(
        ready_rx.await.unwrap_or(false),
        "model initialization failed"
    );
    let router = Router::new()
        .route("/health", get(health))
        .route("/slots", get(slots))
        .route("/tokenize", post(tokenize))
        .route("/completion", post(completion))
        .route("/v1/editor/context", post(editor_context))
        .route("/v1/models", get(models))
        .route("/v1/model", post(switch))
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind((args.host, args.port)).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod worker_guard_tests {
    use super::*;
    #[test]
    fn syntax_guard_requires_the_exact_prepared_prompt() {
        let state = context::EditorState {
            file_id: "synthetic.rs".into(),
            filetype: "rust".into(),
            source: "fn main() { let value = 1; }\n".into(),
            target_row: 0,
            cursor_col: 0,
            history: vec![],
            relevant: vec![],
        };
        let prepared = Some(("prepared".into(), state));
        let invalid = json!({"kind":"replace_line","text":"fn main() { let value = ; }"});
        let mut action = Some(invalid.clone());
        assert_eq!(
            validate_action(true, "other", &prepared, &mut action)["status"],
            "unavailable"
        );
        assert_eq!(action, Some(invalid));
        assert_eq!(
            validate_action(true, "prepared", &prepared, &mut action)["status"],
            "rejected"
        );
        assert!(action.is_none());
        let mut action = Some(json!({"kind":"replace_line","text":"fn main() { let value = 2; }"}));
        assert_eq!(
            validate_action(true, "prepared", &prepared, &mut action)["status"],
            "passed"
        );
        assert!(action.is_some());
    }

    #[test]
    fn experimental_syntax_opt_out_preserves_proposal() {
        let invalid = json!({"kind":"replace_line","text":"fn main() { let value = ; }"});
        let mut action = Some(invalid.clone());
        assert_eq!(
            validate_action(false, "prompt", &None, &mut action)["status"],
            "disabled"
        );
        assert_eq!(action, Some(invalid));
        let mut action = None;
        validate_action(false, "prompt", &None, &mut action);
        assert!(
            action.is_none(),
            "disabling syntax checks must not invent a decoded action"
        );
        assert!(Args::try_parse_from(["engine"]).unwrap().syntax_validation);
        assert!(
            !Args::try_parse_from(["engine", "--syntax-validation", "false"])
                .unwrap()
                .syntax_validation
        );
    }
}
