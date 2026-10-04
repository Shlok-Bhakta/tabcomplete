mod context;
mod embedded;
mod syntax_guard;
// The research FIM profile is enabled only by an explicit selected artifact.
#[allow(dead_code)]
mod fim_v1;
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
    token_type::LlamaTokenAttr,
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
    #[serde(default)]
    fim_profile: Option<fim_v1::ServingProfile>,
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
    #[serde(default)]
    request_id: Option<String>,
    #[serde(default)]
    context_hash: Option<String>,
    #[serde(default)]
    completion_mode: Option<String>,
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
    if let Some(payload) = &profile.embedded {
        // The executable footer and appended GGUF were bounded and hash-verified
        // before worker startup. The FIM profile is part of that same footer.
        ensure!(
            profile.sha256 == payload.sha256 && profile.protocol == payload.protocol,
            "embedded model identity mismatch"
        );
        if profile.protocol == fim_v1::WIRE_VERSION {
            let embedded_profile = payload
                .fim_profile
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("embedded FIM profile is required"))?;
            let selected_profile = profile
                .fim_profile
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("FIM profile identity is required"))?;
            ensure!(
                selected_profile == embedded_profile,
                "embedded FIM profile identity mismatch"
            );
            fim_v1::validate_serving_profile(selected_profile)?;
        } else {
            ensure!(
                profile.fim_profile.is_none(),
                "unexpected FIM profile identity"
            );
        }
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
            "single-line-edit-v1" | "sweep-full-file-v1" | fim_v1::WIRE_VERSION
        ),
        "unsupported protocol"
    );
    if profile.protocol == fim_v1::WIRE_VERSION {
        fim_v1::validate_serving_profile(
            profile
                .fim_profile
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("FIM profile identity is required"))?,
        )?;
    } else {
        ensure!(
            profile.fim_profile.is_none(),
            "unexpected FIM profile identity"
        );
    }
    Ok(())
}
fn output_limit(args: &Args, p: &Profile) -> usize {
    if let Some(payload) = &p.embedded {
        return payload.output_tokens;
    }
    if p.protocol == "sweep-full-file-v1" {
        args.output_tokens.clamp(192, 512)
    } else if p.protocol == fim_v1::WIRE_VERSION {
        fim_v1::OUTPUT_TOKEN_CAP
    } else {
        args.output_tokens.min(64)
    }
}

fn generation_renders_special_tokens(protocol: &str) -> bool {
    protocol == fim_v1::WIRE_VERSION
}

fn public_context_error(error: &anyhow::Error) -> &'static str {
    if error
        .to_string()
        .contains("not NFC-normalized; tokenizer parity is unavailable")
    {
        "fim_context_unsupported_non_nfc"
    } else {
        "editor context outside contract"
    }
}

struct FimContextBinding {
    prepared: fim_v1::Prepared,
    request_id: String,
    context_hash: String,
    completion_mode: String,
}

fn token_piece(model: &LlamaModel, token: LlamaToken, special: bool) -> Result<Vec<u8>> {
    Ok(
        match model.token_to_piece_bytes(token, 256, special, None) {
            Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(size))
                if size.unsigned_abs() <= 4096 =>
            {
                model.token_to_piece_bytes(token, size.unsigned_abs() as usize, special, None)?
            }
            result => result?,
        },
    )
}

fn special_token_piece(model: &LlamaModel, token: LlamaToken) -> Result<String> {
    let bytes = token_piece(model, token, true)?;
    Ok(std::str::from_utf8(&bytes)?.to_owned())
}

fn verify_prompt_round_trip(model: &LlamaModel, prompt: &str) -> Result<Vec<LlamaToken>> {
    let tokens = model.str_to_token(prompt, AddBos::Never)?;
    let mut decoded = Vec::with_capacity(prompt.len());
    for token in &tokens {
        decoded.extend(token_piece(model, *token, true)?);
    }
    ensure!(
        decoded == prompt.as_bytes(),
        "FIM prompt does not round-trip through the selected native tokenizer"
    );
    Ok(tokens)
}

fn native_token_pieces(model: &LlamaModel, text: &str) -> Result<Vec<Vec<u8>>> {
    let tokens = model.str_to_token(text, AddBos::Never)?;
    tokens
        .into_iter()
        .map(|token| token_piece(model, token, true))
        .collect()
}

fn verify_fim_tokenizer(
    model: &LlamaModel,
    profile: &fim_v1::ServingProfile,
) -> Result<fim_v1::TokenContract> {
    fim_v1::validate_serving_profile(profile)?;
    let tokenizer = &profile.tokenizer;
    ensure!(
        tokenizer
            .tokenizer_vocab_ids
            .last()
            .is_some_and(|id| *id < model.n_vocab()),
        "tokenizer vocabulary exceeds the selected model"
    );
    ensure!(
        model.token_eos().0 == tokenizer.eos_id,
        "FIM EOS token mismatch"
    );
    for (spelling, expected) in [
        (fim_v1::FIM_PREFIX, tokenizer.fim_prefix_id),
        (fim_v1::FIM_SUFFIX, tokenizer.fim_suffix_id),
        (fim_v1::FIM_MIDDLE, tokenizer.fim_middle_id),
    ] {
        let encoded = model.str_to_token(spelling, AddBos::Never)?;
        ensure!(
            encoded.len() == 1 && encoded[0].0 == expected,
            "FIM marker tokenization mismatch"
        );
    }
    let mut actual = Vec::new();
    for id in 0..model.n_vocab() {
        let token = LlamaToken(id);
        let attributes = model.token_attr(token);
        let known = tokenizer.tokenizer_vocab_ids.binary_search(&id).is_ok();
        ensure!(
            if known {
                !attributes.intersects(LlamaTokenAttr::Unknown | LlamaTokenAttr::Unused)
            } else {
                attributes.intersects(LlamaTokenAttr::Unknown | LlamaTokenAttr::Unused)
            },
            "tokenizer vocabulary IDs disagree with the selected model"
        );
        if attributes.intersects(LlamaTokenAttr::Control | LlamaTokenAttr::UserDefined) {
            actual.push(fim_v1::SpecialToken {
                id,
                spelling: special_token_piece(model, token)?,
            });
        }
    }
    ensure!(
        actual == tokenizer.special_tokens,
        "FIM control-token inventory mismatch"
    );
    fim_v1::TokenContract::from_profile(tokenizer)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkerFailureStage {
    BackendInit,
    ModelIdentityVerify,
    ModelContainerOpen,
    ModelLoad,
    FimProfileIdentity,
    FimTokenizerVerify,
    RuntimeConfiguration,
    ModelContextCreate,
    ModelSelectionCommit,
    RuntimeIdentityBuild,
    WorkerLoop,
}

impl WorkerFailureStage {
    fn stage(self) -> &'static str {
        match self {
            Self::BackendInit => "backend_init",
            Self::ModelIdentityVerify => "model_identity_verify",
            Self::ModelContainerOpen => "model_container_open",
            Self::ModelLoad => "model_load",
            Self::FimProfileIdentity => "fim_profile_identity",
            Self::FimTokenizerVerify => "fim_tokenizer_verify",
            Self::RuntimeConfiguration => "runtime_configuration",
            Self::ModelContextCreate => "model_context_create",
            Self::ModelSelectionCommit => "model_selection_commit",
            Self::RuntimeIdentityBuild => "runtime_identity_build",
            Self::WorkerLoop => "worker_loop",
        }
    }

    fn code(self) -> &'static str {
        match self {
            Self::BackendInit => "backend_init_failed",
            Self::ModelIdentityVerify => "model_identity_verification_failed",
            Self::ModelContainerOpen => "model_container_open_failed",
            Self::ModelLoad => "gguf_model_load_failed",
            Self::FimProfileIdentity => "fim_profile_identity_failed",
            Self::FimTokenizerVerify => "fim_tokenizer_contract_failed",
            Self::RuntimeConfiguration => "runtime_configuration_failed",
            Self::ModelContextCreate => "model_context_initialization_failed",
            Self::ModelSelectionCommit => "model_selection_commit_failed",
            Self::RuntimeIdentityBuild => "runtime_identity_build_failed",
            Self::WorkerLoop => "model_worker_failed",
        }
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
    let mut failure_stage = WorkerFailureStage::BackendInit;
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
            failure_stage = WorkerFailureStage::ModelIdentityVerify;
            verify_model(&profile)?;
            let load = Instant::now();
            // Retain the borrowed FILE until after model/context destruction.
            failure_stage = WorkerFailureStage::ModelContainerOpen;
            let embedded_file = profile
                .embedded
                .as_ref()
                .map(embedded::Payload::open_stream)
                .transpose()?;
            let model_params = LlamaModelParams::default()
                .with_n_gpu_layers(0)
                .with_use_mmap(true);
            failure_stage = WorkerFailureStage::ModelLoad;
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
            let fim_contract = if profile.protocol == fim_v1::WIRE_VERSION {
                failure_stage = WorkerFailureStage::FimProfileIdentity;
                let fim_profile = profile
                    .fim_profile
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("FIM profile identity is required"))?;
                failure_stage = WorkerFailureStage::FimTokenizerVerify;
                Some(verify_fim_tokenizer(&model, fim_profile)?)
            } else {
                None
            };
            failure_stage = WorkerFailureStage::RuntimeConfiguration;
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
            failure_stage = WorkerFailureStage::ModelContextCreate;
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
                failure_stage = WorkerFailureStage::ModelSelectionCommit;
                alias = fallback_alias
                    .take()
                    .ok_or_else(|| anyhow::anyhow!("missing switch rollback"))?;
                switch_failed = true;
                continue;
            }
            failure_stage = WorkerFailureStage::RuntimeIdentityBuild;
            let context_layout = if profile.protocol == fim_v1::WIRE_VERSION {
                fim_v1::CONTEXT_LAYOUT
            } else {
                context::effective_layout(&profile.protocol, &args.context_layout)?
            };
            let mut runtime_args = args.clone();
            runtime_args.context_layout = context_layout.into();
            if profile.protocol == fim_v1::WIRE_VERSION {
                runtime_args.output_tokens = fim_v1::OUTPUT_TOKEN_CAP;
            }
            let mut identity = json!({"status":"ok","alias":alias,"model_sha256":profile.sha256,"model_protocol":profile.protocol,
                "context_layout":context_layout,
                "model_embedded":app.embedded_mode,"model_switch_supported":!app.embedded_mode,
                "model_selection":if app.embedded_mode{"declarative"}else{"research-registry"},
                "model_storage":if app.embedded_mode{"executable-mmap"}else{"external-file-mmap"},
                "backend":"llama.cpp CPU via Rust","llama_cpp_2":"0.1.157","llama_cpp_sys_2":"0.1.158",
                "runtime_config_hash":digest(&serde_json::to_vec(&(&runtime_args,&profile))?),"threads":args.threads,"prompt_threads":args.prompt_threads,
                "context_size":args.context_size,"batch_size":args.batch_size,"microbatch_size":args.microbatch_size,"input_tokens":args.input_tokens,"output_tokens":output_limit(&args,&profile),
                "cache_type":args.cache_type,"syntax_validation":args.syntax_validation,
                "saved_contexts":0,"active_slots":1,"load_ms":load.elapsed().as_secs_f64()*1000.});
            if let Some(fim_profile) = &profile.fim_profile {
                identity["fim_profile"] = json!(fim_profile);
                identity["completion_mode"] = json!(fim_v1::COMPLETION_MODE);
                identity["tokenizer_id"] = json!(fim_profile.tokenizer.tokenizer_id);
                identity["tokenizer_revision"] = json!(fim_profile.tokenizer.tokenizer_revision);
                identity["tokenizer_sha256"] = json!(fim_profile.tokenizer.tokenizer_sha256);
                identity["tokenizer_contract_sha256"] =
                    json!(fim_profile.tokenizer.tokenizer_contract_sha256);
                identity["tokenizer_vocab_size"] =
                    json!(fim_profile.tokenizer.tokenizer_vocab_size);
                identity["tokenizer_vocab_ids_sha256"] =
                    json!(fim_profile.tokenizer.tokenizer_vocab_ids_sha256);
                identity["fim_token_ids"] = json!({
                    "eos": fim_profile.tokenizer.eos_id,
                    "fim_prefix": fim_profile.tokenizer.fim_prefix_id,
                    "fim_suffix": fim_profile.tokenizer.fim_suffix_id,
                    "fim_middle": fim_profile.tokenizer.fim_middle_id,
                });
            }
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
            failure_stage = WorkerFailureStage::WorkerLoop;
            let mut cached: Vec<LlamaToken> = Vec::new();
            let mut cache_repo = String::new();
            let mut prepared_context: Option<(String, context::EditorState)> = None;
            let mut prepared_fim: Option<FimContextBinding> = None;
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
                        prepared_fim = None;
                        if reply.is_closed() {
                            continue;
                        }
                        let result = if let Some(token_contract) = &fim_contract {
                            let request_id = request
                                .request_id
                                .as_deref()
                                .ok_or_else(|| anyhow::anyhow!("missing FIM request identity"));
                            let completion_mode = request
                                .completion_mode
                                .as_deref()
                                .ok_or_else(|| anyhow::anyhow!("missing FIM completion mode"));
                            request_id
                                .and_then(|request_id| {
                                    context::validate_fim_editor_state(&request.state)?;
                                    ensure!(
                                        completion_mode? == fim_v1::COMPLETION_MODE,
                                        "FIM completion mode mismatch"
                                    );
                                    Ok(request_id)
                                })
                                .and_then(|request_id| {
                                    let mut prepared = fim_v1::prepare(
                                        &request.state.source,
                                        request.state.target_row,
                                        request.state.cursor_col,
                                        token_contract,
                                    )?;
                                    let prefix_source = &request.state.source
                                        [..prepared.model_hole_range.start_byte];
                                    let suffix_source =
                                        &request.state.source[prepared.model_hole_range.end_byte..];
                                    let prefix_window = fim_v1::crop_context(
                                        prefix_source,
                                        fim_v1::PREFIX_CONTEXT_TOKEN_LIMIT,
                                        fim_v1::ContextSide::KeepRight,
                                        |segment| native_token_pieces(&model, segment),
                                    )?;
                                    let suffix_window = fim_v1::crop_context(
                                        suffix_source,
                                        fim_v1::SUFFIX_CONTEXT_TOKEN_LIMIT,
                                        fim_v1::ContextSide::KeepLeft,
                                        |segment| native_token_pieces(&model, segment),
                                    )?;
                                    prepared.set_bounded_window(
                                        &request.state.source,
                                        prefix_window.start_byte,
                                        suffix_window.end_byte + prepared.model_hole_range.end_byte,
                                        prefix_window.token_count,
                                        suffix_window.token_count,
                                    )?;
                                    let prompt_tokens =
                                        verify_prompt_round_trip(&model, &prepared.prompt)?.len();
                                    ensure!(
                                        prompt_tokens > 0
                                            && prompt_tokens <= args.input_tokens
                                            && prompt_tokens
                                                == prefix_window.token_count
                                                    + suffix_window.token_count
                                                    + 3
                                            && prompt_tokens + fim_v1::OUTPUT_TOKEN_CAP
                                                <= args.input_tokens
                                            && prompt_tokens + fim_v1::OUTPUT_TOKEN_CAP
                                                <= args.context_size as usize,
                                        "FIM prompt exceeds the configured token budget"
                                    );
                                    let context_hash = fim_v1::context_digest(
                                        request_id,
                                        &request.state.source,
                                        request.state.target_row,
                                        request.state.cursor_col,
                                        &prepared,
                                    )?;
                                    let mut value = serde_json::to_value(&prepared)?;
                                    value["prompt_tokens"] = json!(prompt_tokens);
                                    value["prefix_context_tokens"] =
                                        json!(prepared.prefix_token_count);
                                    value["suffix_context_tokens"] =
                                        json!(prepared.suffix_token_count);
                                    value["filetype_training_scope"] = json!(
                                        fim_v1::filetype_training_scope(&request.state.filetype)
                                    );
                                    value["request_id"] = json!(request_id);
                                    value["completion_mode"] = json!(fim_v1::COMPLETION_MODE);
                                    value["context_hash"] = json!(context_hash);
                                    value["window"] = Value::Null;
                                    value["selected_buffers"] = json!([]);
                                    prepared_context =
                                        Some((prepared.prompt.clone(), request.state.clone()));
                                    prepared_fim = Some(FimContextBinding {
                                        prepared,
                                        request_id: request_id.to_owned(),
                                        context_hash,
                                        completion_mode: fim_v1::COMPLETION_MODE.into(),
                                    });
                                    Ok(value)
                                })
                        } else {
                            context::prepare_layout(
                                &request,
                                &profile.protocol,
                                &args.context_layout,
                                args.input_tokens,
                                |p| Ok(model.str_to_token(p, AddBos::Always)?.len()),
                            )
                            .and_then(|p| {
                                prepared_context = Some((p.prompt.clone(), request.state.clone()));
                                Ok(serde_json::to_value(p)?)
                            })
                        };
                        if result.is_err() {
                            prepared_context = None;
                            prepared_fim = None;
                        }
                        let result = result.map(|mut p| {
                            p["model_identity"] = identity.clone();
                            if fim_contract.is_none() {
                                p["context_hash"] =
                                    json!(digest(p["prompt"].as_str().unwrap().as_bytes()));
                            }
                            p
                        });
                        let _ =
                            reply.send(result.map_err(|error| public_context_error(&error).into()));
                    }
                    Job::Tokenize(prompt, special, reply) => {
                        prepared_fim = None;
                        let result=model.str_to_token(&prompt,if special{AddBos::Always}else{AddBos::Never})
                            .map(|tokens|json!({"tokens":tokens.iter().map(|t|t.0).collect::<Vec<_>>()}))
                            .map_err(|_|"tokenization failed".into());
                        let _ = reply.send(result);
                    }
                    Job::Generate(request, reply) => {
                        let fim_binding = if profile.protocol == fim_v1::WIRE_VERSION {
                            prepared_fim.take()
                        } else {
                            None
                        };
                        let result = (|| -> Result<()> {
                            let started = Instant::now();
                            let is_fim = profile.protocol == fim_v1::WIRE_VERSION;
                            let tokens = if is_fim {
                                verify_prompt_round_trip(&model, &request.prompt)?
                            } else {
                                model.str_to_token(&request.prompt, AddBos::Always)?
                            };
                            let limit = request
                                .n_predict
                                .unwrap_or(output_limit(&args, &profile))
                                .min(output_limit(&args, &profile));
                            ensure!(
                                limit > 0 && request.repository_identity.len() <= 4096,
                                "invalid generation bounds"
                            );
                            if is_fim {
                                let binding = fim_binding.as_ref().ok_or_else(|| {
                                    anyhow::anyhow!("FIM generation has no prepared context")
                                })?;
                                ensure!(
                                    request.request_id.as_deref()
                                        == Some(binding.request_id.as_str())
                                        && request.context_hash.as_deref()
                                            == Some(binding.context_hash.as_str())
                                        && request.completion_mode.as_deref()
                                            == Some(binding.completion_mode.as_str())
                                        && request.prompt == binding.prepared.prompt,
                                    "FIM request does not match its prepared context"
                                );
                                ensure!(
                                    request.n_predict == Some(fim_v1::OUTPUT_TOKEN_CAP)
                                        && limit == fim_v1::OUTPUT_TOKEN_CAP,
                                    "FIM output cap mismatch"
                                );
                            }
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
                            let mut terminal_token_id = None;
                            let mut sampled_token_ids = Vec::new();
                            for _ in 0..limit {
                                if reply.is_closed() {
                                    return Ok(());
                                }
                                let token = sampler.sample(&ctx, -1);
                                if model.is_eog_token(token) {
                                    eos = true;
                                    terminal_token_id = Some(token.0);
                                    break;
                                }
                                sampler.accept(token);
                                predicted += 1;
                                if is_fim {
                                    sampled_token_ids.push(token.0);
                                }
                                let piece = match model.token_to_piece_bytes(
                                    token,
                                    128,
                                    generation_renders_special_tokens(&profile.protocol),
                                    None,
                                ) {
                                    Err(
                                        llama_cpp_2::TokenToStringError::InsufficientBufferSpace(n),
                                    ) => model.token_to_piece_bytes(
                                        token,
                                        n.unsigned_abs() as usize,
                                        generation_renders_special_tokens(&profile.protocol),
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
                            let mut completion_failure = None;
                            let mut action = if is_fim {
                                let binding = fim_binding.as_ref().ok_or_else(|| {
                                    anyhow::anyhow!("FIM generation lost its prepared context")
                                })?;
                                let raw = std::str::from_utf8(&bytes)?;
                                let token_contract = fim_contract.as_ref().ok_or_else(|| {
                                    anyhow::anyhow!("FIM tokenizer contract unavailable")
                                })?;
                                match fim_v1::check_completion(
                                    &binding.prepared,
                                    &profile.protocol,
                                    output_limit(&args, &profile),
                                    raw,
                                    &sampled_token_ids,
                                    terminal_token_id,
                                    token_contract,
                                )? {
                                    fim_v1::CompletionCheck::Valid(decoded) => {
                                        Some(serde_json::to_value(decoded.action)?)
                                    }
                                    fim_v1::CompletionCheck::Invalid(failure) => {
                                        completion_failure = Some(failure.code());
                                        None
                                    }
                                }
                            } else if eos {
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
                            let action_validation = if is_fim {
                                match completion_failure {
                                    Some(code) => {
                                        json!({"policy":"q25-fim-completion-v1","status":"invalid","code":code})
                                    }
                                    None => {
                                        json!({"policy":"q25-fim-completion-v1","status":"not_applicable"})
                                    }
                                }
                            } else {
                                validate_action(
                                    args.syntax_validation,
                                    &request.prompt,
                                    &prepared_context,
                                    &mut action,
                                )
                            };
                            let stop_type = if is_fim {
                                match terminal_token_id {
                                    Some(id) if id == fim_v1::EOS_TOKEN_ID => "eos",
                                    Some(_) => "control",
                                    None => "limit",
                                }
                            } else if eos {
                                "eos"
                            } else {
                                "limit"
                            };
                            let mut terminal = json!({"content":"","stop":true,"stop_type":stop_type,"tokens_predicted":predicted,
                                "canonical_action":action,"action_validation":action_validation,"model_protocol":profile.protocol,"model_sha256":profile.sha256,
                                "context_layout":context_layout,
                                "timings":{"cache_n":common,"prompt_n":tokens.len()-common,"prompt_ms":prompt_ms,"predicted_n":predicted,
                                    "predicted_ms":generated.elapsed().as_secs_f64()*1000.,"total_ms":started.elapsed().as_secs_f64()*1000.}});
                            if let Some(binding) = &fim_binding {
                                let fim_profile =
                                    profile.fim_profile.as_ref().ok_or_else(|| {
                                        anyhow::anyhow!("FIM profile identity unavailable")
                                    })?;
                                terminal["terminal_token_id"] =
                                    terminal_token_id.map_or(Value::Null, |id| json!(id));
                                terminal["sampled_token_ids"] = json!(sampled_token_ids);
                                terminal["request_id"] = json!(binding.request_id);
                                terminal["context_hash"] = json!(binding.context_hash);
                                terminal["completion_mode"] = json!(binding.completion_mode);
                                terminal["tokenizer_sha256"] =
                                    json!(binding.prepared.tokenizer_sha256);
                                terminal["tokenizer_contract_sha256"] =
                                    json!(binding.prepared.tokenizer_contract_sha256);
                                terminal["artifact_manifest_sha256"] =
                                    json!(fim_profile.artifact_manifest_sha256);
                                terminal["tokenizer_id"] =
                                    json!(fim_profile.tokenizer.tokenizer_id);
                                terminal["tokenizer_revision"] =
                                    json!(fim_profile.tokenizer.tokenizer_revision);
                                terminal["tokenizer_vocab_size"] =
                                    json!(fim_profile.tokenizer.tokenizer_vocab_size);
                                terminal["tokenizer_vocab_ids_sha256"] =
                                    json!(fim_profile.tokenizer.tokenizer_vocab_ids_sha256);
                                terminal["fim_token_ids"] = json!({
                                    "eos": fim_v1::EOS_TOKEN_ID,
                                    "fim_prefix": fim_v1::FIM_PREFIX_TOKEN_ID,
                                    "fim_suffix": fim_v1::FIM_SUFFIX_TOKEN_ID,
                                    "fim_middle": fim_v1::FIM_MIDDLE_TOKEN_ID,
                                });
                            }
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
        eprintln!(
            "tabcomplete_model_worker_failure stage={} code={}",
            failure_stage.stage(),
            failure_stage.code()
        );
        *app.identity.lock().unwrap() = json!({
            "status":"failed",
            "failure_stage":failure_stage.stage(),
            "failure_code":failure_stage.code()
        });
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
        Ok(Err(error)) => {
            let public_error = if error == "fim_context_unsupported_non_nfc" {
                "fim_context_unsupported_non_nfc"
            } else {
                "request outside contract"
            };
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"error":public_error})),
            )
                .into_response()
        }
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
async fn fim_tokenizer(State(app): State<App>) -> Response {
    let identity = app.identity.lock().unwrap().clone();
    if identity["status"] != "ok" || identity["model_protocol"] != fim_v1::WIRE_VERSION {
        return StatusCode::NOT_FOUND.into_response();
    }
    let Some(profile) = identity["alias"]
        .as_str()
        .and_then(|alias| app.registry.get(alias))
        .and_then(|profile| profile.fim_profile.as_ref())
    else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    Json(json!({
        "alias":identity["alias"],
        "model_sha256":identity["model_sha256"],
        "artifact_manifest_sha256":profile.artifact_manifest_sha256,
        "tokenizer":profile.tokenizer,
        "tokenizer_vocab_ids":profile.tokenizer.tokenizer_vocab_ids,
    }))
    .into_response()
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
        fim_profile: payload
            .as_ref()
            .and_then(|embedded| embedded.fim_profile.clone()),
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
        .route("/v1/fim-tokenizer", get(fim_tokenizer))
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
    fn startup_failure_codes_distinguish_load_tokenizer_and_context_stages() {
        let cases = [
            (
                WorkerFailureStage::ModelLoad,
                "model_load",
                "gguf_model_load_failed",
            ),
            (
                WorkerFailureStage::FimTokenizerVerify,
                "fim_tokenizer_verify",
                "fim_tokenizer_contract_failed",
            ),
            (
                WorkerFailureStage::ModelContextCreate,
                "model_context_create",
                "model_context_initialization_failed",
            ),
        ];
        for &(stage, expected_stage, expected_code) in &cases {
            assert_eq!(stage.stage(), expected_stage);
            assert_eq!(stage.code(), expected_code);
            assert!(
                stage
                    .stage()
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
            );
            assert!(
                stage
                    .code()
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
            );
        }
        assert_ne!(cases[0].2, cases[1].2);
        assert_ne!(cases[1].2, cases[2].2);
    }

    fn synthetic_fim_profile() -> fim_v1::ServingProfile {
        let tokenizer_vocab_ids = vec![
            fim_v1::EOS_TOKEN_ID,
            fim_v1::FIM_PREFIX_TOKEN_ID,
            fim_v1::FIM_MIDDLE_TOKEN_ID,
            fim_v1::FIM_SUFFIX_TOKEN_ID,
        ];
        let tokenizer_vocab_ids_sha256 = digest(
            tokenizer_vocab_ids
                .iter()
                .map(|id| format!("{id}\n"))
                .collect::<String>()
                .as_bytes(),
        );
        let mut tokenizer = fim_v1::TokenizerProfile {
            tokenizer_id: "synthetic/q25-fim".into(),
            tokenizer_revision: "synthetic-revision".into(),
            tokenizer_sha256: "a".repeat(64),
            tokenizer_contract_sha256: String::new(),
            tokenizer_vocab_size: tokenizer_vocab_ids.len(),
            tokenizer_vocab_ids_sha256,
            tokenizer_vocab_ids,
            eos_id: fim_v1::EOS_TOKEN_ID,
            fim_prefix_id: fim_v1::FIM_PREFIX_TOKEN_ID,
            fim_suffix_id: fim_v1::FIM_SUFFIX_TOKEN_ID,
            fim_middle_id: fim_v1::FIM_MIDDLE_TOKEN_ID,
            completion_mode: fim_v1::COMPLETION_MODE.into(),
            special_tokens: vec![
                fim_v1::SpecialToken {
                    id: fim_v1::EOS_TOKEN_ID,
                    spelling: fim_v1::EOS_SPELLING.into(),
                },
                fim_v1::SpecialToken {
                    id: fim_v1::FIM_PREFIX_TOKEN_ID,
                    spelling: fim_v1::FIM_PREFIX.into(),
                },
                fim_v1::SpecialToken {
                    id: fim_v1::FIM_MIDDLE_TOKEN_ID,
                    spelling: fim_v1::FIM_MIDDLE.into(),
                },
                fim_v1::SpecialToken {
                    id: fim_v1::FIM_SUFFIX_TOKEN_ID,
                    spelling: fim_v1::FIM_SUFFIX.into(),
                },
            ],
        };
        tokenizer.tokenizer_contract_sha256 =
            digest(&fim_v1::tokenizer_contract_bytes(&tokenizer).unwrap());
        fim_v1::ServingProfile {
            artifact_manifest_sha256: "b".repeat(64),
            tokenizer,
        }
    }

    fn synthetic_fim_payload(fim_profile: fim_v1::ServingProfile) -> embedded::Payload {
        embedded::Payload {
            version: 1,
            offset: 4096,
            length: 8,
            sha256: "c".repeat(64),
            alias: "q25-fim".into(),
            protocol: fim_v1::WIRE_VERSION.into(),
            output_tokens: fim_v1::OUTPUT_TOKEN_CAP,
            context_size: 2304,
            input_tokens: 1024,
            batch_size: 256,
            microbatch_size: 64,
            threads: 4,
            cache_type: "f16".into(),
            context_layout: fim_v1::CONTEXT_LAYOUT.into(),
            fim_profile: Some(fim_profile),
        }
    }

    #[test]
    fn embedded_fim_profile_must_match_the_verified_footer() {
        let fim_profile = synthetic_fim_profile();
        let payload = synthetic_fim_payload(fim_profile.clone());
        let profile = Profile {
            path: "/proc/self/exe".into(),
            sha256: payload.sha256.clone(),
            protocol: payload.protocol.clone(),
            fim_profile: Some(fim_profile),
            embedded: Some(payload.clone()),
        };
        assert!(verify_model(&profile).is_ok());

        let mut mismatched = profile;
        mismatched
            .fim_profile
            .as_mut()
            .unwrap()
            .artifact_manifest_sha256 = "d".repeat(64);
        assert!(verify_model(&mismatched).is_err());

        let mut wrong_model = Profile {
            path: "/proc/self/exe".into(),
            sha256: "e".repeat(64),
            protocol: payload.protocol.clone(),
            fim_profile: payload.fim_profile.clone(),
            embedded: Some(payload),
        };
        assert!(verify_model(&wrong_model).is_err());
        wrong_model.sha256 = "c".repeat(64);
        wrong_model.protocol = "single-line-edit-v1".into();
        assert!(verify_model(&wrong_model).is_err());
    }

    #[test]
    fn only_fim_generation_renders_control_token_spellings() {
        assert!(generation_renders_special_tokens(fim_v1::WIRE_VERSION));
        assert!(!generation_renders_special_tokens("single-line-edit-v1"));
        assert!(!generation_renders_special_tokens("sweep-full-file-v1"));
    }

    #[test]
    fn context_refusal_exposes_only_the_nfc_unsupported_code() {
        let nfc_refusal = anyhow::anyhow!(
            "retained FIM prefix is not NFC-normalized; tokenizer parity is unavailable"
        );
        assert_eq!(
            public_context_error(&nfc_refusal),
            "fim_context_unsupported_non_nfc"
        );
        let arbitrary = anyhow::anyhow!("private source text must stay hidden");
        assert_eq!(
            public_context_error(&arbitrary),
            "editor context outside contract"
        );
    }

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
