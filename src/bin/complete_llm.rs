//! Interactive text completion for a production language-model checkpoint.
//!
//! This binary is intentionally separate from `train_llm`: it loads only the
//! model record, converts it to the inference backend, and never opens the
//! optimizer or packed corpora. By default it follows the atomic
//! `RUN/checkpoints/latest.json` pointer written by the trainer.
//!
//! The current checkpoint is a base next-token model, not a chat-SFT model.
//! The REPL consequently inserts no role labels or chat-control tokens. It
//! prepends the trained `<|doc|>` boundary once when a stream starts, then
//! carries CQ fast-weight memory through input fragments and generated tokens.
//! `/reset` starts another document.

use bdh_cq_llm::precision::ProjectionBackend;
use bdh_cq_llm::pretrain::{PretrainConfig, hex_digest, sha256_file};
use bdh_cq_llm::{Bdh, BdhConfig, BdhForwardOptions, Memory, ModelInput};
use burn::{
    backend::{Autodiff, Vulkan, wgpu::WgpuDevice},
    module::{AutodiffModule, Module},
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{DType, Int, Tensor, TensorData, backend::Backend},
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use std::{
    env,
    error::Error,
    fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
};
use tokenizers::Tokenizer;

type InferenceBackend = Vulkan<f32, i32>;
type TrainingBackend = Autodiff<InferenceBackend>;
type CheckpointRecorder = BinFileRecorder<FullPrecisionSettings>;
type AnyError = Box<dyn Error + Send + Sync>;

const DEFAULT_CONFIG: &str = "configs/rx6700-cq.json";
const DEFAULT_MAX_NEW_TOKENS: usize = 128;
const DEFAULT_TEMPERATURE: f64 = 0.8;
const DEFAULT_TOP_K: usize = 50;

#[derive(Debug, Clone, PartialEq)]
struct Arguments {
    config_path: PathBuf,
    checkpoint: Option<PathBuf>,
    device_index: usize,
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    seed: Option<u64>,
}

impl Default for Arguments {
    fn default() -> Self {
        Self {
            config_path: PathBuf::from(DEFAULT_CONFIG),
            checkpoint: None,
            device_index: 0,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            temperature: DEFAULT_TEMPERATURE,
            top_k: DEFAULT_TOP_K,
            seed: None,
        }
    }
}

/// Minimal part of the trainer's atomic latest-checkpoint pointer.
#[derive(Debug, Deserialize)]
struct LatestPointer {
    checkpoint_dir: String,
    optimizer_step: u64,
}

/// Metadata needed to identify and validate a checkpoint for inference.
#[derive(Debug, Deserialize)]
struct CheckpointState {
    config_sha256: String,
    tokenizer_sha256: String,
    optimizer_step: u64,
    tokens_seen: u64,
}

fn main() -> Result<(), AnyError> {
    let arguments = parse_arguments(env::args().skip(1))?;
    let config = PretrainConfig::from_path(&arguments.config_path)?;
    let tokenizer = Tokenizer::from_file(&config.tokenizer)?;
    let vocabulary = tokenizer.get_vocab_size(true);
    let checkpoint = resolve_checkpoint(&config.run_dir, arguments.checkpoint.as_deref())?;
    let state = read_checkpoint_state(&checkpoint)?;
    validate_artifacts(&arguments.config_path, &config.tokenizer, &state)?;

    let document_token = required_token(&tokenizer, "<|doc|>")?;
    // Only <|doc|> occurs in base pretraining. Sampling the other reserved
    // tokens would expose essentially untrained output rows, so mask them.
    let banned_tokens = [
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|eot|>",
    ]
    .into_iter()
    .map(|token| required_token(&tokenizer, token))
    .collect::<Result<Vec<_>, _>>()?;

    let device = WgpuDevice::DiscreteGpu(arguments.device_index);
    TrainingBackend::seed(&device, config.seed);
    println!(
        "loading checkpoint step {} ({} training tokens) from {}",
        state.optimizer_step,
        state.tokens_seen,
        checkpoint.display()
    );
    println!(
        "initializing Vulkan device {:?}; vocab={}, local chunk={}",
        device, vocabulary, config.sequence_length
    );

    // Checkpoints are written from Autodiff<Vulkan>. Load that exact record
    // type first, then strip autodiff tracking for generation.
    let training_model = BdhConfig::new(vocabulary, config.model.dim)
        .with_depth(config.model.depth)
        .with_heads(config.model.heads)
        .with_dim_qk_heads(config.model.dim_qk_heads)
        .with_rotary_dim(config.model.rotary_dim)
        .with_tie_embeddings(config.model.tie_embeddings)
        .with_attn_residual(config.model.attn_residual)
        .with_attn_residual_tied(config.model.attn_residual_tied)
        .with_attn_residual_heads(config.model.attn_residual_heads)
        .with_attn_residual_depth_bias_distance(config.model.attn_residual_depth_bias_distance)
        .with_gated_neuron_state(config.model.gated_neuron_state)
        .with_gated_neuron_state_initial_update(config.model.gated_neuron_state_initial_update)
        .with_gated_neuron_state_initial_injection(
            config.model.gated_neuron_state_initial_injection,
        )
        .with_normalize_each_depth(config.model.normalize_each_depth)
        .with_cq_memory_decay(config.model.cq_memory_decay)
        .with_cq_memory_initial_rho(config.model.cq_memory_initial_rho)
        .init::<TrainingBackend>(&device)?;
    let recorder = CheckpointRecorder::default();
    let record = recorder.load(checkpoint.join("model"), &device)?;
    let model: Bdh<InferenceBackend> = training_model.load_record(record).valid();

    let seed = arguments.seed.unwrap_or(config.seed);
    let mut rng = StdRng::seed_from_u64(seed);
    println!(
        "ready: text completion, temperature={}, top-k={}, max-new-tokens={}, seed={}",
        arguments.temperature, arguments.top_k, arguments.max_new_tokens, seed
    );
    println!("commands: /reset, /status, /help, /quit");
    println!("В модель поступает <|doc|> только в начале, затем ровно введённый текст.\n");

    run_repl(
        &model,
        &tokenizer,
        &device,
        config.sequence_length,
        document_token,
        &banned_tokens,
        &arguments,
        &mut rng,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_repl(
    model: &Bdh<InferenceBackend>,
    tokenizer: &Tokenizer,
    device: &WgpuDevice,
    chunk_size: usize,
    document_token: usize,
    banned_tokens: &[usize],
    arguments: &Arguments,
    rng: &mut StdRng,
) -> Result<(), AnyError> {
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let mut memory: Option<CompletionState<InferenceBackend>> = None;
    let mut document_start_pending = true;

    loop {
        print!("text> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            println!();
            break;
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        match line {
            "/quit" | "/exit" => break,
            "/reset" => {
                memory = None;
                document_start_pending = true;
                println!("CQ-memory сброшена; следующий ввод начнёт новый документ.");
                continue;
            }
            "/status" => {
                println!(
                    "stream memory: {} tokens",
                    memory.as_ref().map_or(0, |value| {
                        value
                            .committed
                            .as_ref()
                            .and_then(|m| m.position_offsets.first().copied())
                            .unwrap_or(0)
                            + value.pending.len()
                    })
                );
                continue;
            }
            "/help" => {
                println!("Введите следующий фрагмент текста или команду:");
                println!("  /reset  — начать новый независимый документ");
                println!("  /status — показать длину CQ-контекста");
                println!("  /quit   — выйти");
                continue;
            }
            _ if line.starts_with('/') => {
                println!("неизвестная команда; /help покажет список");
                continue;
            }
            _ => {}
        }

        // `false` disables tokenizer post-processing. Apart from the explicit
        // document boundary below, these are exactly the user's token IDs.
        let encoding = tokenizer.encode(line, false)?;
        let prompt_tokens =
            prepare_prompt_tokens(encoding.get_ids(), document_start_pending, document_token);
        document_start_pending = false;

        // The first forward in a fresh process may spend noticeable time in
        // lazy Vulkan kernel compilation/autotuning. Print the UI prefix before
        // that work so an interactive terminal does not appear frozen.
        print!("continuation> ");
        io::stdout().flush()?;
        let (next_memory, logits) =
            ingest_tokens(model, memory.take(), &prompt_tokens, chunk_size, device)?;
        let (generated, next_memory, document_ended) = generate_tokens(
            model,
            next_memory,
            logits,
            document_token,
            banned_tokens,
            arguments.max_new_tokens,
            arguments.temperature,
            arguments.top_k,
            rng,
            device,
        )?;
        if document_ended {
            // The sampled marker is not committed to the previous document's
            // memory. The next input receives one fresh <|doc|>, exactly like
            // the beginning of a packed pretraining document.
            memory = None;
            document_start_pending = true;
        } else {
            memory = Some(next_memory);
        }

        let text = tokenizer.decode(
            &generated
                .iter()
                .map(|token| *token as u32)
                .collect::<Vec<_>>(),
            true,
        )?;
        if text.trim().is_empty() {
            println!("[пустое или пробельное продолжение]");
        } else {
            println!("{text}");
        }
        if document_ended {
            println!("[модель завершила документ; следующий ввод начнёт новый]");
        }
    }
    Ok(())
}

/// Convert one already-tokenized user fragment into stream input.
///
/// The document marker is the sole implicit token and appears only at stream
/// start. Later fragments are appended without spaces, newlines, or labels.
fn prepare_prompt_tokens(
    encoded: &[u32],
    document_start: bool,
    document_token: usize,
) -> Vec<usize> {
    let mut tokens = Vec::with_capacity(encoded.len() + usize::from(document_start));
    if document_start {
        tokens.push(document_token);
    }
    tokens.extend(encoded.iter().map(|token| *token as usize));
    tokens
}

/// CQ contains only completed chunks. The unfinished token prefix remains
/// explicit so every next-token prediction retains local causal attention.
/// Re-evaluating that prefix must always start from the SAME committed memory:
/// feeding tentative output memory back would duplicate writes and retention.
struct CompletionState<B: Backend> {
    committed: Option<Memory<B>>,
    pending: Vec<usize>,
    chunk_size: usize,
}

/// Append prompt/generated IDs, committing CQ exactly once per full chunk.
/// Partial chunks are recomputed (no KV-cache implementation) and their
/// tentative memory is discarded. This preserves chunk-level CQ decay and
/// RoPE offsets, including when a later REPL fragment extends the same chunk.
fn ingest_tokens<B: ProjectionBackend>(
    model: &Bdh<B>,
    memory: Option<CompletionState<B>>,
    tokens: &[usize],
    chunk_size: usize,
    device: &B::Device,
) -> Result<(CompletionState<B>, Tensor<B, 3>), AnyError> {
    if tokens.is_empty() || chunk_size == 0 {
        return Err("prompt tokens and chunk size must be non-empty".into());
    }
    let mut state = memory.unwrap_or(CompletionState {
        committed: None,
        pending: Vec::new(),
        chunk_size,
    });
    if state.chunk_size != chunk_size {
        return Err("cannot change chunk size within a completion stream".into());
    }
    let mut last_logits = None;
    let mut remaining = tokens;
    while !remaining.is_empty() {
        let count = remaining.len().min(chunk_size - state.pending.len());
        state.pending.extend_from_slice(&remaining[..count]);
        remaining = &remaining[count..];
        let input = Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                state.pending.iter().map(|t| *t as i64).collect::<Vec<_>>(),
                [1, state.pending.len()],
            ),
            device,
        );
        let output = model.forward(
            ModelInput::TokenIds(input),
            state.committed.clone(),
            BdhForwardOptions::default(),
        )?;
        last_logits = output.logits;
        if state.pending.len() == chunk_size {
            state.committed = Some(output.memory);
            state.pending.clear();
        }
    }
    Ok((state, last_logits.expect("default forward returns logits")))
}

/// Autoregress using discrete token IDs, matching language-model pretraining.
///
/// `ReasoningWrapper::generate` deliberately uses raw embeddings to reproduce
/// upstream latent-reasoning behavior. That is not the path trained by
/// `train_llm`, so text completion appends discrete IDs and recomputes the
/// current chunk prefix after each sample instead.
#[allow(clippy::too_many_arguments)]
fn generate_tokens(
    model: &Bdh<InferenceBackend>,
    mut memory: CompletionState<InferenceBackend>,
    mut logits: Tensor<InferenceBackend, 3>,
    document_token: usize,
    banned_tokens: &[usize],
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    rng: &mut StdRng,
    device: &WgpuDevice,
) -> Result<(Vec<usize>, CompletionState<InferenceBackend>, bool), AnyError> {
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut document_ended = false;
    for _ in 0..max_new_tokens {
        let values = last_token_logits(logits)?;
        let token = sample_from_values(&values, temperature, top_k, banned_tokens, rng)?;
        // <|doc|> is the only trained document terminator. Do not commit it to
        // the completed stream's memory; the REPL starts a new document on the
        // next user fragment.
        if token == document_token {
            document_ended = true;
            break;
        }
        generated.push(token);

        let chunk_size = memory.chunk_size;
        (memory, logits) = ingest_tokens(model, Some(memory), &[token], chunk_size, device)?;
    }
    Ok((generated, memory, document_ended))
}

/// Copy only the final position's vocabulary logits back to the host for
/// scalar sampling. Model and CQ state remain resident on the GPU.
fn last_token_logits(logits: Tensor<InferenceBackend, 3>) -> Result<Vec<f32>, AnyError> {
    let [_, sequence, vocabulary] = logits.dims();
    Ok(logits
        .slice([0..1, sequence - 1..sequence, 0..vocabulary])
        .reshape([vocabulary])
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()?)
}

/// Temperature/top-k categorical sampler with explicit masking for untrained
/// reserved-token rows. Temperature zero selects deterministic greedy output.
fn sample_from_values(
    values: &[f32],
    temperature: f64,
    top_k: usize,
    banned_tokens: &[usize],
    rng: &mut StdRng,
) -> Result<usize, AnyError> {
    if values.is_empty() {
        return Err("cannot sample an empty vocabulary".into());
    }
    let mut candidates = values
        .iter()
        .copied()
        .enumerate()
        .filter(|(token, value)| !banned_tokens.contains(token) && value.is_finite())
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err("all vocabulary logits were masked or non-finite".into());
    }
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    let keep = if top_k == 0 {
        candidates.len()
    } else {
        top_k.min(candidates.len())
    };
    candidates.truncate(keep);

    if temperature == 0.0 {
        return Ok(candidates[0].0);
    }
    let maximum = candidates[0].1 as f64;
    let weights = candidates
        .iter()
        .map(|(_, value)| (((*value as f64) - maximum) / temperature).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err("sampling probabilities are non-finite".into());
    }
    let mut draw = rng.random::<f64>() * total;
    for ((token, _), weight) in candidates.iter().zip(weights) {
        draw -= weight;
        if draw <= 0.0 {
            return Ok(*token);
        }
    }
    Ok(candidates.last().expect("non-empty candidates").0)
}

fn required_token(tokenizer: &Tokenizer, token: &str) -> Result<usize, AnyError> {
    tokenizer
        .token_to_id(token)
        .map(|id| id as usize)
        .ok_or_else(|| format!("tokenizer has no required token {token}").into())
}

fn resolve_checkpoint(run_dir: &Path, explicit: Option<&Path>) -> Result<PathBuf, AnyError> {
    if let Some(checkpoint) = explicit {
        return checkpoint
            .is_dir()
            .then(|| checkpoint.to_owned())
            .ok_or_else(|| {
                format!(
                    "checkpoint directory {} does not exist",
                    checkpoint.display()
                )
                .into()
            });
    }
    let checkpoints = run_dir.join("checkpoints");
    let pointer_path = checkpoints.join("latest.json");
    let pointer: LatestPointer =
        serde_json::from_slice(&fs::read(&pointer_path).map_err(|error| {
            format!(
                "cannot read latest checkpoint pointer {}: {error}",
                pointer_path.display()
            )
        })?)?;
    let directory = checkpoints.join(&pointer.checkpoint_dir);
    let state = read_checkpoint_state(&directory)?;
    if state.optimizer_step != pointer.optimizer_step {
        return Err("latest checkpoint pointer/state step mismatch".into());
    }
    Ok(directory)
}

fn read_checkpoint_state(checkpoint: &Path) -> Result<CheckpointState, AnyError> {
    let state_path = checkpoint.join("state.json");
    let state: CheckpointState =
        serde_json::from_slice(&fs::read(&state_path).map_err(|error| {
            format!(
                "cannot read checkpoint state {}: {error}",
                state_path.display()
            )
        })?)?;
    if !checkpoint.join("model.bin").is_file() {
        return Err(format!("checkpoint {} has no model.bin", checkpoint.display()).into());
    }
    Ok(state)
}

fn validate_artifacts(
    config_path: &Path,
    tokenizer_path: &Path,
    state: &CheckpointState,
) -> Result<(), AnyError> {
    let config_hash = hex_digest(&sha256_file(config_path)?);
    if state.config_sha256 != config_hash {
        return Err("checkpoint was created from a different config file".into());
    }
    let tokenizer_hash = hex_digest(&sha256_file(tokenizer_path)?);
    if state.tokenizer_sha256 != tokenizer_hash {
        return Err("checkpoint was created with a different tokenizer".into());
    }
    Ok(())
}

fn parse_arguments<I>(arguments: I) -> Result<Arguments, AnyError>
where
    I: IntoIterator<Item = String>,
{
    let mut parsed = Arguments::default();
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{argument} requires a value"))
        };
        match argument.as_str() {
            "--config" => parsed.config_path = PathBuf::from(value()?),
            "--checkpoint" => parsed.checkpoint = Some(PathBuf::from(value()?)),
            "--device" => parsed.device_index = value()?.parse()?,
            "--max-new-tokens" => parsed.max_new_tokens = value()?.parse()?,
            "--temperature" => parsed.temperature = value()?.parse()?,
            "--top-k" => parsed.top_k = value()?.parse()?,
            "--seed" => parsed.seed = Some(value()?.parse()?),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument {unknown}; use --help").into()),
        }
    }
    if parsed.max_new_tokens == 0 {
        return Err("--max-new-tokens must be greater than zero".into());
    }
    if !parsed.temperature.is_finite() || parsed.temperature < 0.0 {
        return Err("--temperature must be finite and non-negative".into());
    }
    Ok(parsed)
}

fn print_help() {
    println!(
        "Interactive text completion with a production BDH-CQ checkpoint\n\
         \n\
         Usage:\n\
           cargo run --release --bin complete_llm -- [OPTIONS]\n\
         \n\
         Options:\n\
           --config PATH          training config [{DEFAULT_CONFIG}]\n\
           --checkpoint PATH      explicit step directory instead of latest.json\n\
           --device INDEX         discrete Vulkan device [0]\n\
           --max-new-tokens N     response limit [{DEFAULT_MAX_NEW_TOKENS}]\n\
           --temperature FLOAT    0 for greedy [{DEFAULT_TEMPERATURE}]\n\
           --top-k N              candidates; 0 disables [{DEFAULT_TOP_K}]\n\
           --seed N               sampler seed [training seed]\n\
           -h, --help             show this help"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_prefix_replay_matches_fixed_chunk_forward() {
        type B = burn::backend::NdArray<f32>;
        let device = Default::default();
        let model = BdhConfig::new(32, 8)
            .with_heads(2)
            .with_depth(2)
            .with_dim_qk_heads(16)
            .with_rotary_dim(4)
            .with_tie_embeddings(true)
            .with_gated_neuron_state(true)
            .with_attn_residual(true)
            .with_attn_residual_heads(2)
            .with_normalize_each_depth(true)
            .with_cq_memory_decay(true)
            .init::<B>(&device)
            .unwrap();
        let ids = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut state = None;
        for end in 1..=ids.len() {
            let (next, actual) =
                ingest_tokens(&model, state, &ids[end - 1..end], 4, &device).unwrap();
            assert_eq!(next.pending.len(), end % 4);
            assert_eq!(
                next.committed.as_ref().map_or(0, |m| m.position_offsets[0]),
                end / 4 * 4
            );
            // Independent reference: full chunks followed by one partial
            // prefix, with normal model.forward calls (not the ingestion API).
            let mut reference_memory = None;
            let mut expected = None;
            for chunk in ids[..end].chunks(4) {
                let input = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(
                        chunk.iter().map(|t| *t as i64).collect::<Vec<_>>(),
                        [1, chunk.len()],
                    ),
                    &device,
                );
                let output = model
                    .forward(
                        ModelInput::TokenIds(input),
                        reference_memory,
                        BdhForwardOptions::default(),
                    )
                    .unwrap();
                reference_memory = Some(output.memory);
                expected = output.logits;
            }
            let actual = actual.into_data().to_vec::<f32>().unwrap();
            let expected = expected.unwrap().into_data().to_vec::<f32>().unwrap();
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 1e-5);
            }
            state = Some(next);
        }
        // A whole user fragment must also fill an existing partial chunk,
        // not implicitly start a new CQ chunk at the REPL turn boundary.
        let (first, _) = ingest_tokens(&model, None, &ids[..3], 4, &device).unwrap();
        let (joined, logits) = ingest_tokens(&model, Some(first), &ids[3..], 4, &device).unwrap();
        let (whole, expected) = ingest_tokens(&model, None, &ids, 4, &device).unwrap();
        assert_eq!(joined.pending, whole.pending);
        assert_eq!(
            logits.into_data().to_vec::<f32>().unwrap(),
            expected.into_data().to_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn document_marker_is_the_only_implicit_prompt_token() {
        assert_eq!(prepare_prompt_tokens(&[10, 11], true, 3), [3, 10, 11]);
        assert_eq!(prepare_prompt_tokens(&[12, 13], false, 3), [12, 13]);
    }

    #[test]
    fn greedy_sampling_honors_the_reserved_token_mask() {
        let mut rng = StdRng::seed_from_u64(7);
        let token = sample_from_values(&[100.0, 3.0, 2.0], 0.0, 1, &[0], &mut rng).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn top_k_sampling_cannot_escape_the_kept_set() {
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..32 {
            let token =
                sample_from_values(&[4.0, 3.0, 2.0, 100.0], 1.0, 2, &[3], &mut rng).unwrap();
            assert!(token == 0 || token == 1);
        }
    }

    #[test]
    fn cli_defaults_and_overrides_are_stable() {
        assert_eq!(parse_arguments(Vec::new()).unwrap(), Arguments::default());
        let parsed = parse_arguments([
            "--temperature".to_owned(),
            "0".to_owned(),
            "--top-k".to_owned(),
            "0".to_owned(),
            "--max-new-tokens".to_owned(),
            "7".to_owned(),
        ])
        .unwrap();
        assert_eq!(parsed.temperature, 0.0);
        assert_eq!(parsed.top_k, 0);
        assert_eq!(parsed.max_new_tokens, 7);
    }
}
