//! Production single-GPU pretraining for the packed Russian corpus.
//!
//! The first curriculum stage can train independent local chunks. After the
//! configured token threshold, adjacent chunks in the same shuffled work block
//! share BDH-CQ fast-weight memory. `<|doc|>` starts a new document and resets
//! that memory. Autograd history is detached independently from memory values,
//! so documents may be long while truncated BPTT remains bounded.

use bdh_cq_llm::{
    Bdh, BdhConfig, BdhForwardOptions, Memory, ModelInput,
    pretrain::{
        CorpusSource, CurriculumPhase, PackedCorpus, PretrainConfig, TrainingSchedule, hex_digest,
        sha256_file, token_file,
    },
};
use burn::{
    backend::{Autodiff, Vulkan, wgpu::WgpuDevice},
    grad_clipping::GradientClippingConfig,
    module::{AutodiffModule, Module},
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    ops::Range,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};
use tokenizers::Tokenizer;

type InferenceBackend = Vulkan<f32, i32>;
type TrainingBackend = Autodiff<InferenceBackend>;
type CheckpointRecorder = BinFileRecorder<FullPrecisionSettings>;
type AnyError = Box<dyn std::error::Error>;

/// Set by the async-signal-safe SIGINT handler and polled between GPU steps.
static INTERRUPT_REQUESTED: AtomicBool = AtomicBool::new(false);

#[derive(Debug)]
struct Arguments {
    config_path: PathBuf,
    import_checkpoint: Option<PathBuf>,
    max_steps: Option<u64>,
    device_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunState {
    format_version: u32,
    config_sha256: String,
    tokenizer_sha256: String,
    #[serde(default)]
    packed_corpora_sha256: String,
    optimizer_step: u64,
    tokens_seen: u64,
    examples_seen: u64,
    block_index: usize,
    sequence_in_block: usize,
    best_validation_loss: Option<f32>,
    #[serde(default)]
    best_stateful_validation_loss: Option<f32>,
    #[serde(default)]
    cq_activation_tokens: Option<u64>,
    #[serde(default)]
    document_resets: u64,
    #[serde(default)]
    block_resets: u64,
    /// Extra memory resets made so a checkpoint taken just after a partial
    /// work-block boundary has exactly reproducible resume semantics.
    #[serde(default)]
    checkpoint_resets: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct LatestPointer {
    checkpoint_dir: String,
    optimizer_step: u64,
}

#[derive(Debug, Serialize)]
struct LogEvent<'a> {
    event: &'a str,
    step: u64,
    tokens_seen: u64,
    phase: CurriculumPhase,
    mode: &'a str,
    loss: f32,
    learning_rate: f64,
    tokens_per_second: f64,
    elapsed_seconds: f64,
    memory_tokens: usize,
    document_resets: u64,
    block_resets: u64,
    memory_rms: Option<f32>,
    memory_abs_max: Option<f32>,
}

struct TokenLoader {
    sequence_length: usize,
    schedule: TrainingSchedule,
    corpora: BTreeMap<CorpusSource, PackedCorpus>,
    block_index: usize,
    sequence_in_block: usize,
}

struct TokenBatch {
    inputs: Vec<i64>,
    targets: Vec<i64>,
    batch_size: usize,
    phase: CurriculumPhase,
    block_started: bool,
    block_ended: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DocumentSegment {
    range: Range<usize>,
    reset_before: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct SourceValidationMetrics {
    memoryless_loss: f32,
    memoryless_perplexity: f32,
    memoryless_bits_per_byte: f32,
    stateful_loss: Option<f32>,
    stateful_perplexity: Option<f32>,
    stateful_bits_per_byte: Option<f32>,
    target_tokens: u64,
    decoded_utf8_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct ValidationMetrics {
    memoryless: f32,
    stateful: Option<f32>,
    per_source: BTreeMap<CorpusSource, SourceValidationMetrics>,
}

#[derive(Debug, Serialize)]
struct ValidationLogEvent<'a> {
    event: &'a str,
    step: u64,
    tokens_seen: u64,
    mode: &'a str,
    memoryless_loss: f32,
    stateful_loss: Option<f32>,
    selected_loss: f32,
    best_selected_loss: f32,
    improved: bool,
    per_source: &'a BTreeMap<CorpusSource, SourceValidationMetrics>,
}

#[derive(Debug, Default, Clone, Copy)]
struct ValidationAccumulator {
    memoryless_loss_sum: f64,
    stateful_loss_sum: f64,
    target_tokens: u64,
    decoded_utf8_bytes: u64,
}

/// Remembers boundaries that occur *inside* a gradient-accumulation step.
///
/// Packed source blocks are not all multiples of gradient accumulation. A
/// boundary can therefore occur on microbatch 16 while the optimizer update is
/// only made on microbatch 32. Looking only at the final microbatch loses that
/// boundary and was the reason STOP and periodic checkpoints never fired.
#[derive(Debug, Default)]
struct CheckpointGate {
    pending: bool,
    crossed_block_boundary: bool,
}

impl CheckpointGate {
    fn observe_batch(&mut self, block_ended: bool) {
        self.crossed_block_boundary |= block_ended;
    }

    fn request(&mut self, requested: bool) {
        self.pending |= requested;
    }

    /// Consume one optimizer step and report whether it is safe to save now.
    fn finish_optimizer_step(&mut self, stateful: bool) -> bool {
        let save_now = self.pending && (!stateful || self.crossed_block_boundary);
        self.crossed_block_boundary = false;
        if save_now {
            self.pending = false;
        }
        save_now
    }
}

#[cfg(unix)]
extern "C" fn handle_sigint(_signal: i32) {
    INTERRUPT_REQUESTED.store(true, Ordering::Relaxed);
}

#[cfg(unix)]
unsafe extern "C" {
    fn signal(signal: i32, handler: usize) -> usize;
}

/// Turn the first Ctrl+C into the same graceful request as a `STOP` file.
///
/// The handler performs only one lock-free atomic store. Model recording and
/// all filesystem work remain in the normal training thread.
#[cfg(unix)]
fn install_interrupt_handler() -> Result<(), AnyError> {
    const SIGINT: i32 = 2;
    const SIG_ERR: usize = usize::MAX;
    // SAFETY: `handle_sigint` has C signal-handler ABI and remains valid for
    // the process lifetime. The handler itself only stores an atomic flag.
    let previous = unsafe { signal(SIGINT, handle_sigint as *const () as usize) };
    if previous == SIG_ERR {
        Err("failed to install the SIGINT checkpoint handler".into())
    } else {
        Ok(())
    }
}

#[cfg(not(unix))]
fn install_interrupt_handler() -> Result<(), AnyError> {
    Ok(())
}

fn main() -> Result<(), AnyError> {
    install_interrupt_handler()?;
    let arguments = parse_arguments()?;
    let config = PretrainConfig::from_path(&arguments.config_path)?;
    let config_sha256 = hex_digest(&sha256_file(&arguments.config_path)?);
    let tokenizer_sha256_bytes = sha256_file(&config.tokenizer)?;
    let tokenizer_sha256 = hex_digest(&tokenizer_sha256_bytes);
    let packed_corpora_sha256 = packed_corpora_fingerprint(&config)?;
    let tokenizer = Tokenizer::from_file(&config.tokenizer)
        .map_err(|error| format!("cannot load tokenizer: {error}"))?;
    let vocabulary = tokenizer.get_vocab_size(true);
    let document_token = tokenizer
        .token_to_id("<|doc|>")
        .ok_or("tokenizer has no <|doc|> token")? as i64;
    let padding_token = tokenizer
        .token_to_id("<|pad|>")
        .ok_or("tokenizer has no <|pad|> token")? as i64;

    fs::create_dir_all(&config.run_dir)?;
    freeze_config(&config.run_dir, &arguments.config_path, &config_sha256)?;
    let schedule = TrainingSchedule::build(&config)?;
    let mut loader = TokenLoader::open(&config, schedule, vocabulary, tokenizer_sha256_bytes)?;

    let device = WgpuDevice::DiscreteGpu(arguments.device_index);
    TrainingBackend::seed(&device, config.seed);
    println!(
        "initializing Vulkan device {:?}; vocab={}, local context={}, schedule={} tokens",
        device, vocabulary, config.sequence_length, loader.schedule.effective_tokens
    );
    let mut model = BdhConfig::new(vocabulary, config.model.dim)
        .with_depth(config.model.depth)
        .with_heads(config.model.heads)
        .with_dim_qk_heads(config.model.dim_qk_heads)
        .with_rotary_dim(config.model.rotary_dim)
        .with_tie_embeddings(config.model.tie_embeddings)
        .with_attn_residual(config.model.attn_residual)
        .with_attn_residual_tied(config.model.attn_residual_tied)
        .with_attn_residual_depth_bias_distance(config.model.attn_residual_depth_bias_distance)
        .with_gated_neuron_state(config.model.gated_neuron_state)
        .with_cq_memory_decay(config.model.cq_memory_decay)
        .with_cq_memory_retention(config.model.cq_memory_retention)
        .init::<TrainingBackend>(&device)?;
    println!("model parameters: {}", model.num_params());
    let mut optimizer = AdamWConfig::new()
        .with_beta_1(config.optimizer.beta_1)
        .with_beta_2(config.optimizer.beta_2)
        .with_weight_decay(config.optimizer.weight_decay)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(
            config.optimizer.gradient_clip_norm,
        )))
        .init();

    let mut state = RunState {
        format_version: if config.format_version >= 2 { 3 } else { 2 },
        config_sha256: config_sha256.clone(),
        tokenizer_sha256: tokenizer_sha256.clone(),
        packed_corpora_sha256: packed_corpora_sha256.clone(),
        optimizer_step: 0,
        tokens_seen: 0,
        examples_seen: 0,
        block_index: 0,
        sequence_in_block: 0,
        best_validation_loss: None,
        best_stateful_validation_loss: None,
        cq_activation_tokens: None,
        document_resets: 0,
        block_resets: 0,
        checkpoint_resets: 0,
    };

    if let Some((checkpoint, saved_state)) = latest_checkpoint(&config.run_dir)? {
        if arguments.import_checkpoint.is_some() {
            return Err("--import-checkpoint is only valid for an empty run directory".into());
        }
        validate_resume_state(
            &saved_state,
            &config_sha256,
            &tokenizer_sha256,
            &packed_corpora_sha256,
        )?;
        (model, optimizer) = load_checkpoint(&checkpoint, &device, model, optimizer)?;
        state = saved_state;
        loader.restore(state.block_index, state.sequence_in_block)?;
        println!(
            "resumed CQ run at step {}, {} tokens from {}",
            state.optimizer_step,
            state.tokens_seen,
            checkpoint.display()
        );
    } else if let Some(checkpoint) = &arguments.import_checkpoint {
        let imported = import_state(
            checkpoint,
            &config,
            &config_sha256,
            &tokenizer_sha256,
            &packed_corpora_sha256,
        )?;
        (model, optimizer) = load_checkpoint(checkpoint, &device, model, optimizer)?;
        state = imported;
        loader.restore(state.block_index, state.sequence_in_block)?;
        println!(
            "imported base checkpoint step {}, {} tokens from {}",
            state.optimizer_step,
            state.tokens_seen,
            checkpoint.display()
        );
    }

    if config.memory.is_stateful(state.tokens_seen) && state.cq_activation_tokens.is_none() {
        state.cq_activation_tokens = Some(state.tokens_seen);
        println!(
            "stateful CQ activates at imported cursor {} (configured threshold {})",
            state.tokens_seen, config.memory.stateful_after_tokens
        );
    }

    train(
        model,
        optimizer,
        state,
        loader,
        &config,
        vocabulary,
        document_token,
        padding_token,
        &tokenizer,
        &device,
        arguments.max_steps,
    )
}

#[allow(clippy::too_many_arguments)]
fn train<O>(
    mut model: Bdh<TrainingBackend>,
    mut optimizer: O,
    mut state: RunState,
    mut loader: TokenLoader,
    config: &PretrainConfig,
    vocabulary: usize,
    document_token: i64,
    padding_token: i64,
    tokenizer: &Tokenizer,
    device: &WgpuDevice,
    max_steps: Option<u64>,
) -> Result<(), AnyError>
where
    O: Optimizer<Bdh<TrainingBackend>, TrainingBackend>,
{
    let stop_at_step = max_steps.map(|additional| state.optimizer_step.saturating_add(additional));
    let criterion = CrossEntropyLossConfig::new().init(device);
    let padded_criterion = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![padding_token as usize]))
        .init(device);
    let mut accumulator = GradientsAccumulator::new();
    let mut accumulated_micro_batches = 0_usize;
    let mut accumulated_loss = 0.0_f32;
    let mut pending_bptt_loss: Option<Tensor<TrainingBackend, 1>> = None;
    let mut chunks_in_graph = 0_usize;
    let mut memories = (0..config.optimizer.micro_batch_size)
        .map(|_| None)
        .collect::<Vec<Option<Memory<TrainingBackend>>>>();
    let mut interval_tokens = 0_u64;
    let training_start = Instant::now();
    let mut interval_start = Instant::now();
    let log_path = config.run_dir.join("train.jsonl");
    let mut stop_pending = false;
    let mut max_stop_pending = false;
    let mut best_checkpoint_pending = false;
    let mut checkpoint_gate = CheckpointGate::default();

    loop {
        let Some(batch) = loader.next_batch(config.optimizer.micro_batch_size)? else {
            flush_bptt(&model, &mut accumulator, &mut pending_bptt_loss);
            checkpoint(&config.run_dir, &model, &optimizer, &state)?;
            println!("training schedule complete; final checkpoint saved");
            break;
        };
        let stateful = config.memory.is_stateful(state.tokens_seen);
        if stateful && state.cq_activation_tokens.is_none() {
            state.cq_activation_tokens = Some(state.tokens_seen);
            memories.iter_mut().for_each(|memory| *memory = None);
            chunks_in_graph = 0;
            println!("stateful CQ activated at {} tokens", state.tokens_seen);
        }
        if stateful && batch.block_started && config.memory.reset_on_work_block {
            memories.iter_mut().for_each(|memory| *memory = None);
            chunks_in_graph = 0;
            state.block_resets += 1;
        }

        let chunk_loss = if stateful {
            stateful_chunk_loss(
                &model,
                &padded_criterion,
                &batch,
                &mut memories,
                document_token,
                padding_token,
                config.memory.reset_on_document,
                &mut state.document_resets,
                vocabulary,
                device,
            )?
        } else {
            memories.iter_mut().for_each(|memory| *memory = None);
            memoryless_chunk_loss(&model, &criterion, &batch, vocabulary, device)?
        };
        let loss_value = chunk_loss.clone().to_data().to_vec::<f32>()?[0];
        if !loss_value.is_finite() {
            checkpoint(&config.run_dir, &model, &optimizer, &state)?;
            return Err(format!("non-finite loss {loss_value}; emergency checkpoint saved").into());
        }

        let scaled = chunk_loss / config.optimizer.gradient_accumulation as f64;
        pending_bptt_loss = Some(match pending_bptt_loss {
            Some(previous) => previous + scaled,
            None => scaled,
        });
        accumulated_micro_batches += 1;
        accumulated_loss += loss_value;
        chunks_in_graph += 1;
        let batch_tokens = (batch.batch_size * config.sequence_length) as u64;
        state.tokens_seen += batch_tokens;
        state.examples_seen += batch.batch_size as u64;
        interval_tokens += batch_tokens;
        checkpoint_gate.observe_batch(batch.block_ended);

        let graph_boundary = !stateful
            || chunks_in_graph >= config.memory.chunks_per_detach
            || batch.block_ended
            || accumulated_micro_batches == config.optimizer.gradient_accumulation;
        if graph_boundary {
            flush_bptt(&model, &mut accumulator, &mut pending_bptt_loss);
            for memory in &mut memories {
                *memory = memory.take().map(Memory::detach);
            }
            chunks_in_graph = 0;
        }
        if batch.block_ended && config.memory.reset_on_work_block {
            memories.iter_mut().for_each(|memory| *memory = None);
        }

        if accumulated_micro_batches < config.optimizer.gradient_accumulation {
            continue;
        }

        let learning_rate = learning_rate(
            config,
            state.tokens_seen,
            loader.schedule.phase_one_tokens,
            loader.schedule.effective_tokens,
        );
        model = optimizer.step(learning_rate, model, accumulator.grads());
        state.optimizer_step += 1;
        state.block_index = loader.block_index;
        state.sequence_in_block = loader.sequence_in_block;
        let mean_loss = accumulated_loss / accumulated_micro_batches as f32;
        accumulated_micro_batches = 0;
        accumulated_loss = 0.0;

        let stop_requested =
            config.run_dir.join("STOP").is_file() || INTERRUPT_REQUESTED.load(Ordering::Relaxed);
        if stop_requested && !stop_pending {
            println!(
                "stop requested; waiting for the next safe work-block boundary (at most one block)"
            );
        }
        stop_pending |= stop_requested;
        max_stop_pending |= stop_at_step.is_some_and(|limit| state.optimizer_step >= limit);

        if state.optimizer_step.is_multiple_of(config.log_every_steps) {
            let interval_seconds = interval_start.elapsed().as_secs_f64().max(1e-6);
            let (memory_rms, memory_abs_max) = memory_statistics(&memories)?;
            if memory_rms.is_some_and(|value| !value.is_finite())
                || memory_abs_max.is_some_and(|value| !value.is_finite())
            {
                checkpoint(&config.run_dir, &model, &optimizer, &state)?;
                return Err("non-finite CQ memory statistic; emergency checkpoint saved".into());
            }
            let event = LogEvent {
                event: "train",
                step: state.optimizer_step,
                tokens_seen: state.tokens_seen,
                phase: batch.phase,
                mode: if stateful {
                    "stateful_cq"
                } else {
                    "memoryless"
                },
                loss: mean_loss,
                learning_rate,
                tokens_per_second: interval_tokens as f64 / interval_seconds,
                elapsed_seconds: training_start.elapsed().as_secs_f64(),
                memory_tokens: memories
                    .iter()
                    .flatten()
                    .map(|value| value.tokens_seen)
                    .max()
                    .unwrap_or(0),
                document_resets: state.document_resets,
                block_resets: state.block_resets,
                memory_rms,
                memory_abs_max,
            };
            append_json_line(&log_path, &event)?;
            println!(
                "step {:>7} | {:?} | {} | tokens {:>10} | loss {:.5} | lr {:.3e} | {:.0} tok/s | memory {}",
                event.step,
                event.phase,
                event.mode,
                event.tokens_seen,
                event.loss,
                event.learning_rate,
                event.tokens_per_second,
                event.memory_tokens,
            );
            interval_tokens = 0;
            interval_start = Instant::now();
        }

        if state
            .optimizer_step
            .is_multiple_of(config.validation_every_steps)
            && !stop_pending
            && !max_stop_pending
        {
            let validation = validate(
                &model,
                &mut loader,
                config,
                vocabulary,
                document_token,
                padding_token,
                tokenizer,
                device,
                stateful,
            )?;
            let selected = validation.stateful.unwrap_or(validation.memoryless);
            let previous_best = if stateful {
                state.best_stateful_validation_loss
            } else {
                state.best_validation_loss
            };
            let improved = previous_best.is_none_or(|old| selected < old);
            let best = if stateful {
                state.best_stateful_validation_loss = Some(
                    state
                        .best_stateful_validation_loss
                        .map_or(selected, |old| old.min(selected)),
                );
                state.best_stateful_validation_loss.expect("just assigned")
            } else {
                state.best_validation_loss = Some(
                    state
                        .best_validation_loss
                        .map_or(selected, |old| old.min(selected)),
                );
                state.best_validation_loss.expect("just assigned")
            };
            best_checkpoint_pending |= improved;
            checkpoint_gate.request(improved);
            append_json_line(
                &log_path,
                &ValidationLogEvent {
                    event: "validation",
                    step: state.optimizer_step,
                    tokens_seen: state.tokens_seen,
                    mode: if stateful {
                        "stateful_cq"
                    } else {
                        "memoryless"
                    },
                    memoryless_loss: validation.memoryless,
                    stateful_loss: validation.stateful,
                    selected_loss: selected,
                    best_selected_loss: best,
                    improved,
                    per_source: &validation.per_source,
                },
            )?;
            println!(
                "validation step {}: memoryless {:.5}, stateful {:?}, best {:.5}",
                state.optimizer_step, validation.memoryless, validation.stateful, best,
            );
            for (source, metrics) in &validation.per_source {
                println!(
                    "  {:<12} | loss {:.5} / {:?} | ppl {:.1} / {:?} | BPB {:.3} / {:?}",
                    source.as_str(),
                    metrics.memoryless_loss,
                    metrics.stateful_loss,
                    metrics.memoryless_perplexity,
                    metrics.stateful_perplexity,
                    metrics.memoryless_bits_per_byte,
                    metrics.stateful_bits_per_byte,
                );
            }
        }

        let periodic = state
            .optimizer_step
            .is_multiple_of(config.checkpoint_every_steps);
        checkpoint_gate.request(periodic || stop_pending || max_stop_pending);
        let safe_boundary = checkpoint_gate.finish_optimizer_step(stateful);
        if safe_boundary {
            // A partial source block may already have started later in this
            // optimizer accumulation. Its Memory cannot be serialized by the
            // model recorder, so make the reset explicit both now and after
            // resume. Model/optimizer/corpus cursor remain fully checkpointed.
            if stateful {
                memories.iter_mut().for_each(|memory| *memory = None);
                chunks_in_graph = 0;
                state.checkpoint_resets += 1;
            }
            checkpoint(&config.run_dir, &model, &optimizer, &state)?;
            if best_checkpoint_pending {
                write_json_atomically(
                    config.run_dir.join("checkpoints/best.json"),
                    &LatestPointer {
                        checkpoint_dir: format!("step-{:012}", state.optimizer_step),
                        optimizer_step: state.optimizer_step,
                    },
                )?;
                best_checkpoint_pending = false;
            }
            println!(
                "checkpoint saved at safe block boundary, step {} ({} tokens)",
                state.optimizer_step, state.tokens_seen
            );
        }
        if (stop_pending || max_stop_pending) && safe_boundary {
            if INTERRUPT_REQUESTED.load(Ordering::Relaxed) {
                println!("SIGINT handled cleanly after saving the checkpoint");
            } else if stop_pending {
                println!(
                    "{} detected; stopped cleanly (remove it before resume)",
                    config.run_dir.join("STOP").display()
                );
            } else {
                println!("requested --max-steps reached at a safe block boundary");
            }
            break;
        }
    }
    Ok(())
}

fn memoryless_chunk_loss(
    model: &Bdh<TrainingBackend>,
    criterion: &burn::nn::loss::CrossEntropyLoss<TrainingBackend>,
    batch: &TokenBatch,
    vocabulary: usize,
    device: &WgpuDevice,
) -> Result<Tensor<TrainingBackend, 1>, AnyError> {
    let sequence = batch.inputs.len() / batch.batch_size;
    let inputs =
        ids_tensor::<TrainingBackend>(batch.inputs.clone(), batch.batch_size, sequence, device);
    let targets =
        ids_tensor::<TrainingBackend>(batch.targets.clone(), batch.batch_size, sequence, device)
            .reshape([batch.batch_size * sequence]);
    let logits = model
        .forward(ModelInput::TokenIds(inputs), None, Default::default())?
        .logits
        .expect("default BDH forward requests logits")
        .reshape([batch.batch_size * sequence, vocabulary]);
    Ok(criterion.forward(logits, targets))
}

#[allow(clippy::too_many_arguments)]
fn stateful_chunk_loss(
    model: &Bdh<TrainingBackend>,
    criterion: &burn::nn::loss::CrossEntropyLoss<TrainingBackend>,
    batch: &TokenBatch,
    memories: &mut [Option<Memory<TrainingBackend>>],
    document_token: i64,
    padding_token: i64,
    reset_on_document: bool,
    document_resets: &mut u64,
    vocabulary: usize,
    device: &WgpuDevice,
) -> Result<Tensor<TrainingBackend, 1>, AnyError> {
    if memories.len() != batch.batch_size {
        return Err(format!(
            "stateful memory lanes changed from {} to {}",
            memories.len(),
            batch.batch_size
        )
        .into());
    }
    let sequence = batch.inputs.len() / batch.batch_size;
    let mut combined: Option<Tensor<TrainingBackend, 1>> = None;
    for (lane, memory) in memories.iter_mut().enumerate() {
        let start = lane * sequence;
        let end = start + sequence;
        let lane_batch = TokenBatch {
            inputs: batch.inputs[start..end].to_vec(),
            targets: batch.targets[start..end].to_vec(),
            batch_size: 1,
            phase: batch.phase,
            block_started: batch.block_started,
            block_ended: batch.block_ended,
        };
        let mut lane_loss: Option<Tensor<TrainingBackend, 1>> = None;
        for segment in document_segments(&lane_batch.inputs, document_token, reset_on_document) {
            if segment.reset_before {
                *memory = None;
                *document_resets += 1;
            }
            let length = segment.range.len();
            let physical_length = segment_bucket_length(length, sequence);
            let (input_ids, target_ids) =
                padded_segment_tokens(&lane_batch, segment.range, physical_length, padding_token);
            let inputs = ids_tensor::<TrainingBackend>(input_ids, 1, physical_length, device);
            let targets = ids_tensor::<TrainingBackend>(target_ids, 1, physical_length, device)
                .reshape([physical_length]);
            let output = model.forward(
                ModelInput::TokenIds(inputs),
                memory.take(),
                BdhForwardOptions {
                    valid_sequence_length: Some(length),
                    ..Default::default()
                },
            )?;
            let logits = output
                .logits
                .expect("default BDH forward requests logits")
                .reshape([physical_length, vocabulary]);
            // Burn masks padded labels to zero but averages over the physical
            // length. This restores a mean over the original logical chunk.
            let weighted =
                criterion.forward(logits, targets) * (physical_length as f64 / sequence as f64);
            lane_loss = Some(match lane_loss {
                Some(previous) => previous + weighted,
                None => weighted,
            });
            *memory = Some(output.memory);
        }
        let lane_loss =
            lane_loss.ok_or("a stateful lane cannot be empty")? / batch.batch_size as f64;
        combined = Some(match combined {
            Some(previous) => previous + lane_loss,
            None => lane_loss,
        });
    }
    combined.ok_or_else(|| "a training chunk cannot be empty".into())
}

/// Map arbitrary document fragments onto a tiny set of physical GPU shapes.
///
/// Production chunks are 256 positions, yielding only
/// `16, 32, 64, 128, 256`. A non-standard final maximum is retained as one
/// additional bucket, which keeps tests and alternate configs well-defined.
fn segment_bucket_length(valid_length: usize, maximum: usize) -> usize {
    assert!(valid_length > 0 && valid_length <= maximum);
    [16, 32, 64, 128, 256]
        .into_iter()
        .find(|bucket| *bucket >= valid_length && *bucket <= maximum)
        .unwrap_or(maximum)
}

/// Copy one real segment and fill only its trailing physical slots with pad.
fn padded_segment_tokens(
    batch: &TokenBatch,
    range: Range<usize>,
    physical_length: usize,
    padding_token: i64,
) -> (Vec<i64>, Vec<i64>) {
    let valid_length = range.len();
    debug_assert!(physical_length >= valid_length);
    let mut inputs = vec![padding_token; physical_length];
    let mut targets = vec![padding_token; physical_length];
    inputs[..valid_length].copy_from_slice(&batch.inputs[range.clone()]);
    targets[..valid_length].copy_from_slice(&batch.targets[range]);
    (inputs, targets)
}

fn flush_bptt(
    model: &Bdh<TrainingBackend>,
    accumulator: &mut GradientsAccumulator<Bdh<TrainingBackend>>,
    pending: &mut Option<Tensor<TrainingBackend, 1>>,
) {
    if let Some(loss) = pending.take() {
        let gradients = GradientsParams::from_grads(loss.backward(), model);
        accumulator.accumulate(model, gradients);
    }
}

fn document_segments(inputs: &[i64], document_token: i64, reset: bool) -> Vec<DocumentSegment> {
    if inputs.is_empty() {
        return Vec::new();
    }
    if !reset {
        return vec![DocumentSegment {
            range: 0..inputs.len(),
            reset_before: false,
        }];
    }
    let starts = std::iter::once(0)
        .chain(
            inputs
                .iter()
                .enumerate()
                .skip(1)
                .filter_map(|(index, token)| (*token == document_token).then_some(index)),
        )
        .collect::<Vec<_>>();
    starts
        .iter()
        .enumerate()
        .map(|(index, start)| DocumentSegment {
            range: *start..starts.get(index + 1).copied().unwrap_or(inputs.len()),
            reset_before: inputs[*start] == document_token,
        })
        .collect()
}

fn memory_statistics(
    memories: &[Option<Memory<TrainingBackend>>],
) -> Result<(Option<f32>, Option<f32>), AnyError> {
    let mut square_sum = 0.0_f64;
    let mut elements = 0_usize;
    let mut abs_max = 0.0_f32;
    for memory in memories.iter().flatten() {
        for weight in memory.fast_weights.iter().flatten() {
            let count = weight.dims().into_iter().product::<usize>();
            let mean_square = weight
                .clone()
                .powf_scalar(2.0)
                .mean()
                .to_data()
                .to_vec::<f32>()?[0];
            let maximum = weight.clone().abs().max().to_data().to_vec::<f32>()?[0];
            square_sum += f64::from(mean_square) * count as f64;
            elements += count;
            abs_max = abs_max.max(maximum);
        }
    }
    if elements == 0 {
        Ok((None, None))
    } else {
        Ok((
            Some((square_sum / elements as f64).sqrt() as f32),
            Some(abs_max),
        ))
    }
}

impl TokenLoader {
    fn open(
        config: &PretrainConfig,
        schedule: TrainingSchedule,
        vocabulary: usize,
        tokenizer_sha256: [u8; 32],
    ) -> Result<Self, String> {
        let mut corpora = BTreeMap::new();
        for source in CorpusSource::all() {
            let corpus = PackedCorpus::open(token_file(config, source))?;
            let budget = config.budget(source)?;
            if corpus.header.source != source
                || corpus.header.vocab_size != vocabulary as u32
                || corpus.header.train_tokens != budget.train_tokens
                || corpus.header.validation_tokens != budget.validation_tokens
                || corpus.header.tokenizer_sha256 != tokenizer_sha256
            {
                return Err(format!(
                    "packed {} header does not match config/tokenizer; repack the data",
                    source.as_str()
                ));
            }
            corpora.insert(source, corpus);
        }
        Ok(Self {
            sequence_length: config.sequence_length,
            schedule,
            corpora,
            block_index: 0,
            sequence_in_block: 0,
        })
    }

    fn restore(&mut self, block_index: usize, sequence_in_block: usize) -> Result<(), String> {
        if block_index > self.schedule.blocks.len() {
            return Err("checkpoint block cursor exceeds training schedule".into());
        }
        if block_index < self.schedule.blocks.len()
            && sequence_in_block > self.schedule.blocks[block_index].sequences
        {
            return Err("checkpoint sequence cursor exceeds training schedule block".into());
        }
        self.block_index = block_index;
        self.sequence_in_block = sequence_in_block;
        self.normalize_cursor();
        Ok(())
    }

    fn normalize_cursor(&mut self) {
        while self.block_index < self.schedule.blocks.len()
            && self.sequence_in_block >= self.schedule.blocks[self.block_index].sequences
        {
            self.block_index += 1;
            self.sequence_in_block = 0;
        }
    }

    fn next_batch(&mut self, requested: usize) -> Result<Option<TokenBatch>, String> {
        self.normalize_cursor();
        let current_block = self.block_index;
        let sequence_before = self.sequence_in_block;
        let Some(block) = self.schedule.blocks.get(current_block).cloned() else {
            return Ok(None);
        };
        if !block.sequences.is_multiple_of(requested) || !sequence_before.is_multiple_of(requested)
        {
            return Err(format!(
                "work block of {} sequences cannot form {requested} stable lanes at cursor {sequence_before}",
                block.sequences
            ));
        }
        let batch_size = requested;
        let lane_length = block.sequences / batch_size;
        let lane_step = sequence_before / batch_size;
        if lane_step >= lane_length {
            return Err("normalized loader cursor exceeded its work block".into());
        }
        let mut inputs = Vec::with_capacity(batch_size * self.sequence_length);
        let mut targets = Vec::with_capacity(batch_size * self.sequence_length);
        // Each row advances through one contiguous stripe of the work block.
        // Consequently row `lane` at the next call receives the immediately
        // following chunk instead of skipping over the other batch rows.
        for lane in 0..batch_size {
            let sequence =
                striped_sequence_index(block.sequences, batch_size, sequence_before, lane);
            debug_assert_eq!(sequence, lane * lane_length + lane_step);
            let start = block.token_start + (sequence * self.sequence_length) as u64;
            let tokens = self
                .corpora
                .get_mut(&block.source)
                .expect("all schedule sources were opened")
                .read_tokens(start, self.sequence_length + 1)?;
            inputs.extend(
                tokens[..self.sequence_length]
                    .iter()
                    .map(|id| i64::from(*id)),
            );
            targets.extend(
                tokens[1..self.sequence_length + 1]
                    .iter()
                    .map(|id| i64::from(*id)),
            );
        }
        self.sequence_in_block += batch_size;
        self.normalize_cursor();
        Ok(Some(TokenBatch {
            inputs,
            targets,
            batch_size,
            phase: block.phase,
            block_started: sequence_before == 0,
            block_ended: self.block_index != current_block,
        }))
    }

    fn validation_batch(
        &mut self,
        source: CorpusSource,
        sequence_index: u64,
        requested: usize,
    ) -> Result<TokenBatch, String> {
        let corpus = self
            .corpora
            .get_mut(&source)
            .expect("all validation sources were opened");
        let available = (corpus.header.validation_tokens - 1) / self.sequence_length as u64;
        let first = sequence_index % available;
        let batch_size = requested.min((available - first) as usize);
        let start = corpus.header.train_tokens + first * self.sequence_length as u64;
        let tokens = corpus.read_tokens(start, batch_size * self.sequence_length + 1)?;
        let mut inputs = Vec::with_capacity(batch_size * self.sequence_length);
        let mut targets = Vec::with_capacity(batch_size * self.sequence_length);
        for sequence in 0..batch_size {
            let offset = sequence * self.sequence_length;
            inputs.extend(
                tokens[offset..offset + self.sequence_length]
                    .iter()
                    .map(|id| i64::from(*id)),
            );
            targets.extend(
                tokens[offset + 1..offset + self.sequence_length + 1]
                    .iter()
                    .map(|id| i64::from(*id)),
            );
        }
        Ok(TokenBatch {
            inputs,
            targets,
            batch_size,
            phase: CurriculumPhase::General,
            block_started: first == 0,
            block_ended: false,
        })
    }
}

/// Map a stable document lane to its next contiguous sequence in a block.
fn striped_sequence_index(
    block_sequences: usize,
    batch_size: usize,
    sequence_cursor: usize,
    lane: usize,
) -> usize {
    debug_assert!(block_sequences.is_multiple_of(batch_size));
    debug_assert!(sequence_cursor.is_multiple_of(batch_size));
    debug_assert!(lane < batch_size);
    lane * (block_sequences / batch_size) + sequence_cursor / batch_size
}

#[allow(clippy::too_many_arguments)]
fn validate(
    model: &Bdh<TrainingBackend>,
    loader: &mut TokenLoader,
    config: &PretrainConfig,
    vocabulary: usize,
    document_token: i64,
    padding_token: i64,
    tokenizer: &Tokenizer,
    device: &WgpuDevice,
    stateful: bool,
) -> Result<ValidationMetrics, AnyError> {
    let inference = model.clone().valid();
    let criterion = CrossEntropyLossConfig::new().init(device);
    let padded_criterion = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![padding_token as usize]))
        .init(device);
    let mut accumulators = BTreeMap::<CorpusSource, ValidationAccumulator>::new();
    for source in CorpusSource::all() {
        accumulators.insert(source, ValidationAccumulator::default());
    }
    let mut memories: BTreeMap<CorpusSource, Option<Memory<InferenceBackend>>> = BTreeMap::new();
    for source in CorpusSource::all() {
        memories.insert(source, None);
    }
    for index in 0..config.validation_batches {
        let source = CorpusSource::all()[index % 3];
        // Stateful validation keeps one exactly contiguous stream per source.
        // Training may use several striped document lanes, but batching those
        // here would require one rotary cursor per row and would change the
        // metric when `micro_batch_size` changes.
        let sequence_index = (index / 3) as u64;
        let batch = loader.validation_batch(source, sequence_index, 1)?;
        let memoryless = inference_chunk_loss(
            &inference, &criterion, &batch, None, None, vocabulary, device,
        )?
        .0;
        let target_tokens = batch.targets.len() as u64;
        let target_ids = batch
            .targets
            .iter()
            .map(|id| u32::try_from(*id))
            .collect::<Result<Vec<_>, _>>()?;
        // `skip_special_tokens=true` removes <|doc|>/<|pad|>, which do not
        // correspond to bytes in the source document. ByteLevel decoding of
        // the complete chunk preserves the original UTF-8 byte accounting.
        let decoded_utf8_bytes = tokenizer
            .decode(&target_ids, true)
            .map_err(|error| format!("cannot decode validation targets: {error}"))?
            .len() as u64;
        let accumulator = accumulators
            .get_mut(&source)
            .expect("all validation sources have accumulators");
        accumulator.memoryless_loss_sum += f64::from(memoryless) * target_tokens as f64;
        accumulator.target_tokens += target_tokens;
        accumulator.decoded_utf8_bytes += decoded_utf8_bytes;
        if stateful {
            let mut memory = memories.remove(&source).flatten();
            let mut weighted = 0.0_f32;
            for segment in document_segments(&batch.inputs, document_token, true) {
                if segment.reset_before {
                    memory = None;
                }
                let length = segment.range.len();
                let physical_length = segment_bucket_length(length, batch.inputs.len());
                let (inputs, targets) =
                    padded_segment_tokens(&batch, segment.range, physical_length, padding_token);
                let segment_batch = TokenBatch {
                    inputs,
                    targets,
                    batch_size: 1,
                    phase: batch.phase,
                    block_started: false,
                    block_ended: false,
                };
                let (value, next) = inference_chunk_loss(
                    &inference,
                    &padded_criterion,
                    &segment_batch,
                    memory,
                    Some(length),
                    vocabulary,
                    device,
                )?;
                weighted += value * physical_length as f32 / batch.inputs.len() as f32;
                memory = next;
            }
            accumulator.stateful_loss_sum += f64::from(weighted) * target_tokens as f64;
            memories.insert(source, memory);
        }
    }
    let mut per_source = BTreeMap::new();
    let mut total = ValidationAccumulator::default();
    for source in CorpusSource::all() {
        let accumulator = accumulators[&source];
        if accumulator.target_tokens == 0 || accumulator.decoded_utf8_bytes == 0 {
            return Err(format!("validation source {} produced no text", source.as_str()).into());
        }
        let memoryless_loss =
            (accumulator.memoryless_loss_sum / accumulator.target_tokens as f64) as f32;
        let stateful_loss = stateful
            .then_some((accumulator.stateful_loss_sum / accumulator.target_tokens as f64) as f32);
        per_source.insert(
            source,
            validation_source_metrics(memoryless_loss, stateful_loss, accumulator),
        );
        total.memoryless_loss_sum += accumulator.memoryless_loss_sum;
        total.stateful_loss_sum += accumulator.stateful_loss_sum;
        total.target_tokens += accumulator.target_tokens;
        total.decoded_utf8_bytes += accumulator.decoded_utf8_bytes;
    }
    Ok(ValidationMetrics {
        memoryless: (total.memoryless_loss_sum / total.target_tokens as f64) as f32,
        stateful: stateful.then_some((total.stateful_loss_sum / total.target_tokens as f64) as f32),
        per_source,
    })
}

fn validation_source_metrics(
    memoryless_loss: f32,
    stateful_loss: Option<f32>,
    accumulator: ValidationAccumulator,
) -> SourceValidationMetrics {
    let bits_per_byte = |loss: f32| {
        (f64::from(loss) * accumulator.target_tokens as f64
            / (std::f64::consts::LN_2 * accumulator.decoded_utf8_bytes as f64)) as f32
    };
    SourceValidationMetrics {
        memoryless_loss,
        memoryless_perplexity: memoryless_loss.exp(),
        memoryless_bits_per_byte: bits_per_byte(memoryless_loss),
        stateful_loss,
        stateful_perplexity: stateful_loss.map(f32::exp),
        stateful_bits_per_byte: stateful_loss.map(bits_per_byte),
        target_tokens: accumulator.target_tokens,
        decoded_utf8_bytes: accumulator.decoded_utf8_bytes,
    }
}

fn inference_chunk_loss(
    model: &Bdh<InferenceBackend>,
    criterion: &burn::nn::loss::CrossEntropyLoss<InferenceBackend>,
    batch: &TokenBatch,
    memory: Option<Memory<InferenceBackend>>,
    valid_sequence_length: Option<usize>,
    vocabulary: usize,
    device: &WgpuDevice,
) -> Result<(f32, Option<Memory<InferenceBackend>>), AnyError> {
    let sequence = batch.inputs.len() / batch.batch_size;
    let inputs =
        ids_tensor::<InferenceBackend>(batch.inputs.clone(), batch.batch_size, sequence, device);
    let targets =
        ids_tensor::<InferenceBackend>(batch.targets.clone(), batch.batch_size, sequence, device)
            .reshape([batch.batch_size * sequence]);
    let output = model.forward(
        ModelInput::TokenIds(inputs),
        memory,
        BdhForwardOptions {
            valid_sequence_length,
            ..Default::default()
        },
    )?;
    let logits = output
        .logits
        .expect("default BDH forward requests logits")
        .reshape([batch.batch_size * sequence, vocabulary]);
    let value = criterion
        .forward(logits, targets)
        .to_data()
        .to_vec::<f32>()?[0];
    Ok((value, Some(output.memory)))
}

fn ids_tensor<B: Backend>(
    ids: Vec<i64>,
    batch: usize,
    sequence: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    Tensor::from_data(TensorData::new(ids, [batch, sequence]), device)
}

fn learning_rate(
    config: &PretrainConfig,
    tokens_seen: u64,
    phase_one_tokens: u64,
    total_tokens: u64,
) -> f64 {
    if config.optimizer.focus_max_learning_rate > 0.0 && tokens_seen >= phase_one_tokens {
        let focus_tokens = tokens_seen.saturating_sub(phase_one_tokens);
        let focus_min = if config.optimizer.focus_min_learning_rate > 0.0 {
            config.optimizer.focus_min_learning_rate
        } else {
            config.optimizer.min_learning_rate
        };
        let focus_max = config.optimizer.focus_max_learning_rate;
        if focus_tokens < config.optimizer.focus_warmup_tokens {
            let progress = focus_tokens as f64 / config.optimizer.focus_warmup_tokens.max(1) as f64;
            return config.optimizer.min_learning_rate
                + (focus_max - config.optimizer.min_learning_rate) * progress;
        }
        let decay_tokens = total_tokens
            .saturating_sub(phase_one_tokens)
            .saturating_sub(config.optimizer.focus_warmup_tokens)
            .max(1);
        let progress = focus_tokens
            .saturating_sub(config.optimizer.focus_warmup_tokens)
            .min(decay_tokens) as f64
            / decay_tokens as f64;
        let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        return focus_min + (focus_max - focus_min) * cosine;
    }

    if tokens_seen < config.optimizer.warmup_tokens {
        return config.optimizer.max_learning_rate * tokens_seen as f64
            / config.optimizer.warmup_tokens.max(1) as f64;
    }
    let decay_end = if config.optimizer.focus_max_learning_rate > 0.0 {
        phase_one_tokens
    } else {
        total_tokens
    };
    let decay_tokens = decay_end
        .saturating_sub(config.optimizer.warmup_tokens)
        .max(1);
    let progress = tokens_seen
        .saturating_sub(config.optimizer.warmup_tokens)
        .min(decay_tokens) as f64
        / decay_tokens as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    config.optimizer.min_learning_rate
        + (config.optimizer.max_learning_rate - config.optimizer.min_learning_rate) * cosine
}

fn load_checkpoint<O>(
    checkpoint: &Path,
    device: &WgpuDevice,
    model: Bdh<TrainingBackend>,
    optimizer: O,
) -> Result<(Bdh<TrainingBackend>, O), AnyError>
where
    O: Optimizer<Bdh<TrainingBackend>, TrainingBackend>,
{
    let recorder = CheckpointRecorder::default();
    let model_record = recorder.load(checkpoint.join("model"), device)?;
    let model = model.load_record(model_record);
    let optimizer_record = recorder.load(checkpoint.join("optimizer"), device)?;
    let optimizer = optimizer.load_record(optimizer_record);
    Ok((model, optimizer))
}

fn import_state(
    checkpoint: &Path,
    config: &PretrainConfig,
    config_sha256: &str,
    tokenizer_sha256: &str,
    packed_corpora_sha256: &str,
) -> Result<RunState, AnyError> {
    let mut state: RunState = serde_json::from_slice(&fs::read(checkpoint.join("state.json"))?)?;
    if state.format_version != 1 && state.format_version != 2 && state.format_version != 3 {
        return Err(format!(
            "cannot import checkpoint state version {}",
            state.format_version
        )
        .into());
    }
    if state.tokenizer_sha256 != tokenizer_sha256
        || state.packed_corpora_sha256 != packed_corpora_sha256
    {
        return Err("imported checkpoint data/tokenizer fingerprints do not match".into());
    }
    let run_dir = checkpoint
        .parent()
        .and_then(Path::parent)
        .ok_or("checkpoint must be RUN/checkpoints/step-N")?;
    let previous_config_path = run_dir.join("config.json");
    if hex_digest(&sha256_file(&previous_config_path)?) != state.config_sha256 {
        return Err("imported checkpoint frozen config hash does not match state".into());
    }
    let previous = PretrainConfig::from_path(previous_config_path)?;
    if !config.continuation_compatible_with(&previous) {
        return Err("new config changes more than run_dir/memory policy; refusing import".into());
    }
    state.format_version = if config.format_version >= 2 { 3 } else { 2 };
    state.config_sha256 = config_sha256.to_owned();
    state.cq_activation_tokens = config
        .memory
        .is_stateful(state.tokens_seen)
        .then_some(state.tokens_seen);
    state.document_resets = 0;
    state.block_resets = 0;
    Ok(state)
}

fn checkpoint<O>(
    run_dir: &Path,
    model: &Bdh<TrainingBackend>,
    optimizer: &O,
    state: &RunState,
) -> Result<(), AnyError>
where
    O: Optimizer<Bdh<TrainingBackend>, TrainingBackend>,
{
    let checkpoints = run_dir.join("checkpoints");
    fs::create_dir_all(&checkpoints)?;
    let name = format!("step-{:012}", state.optimizer_step);
    let final_dir = checkpoints.join(&name);
    let temporary = checkpoints.join(format!("{name}.partial"));
    if temporary.exists() {
        fs::remove_dir_all(&temporary)?;
    }
    fs::create_dir(&temporary)?;
    let recorder = CheckpointRecorder::default();
    recorder.record(model.clone().into_record(), temporary.join("model"))?;
    recorder.record(optimizer.to_record(), temporary.join("optimizer"))?;
    write_json(temporary.join("state.json"), state)?;
    if final_dir.exists() {
        fs::remove_dir_all(&final_dir)?;
    }
    fs::rename(&temporary, &final_dir)?;
    write_json_atomically(
        checkpoints.join("latest.json"),
        &LatestPointer {
            checkpoint_dir: name,
            optimizer_step: state.optimizer_step,
        },
    )?;
    prune_checkpoints(&checkpoints, 2)?;
    Ok(())
}

fn latest_checkpoint(run_dir: &Path) -> Result<Option<(PathBuf, RunState)>, AnyError> {
    let checkpoints = run_dir.join("checkpoints");
    let pointer_path = checkpoints.join("latest.json");
    if !pointer_path.is_file() {
        return Ok(None);
    }
    let pointer: LatestPointer = serde_json::from_slice(&fs::read(&pointer_path)?)?;
    let directory = checkpoints.join(pointer.checkpoint_dir);
    let state: RunState = serde_json::from_slice(&fs::read(directory.join("state.json"))?)?;
    if state.optimizer_step != pointer.optimizer_step {
        return Err("latest checkpoint pointer/state step mismatch".into());
    }
    Ok(Some((directory, state)))
}

fn validate_resume_state(
    state: &RunState,
    config_sha256: &str,
    tokenizer_sha256: &str,
    packed_corpora_sha256: &str,
) -> Result<(), String> {
    if state.format_version != 2 && state.format_version != 3 {
        return Err(format!(
            "CQ run requires checkpoint state version 2 or 3, got {}",
            state.format_version
        ));
    }
    if state.config_sha256 != config_sha256 {
        return Err("checkpoint was created with a different configuration".into());
    }
    if state.tokenizer_sha256 != tokenizer_sha256 {
        return Err("checkpoint was created with a different tokenizer".into());
    }
    if state.packed_corpora_sha256 != packed_corpora_sha256 {
        return Err("checkpoint was created with different packed corpus bytes".into());
    }
    Ok(())
}

fn packed_corpora_fingerprint(config: &PretrainConfig) -> Result<String, AnyError> {
    let mut combined = Sha256::new();
    for source in CorpusSource::all() {
        let digest = sha256_file(token_file(config, source))?;
        combined.update(source.as_str().as_bytes());
        combined.update(digest);
    }
    let fingerprint: [u8; 32] = combined.finalize().into();
    let fingerprint = hex_digest(&fingerprint);
    println!("packed corpora fingerprint: {fingerprint}");
    Ok(fingerprint)
}

fn freeze_config(run_dir: &Path, source: &Path, expected_sha256: &str) -> Result<(), AnyError> {
    let destination = run_dir.join("config.json");
    if destination.exists() {
        if hex_digest(&sha256_file(&destination)?) != expected_sha256 {
            return Err(format!("{} differs from requested config", destination.display()).into());
        }
    } else {
        fs::copy(source, destination)?;
    }
    Ok(())
}

fn prune_checkpoints(checkpoints: &Path, keep: usize) -> Result<(), std::io::Error> {
    let protected_best = fs::read(checkpoints.join("best.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<LatestPointer>(&bytes).ok())
        .map(|pointer| pointer.checkpoint_dir);
    let mut directories = fs::read_dir(checkpoints)?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("step-"))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    directories.sort();
    let newest = directories
        .iter()
        .rev()
        .take(keep)
        .cloned()
        .collect::<Vec<_>>();
    for directory in directories {
        let name = directory
            .file_name()
            .map(|value| value.to_string_lossy().into_owned());
        if newest.contains(&directory) || name == protected_best {
            continue;
        }
        fs::remove_dir_all(directory)?;
    }
    Ok(())
}

fn append_json_line(path: &Path, value: &impl Serialize) -> Result<(), AnyError> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn write_json(path: PathBuf, value: &impl Serialize) -> Result<(), AnyError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

fn write_json_atomically(path: PathBuf, value: &impl Serialize) -> Result<(), AnyError> {
    let temporary = path.with_extension("json.partial");
    write_json(temporary.clone(), value)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut config_path = PathBuf::from("configs/rx6700.json");
    let mut import_checkpoint = None;
    let mut max_steps = None;
    let mut device_index = 0;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--config" => config_path = args.next().ok_or("--config needs a path")?.into(),
            "--import-checkpoint" => {
                import_checkpoint = Some(
                    args.next()
                        .ok_or("--import-checkpoint needs a path")?
                        .into(),
                );
            }
            "--max-steps" => {
                max_steps = Some(
                    args.next()
                        .ok_or("--max-steps needs a number")?
                        .parse()
                        .map_err(|error| format!("invalid --max-steps: {error}"))?,
                );
            }
            "--device" => {
                device_index = args
                    .next()
                    .ok_or("--device needs an index")?
                    .parse()
                    .map_err(|error| format!("invalid --device: {error}"))?;
            }
            "-h" | "--help" => {
                println!(
                    "train_llm [--config PATH] [--import-checkpoint DIR] [--device INDEX] [--max-steps N]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}; use --help")),
        }
    }
    Ok(Arguments {
        config_path,
        import_checkpoint,
        max_steps,
        device_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_marker_starts_a_reset_segment() {
        let segments = document_segments(&[10, 11, 3, 20, 3, 30], 3, true);
        assert_eq!(
            segments,
            vec![
                DocumentSegment {
                    range: 0..2,
                    reset_before: false
                },
                DocumentSegment {
                    range: 2..4,
                    reset_before: true
                },
                DocumentSegment {
                    range: 4..6,
                    reset_before: true
                },
            ]
        );
    }

    #[test]
    fn marker_at_chunk_start_resets_without_empty_segment() {
        assert_eq!(
            document_segments(&[3, 10, 11], 3, true),
            vec![DocumentSegment {
                range: 0..3,
                reset_before: true
            }]
        );
    }

    #[test]
    fn disabled_document_reset_keeps_one_segment() {
        assert_eq!(
            document_segments(&[3, 10, 3], 3, false),
            vec![DocumentSegment {
                range: 0..3,
                reset_before: false
            }]
        );
    }

    #[test]
    fn document_segments_use_only_bounded_gpu_buckets() {
        assert_eq!(segment_bucket_length(1, 256), 16);
        assert_eq!(segment_bucket_length(16, 256), 16);
        assert_eq!(segment_bucket_length(17, 256), 32);
        assert_eq!(segment_bucket_length(64, 256), 64);
        assert_eq!(segment_bucket_length(65, 256), 128);
        assert_eq!(segment_bucket_length(129, 256), 256);
        assert_eq!(segment_bucket_length(256, 256), 256);
        assert_eq!(segment_bucket_length(129, 192), 192);
    }

    #[test]
    fn checkpoint_gate_remembers_a_boundary_before_the_final_microbatch() {
        let mut gate = CheckpointGate::default();

        // A request without a boundary must persist into later steps.
        gate.request(true);
        assert!(!gate.finish_optimizer_step(true));
        assert!(gate.pending);

        // The next boundary occurs halfway through an optimizer step. Later
        // microbatches must not erase it.
        gate.observe_batch(true);
        gate.observe_batch(false);
        assert!(gate.finish_optimizer_step(true));
        assert!(!gate.pending);
    }

    #[test]
    fn memoryless_checkpoint_does_not_wait_for_a_block_boundary() {
        let mut gate = CheckpointGate::default();
        gate.request(true);
        assert!(gate.finish_optimizer_step(false));
    }

    #[test]
    fn striped_lanes_advance_contiguously_without_cross_talk() {
        assert_eq!(
            (0..4)
                .map(|lane| striped_sequence_index(256, 4, 0, lane))
                .collect::<Vec<_>>(),
            vec![0, 64, 128, 192]
        );
        assert_eq!(
            (0..4)
                .map(|lane| striped_sequence_index(256, 4, 4, lane))
                .collect::<Vec<_>>(),
            vec![1, 65, 129, 193]
        );
        for lane in 0..4 {
            let positions = (0..64)
                .map(|step| striped_sequence_index(256, 4, step * 4, lane))
                .collect::<Vec<_>>();
            assert!(positions.windows(2).all(|pair| pair[1] == pair[0] + 1));
        }
    }

    #[test]
    fn v2_learning_rate_rewarms_at_the_focus_boundary() {
        let config = PretrainConfig::from_path("configs/rx6700-v2.json").unwrap();
        let schedule = TrainingSchedule::build(&config).unwrap();
        let phase_one = schedule.phase_one_tokens;
        let total = schedule.effective_tokens;
        let close = |actual: f64, expected: f64| {
            assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}")
        };

        close(learning_rate(&config, 0, phase_one, total), 0.0);
        close(
            learning_rate(&config, config.optimizer.warmup_tokens, phase_one, total),
            config.optimizer.max_learning_rate,
        );
        close(
            learning_rate(&config, phase_one, phase_one, total),
            config.optimizer.min_learning_rate,
        );
        close(
            learning_rate(
                &config,
                phase_one + config.optimizer.focus_warmup_tokens,
                phase_one,
                total,
            ),
            config.optimizer.focus_max_learning_rate,
        );
        close(
            learning_rate(&config, total, phase_one, total),
            config.optimizer.focus_min_learning_rate,
        );
    }

    #[test]
    fn bits_per_byte_uses_cross_entropy_nats_and_decoded_bytes() {
        let metrics = validation_source_metrics(
            std::f32::consts::LN_2,
            Some(2.0 * std::f32::consts::LN_2),
            ValidationAccumulator {
                memoryless_loss_sum: 0.0,
                stateful_loss_sum: 0.0,
                target_tokens: 100,
                decoded_utf8_bytes: 200,
            },
        );
        assert!((metrics.memoryless_bits_per_byte - 0.5).abs() < 1e-6);
        assert!((metrics.stateful_bits_per_byte.unwrap() - 1.0).abs() < 1e-6);
    }
}
