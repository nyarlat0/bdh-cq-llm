//! Production-oriented single-GPU pretraining loop for the packed 1B-token run.
//!
//! This trains the core BDH next-token model.  The latent-reasoning wrapper is
//! intentionally not enabled during base pretraining: it is a later supervised
//! or task-specific stage, while all one billion tokens teach the recurrent BDH
//! block and its associative state ordinary language modelling first.
//!
//! The loop targets an RX 6700 XT through Burn's Vulkan backend.  It supports
//! deterministic block shuffling, gradient accumulation, AdamW, token-based
//! warmup/cosine decay, held-out loss, checkpoint resume and a graceful `STOP`
//! file.  Checkpoints are written into a new directory before the atomic
//! `latest.json` pointer changes; the two newest are retained.

use bdh_cq_llm::{
    Bdh, BdhConfig, ModelInput,
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
    path::{Path, PathBuf},
    time::Instant,
};
use tokenizers::Tokenizer;

type InferenceBackend = Vulkan<f32, i32>;
type TrainingBackend = Autodiff<InferenceBackend>;
type CheckpointRecorder = BinFileRecorder<FullPrecisionSettings>;

#[derive(Debug)]
struct Arguments {
    config_path: PathBuf,
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
    loss: f32,
    learning_rate: f64,
    tokens_per_second: f64,
    elapsed_seconds: f64,
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = parse_arguments()?;
    let config = PretrainConfig::from_path(&arguments.config_path)?;
    let config_sha256 = hex_digest(&sha256_file(&arguments.config_path)?);
    let tokenizer_sha256_bytes = sha256_file(&config.tokenizer)?;
    let tokenizer_sha256 = hex_digest(&tokenizer_sha256_bytes);
    let packed_corpora_sha256 = packed_corpora_fingerprint(&config)?;
    let tokenizer = Tokenizer::from_file(&config.tokenizer)
        .map_err(|error| format!("cannot load tokenizer: {error}"))?;
    let vocabulary = tokenizer.get_vocab_size(true);

    fs::create_dir_all(&config.run_dir)?;
    freeze_config(&config.run_dir, &arguments.config_path, &config_sha256)?;
    let schedule = TrainingSchedule::build(&config)?;
    let mut loader = TokenLoader::open(&config, schedule, vocabulary, tokenizer_sha256_bytes)?;

    let device = WgpuDevice::DiscreteGpu(arguments.device_index);
    TrainingBackend::seed(&device, config.seed);
    println!(
        "initializing Vulkan device {:?}; vocab={}, context={}, effective schedule={} tokens",
        device, vocabulary, config.sequence_length, loader.schedule.effective_tokens
    );
    let mut model = BdhConfig::new(vocabulary, config.model.dim)
        .with_depth(config.model.depth)
        .with_heads(config.model.heads)
        .with_dim_qk_heads(config.model.dim_qk_heads)
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
        format_version: 1,
        config_sha256: config_sha256.clone(),
        tokenizer_sha256: tokenizer_sha256.clone(),
        packed_corpora_sha256: packed_corpora_sha256.clone(),
        optimizer_step: 0,
        tokens_seen: 0,
        examples_seen: 0,
        block_index: 0,
        sequence_in_block: 0,
        best_validation_loss: None,
    };
    if let Some((checkpoint, saved_state)) = latest_checkpoint(&config.run_dir)? {
        validate_resume_state(
            &saved_state,
            &config_sha256,
            &tokenizer_sha256,
            &packed_corpora_sha256,
        )?;
        let recorder = CheckpointRecorder::default();
        let model_record = recorder.load(checkpoint.join("model"), &device)?;
        model = model.load_record(model_record);
        let optimizer_record = recorder.load(checkpoint.join("optimizer"), &device)?;
        optimizer = optimizer.load_record(optimizer_record);
        state = saved_state;
        loader.restore(state.block_index, state.sequence_in_block)?;
        println!(
            "resumed step {} at {} tokens from {}",
            state.optimizer_step,
            state.tokens_seen,
            checkpoint.display()
        );
    }

    let stop_at_step = arguments
        .max_steps
        .map(|additional| state.optimizer_step.saturating_add(additional));
    let criterion = CrossEntropyLossConfig::new().init(&device);
    let mut accumulator = GradientsAccumulator::new();
    let mut accumulated = 0_usize;
    let mut accumulated_loss = 0.0_f32;
    let mut interval_tokens = 0_u64;
    let training_start = Instant::now();
    let mut interval_start = Instant::now();
    let log_path = config.run_dir.join("train.jsonl");

    loop {
        if stop_at_step.is_some_and(|limit| state.optimizer_step >= limit) {
            checkpoint(&config.run_dir, &model, &optimizer, &state)?;
            println!("requested --max-steps reached; checkpoint saved");
            break;
        }
        let Some(batch) = loader.next_batch(config.optimizer.micro_batch_size)? else {
            checkpoint(&config.run_dir, &model, &optimizer, &state)?;
            println!("training schedule complete; final checkpoint saved");
            break;
        };
        let batch_tokens = (batch.batch_size * config.sequence_length) as u64;
        let inputs = ids_tensor::<TrainingBackend>(
            batch.inputs,
            batch.batch_size,
            config.sequence_length,
            &device,
        );
        let targets = ids_tensor::<TrainingBackend>(
            batch.targets,
            batch.batch_size,
            config.sequence_length,
            &device,
        )
        .reshape([batch.batch_size * config.sequence_length]);
        let logits = model
            .forward(ModelInput::TokenIds(inputs), None, Default::default())?
            .logits
            .expect("default BDH forward requests logits")
            .reshape([batch.batch_size * config.sequence_length, vocabulary]);
        let loss = criterion.forward(logits, targets);
        let loss_value = loss.clone().to_data().to_vec::<f32>()?[0];
        if !loss_value.is_finite() {
            checkpoint(&config.run_dir, &model, &optimizer, &state)?;
            return Err(format!("non-finite loss {loss_value}; emergency checkpoint saved").into());
        }
        let scaled_loss = loss / config.optimizer.gradient_accumulation as f64;
        let gradients = GradientsParams::from_grads(scaled_loss.backward(), &model);
        accumulator.accumulate(&model, gradients);
        accumulated += 1;
        accumulated_loss += loss_value;
        state.tokens_seen += batch_tokens;
        state.examples_seen += batch.batch_size as u64;
        interval_tokens += batch_tokens;

        if accumulated < config.optimizer.gradient_accumulation {
            continue;
        }

        let learning_rate =
            learning_rate(&config, state.tokens_seen, loader.schedule.effective_tokens);
        model = optimizer.step(learning_rate, model, accumulator.grads());
        state.optimizer_step += 1;
        state.block_index = loader.block_index;
        state.sequence_in_block = loader.sequence_in_block;
        let mean_loss = accumulated_loss / accumulated as f32;
        accumulated = 0;
        accumulated_loss = 0.0;

        if state.optimizer_step.is_multiple_of(config.log_every_steps) {
            let interval_seconds = interval_start.elapsed().as_secs_f64().max(1e-6);
            let event = LogEvent {
                event: "train",
                step: state.optimizer_step,
                tokens_seen: state.tokens_seen,
                phase: batch.phase,
                loss: mean_loss,
                learning_rate,
                tokens_per_second: interval_tokens as f64 / interval_seconds,
                elapsed_seconds: training_start.elapsed().as_secs_f64(),
            };
            append_json_line(&log_path, &event)?;
            println!(
                "step {:>7} | {:?} | tokens {:>10} | loss {:.5} | lr {:.3e} | {:.0} tok/s",
                event.step,
                event.phase,
                event.tokens_seen,
                event.loss,
                event.learning_rate,
                event.tokens_per_second
            );
            interval_tokens = 0;
            interval_start = Instant::now();
        }

        if state
            .optimizer_step
            .is_multiple_of(config.validation_every_steps)
        {
            let validation = validate(&model, &mut loader, &config, vocabulary, &device)?;
            state.best_validation_loss = Some(
                state
                    .best_validation_loss
                    .map_or(validation, |best| best.min(validation)),
            );
            println!(
                "validation at step {}: loss {:.5} (best {:.5})",
                state.optimizer_step,
                validation,
                state.best_validation_loss.expect("just assigned")
            );
        }

        let stop_requested = config.run_dir.join("STOP").is_file();
        if state
            .optimizer_step
            .is_multiple_of(config.checkpoint_every_steps)
            || stop_requested
        {
            checkpoint(&config.run_dir, &model, &optimizer, &state)?;
            println!("checkpoint saved at step {}", state.optimizer_step);
        }
        if stop_requested {
            println!(
                "{} detected; stopped cleanly (remove it before resume)",
                config.run_dir.join("STOP").display()
            );
            break;
        }
    }
    Ok(())
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
            return Err("checkpoint sequence cursor exceeds its work block".into());
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
        let Some(block) = self.schedule.blocks.get(self.block_index).cloned() else {
            return Ok(None);
        };
        let batch_size = requested.min(block.sequences - self.sequence_in_block);
        let start = block.token_start + (self.sequence_in_block * self.sequence_length) as u64;
        let count = batch_size * self.sequence_length + 1;
        let tokens = self
            .corpora
            .get_mut(&block.source)
            .expect("all schedule sources were opened")
            .read_tokens(start, count)?;
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
        self.sequence_in_block += batch_size;
        self.normalize_cursor();
        Ok(Some(TokenBatch {
            inputs,
            targets,
            batch_size,
            phase: block.phase,
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
        let available_sequences =
            (corpus.header.validation_tokens - 1) / self.sequence_length as u64;
        let first = sequence_index % available_sequences;
        let batch_size = requested.min((available_sequences - first) as usize);
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
        })
    }
}

fn validate(
    model: &Bdh<TrainingBackend>,
    loader: &mut TokenLoader,
    config: &PretrainConfig,
    vocabulary: usize,
    device: &WgpuDevice,
) -> Result<f32, Box<dyn std::error::Error>> {
    let inference = model.clone().valid();
    let criterion = CrossEntropyLossConfig::new().init(device);
    let mut total = 0.0_f32;
    for index in 0..config.validation_batches {
        let source = CorpusSource::all()[index % 3];
        let sequence_index = (index / 3 * config.optimizer.micro_batch_size) as u64;
        let batch =
            loader.validation_batch(source, sequence_index, config.optimizer.micro_batch_size)?;
        let inputs = ids_tensor::<InferenceBackend>(
            batch.inputs,
            batch.batch_size,
            config.sequence_length,
            device,
        );
        let targets = ids_tensor::<InferenceBackend>(
            batch.targets,
            batch.batch_size,
            config.sequence_length,
            device,
        )
        .reshape([batch.batch_size * config.sequence_length]);
        let logits = inference
            .forward(ModelInput::TokenIds(inputs), None, Default::default())?
            .logits
            .expect("default BDH forward requests logits")
            .reshape([batch.batch_size * config.sequence_length, vocabulary]);
        total += criterion
            .forward(logits, targets)
            .to_data()
            .to_vec::<f32>()?[0];
    }
    Ok(total / config.validation_batches as f32)
}

fn ids_tensor<B: Backend>(
    ids: Vec<i64>,
    batch: usize,
    sequence: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    Tensor::from_data(TensorData::new(ids, [batch, sequence]), device)
}

fn learning_rate(config: &PretrainConfig, tokens_seen: u64, total_tokens: u64) -> f64 {
    if tokens_seen < config.optimizer.warmup_tokens {
        return config.optimizer.max_learning_rate * tokens_seen as f64
            / config.optimizer.warmup_tokens.max(1) as f64;
    }
    let decay_tokens = total_tokens
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

fn checkpoint<O>(
    run_dir: &Path,
    model: &Bdh<TrainingBackend>,
    optimizer: &O,
    state: &RunState,
) -> Result<(), Box<dyn std::error::Error>>
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
    let pointer = LatestPointer {
        checkpoint_dir: name,
        optimizer_step: state.optimizer_step,
    };
    write_json_atomically(checkpoints.join("latest.json"), &pointer)?;
    prune_checkpoints(&checkpoints, 2)?;
    Ok(())
}

fn latest_checkpoint(
    run_dir: &Path,
) -> Result<Option<(PathBuf, RunState)>, Box<dyn std::error::Error>> {
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
    if state.format_version != 1 {
        return Err(format!(
            "unsupported checkpoint state version {}",
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

fn packed_corpora_fingerprint(
    config: &PretrainConfig,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut combined = Sha256::new();
    for source in CorpusSource::all() {
        let path = token_file(config, source);
        let digest = sha256_file(&path)?;
        combined.update(source.as_str().as_bytes());
        combined.update(digest);
    }
    let fingerprint: [u8; 32] = combined.finalize().into();
    let fingerprint = hex_digest(&fingerprint);
    println!("packed corpora fingerprint: {fingerprint}");
    Ok(fingerprint)
}

fn freeze_config(
    run_dir: &Path,
    source: &Path,
    expected_sha256: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let destination = run_dir.join("config.json");
    if destination.exists() {
        let actual = hex_digest(&sha256_file(&destination)?);
        if actual != expected_sha256 {
            return Err(format!(
                "{} differs from requested config; choose another run_dir",
                destination.display()
            )
            .into());
        }
    } else {
        fs::copy(source, destination)?;
    }
    Ok(())
}

fn prune_checkpoints(checkpoints: &Path, keep: usize) -> Result<(), std::io::Error> {
    let mut directories = fs::read_dir(checkpoints)?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("step-"))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    directories.sort();
    let remove_count = directories.len().saturating_sub(keep);
    for directory in directories.into_iter().take(remove_count) {
        fs::remove_dir_all(directory)?;
    }
    Ok(())
}

fn append_json_line(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn write_json(path: PathBuf, value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

fn write_json_atomically(
    path: PathBuf,
    value: &impl Serialize,
) -> Result<(), Box<dyn std::error::Error>> {
    let temporary = path.with_extension("json.partial");
    write_json(temporary.clone(), value)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut config_path = PathBuf::from("configs/rx6700.json");
    let mut max_steps = None;
    let mut device_index = 0;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--config" => config_path = args.next().ok_or("--config needs a path")?.into(),
            "--max-steps" => {
                max_steps = Some(
                    args.next()
                        .ok_or("--max-steps needs a number")?
                        .parse()
                        .map_err(|error| format!("invalid --max-steps: {error}"))?,
                )
            }
            "--device" => {
                device_index = args
                    .next()
                    .ok_or("--device needs an index")?
                    .parse()
                    .map_err(|error| format!("invalid --device: {error}"))?
            }
            "-h" | "--help" => {
                println!("train_llm [--config PATH] [--device INDEX] [--max-steps N]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}; use --help")),
        }
    }
    Ok(Arguments {
        config_path,
        max_steps,
        device_index,
    })
}
