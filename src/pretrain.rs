//! Reproducible data and run configuration for full language-model pretraining.
//!
//! Raw dataset adapters emit UTF-8 documents.  The packing step tokenizes them
//! once and stores token ids as little-endian `u16`: the fixed 32,768-token
//! vocabulary fits exactly, and training never has to repeat expensive BPE.
//! A small binary header and a JSON manifest make accidental tokenizer/data
//! mismatches fail before the first GPU allocation.

use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

/// Magic at the start of every packed token file.
pub const PACKED_MAGIC: [u8; 8] = *b"BDHCQT01";
/// Current packed-corpus format version.
pub const PACKED_FORMAT_VERSION: u32 = 1;
/// Byte length of [`PackedHeader`]'s stable on-disk representation.
pub const PACKED_HEADER_BYTES: u64 = 72;

/// One of the three data sources in the 1B-token curriculum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusSource {
    /// Russian split of `epfml/FineWeb2-HQ`.
    Fineweb2Hq,
    /// Local `IlyaGusev/ficbook` Parquet files.
    Ficbook,
    /// Local concatenation of `Imperius/ru-classic`.
    RuClassic,
}

impl CorpusSource {
    /// Stable file/config name for this source.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fineweb2Hq => "fineweb2_hq",
            Self::Ficbook => "ficbook",
            Self::RuClassic => "ru_classic",
        }
    }

    /// Stable one-byte code stored in packed headers and stream frames.
    pub const fn code(self) -> u8 {
        match self {
            Self::Fineweb2Hq => 0,
            Self::Ficbook => 1,
            Self::RuClassic => 2,
        }
    }

    /// Convert a stable source code into an enum value.
    pub fn from_code(code: u8) -> Result<Self, String> {
        match code {
            0 => Ok(Self::Fineweb2Hq),
            1 => Ok(Self::Ficbook),
            2 => Ok(Self::RuClassic),
            _ => Err(format!("unknown packed corpus source code {code}")),
        }
    }

    /// Sources in their documented curriculum order.
    pub const fn all() -> [Self; 3] {
        [Self::Fineweb2Hq, Self::Ficbook, Self::RuClassic]
    }
}

/// Requested training and held-out token counts for one source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceBudget {
    /// Tokens made available to the training schedule.
    pub train_tokens: u64,
    /// Tokens kept after the training range for loss evaluation only.
    pub validation_tokens: u64,
}

/// BDH dimensions selected for a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Communication/value width `D`.
    pub dim: usize,
    /// Number of applications of the shared recurrent block.
    pub depth: usize,
    /// Number of independent positive-feature heads.
    pub heads: usize,
    /// Total positive Q/K feature width across all heads.
    pub dim_qk_heads: usize,
}

/// AdamW and batch-scheduling settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Number of independent sequences passed to one forward call.
    pub micro_batch_size: usize,
    /// Micro-batches summed before one optimizer update.
    pub gradient_accumulation: usize,
    /// Peak learning rate after warmup.
    pub max_learning_rate: f64,
    /// Final cosine-decay learning rate.
    pub min_learning_rate: f64,
    /// Number of processed tokens used for linear warmup.
    pub warmup_tokens: u64,
    /// Adam first-moment decay.
    pub beta_1: f32,
    /// Adam second-moment decay.
    pub beta_2: f32,
    /// Decoupled AdamW weight decay.
    pub weight_decay: f32,
    /// Global gradient norm limit.
    pub gradient_clip_norm: f32,
}

/// Complete, serializable contract shared by the packer and trainer.
/// Stateful contextual-memory curriculum used after the language warm-up.
///
/// A zero `stateful_after_tokens` enables memory from the beginning.  The
/// default deliberately disables it so historical configs retain their exact
/// memoryless behavior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryTrainingConfig {
    /// Global token count at which adjacent chunks start sharing CQ memory.
    pub stateful_after_tokens: u64,
    /// Number of 256-token chunks kept in one truncated-BPTT graph.
    pub chunks_per_detach: usize,
    /// Reset memory whenever the packed `<|doc|>` marker starts a new document.
    pub reset_on_document: bool,
    /// Reset at shuffled work-block boundaries, which are not text-contiguous.
    pub reset_on_work_block: bool,
}

impl Default for MemoryTrainingConfig {
    fn default() -> Self {
        Self {
            stateful_after_tokens: u64::MAX,
            chunks_per_detach: 1,
            reset_on_document: true,
            reset_on_work_block: true,
        }
    }
}

impl MemoryTrainingConfig {
    /// Whether the current global token cursor has entered the CQ stage.
    pub fn is_stateful(&self, tokens_seen: u64) -> bool {
        tokens_seen >= self.stateful_after_tokens
    }
}

/// Complete, serializable contract shared by the packer and trainer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PretrainConfig {
    /// Configuration schema version; currently `1`.
    pub format_version: u32,
    /// Path to the trained Hugging Face tokenizer JSON.
    pub tokenizer: PathBuf,
    /// Directory containing packed token files and manifests.
    pub packed_dir: PathBuf,
    /// Directory containing checkpoints, logs and the frozen config.
    pub run_dir: PathBuf,
    /// Random seed for source shuffling and parameter initialization.
    pub seed: u64,
    /// Tokens in each next-token training example.
    pub sequence_length: usize,
    /// Adjacent examples kept together as one shuffled I/O block.
    pub block_sequences: usize,
    /// Exact per-source token budgets, keyed by stable source name.
    pub sources: BTreeMap<String, SourceBudget>,
    /// Ficbook tokens mixed into phase one; the remainder forms phase two.
    pub ficbook_phase_one_tokens: u64,
    /// Architecture dimensions.
    pub model: ModelConfig,
    /// Optimizer and batching settings.
    pub optimizer: OptimizerConfig,
    /// Optional persistent-memory curriculum. Missing means memoryless training.
    #[serde(default)]
    pub memory: MemoryTrainingConfig,
    /// Save `latest` model/optimizer/state after this many updates.
    pub checkpoint_every_steps: u64,
    /// Evaluate held-out source ranges after this many updates.
    pub validation_every_steps: u64,
    /// Number of validation micro-batches per evaluation.
    pub validation_batches: usize,
    /// Print and append a JSONL event after this many updates.
    pub log_every_steps: u64,
}

impl PretrainConfig {
    /// Parse and validate a JSON run configuration.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .map_err(|error| format!("cannot read config {}: {error}", path.display()))?;
        let config: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("invalid config {}: {error}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    /// Return one required source budget.
    pub fn budget(&self, source: CorpusSource) -> Result<&SourceBudget, String> {
        self.sources
            .get(source.as_str())
            .ok_or_else(|| format!("configuration is missing source {}", source.as_str()))
    }

    /// Reject inconsistent dimensions, budgets and schedules before work starts.
    pub fn validate(&self) -> Result<(), String> {
        if self.format_version != 1 {
            return Err(format!(
                "unsupported config format {}, expected 1",
                self.format_version
            ));
        }
        if self.sequence_length == 0
            || self.block_sequences == 0
            || self.optimizer.micro_batch_size == 0
            || self.optimizer.gradient_accumulation == 0
            || self.checkpoint_every_steps == 0
            || self.validation_every_steps == 0
            || self.validation_batches == 0
            || self.log_every_steps == 0
        {
            return Err("sequence, batch, validation and interval sizes must be non-zero".into());
        }
        if !self
            .block_sequences
            .is_multiple_of(self.optimizer.micro_batch_size)
        {
            return Err("block_sequences must be divisible by micro_batch_size".into());
        }
        if self.memory.chunks_per_detach == 0 {
            return Err("memory.chunks_per_detach must be non-zero".into());
        }
        if self.memory.stateful_after_tokens != u64::MAX && self.optimizer.micro_batch_size != 1 {
            return Err("stateful CQ training currently requires micro_batch_size = 1".into());
        }
        if self.memory.stateful_after_tokens != u64::MAX
            && !self
                .optimizer
                .gradient_accumulation
                .is_multiple_of(self.memory.chunks_per_detach)
        {
            return Err(
                "gradient_accumulation must be divisible by memory.chunks_per_detach".into(),
            );
        }
        if self.model.heads == 0 || !self.model.dim_qk_heads.is_multiple_of(self.model.heads) {
            return Err("dim_qk_heads must be divisible by a non-zero head count".into());
        }
        for source in CorpusSource::all() {
            let budget = self.budget(source)?;
            if budget.train_tokens <= self.sequence_length as u64
                || budget.validation_tokens <= self.sequence_length as u64
            {
                return Err(format!(
                    "{} train/validation ranges must each exceed sequence_length",
                    source.as_str()
                ));
            }
        }
        let ficbook = self.budget(CorpusSource::Ficbook)?;
        if self.ficbook_phase_one_tokens == 0
            || self.ficbook_phase_one_tokens >= ficbook.train_tokens
        {
            return Err("ficbook_phase_one_tokens must split the Ficbook training range".into());
        }
        if !(0.0 < self.optimizer.min_learning_rate
            && self.optimizer.min_learning_rate <= self.optimizer.max_learning_rate)
        {
            return Err("learning rates must satisfy 0 < min <= max".into());
        }
        Ok(())
    }

    /// Total requested training-token budget.
    pub fn total_train_tokens(&self) -> Result<u64, String> {
        CorpusSource::all()
            .into_iter()
            .try_fold(0_u64, |sum, source| {
                self.budget(source).map(|budget| sum + budget.train_tokens)
            })
    }

    /// Check that a checkpoint may continue under a new run contract.
    ///
    /// Continuation may change only the output directory and the newly added
    /// memory policy. Model, optimizer, corpus and deterministic schedule stay
    /// byte-for-byte compatible at the saved cursor.
    pub fn continuation_compatible_with(&self, previous: &Self) -> bool {
        self.format_version == previous.format_version
            && self.tokenizer == previous.tokenizer
            && self.packed_dir == previous.packed_dir
            && self.seed == previous.seed
            && self.sequence_length == previous.sequence_length
            && self.block_sequences == previous.block_sequences
            && self.sources == previous.sources
            && self.ficbook_phase_one_tokens == previous.ficbook_phase_one_tokens
            && self.model == previous.model
            && self.optimizer == previous.optimizer
            && self.checkpoint_every_steps == previous.checkpoint_every_steps
            && self.validation_every_steps == previous.validation_every_steps
            && self.validation_batches == previous.validation_batches
            && self.log_every_steps == previous.log_every_steps
    }
}

/// Fixed metadata serialized at the start of each `.tokens` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedHeader {
    /// Vocabulary size used while packing.
    pub vocab_size: u32,
    /// Dataset represented by the payload.
    pub source: CorpusSource,
    /// Number of leading payload tokens assigned to training.
    pub train_tokens: u64,
    /// Number of trailing payload tokens assigned to validation.
    pub validation_tokens: u64,
    /// SHA-256 of the exact tokenizer JSON.
    pub tokenizer_sha256: [u8; 32],
}

impl PackedHeader {
    /// Serialize the stable 72-byte little-endian representation.
    pub fn to_bytes(&self) -> [u8; PACKED_HEADER_BYTES as usize] {
        let mut bytes = [0_u8; PACKED_HEADER_BYTES as usize];
        bytes[0..8].copy_from_slice(&PACKED_MAGIC);
        bytes[8..12].copy_from_slice(&PACKED_FORMAT_VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.vocab_size.to_le_bytes());
        bytes[16] = self.source.code();
        bytes[24..32].copy_from_slice(&self.train_tokens.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.validation_tokens.to_le_bytes());
        bytes[40..72].copy_from_slice(&self.tokenizer_sha256);
        bytes
    }

    /// Parse and validate the stable on-disk representation.
    pub fn from_bytes(bytes: [u8; PACKED_HEADER_BYTES as usize]) -> Result<Self, String> {
        if bytes[0..8] != PACKED_MAGIC {
            return Err("invalid packed-token magic".into());
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().expect("fixed slice"));
        if version != PACKED_FORMAT_VERSION {
            return Err(format!("unsupported packed-token version {version}"));
        }
        Ok(Self {
            vocab_size: u32::from_le_bytes(bytes[12..16].try_into().expect("fixed slice")),
            source: CorpusSource::from_code(bytes[16])?,
            train_tokens: u64::from_le_bytes(bytes[24..32].try_into().expect("fixed slice")),
            validation_tokens: u64::from_le_bytes(bytes[32..40].try_into().expect("fixed slice")),
            tokenizer_sha256: bytes[40..72].try_into().expect("fixed slice"),
        })
    }
}

/// Human-readable provenance accompanying one packed file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedManifest {
    /// Packed format version.
    pub format_version: u32,
    /// Dataset represented by the packed file.
    pub source: CorpusSource,
    /// Path of the packed file, relative to the working directory.
    pub token_file: PathBuf,
    /// Vocabulary size observed in the tokenizer.
    pub vocab_size: u32,
    /// Exact number of training tokens written.
    pub train_tokens: u64,
    /// Exact number of validation tokens written.
    pub validation_tokens: u64,
    /// Number of source documents consumed, including the final one.
    pub documents: u64,
    /// UTF-8 bytes consumed before tokenization.
    pub utf8_bytes: u64,
    /// Hex SHA-256 of the tokenizer JSON.
    pub tokenizer_sha256: String,
    /// Hex SHA-256 of the packed token payload (header excluded).
    pub payload_sha256: String,
    /// Explicit statement about transformations applied by the adapter.
    pub content_policy: String,
}

/// One validated packed corpus with random-access sequence reads.
#[derive(Debug)]
pub struct PackedCorpus {
    file: File,
    path: PathBuf,
    /// Header parsed from the token file.
    pub header: PackedHeader,
}

impl PackedCorpus {
    /// Open a packed file and check that its byte size matches its header.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)
            .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
        let mut header_bytes = [0_u8; PACKED_HEADER_BYTES as usize];
        file.read_exact(&mut header_bytes)
            .map_err(|error| format!("cannot read {} header: {error}", path.display()))?;
        let header = PackedHeader::from_bytes(header_bytes)?;
        let expected = PACKED_HEADER_BYTES + 2 * (header.train_tokens + header.validation_tokens);
        let actual = file
            .metadata()
            .map_err(|error| format!("cannot stat {}: {error}", path.display()))?
            .len();
        if actual != expected {
            return Err(format!(
                "{} has {actual} bytes, header requires {expected}",
                path.display()
            ));
        }
        Ok(Self { file, path, header })
    }

    /// Read a contiguous token range, decoding little-endian `u16` ids.
    pub fn read_tokens(&mut self, token_offset: u64, count: usize) -> Result<Vec<u16>, String> {
        let end = token_offset
            .checked_add(count as u64)
            .ok_or_else(|| "token range overflow".to_string())?;
        let total = self.header.train_tokens + self.header.validation_tokens;
        if end > total {
            return Err(format!(
                "read {token_offset}..{end} exceeds {} tokens in {}",
                total,
                self.path.display()
            ));
        }
        self.file
            .seek(SeekFrom::Start(PACKED_HEADER_BYTES + token_offset * 2))
            .map_err(|error| format!("cannot seek {}: {error}", self.path.display()))?;
        let mut bytes = vec![0_u8; count * 2];
        self.file
            .read_exact(&mut bytes)
            .map_err(|error| format!("cannot read {}: {error}", self.path.display()))?;
        Ok(bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect())
    }
}

/// Curriculum phase attached to each shuffled I/O block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurriculumPhase {
    /// 650M FineWeb + 50M Ficbook + 50M classics, shuffled by blocks.
    General,
    /// Remaining 250M Ficbook tokens for style/domain specialization.
    FicbookFocus,
}

/// A contiguous group of examples read together for I/O locality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkBlock {
    /// Curriculum stage to which this block belongs.
    pub phase: CurriculumPhase,
    /// Packed corpus holding the block.
    pub source: CorpusSource,
    /// Offset of the first token in the packed file.
    pub token_start: u64,
    /// Number of non-overlapping next-token examples in the block.
    pub sequences: usize,
}

/// Deterministic block-shuffled schedule reconstructed during resume.
#[derive(Debug, Clone)]
pub struct TrainingSchedule {
    /// Phase-one blocks followed by phase-two blocks.
    pub blocks: Vec<WorkBlock>,
    /// Number of target tokens actually used after dropping range remainders.
    pub effective_tokens: u64,
    /// Index of the first phase-two block.
    pub phase_two_block: usize,
}

impl TrainingSchedule {
    /// Build the exact two-phase schedule described by [`PretrainConfig`].
    pub fn build(config: &PretrainConfig) -> Result<Self, String> {
        let mut phase_one = Vec::new();
        let mut phase_two = Vec::new();
        add_range_blocks(
            &mut phase_one,
            config,
            CurriculumPhase::General,
            CorpusSource::Fineweb2Hq,
            0,
            config.budget(CorpusSource::Fineweb2Hq)?.train_tokens,
        );
        add_range_blocks(
            &mut phase_one,
            config,
            CurriculumPhase::General,
            CorpusSource::Ficbook,
            0,
            config.ficbook_phase_one_tokens,
        );
        add_range_blocks(
            &mut phase_one,
            config,
            CurriculumPhase::General,
            CorpusSource::RuClassic,
            0,
            config.budget(CorpusSource::RuClassic)?.train_tokens,
        );
        add_range_blocks(
            &mut phase_two,
            config,
            CurriculumPhase::FicbookFocus,
            CorpusSource::Ficbook,
            config.ficbook_phase_one_tokens,
            config.budget(CorpusSource::Ficbook)?.train_tokens,
        );

        // Every block contains whole micro-batches.  Trim at most
        // `gradient_accumulation - 1` final micro-batches so the schedule ends
        // exactly on an optimizer boundary and resume can represent its cursor
        // without serializing an in-flight gradient accumulator.
        let batch = config.optimizer.micro_batch_size;
        let total_micro_batches = phase_one
            .iter()
            .chain(&phase_two)
            .map(|block| block.sequences / batch)
            .sum::<usize>();
        let excess_batches = total_micro_batches % config.optimizer.gradient_accumulation;
        trim_tail_sequences(&mut phase_two, excess_batches * batch);

        phase_one.shuffle(&mut StdRng::seed_from_u64(config.seed));
        phase_two.shuffle(&mut StdRng::seed_from_u64(config.seed ^ 0xf1cb_00c5));
        let phase_two_block = phase_one.len();
        phase_one.extend(phase_two);
        let effective_tokens = phase_one
            .iter()
            .map(|block| block.sequences as u64 * config.sequence_length as u64)
            .sum();
        Ok(Self {
            blocks: phase_one,
            effective_tokens,
            phase_two_block,
        })
    }
}

fn add_range_blocks(
    output: &mut Vec<WorkBlock>,
    config: &PretrainConfig,
    phase: CurriculumPhase,
    source: CorpusSource,
    start: u64,
    end: u64,
) {
    let sequence = config.sequence_length as u64;
    let total_sequences = end.saturating_sub(start).saturating_sub(1) / sequence;
    let micro_batch = config.optimizer.micro_batch_size as u64;
    let total_sequences = total_sequences - total_sequences % micro_batch;
    let block_sequences = config.block_sequences as u64;
    let mut first_sequence = 0;
    while first_sequence < total_sequences {
        let sequences = (total_sequences - first_sequence).min(block_sequences) as usize;
        output.push(WorkBlock {
            phase,
            source,
            token_start: start + first_sequence * sequence,
            sequences,
        });
        first_sequence += sequences as u64;
    }
}

fn trim_tail_sequences(blocks: &mut Vec<WorkBlock>, mut sequences: usize) {
    while sequences > 0 {
        let last = blocks
            .last_mut()
            .expect("phase two is larger than one accumulation remainder");
        if last.sequences > sequences {
            last.sequences -= sequences;
            break;
        }
        sequences -= last.sequences;
        blocks.pop();
    }
}

/// SHA-256 digest of a file, used to bind tokenizer, packed data and run state.
pub fn sha256_file(path: impl AsRef<Path>) -> Result<[u8; 32], String> {
    let path = path.as_ref();
    let mut file = File::open(path)
        .map_err(|error| format!("cannot open {} for hashing: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("cannot hash {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest.finalize().into())
}

/// Lowercase hexadecimal encoding used in JSON manifests.
pub fn hex_digest(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Canonical path of one packed token file.
pub fn token_file(config: &PretrainConfig, source: CorpusSource) -> PathBuf {
    config
        .packed_dir
        .join(format!("{}.tokens", source.as_str()))
}

/// Canonical path of one packed provenance manifest.
pub fn manifest_file(config: &PretrainConfig, source: CorpusSource) -> PathBuf {
    config
        .packed_dir
        .join(format!("{}.manifest.json", source.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_header_round_trip() {
        let header = PackedHeader {
            vocab_size: 32_768,
            source: CorpusSource::Ficbook,
            train_tokens: 300_000_000,
            validation_tokens: 1_000_000,
            tokenizer_sha256: [0x5a; 32],
        };
        assert_eq!(PackedHeader::from_bytes(header.to_bytes()).unwrap(), header);
    }

    #[test]
    fn real_config_has_requested_billion_tokens() {
        let config = PretrainConfig::from_path("configs/rx6700.json").unwrap();
        assert_eq!(config.total_train_tokens().unwrap(), 1_000_000_000);
        assert_eq!(
            config.ficbook_phase_one_tokens, 50_000_000,
            "the remaining 250M tokens form phase two"
        );
        let schedule = TrainingSchedule::build(&config).unwrap();
        assert!(schedule.phase_two_block > 0);
        assert!(schedule.effective_tokens <= 1_000_000_000);
        assert!(schedule.effective_tokens > 999_000_000);
        let effective_sequences = schedule
            .blocks
            .iter()
            .map(|block| block.sequences)
            .sum::<usize>();
        let micro_batches = effective_sequences / config.optimizer.micro_batch_size;
        assert!(micro_batches.is_multiple_of(config.optimizer.gradient_accumulation));
    }
}
