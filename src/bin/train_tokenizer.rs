//! Train the project tokenizer from the planned Russian corpus mixture.
//!
//! Rust owns BPE training and the resulting `tokenizer.json`. A small Python
//! process is used only as a dataset reader: Hugging Face `datasets` handles
//! remote iteration, while PyArrow projects the local nested Parquet columns.
//! Documents cross the process boundary as length-prefixed UTF-8, so neither
//! newlines nor arbitrary prose need escaping.

use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs::{self, File},
    io::{self, BufReader, Read},
    path::PathBuf,
    process::{Command, Stdio},
};

use bdh_cq_llm::{SPECIAL_TOKENS, TokenizerTrainingConfig, train_byte_level_bpe};
use serde::Serialize;

type AnyError = Box<dyn Error + Send + Sync>;

const STREAM_MAGIC: &[u8; 8] = b"BDHCQDS1";
const STREAM_ERROR: u8 = 254;
const STREAM_END: u8 = 255;
const SOURCE_NAMES: [&str; 3] = ["fineweb2_hq", "ficbook", "ru_classic"];
const PRETRAINING_TOKEN_BUDGETS: [u64; 3] = [650_000_000, 300_000_000, 50_000_000];
const MAX_FRAME_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Debug)]
struct Args {
    output: PathBuf,
    manifest: PathBuf,
    sample_bytes: u64,
    vocab_size: usize,
    min_frequency: u64,
    max_token_length: usize,
    seed: u64,
    python: String,
    sampler_script: PathBuf,
    fineweb_dataset: String,
    fineweb_config: String,
    fineweb_revision: String,
    ficbook_glob: String,
    ficbook_part_field: String,
    classic_file: PathBuf,
    shuffle_buffer: usize,
    show_progress: bool,
    smoke_fixture: bool,
}

impl Default for Args {
    fn default() -> Self {
        let output = PathBuf::from("artifacts/tokenizer.json");
        Self {
            manifest: PathBuf::from("artifacts/tokenizer.manifest.json"),
            output,
            // This is a raw UTF-8 byte sample, not the 1B-token model corpus.
            // One decimal GB is large enough for stable BPE statistics while
            // avoiding a second copy of the complete pretraining data.
            sample_bytes: 1_000_000_000,
            vocab_size: 32_768,
            min_frequency: 2,
            max_token_length: 64,
            seed: 42,
            python: "python3".to_owned(),
            sampler_script: PathBuf::from("scripts/stream_tokenizer_corpus.py"),
            fineweb_dataset: "epfml/FineWeb2-HQ".to_owned(),
            fineweb_config: "rus_Cyrl".to_owned(),
            // Resolved from `main` on 2026-08-25. Pinning prevents an upstream
            // data update from silently changing a supposedly seeded build.
            fineweb_revision: "c0c06e94fd3a44ae9e802b2b0fc533817601eb5e".to_owned(),
            ficbook_glob: "datasets/ficbook/*.parquet".to_owned(),
            ficbook_part_field: "clean_text".to_owned(),
            classic_file: PathBuf::from("datasets/ru-classic.txt"),
            shuffle_buffer: 10_000,
            show_progress: true,
            smoke_fixture: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct SourceCounts {
    sequences: u64,
    utf8_bytes: u64,
}

/// Iterator over the private framing protocol emitted by the Python sampler.
///
/// Iterator errors cannot be returned through `tokenizers::Tokenizer::train`,
/// whose iterator item is a string rather than a `Result`. We retain the first
/// error and expose it after training, making truncation a hard failure rather
/// than silently accepting a partial corpus.
struct FramedCorpus<R> {
    reader: R,
    counts: [SourceCounts; 3],
    error: Option<io::Error>,
    finished: bool,
}

impl<R: Read> FramedCorpus<R> {
    fn new(mut reader: R) -> io::Result<Self> {
        let mut magic = [0_u8; STREAM_MAGIC.len()];
        reader.read_exact(&mut magic)?;
        if &magic != STREAM_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sampler stream has an unknown protocol header",
            ));
        }
        Ok(Self {
            reader,
            counts: [SourceCounts::default(); 3],
            error: None,
            finished: false,
        })
    }

    fn error(&self) -> Option<&io::Error> {
        self.error.as_ref()
    }
}

impl<R: Read + Send> Iterator for FramedCorpus<R> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let mut source = [0_u8; 1];
        match self.reader.read(&mut source) {
            Ok(0) => {
                return self.fail(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "sampler stream ended without an explicit terminal frame",
                ));
            }
            Ok(1) => {}
            Ok(_) => unreachable!("a one-byte buffer cannot read more than one byte"),
            Err(error) => return self.fail(error),
        }

        let mut length = [0_u8; 8];
        if let Err(error) = self.reader.read_exact(&mut length) {
            return self.fail(error);
        }
        let length = u64::from_le_bytes(length);
        if length > MAX_FRAME_BYTES {
            return self.fail(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("sampler frame is implausibly large: {length} bytes"),
            ));
        }

        let mut bytes = vec![0_u8; length as usize];
        if let Err(error) = self.reader.read_exact(&mut bytes) {
            return self.fail(error);
        }
        let payload = match String::from_utf8(bytes) {
            Ok(document) => document,
            Err(error) => {
                return self.fail(io::Error::new(io::ErrorKind::InvalidData, error));
            }
        };

        if source[0] == STREAM_END {
            if length != 0 {
                return self.fail(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "sampler end frame must have an empty payload",
                ));
            }
            self.finished = true;
            return None;
        }
        if source[0] == STREAM_ERROR {
            return self.fail(io::Error::other(format!(
                "dataset sampler reported: {payload}"
            )));
        }

        let source = usize::from(source[0]);
        if source >= SOURCE_NAMES.len() {
            return self.fail(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("sampler emitted invalid source id {source}"),
            ));
        }

        self.counts[source].sequences += 1;
        self.counts[source].utf8_bytes += length;
        Some(payload)
    }
}

impl<R> FramedCorpus<R> {
    fn fail<T>(&mut self, error: io::Error) -> Option<T> {
        self.error = Some(error);
        self.finished = true;
        None
    }
}

#[derive(Serialize)]
struct Manifest<'a> {
    format_version: u32,
    tokenizer_file: String,
    algorithm: &'static str,
    vocabulary_size_requested: usize,
    vocabulary_size_actual: usize,
    min_frequency: u64,
    max_token_length: usize,
    normalization: &'static str,
    unknown_token: Option<&'static str>,
    seed: u64,
    tokenizer_sample_budget_utf8_bytes: u64,
    source_plan: Vec<SourceManifest>,
    data_locations: DataLocations,
    special_token_ids: BTreeMap<&'a str, u32>,
    curriculum_note: &'static str,
}

#[derive(Serialize)]
struct SourceManifest {
    name: &'static str,
    target_pretraining_tokens: u64,
    tokenizer_sample_budget_utf8_bytes: u64,
    sampled_sequences: u64,
    sampled_utf8_bytes: u64,
}

#[derive(Serialize)]
struct DataLocations {
    fineweb_dataset: String,
    fineweb_config: String,
    fineweb_revision: String,
    ficbook_glob: String,
    ficbook_part_field: String,
    ficbook_metadata_included: bool,
    classic_file: String,
}

fn main() -> Result<(), AnyError> {
    let args = parse_args()?;
    let sample_budgets = split_sample_budget(args.sample_bytes);

    if let Some(parent) = args
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = args
        .manifest
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }

    eprintln!(
        "Tokenizer sample: {} bytes = FineWeb {} + Ficbook {} + ru-classic {}",
        args.sample_bytes, sample_budgets[0], sample_budgets[1], sample_budgets[2]
    );

    let mut command = Command::new(&args.python);
    command
        .arg(&args.sampler_script)
        .arg("--sample-bytes")
        .arg(args.sample_bytes.to_string())
        .arg("--seed")
        .arg(args.seed.to_string())
        .arg("--fineweb-dataset")
        .arg(&args.fineweb_dataset)
        .arg("--fineweb-config")
        .arg(&args.fineweb_config)
        .arg("--fineweb-revision")
        .arg(&args.fineweb_revision)
        .arg("--ficbook-glob")
        .arg(&args.ficbook_glob)
        .arg("--ficbook-part-field")
        .arg(&args.ficbook_part_field)
        .arg("--classic-file")
        .arg(&args.classic_file)
        .arg("--shuffle-buffer")
        .arg(args.shuffle_buffer.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    if args.smoke_fixture {
        command.arg("--smoke-fixture");
    }

    let mut child = command.spawn().map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "failed to start {:?} {:?}: {error}",
                args.python, args.sampler_script
            ),
        )
    })?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("sampler stdout pipe was not created"))?;
    let mut corpus = match FramedCorpus::new(BufReader::new(stdout)) {
        Ok(corpus) => corpus,
        Err(error) => {
            // `FramedCorpus::new` consumed and dropped stdout on error, so the
            // child cannot remain blocked on a full pipe while we reap it.
            let status = child.wait()?;
            return Err(format!(
                "dataset sampler did not start a valid stream ({status}): {error}"
            )
            .into());
        }
    };

    let training_config = TokenizerTrainingConfig {
        vocab_size: args.vocab_size,
        min_frequency: args.min_frequency,
        max_token_length: args.max_token_length,
        show_progress: args.show_progress,
    };
    let training_result = train_byte_level_bpe(&mut corpus, training_config);
    let stream_error = corpus.error().map(ToString::to_string);
    let counts = corpus.counts;
    drop(corpus); // Close the pipe before waiting if BPE stopped early.
    let status = child.wait()?;

    let tokenizer = training_result?;
    if !status.success() {
        return Err(format!(
            "dataset sampler exited with {status}; install scripts/requirements-tokenizer.txt and inspect the error above"
        )
        .into());
    }
    if let Some(error) = stream_error {
        return Err(format!("invalid or truncated sampler stream: {error}").into());
    }

    validate_tokenizer(&tokenizer)?;
    tokenizer.save(&args.output, true)?;

    let special_token_ids = SPECIAL_TOKENS
        .iter()
        .map(|token| {
            tokenizer
                .token_to_id(token)
                .map(|id| (*token, id))
                .ok_or_else(|| format!("trained tokenizer lost reserved token {token}"))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let source_plan = SOURCE_NAMES
        .iter()
        .enumerate()
        .map(|(index, name)| SourceManifest {
            name,
            target_pretraining_tokens: PRETRAINING_TOKEN_BUDGETS[index],
            tokenizer_sample_budget_utf8_bytes: sample_budgets[index],
            sampled_sequences: counts[index].sequences,
            sampled_utf8_bytes: counts[index].utf8_bytes,
        })
        .collect();
    let manifest = Manifest {
        format_version: 1,
        tokenizer_file: args.output.display().to_string(),
        algorithm: "byte-level BPE (GPT-2 regex pre-tokenization)",
        vocabulary_size_requested: args.vocab_size,
        vocabulary_size_actual: tokenizer.get_vocab_size(true),
        min_frequency: args.min_frequency,
        max_token_length: args.max_token_length,
        normalization: "none; input UTF-8 is preserved byte-for-byte",
        unknown_token: None,
        seed: args.seed,
        tokenizer_sample_budget_utf8_bytes: args.sample_bytes,
        source_plan,
        data_locations: DataLocations {
            fineweb_dataset: args.fineweb_dataset.clone(),
            fineweb_config: args.fineweb_config.clone(),
            fineweb_revision: args.fineweb_revision.clone(),
            ficbook_glob: args.ficbook_glob.clone(),
            ficbook_part_field: args.ficbook_part_field.clone(),
            ficbook_metadata_included: false,
            classic_file: args.classic_file.display().to_string(),
        },
        special_token_ids,
        curriculum_note: "Ficbook's 50M first-pass + 250M second-pass split belongs to model-data scheduling; the tokenizer sees the aggregate 300M (30%) domain weight.",
    };
    serde_json::to_writer_pretty(File::create(&args.manifest)?, &manifest)?;

    eprintln!("Saved tokenizer: {}", args.output.display());
    eprintln!("Saved manifest:  {}", args.manifest.display());
    for (index, source) in SOURCE_NAMES.iter().enumerate() {
        eprintln!(
            "  {source:12}: {} sequences, {} UTF-8 bytes",
            counts[index].sequences, counts[index].utf8_bytes
        );
    }
    Ok(())
}

fn validate_tokenizer(tokenizer: &tokenizers::Tokenizer) -> Result<(), AnyError> {
    for (expected, token) in SPECIAL_TOKENS.iter().enumerate() {
        let actual = tokenizer.token_to_id(token);
        if actual != Some(expected as u32) {
            return Err(
                format!("special token {token} must have ID {expected}, got {actual:?}").into(),
            );
        }
    }

    let probe = "  Проверка ё/е, переносов\nи emoji 🦔 без потерь.\t";
    let ids = tokenizer.encode(probe, false)?.get_ids().to_vec();
    let decoded = tokenizer.decode(&ids, false)?;
    if decoded != probe {
        return Err("trained tokenizer failed its lossless UTF-8 round-trip".into());
    }
    Ok(())
}

fn split_sample_budget(total: u64) -> [u64; 3] {
    let fineweb = total.saturating_mul(65) / 100;
    let ficbook = total.saturating_mul(30) / 100;
    [fineweb, ficbook, total - fineweb - ficbook]
}

fn parse_args() -> Result<Args, AnyError> {
    let mut parsed = Args::default();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || -> Result<String, AnyError> {
            args.next()
                .ok_or_else(|| format!("{flag} requires a value").into())
        };
        match flag.as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--output" => parsed.output = PathBuf::from(value()?),
            "--manifest" => parsed.manifest = PathBuf::from(value()?),
            "--sample-bytes" => parsed.sample_bytes = parse_size(&value()?)?,
            "--vocab-size" => parsed.vocab_size = value()?.parse()?,
            "--min-frequency" => parsed.min_frequency = value()?.parse()?,
            "--max-token-length" => parsed.max_token_length = value()?.parse()?,
            "--seed" => parsed.seed = value()?.parse()?,
            "--python" => parsed.python = value()?,
            "--sampler-script" => parsed.sampler_script = PathBuf::from(value()?),
            "--fineweb-dataset" => parsed.fineweb_dataset = value()?,
            "--fineweb-config" => parsed.fineweb_config = value()?,
            "--fineweb-revision" => parsed.fineweb_revision = value()?,
            "--ficbook-glob" => parsed.ficbook_glob = value()?,
            "--ficbook-part-field" => parsed.ficbook_part_field = value()?,
            "--classic-file" => parsed.classic_file = PathBuf::from(value()?),
            "--shuffle-buffer" => parsed.shuffle_buffer = value()?.parse()?,
            "--no-progress" => parsed.show_progress = false,
            "--smoke-fixture" => parsed.smoke_fixture = true,
            unknown => return Err(format!("unknown option {unknown}; use --help").into()),
        }
    }

    if parsed.sample_bytes == 0 {
        return Err("--sample-bytes must be greater than zero".into());
    }
    if parsed.shuffle_buffer == 0 {
        return Err("--shuffle-buffer must be greater than zero".into());
    }
    if !matches!(parsed.ficbook_part_field.as_str(), "clean_text" | "text") {
        return Err("--ficbook-part-field must be clean_text or text".into());
    }
    Ok(parsed)
}

fn parse_size(value: &str) -> Result<u64, AnyError> {
    let compact = value.replace('_', "");
    let uppercase = compact.to_ascii_uppercase();
    let suffixes = [
        ("GIB", 1_073_741_824_u64),
        ("MIB", 1_048_576),
        ("KIB", 1_024),
        ("GB", 1_000_000_000),
        ("MB", 1_000_000),
        ("KB", 1_000),
        ("G", 1_000_000_000),
        ("M", 1_000_000),
        ("K", 1_000),
    ];
    for (suffix, multiplier) in suffixes {
        if let Some(number) = uppercase.strip_suffix(suffix) {
            return Ok(number
                .parse::<u64>()?
                .checked_mul(multiplier)
                .ok_or("--sample-bytes is too large to represent")?);
        }
    }
    Ok(uppercase.parse()?)
}

fn print_help() {
    println!(
        r#"Train the BDH-CQ Russian byte-level BPE tokenizer.

Usage:
  cargo run --release --bin train_tokenizer -- [OPTIONS]

Important options:
  --output PATH                 tokenizer.json (default: artifacts/tokenizer.json)
  --manifest PATH               reproducibility report next to the tokenizer
  --sample-bytes N              stratified UTF-8 sample; supports MB/GB/MiB/GiB
                                (default: 1GB, divided 65% / 30% / 5%)
  --vocab-size N                vocabulary including 8 special tokens (32768)
  --seed N                      deterministic stream/file shuffle seed (42)
  --ficbook-part-field FIELD    clean_text (default) or raw text
  --no-progress                 disable the BPE progress display
  --smoke-fixture               use built-in documents; no network/dependencies

Data-source options:
  --fineweb-dataset ID          default: epfml/FineWeb2-HQ
  --fineweb-config NAME         default: rus_Cyrl
  --fineweb-revision REV        Hub git revision; pin a commit for reproducibility
  --ficbook-glob GLOB           default: datasets/ficbook/*.parquet
  --classic-file PATH           default: datasets/ru-classic.txt
  --shuffle-buffer N            Hugging Face streaming shuffle buffer (10000)
  --python PATH                 Python interpreter (python3)
  --sampler-script PATH         dataset bridge script
"#
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn stream(frames: &[(u8, &str)]) -> Vec<u8> {
        let mut bytes = STREAM_MAGIC.to_vec();
        for (source, text) in frames {
            bytes.push(*source);
            bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
            bytes.extend_from_slice(text.as_bytes());
        }
        bytes.push(STREAM_END);
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes
    }

    #[test]
    fn framing_preserves_multiline_utf8_and_counts_sources() {
        let input = stream(&[(0, "Fine\nWeb"), (1, "Фикбук 🦔"), (2, "Классика")]);
        let mut corpus = FramedCorpus::new(Cursor::new(input)).unwrap();
        assert_eq!(corpus.by_ref().collect::<Vec<_>>().len(), 3);
        assert!(corpus.error().is_none());
        assert_eq!(corpus.counts[1].utf8_bytes, "Фикбук 🦔".len() as u64);
    }

    #[test]
    fn sampler_error_frame_is_not_silent_eof() {
        let mut input = STREAM_MAGIC.to_vec();
        input.push(STREAM_ERROR);
        input.extend_from_slice(&4_u64.to_le_bytes());
        input.extend_from_slice(b"boom");
        let mut corpus = FramedCorpus::new(Cursor::new(input)).unwrap();
        assert_eq!(corpus.next(), None);
        assert!(corpus.error().unwrap().to_string().contains("boom"));
    }

    #[test]
    fn budget_split_has_no_rounding_loss() {
        assert_eq!(
            split_sample_budget(1_000_000_000),
            [650_000_000, 300_000_000, 50_000_000]
        );
        assert_eq!(split_sample_budget(7).iter().sum::<u64>(), 7);
    }

    #[test]
    fn human_sizes_are_unambiguous() {
        assert_eq!(parse_size("1GB").unwrap(), 1_000_000_000);
        assert_eq!(parse_size("1_GiB").unwrap(), 1_073_741_824);
        assert_eq!(parse_size("250M").unwrap(), 250_000_000);
    }
}
