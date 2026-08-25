//! Convert the three document streams into exact, reusable `u16` token files.
//!
//! The command deliberately performs no semantic filtering.  It inserts the
//! trained `<|doc|>` token between adapter frames, applies the already-frozen
//! tokenizer and cuts the last encoded document at the exact configured token
//! count.  A temporary file is renamed only after all bytes are flushed.
//!
//! ```console
//! cargo run --release --bin pack_pretraining_data -- --config configs/rx6700.json \
//!   --python /tmp/bdh-cq-tokenizer-venv/bin/python
//! ```

use bdh_cq_llm::pretrain::{
    CorpusSource, PACKED_FORMAT_VERSION, PackedCorpus, PackedHeader, PackedManifest,
    PretrainConfig, hex_digest, manifest_file, sha256_file, token_file,
};
use sha2::{Digest, Sha256};
use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdout, Command, Stdio},
    time::Instant,
};
use tokenizers::Tokenizer;

const STREAM_MAGIC: [u8; 8] = *b"BDHCQDS1";
const STREAM_ERROR: u8 = 254;
const STREAM_END: u8 = 255;
const MAX_FRAME_BYTES: u64 = 1 << 20;
const LARGE_BYTE_BUDGET: u64 = 100_000_000_000;

#[derive(Debug)]
struct Arguments {
    config: PathBuf,
    python: PathBuf,
    smoke_fixture: bool,
    force: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_arguments()?;
    let config = PretrainConfig::from_path(&args.config)?;
    let tokenizer_bytes = fs::read(&config.tokenizer)?;
    let tokenizer_sha256 = sha256_file(&config.tokenizer)?;
    let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|error| format!("cannot load {}: {error}", config.tokenizer.display()))?;
    let vocab_size = tokenizer.get_vocab_size(true);
    if vocab_size > u16::MAX as usize + 1 {
        return Err(format!("vocabulary {vocab_size} does not fit packed u16 ids").into());
    }
    let doc_token = tokenizer
        .token_to_id("<|doc|>")
        .ok_or("tokenizer has no required <|doc|> token")?;

    fs::create_dir_all(&config.packed_dir)?;
    println!(
        "packing exact token budgets with tokenizer {} (sha256 {})",
        config.tokenizer.display(),
        hex_digest(&tokenizer_sha256)
    );
    for source in CorpusSource::all() {
        pack_source(
            &config,
            source,
            &tokenizer,
            doc_token,
            vocab_size,
            tokenizer_sha256,
            &args,
        )?;
    }
    println!(
        "all packed corpora are ready in {}",
        config.packed_dir.display()
    );
    Ok(())
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut config = PathBuf::from("configs/rx6700.json");
    let mut python = PathBuf::from("python3");
    let mut smoke_fixture = false;
    let mut force = false;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--config" => config = args.next().ok_or("--config needs a path")?.into(),
            "--python" => python = args.next().ok_or("--python needs a path")?.into(),
            "--smoke-fixture" => smoke_fixture = true,
            "--force" => force = true,
            "-h" | "--help" => {
                println!(
                    "pack_pretraining_data [--config PATH] [--python PATH] \
                     [--smoke-fixture] [--force]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}; use --help")),
        }
    }
    Ok(Arguments {
        config,
        python,
        smoke_fixture,
        force,
    })
}

#[allow(clippy::too_many_arguments)]
fn pack_source(
    config: &PretrainConfig,
    source: CorpusSource,
    tokenizer: &Tokenizer,
    doc_token: u32,
    vocab_size: usize,
    tokenizer_sha256: [u8; 32],
    args: &Arguments,
) -> Result<(), Box<dyn std::error::Error>> {
    let budget = config.budget(source)?;
    let wanted = budget.train_tokens + budget.validation_tokens;
    let final_path = token_file(config, source);
    let manifest_path = manifest_file(config, source);
    let partial_path = append_suffix(&final_path, ".partial");

    if final_path.exists() && !args.force {
        let existing = PackedCorpus::open(&final_path)?;
        if existing.header.source != source
            || existing.header.vocab_size != vocab_size as u32
            || existing.header.train_tokens != budget.train_tokens
            || existing.header.validation_tokens != budget.validation_tokens
            || existing.header.tokenizer_sha256 != tokenizer_sha256
        {
            return Err(format!(
                "{} exists but does not match the config; inspect it and rerun with --force",
                final_path.display()
            )
            .into());
        }
        if partial_path.exists() {
            fs::remove_file(&partial_path)?;
        }
        println!(
            "[{}] validated existing {}",
            source.as_str(),
            final_path.display()
        );
        return Ok(());
    }

    let mut output = BufWriter::with_capacity(
        8 * 1024 * 1024,
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&partial_path)?,
    );
    let header = PackedHeader {
        vocab_size: vocab_size as u32,
        source,
        train_tokens: budget.train_tokens,
        validation_tokens: budget.validation_tokens,
        tokenizer_sha256,
    };
    output.write_all(&header.to_bytes())?;

    let (mut child, stdout) = start_stream(source, args)?;
    let mut input = BufReader::with_capacity(2 * 1024 * 1024, stdout);
    let mut magic = [0_u8; 8];
    input.read_exact(&mut magic)?;
    if magic != STREAM_MAGIC {
        terminate(&mut child);
        return Err("dataset adapter returned invalid protocol magic".into());
    }

    let start = Instant::now();
    let mut tokens_written = 0_u64;
    let mut documents = 0_u64;
    let mut utf8_bytes = 0_u64;
    let mut next_report = 10_000_000_u64;
    let mut payload_digest = Sha256::new();
    while tokens_written < wanted {
        let mut tag = [0_u8; 1];
        input.read_exact(&mut tag)?;
        let length = read_u64(&mut input)?;
        if tag[0] == STREAM_END {
            terminate(&mut child);
            return Err(format!(
                "{} ended at {tokens_written} tokens; need {wanted}",
                source.as_str()
            )
            .into());
        }
        if length > MAX_FRAME_BYTES {
            terminate(&mut child);
            return Err(format!("adapter frame is too large: {length} bytes").into());
        }
        let mut bytes = vec![0_u8; length as usize];
        input.read_exact(&mut bytes)?;
        if tag[0] == STREAM_ERROR {
            terminate(&mut child);
            return Err(format!("dataset adapter: {}", String::from_utf8_lossy(&bytes)).into());
        }
        if tag[0] != source.code() {
            terminate(&mut child);
            return Err(format!(
                "adapter emitted source {} while packing {}",
                tag[0],
                source.as_str()
            )
            .into());
        }

        let text = std::str::from_utf8(&bytes)?;
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|error| format!("tokenizer failed: {error}"))?;
        let remaining = (wanted - tokens_written) as usize;
        let ids = std::iter::once(doc_token)
            .chain(encoding.get_ids().iter().copied())
            .take(remaining);
        for id in ids {
            let id = u16::try_from(id)
                .map_err(|_| format!("token id {id} does not fit the packed u16 format"))?;
            let bytes = id.to_le_bytes();
            output.write_all(&bytes)?;
            payload_digest.update(bytes);
            tokens_written += 1;
        }
        documents += 1;
        utf8_bytes += length;

        if tokens_written >= next_report {
            println!(
                "[{}] {:>10.2}%  {tokens_written:>12} / {wanted} tokens ({:.0} tok/s)",
                source.as_str(),
                100.0 * tokens_written as f64 / wanted as f64,
                tokens_written as f64 / start.elapsed().as_secs_f64().max(1e-6)
            );
            next_report += 10_000_000;
        }
    }

    // The adapter is intentionally one-way. Closing/killing it after the exact
    // token boundary prevents a remote iterable from prefetching unused data.
    drop(input);
    terminate(&mut child);
    output.flush()?;
    output.get_ref().sync_all()?;
    drop(output);

    if final_path.exists() {
        fs::remove_file(&final_path)?;
    }
    fs::rename(&partial_path, &final_path)?;
    let manifest = PackedManifest {
        format_version: PACKED_FORMAT_VERSION,
        source,
        token_file: final_path.clone(),
        vocab_size: vocab_size as u32,
        train_tokens: budget.train_tokens,
        validation_tokens: budget.validation_tokens,
        documents,
        utf8_bytes,
        tokenizer_sha256: hex_digest(&tokenizer_sha256),
        payload_sha256: hex_digest(&payload_digest.finalize().into()),
        content_policy: match source {
            CorpusSource::Ficbook => "Only parts[*].clean_text bodies; title, description, tags, rating, and chapter titles excluded; no story/content filtering".into(),
            _ => "No rating, tag, profanity, or semantic-content filtering; source adapter fields only".into(),
        },
    };
    write_json_atomically(&manifest_path, &manifest)?;
    println!(
        "[{}] complete: {tokens_written} tokens, {documents} frames, {:.1}s",
        source.as_str(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn start_stream(
    source: CorpusSource,
    args: &Arguments,
) -> Result<(Child, ChildStdout), Box<dyn std::error::Error>> {
    let mut command = Command::new(&args.python);
    command
        .arg("scripts/stream_tokenizer_corpus.py")
        .arg("--sample-bytes")
        .arg(LARGE_BYTE_BUDGET.to_string())
        .arg("--source")
        .arg(source.as_str())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    if args.smoke_fixture {
        command.arg("--smoke-fixture");
    }
    let mut child = command.spawn().map_err(|error| {
        format!(
            "cannot start dataset adapter with {}: {error}",
            args.python.display()
        )
    })?;
    let stdout = child.stdout.take().ok_or("dataset adapter has no stdout")?;
    Ok((child, stdout))
}

fn terminate(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn read_u64(reader: &mut impl Read) -> Result<u64, std::io::Error> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn write_json_atomically<T: serde::Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    let temporary = append_suffix(path, ".partial");
    let file = File::create(&temporary)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    if path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(temporary, path)?;
    Ok(())
}
