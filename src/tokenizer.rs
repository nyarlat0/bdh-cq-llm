//! Byte-level BPE tokenizer used by the Russian pretraining pipeline.
//!
//! The model corpus is mostly Russian but is not Russian-only: FineWeb contains
//! URLs, code and foreign fragments, while Ficbook contains names, emoji and
//! unconventional punctuation. A byte-level alphabet makes every UTF-8 input
//! representable without an `<unk>` token. BPE then learns frequent multi-byte
//! sequences such as Russian morphemes and whole common words.
//!
//! This module deliberately applies **no Unicode normalization**. In
//! particular, `е` and `ё`, case, combining marks, whitespace and punctuation
//! remain distinguishable. That makes encoding and decoding lossless and does
//! not silently rewrite the source corpus.

use std::collections::HashSet;

use tokenizers::{
    AddedToken, Tokenizer,
    models::{
        TrainerWrapper,
        bpe::{BPE, BpeTrainerBuilder},
    },
    pre_tokenizers::byte_level::ByteLevel,
};

/// Default vocabulary for the first, relatively small BDH-CQ language model.
///
/// 32,768 is large enough to compress Cyrillic prose well while keeping the
/// embedding table and output projection affordable. Revisit this number only
/// after measuring token fertility on held-out samples; a larger vocabulary is
/// not automatically better for a small model.
pub const DEFAULT_VOCAB_SIZE: usize = 32_768;

/// Structural tokens and their intended IDs.
///
/// `tokenizers` inserts special tokens before the byte alphabet, in this exact
/// order. Tests guard the IDs because checkpoints and packed datasets will
/// depend on them. There is intentionally no `<unk>`: all UTF-8 bytes are in
/// the initial alphabet.
pub const SPECIAL_TOKENS: [&str; 8] = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|doc|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|eot|>",
];

/// Hyperparameters that affect the learned vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizerTrainingConfig {
    /// Total vocabulary size, including byte symbols and [`SPECIAL_TOKENS`].
    pub vocab_size: usize,
    /// A candidate pair must occur this many times before it can become a merge.
    pub min_frequency: u64,
    /// Maximum byte-level token length learned by BPE.
    ///
    /// This prevents a repeated identifier or URL from consuming a vocabulary
    /// slot with an enormous, brittle token. It does not truncate input text.
    pub max_token_length: usize,
    /// Whether Hugging Face Tokenizers should render preprocessing progress.
    pub show_progress: bool,
}

impl Default for TokenizerTrainingConfig {
    fn default() -> Self {
        Self {
            vocab_size: DEFAULT_VOCAB_SIZE,
            min_frequency: 2,
            max_token_length: 64,
            show_progress: true,
        }
    }
}

/// Train a lossless byte-level BPE tokenizer from a document iterator.
///
/// Documents are boundaries for parallel preprocessing, not semantic special
/// tokens: this function does not inject `<|doc|>` into training text. The data
/// packer should add token ID 3 between documents after encoding.
///
/// # Why the iterator matters
///
/// `Tokenizer::train` consumes documents incrementally, so callers do not
/// need to materialize a sampled corpus in RAM. The `train_tokenizer` binary
/// connects this iterator to a length-prefixed Python stream that reads remote
/// FineWeb and local Parquet files.
///
/// # Errors
///
/// Returns an error for an impossible vocabulary configuration or if BPE
/// preprocessing/training fails.
pub fn train_byte_level_bpe<I, S>(
    documents: I,
    config: TokenizerTrainingConfig,
) -> tokenizers::Result<Tokenizer>
where
    I: Iterator<Item = S> + Send,
    S: AsRef<str> + Send,
{
    // Eight reserved IDs plus one symbol for each possible byte are required
    // for lossless byte-level encoding. Merges need at least one further slot.
    let minimum_vocab = SPECIAL_TOKENS.len() + 256;
    if config.vocab_size <= minimum_vocab {
        return Err(format!(
            "vocab_size must exceed {minimum_vocab} (special tokens + byte alphabet)"
        )
        .into());
    }
    if config.max_token_length == 0 {
        return Err("max_token_length must be greater than zero".into());
    }

    let byte_level = ByteLevel::new(false, false, true);
    let alphabet: HashSet<char> = ByteLevel::alphabet().into_iter().collect();
    let special_tokens = SPECIAL_TOKENS
        .iter()
        .map(|token| AddedToken::from((*token).to_owned(), true))
        .collect();

    let bpe = BPE::builder()
        // Unknown fallback is unnecessary because `alphabet` covers all bytes.
        .cache_capacity(100_000)
        .build()?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer
        // No normalizer is installed: byte-for-byte text identity is retained.
        .with_pre_tokenizer(Some(byte_level))
        .with_decoder(Some(byte_level));

    let trainer = BpeTrainerBuilder::new()
        .vocab_size(config.vocab_size)
        .min_frequency(config.min_frequency)
        .max_token_length(Some(config.max_token_length))
        .show_progress(config.show_progress)
        .special_tokens(special_tokens)
        .initial_alphabet(alphabet)
        .build();
    let mut trainer: TrainerWrapper = trainer.into();
    tokenizer.train(&mut trainer, documents)?;

    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tokenizer() -> Tokenizer {
        // Repetition lets the tiny test learn actual merges instead of testing
        // only the guaranteed byte alphabet.
        let documents = vec![
            "Привет, мир! Ёжик 🦔\n".repeat(8),
            "Это русская проза — с тире, кавычками «ёлочками» и emoji.\n".repeat(8),
            "<html>foreign fragments & code_like_names()</html>\n".repeat(8),
        ];
        train_byte_level_bpe(
            documents.into_iter(),
            TokenizerTrainingConfig {
                vocab_size: 512,
                min_frequency: 1,
                max_token_length: 32,
                show_progress: false,
            },
        )
        .expect("tiny tokenizer must train")
    }

    #[test]
    fn utf8_round_trip_is_exact() {
        let tokenizer = tiny_tokenizer();
        let input = "  Ёжик\nNSFW-лексика, emoji: 🦔; e\u{301} != é\t";
        let encoding = tokenizer.encode(input, false).expect("encoding must work");
        let decoded = tokenizer
            .decode(encoding.get_ids(), false)
            .expect("decoding must work");
        assert_eq!(decoded, input);
    }

    #[test]
    fn reserved_ids_are_stable_and_unknown_is_absent() {
        let tokenizer = tiny_tokenizer();
        for (expected_id, token) in SPECIAL_TOKENS.iter().enumerate() {
            assert_eq!(tokenizer.token_to_id(token), Some(expected_id as u32));
        }
        assert_eq!(tokenizer.token_to_id("<unk>"), None);
    }
}
