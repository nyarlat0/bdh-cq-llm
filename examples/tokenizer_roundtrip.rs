//! Inspect a trained tokenizer on one piece of text.
//!
//! This is intentionally small enough to read before the corpus sampler. It
//! shows the runtime side of the tokenizer contract: load the self-contained
//! JSON, encode without automatic control tokens, inspect IDs/pieces, and
//! decode back to the original string.

use std::{env, error::Error};

use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut args = env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "artifacts/tokenizer.json".to_owned());
    let text = args.collect::<Vec<_>>().join(" ");
    let text = if text.is_empty() {
        "Привет! Это проверка русского byte-level BPE: ёжик 🦔.".to_owned()
    } else {
        text
    };

    let tokenizer = Tokenizer::from_file(&path)?;
    // `false` is important: control tokens are an explicit responsibility of
    // the future sequence packer, not an implicit tokenizer side effect.
    let encoding = tokenizer.encode(text.as_str(), false)?;
    let decoded = tokenizer.decode(encoding.get_ids(), false)?;

    println!("tokenizer: {path}");
    println!("vocabulary: {}", tokenizer.get_vocab_size(true));
    println!("input bytes: {}", text.len());
    println!("token count: {}", encoding.len());
    println!("ids: {:?}", encoding.get_ids());
    println!("pieces: {:?}", encoding.get_tokens());
    println!("decoded: {decoded:?}");
    println!("lossless: {}", decoded == text);

    Ok(())
}
