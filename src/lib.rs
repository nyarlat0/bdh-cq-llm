#![warn(missing_docs)]
//! A study-oriented Rust port of the public `lucidrains/bdh-cq` reconstruction.
//!
//! # What this crate is—and is not
//!
//! The 2026 BDH-CQ paper specifies the system-level recurrences
//! `S_t = U_theta(S_(t-1), D_t)` and `H_(r+1) = F_theta(H_r, S_K)`, but says
//! that its exact dimensions and update rules are proprietary. Consequently,
//! no public code can reproduce Pathway's evaluated 150M-parameter system from
//! the paper alone.
//!
//! This crate instead ports Phil Wang's public PyTorch interpretation at
//! upstream commit `720f0c62844af2d14c99750faecfa82f05f23ae1`. It combines
//! the original BDH-GPU ideas—positive high-dimensional Q/K features,
//! low-rank communication, and a fixed-size associative state—with recurrent
//! latent reasoning and optional attention residuals.
//!
//! Start with [`model`] for one BDH pass, then [`reasoning`] for the distinction
//! between contextual memory and the latent workspace. [`icq`] and [`tasks`]
//! show how those mechanisms are used for small ARC-style experiments.

pub mod error;
pub mod icq;
pub mod model;
pub mod reasoning;
mod rope;
pub mod tasks;
pub mod tokenizer;

pub use error::BdhError;
pub use model::{
    AttentionResidual, Bdh, BdhConfig, BdhForwardOptions, BdhOutput, FastWeight, Memory,
    ModelInput, compute_attn_residual_depth_bias,
};
pub use reasoning::{
    GenerateOptions, ReasoningForwardOptions, ReasoningOutput, ReasoningWrapper,
    ReasoningWrapperConfig, Stage,
};
pub use tokenizer::{
    DEFAULT_VOCAB_SIZE, SPECIAL_TOKENS, TokenizerTrainingConfig, train_byte_level_bpe,
};
