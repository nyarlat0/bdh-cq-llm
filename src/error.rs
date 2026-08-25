//! Errors raised by configuration and stage-protocol validation.

use thiserror::Error;

/// Recoverable errors in the public BDH-CQ API.
///
/// Tensor backends can still panic on impossible low-level shape operations.
/// The crate validates the common architectural and protocol mistakes before
/// those operations are reached and reports them through this type.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BdhError {
    /// A model hyperparameter combination is internally inconsistent.
    #[error("invalid BDH configuration: {0}")]
    InvalidConfig(String),

    /// A reasoning-stage sequence violates the ingest/think/answer protocol.
    #[error("invalid reasoning stages: {0}")]
    InvalidStages(String),

    /// A caller passed memory whose batch or layer layout does not fit a model.
    #[error("incompatible recurrent memory: {0}")]
    IncompatibleMemory(String),

    /// Generation has no finite stopping condition.
    #[error("invalid generation options: {0}")]
    InvalidGeneration(String),

    /// ARC grid data is empty, ragged, or outside the ten-color vocabulary.
    #[error("invalid ARC grid: {0}")]
    InvalidGrid(String),
}
