//! Interleaved token ingestion, latent reasoning, supervision, and generation.
//!
//! The wrapper is where the paper's two state variables become visibly
//! different:
//!
//! - contextual CQ state `M` is [`Memory::fast_weights`](crate::model::Memory),
//!   accumulated while demonstrations and query tokens are ingested;
//! - workspace `H` is the final position of [`Memory::embeds`](crate::model::Memory),
//!   repeatedly transformed without decoding an intermediate token.
//! - experimental v2 wide state `S` is a separate `[B,H,1,Q]` latent-only
//!   workspace carried inside one `Think(R)` chain and never stored in `Memory`.
//!
//! A stage list such as `Tokens(prompt), Think(8), Tokens(answer)` therefore
//! means “update context, run eight continuous recurrent passes, then
//! teacher-force the answer.”

use burn::{
    module::{Initializer, Module, Param},
    nn::loss::CrossEntropyLossConfig,
    tensor::{DType, Int, Tensor, TensorData, backend::Backend},
};
use rand::RngExt;

use crate::{
    error::BdhError,
    model::{Bdh, BdhForwardOptions, LatentWorkspace, Memory, ModelInput},
};

/// One segment in an interleaved reasoning program.
#[derive(Clone, Debug)]
pub enum Stage<B: Backend> {
    /// Ingest discrete `[B,N]` token ids in parallel.
    Tokens(Tensor<B, 2, Int>),
    /// Ingest an already-continuous `[B,N,D]` segment.
    Embeddings(Tensor<B, 3>),
    /// Reapply the shared model to a one-position continuous workspace.
    Think(usize),
}

impl<B: Backend> Stage<B> {
    fn is_think(&self) -> bool {
        matches!(self, Self::Think(_))
    }
}

/// Construction options for [`ReasoningWrapper`].
#[derive(Clone, Debug, Default)]
pub struct ReasoningWrapperConfig {
    /// Add a learned vector at the start of every latent iteration.
    pub latent_step_embedding: bool,
    /// Optional non-negative label token excluded from cross-entropy.
    /// `None` corresponds to upstream's default `ignore_index = -1`, because
    /// normal token ids cannot contain `-1`.
    pub ignore_token: Option<usize>,
}

impl ReasoningWrapperConfig {
    /// Default wrapper configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable the learned latent-step marker.
    pub fn with_latent_step_embedding(mut self, enabled: bool) -> Self {
        self.latent_step_embedding = enabled;
        self
    }

    /// Choose an optional label token ignored by the training loss.
    pub fn with_ignore_token(mut self, token: Option<usize>) -> Self {
        self.ignore_token = token;
        self
    }

    /// Wrap an initialized BDH model.
    pub fn init<B: Backend>(self, bdh: Bdh<B>, device: &B::Device) -> ReasoningWrapper<B> {
        ReasoningWrapper {
            latent_step_embedding: self
                .latent_step_embedding
                .then(|| Initializer::Zeros.init([bdh.dim()], device)),
            bdh,
            ignore_token: self.ignore_token,
        }
    }
}

/// Controls an interleaved forward program.
#[derive(Clone, Debug)]
pub struct ReasoningForwardOptions {
    /// Compute the latent-plus-final-segment training objective.
    pub compute_loss: bool,
    /// Default memory-write policy for token/embedding stages.
    pub update_memory: bool,
    /// Default memory-write policy for `Think` stages.
    pub update_latent_memory: bool,
    /// Optional override with exactly one flag per stage.
    pub update_memory_per_stage: Option<Vec<bool>>,
    /// Optional vocabulary class weights for cross-entropy.
    pub class_weights: Option<Vec<f32>>,
}

impl Default for ReasoningForwardOptions {
    fn default() -> Self {
        Self {
            compute_loss: false,
            update_memory: true,
            update_latent_memory: true,
            update_memory_per_stage: None,
            class_weights: None,
        }
    }
}

/// Result of an interleaved wrapper call.
#[derive(Clone, Debug)]
pub struct ReasoningOutput<B: Backend> {
    /// Logits from the last token/embedding stage, if one occurred.
    pub logits: Option<Tensor<B, 3>>,
    /// Memory after every requested stage.
    pub memory: Memory<B>,
    /// Scalar-shaped `[1]` cross-entropy, when requested.
    pub loss: Option<Tensor<B, 1>>,
}

/// Generation controls matching the upstream wrapper.
#[derive(Clone, Debug)]
pub struct GenerateOptions {
    /// Maximum tokens to sample. `None` is valid only with `stop_token`.
    pub max_new_tokens: Option<usize>,
    /// Stop immediately after sampling this token (the stop token is returned).
    pub stop_token: Option<usize>,
    /// `0` selects greedy decoding; positive values sample from scaled logits.
    pub temperature: f64,
    /// Upstream's top-k-by-threshold control. `0.9` retains roughly the top 10%.
    pub filter_threshold: f64,
    /// Default memory-write policy for caller-supplied token stages.
    pub update_memory: bool,
    /// Default memory-write policy for caller-supplied latent stages.
    pub update_latent_memory: bool,
    /// Optional one-flag-per-stage override for caller-supplied stages.
    pub update_memory_per_stage: Option<Vec<bool>>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: Some(32),
            stop_token: None,
            temperature: 1.0,
            filter_threshold: 0.9,
            update_memory: true,
            update_latent_memory: true,
            update_memory_per_stage: None,
        }
    }
}

/// Adds the recurrent-latent protocol to a [`Bdh`] model.
#[derive(Module, Debug)]
pub struct ReasoningWrapper<B: Backend> {
    bdh: Bdh<B>,
    latent_step_embedding: Option<Param<Tensor<B, 1>>>,
    ignore_token: Option<usize>,
}

impl<B: Backend> ReasoningWrapper<B> {
    /// Access the underlying shared-depth model.
    pub fn model(&self) -> &Bdh<B> {
        &self.bdh
    }

    /// Execute an arbitrary ingest/think interleaving.
    ///
    /// When `compute_loss` is true, every latent iteration predicts the first
    /// token of the following discrete segment, and every position in the final
    /// segment predicts its successor.  This is exactly the public Python
    /// wrapper's latent supervision—not an objective disclosed by the paper.
    pub fn forward(
        &self,
        stages: &[Stage<B>],
        mut memory: Option<Memory<B>>,
        options: ReasoningForwardOptions,
    ) -> Result<ReasoningOutput<B>, BdhError> {
        if let Some(per_stage) = &options.update_memory_per_stage
            && per_stage.len() != stages.len()
        {
            return Err(BdhError::InvalidStages(format!(
                "update_memory_per_stage has {} flags for {} stages",
                per_stage.len(),
                stages.len()
            )));
        }

        let total_reasoning_iterations = stages
            .iter()
            .map(|stage| match stage {
                Stage::Think(iterations) => *iterations,
                _ => 0,
            })
            .sum();

        let mut logits = None;
        let mut last_tokens = None;
        let mut latent_logits = Vec::new();
        let mut latent_labels = Vec::new();
        let mut num_labeled_latents = 0;
        let mut attention_history = None;

        for (stage_index, stage) in stages.iter().enumerate() {
            let stage_override = options
                .update_memory_per_stage
                .as_ref()
                .map(|flags| flags[stage_index]);

            match stage {
                Stage::Think(iterations) => {
                    let current = memory.as_ref().ok_or_else(|| {
                        BdhError::InvalidStages(
                            "tokens must be ingested before latent reasoning".into(),
                        )
                    })?;
                    let [batch, sequence, dim] = current.embeds.dims();
                    let mut latent =
                        current
                            .embeds
                            .clone()
                            .slice([0..batch, sequence - 1..sequence, 0..dim]);
                    if attention_history.is_none() {
                        attention_history = Some(vec![latent.clone()]);
                    }

                    let update = stage_override.unwrap_or(options.update_latent_memory);
                    // This workspace belongs to this Think chain only. It is
                    // carried through all outer iterations with its autograd
                    // graph intact, then dropped before the next token stage
                    // (or a later independent Think stage). It never enters
                    // Memory.fast_weights and therefore cannot pollute CQ.
                    let mut wide_workspace = LatentWorkspace::new();
                    for _ in 0..*iterations {
                        if let Some(step_embedding) = &self.latent_step_embedding {
                            latent = latent + step_embedding.val().reshape([1, 1, self.bdh.dim()]);
                        }
                        let (output, next_workspace) = self.bdh.forward_latent(
                            latent,
                            memory,
                            wide_workspace,
                            BdhForwardOptions {
                                update_memory: update,
                                return_logits: false,
                                collect_per_pass_hiddens: false,
                                attention_history,
                                total_reasoning_iterations,
                                valid_sequence_length: None,
                                document_starts: None,
                                memory_read_scale: 1.0,
                            },
                        )?;
                        wide_workspace = next_workspace;
                        attention_history = output.attention_history;
                        memory = Some(output.memory);
                        latent = memory.as_ref().unwrap().embeds.clone();

                        if options.compute_loss {
                            latent_logits.push(self.bdh.project_logits(latent.clone()));
                        }
                    }
                }
                Stage::Tokens(ids) => {
                    if options.compute_loss {
                        let newly_unlabeled = latent_logits.len() - num_labeled_latents;
                        if newly_unlabeled > 0 {
                            let [batch, _] = ids.dims();
                            latent_labels.push(
                                ids.clone()
                                    .slice([0..batch, 0..1])
                                    .repeat_dim(1, newly_unlabeled),
                            );
                        }
                        num_labeled_latents = latent_logits.len();
                        last_tokens = Some(ids.clone());
                    }

                    let output = self.bdh.forward(
                        ModelInput::TokenIds(ids.clone()),
                        memory,
                        BdhForwardOptions {
                            update_memory: stage_override.unwrap_or(options.update_memory),
                            return_logits: true,
                            collect_per_pass_hiddens: false,
                            attention_history: None,
                            total_reasoning_iterations: 0,
                            valid_sequence_length: None,
                            document_starts: None,
                            memory_read_scale: 1.0,
                        },
                    )?;
                    logits = output.logits;
                    memory = Some(output.memory);
                }
                Stage::Embeddings(embeddings) => {
                    if options.compute_loss {
                        return Err(BdhError::InvalidStages(
                            "loss requires discrete labels, but an embedding stage was supplied"
                                .into(),
                        ));
                    }
                    let output = self.bdh.forward(
                        ModelInput::Embeddings(embeddings.clone()),
                        memory,
                        BdhForwardOptions {
                            update_memory: stage_override.unwrap_or(options.update_memory),
                            return_logits: true,
                            collect_per_pass_hiddens: false,
                            attention_history: None,
                            total_reasoning_iterations: 0,
                            valid_sequence_length: None,
                            document_starts: None,
                            memory_read_scale: 1.0,
                        },
                    )?;
                    logits = output.logits;
                    memory = Some(output.memory);
                }
            }
        }

        let memory = memory.ok_or_else(|| {
            BdhError::InvalidStages("no initial memory and no input stages were provided".into())
        })?;
        let loss = if options.compute_loss {
            if stages.last().is_some_and(Stage::is_think) {
                return Err(BdhError::InvalidStages(
                    "latent reasoning cannot be the final stage when computing loss".into(),
                ));
            }
            let final_logits = logits.as_ref().ok_or_else(|| {
                BdhError::InvalidStages(
                    "a discrete answer stage must follow latent reasoning".into(),
                )
            })?;
            let labels = last_tokens.expect("validated discrete final stage");
            Some(self.training_loss(
                final_logits.clone(),
                labels,
                latent_logits,
                latent_labels,
                options.class_weights,
            )?)
        } else {
            None
        };

        Ok(ReasoningOutput {
            logits,
            memory,
            loss,
        })
    }

    fn training_loss(
        &self,
        final_logits: Tensor<B, 3>,
        final_tokens: Tensor<B, 2, Int>,
        latent_logits: Vec<Tensor<B, 3>>,
        latent_labels: Vec<Tensor<B, 2, Int>>,
        class_weights: Option<Vec<f32>>,
    ) -> Result<Tensor<B, 1>, BdhError> {
        let [batch, answer_len, vocabulary] = final_logits.dims();
        let mut all_logits =
            final_logits.slice([0..batch, 0..answer_len.saturating_sub(1), 0..vocabulary]);
        let mut labels = final_tokens.slice([0..batch, 1..answer_len]);

        if !latent_logits.is_empty() {
            if latent_labels.is_empty() {
                return Err(BdhError::InvalidStages(
                    "latent predictions have no following token segment labels".into(),
                ));
            }
            let latent_logits = Tensor::cat(latent_logits, 1);
            let latent_labels = Tensor::cat(latent_labels, 1);
            all_logits = Tensor::cat(vec![latent_logits, all_logits], 1);
            labels = Tensor::cat(vec![latent_labels, labels], 1);
        }

        let positions = all_logits.dims()[1];
        if positions == 0 {
            return Err(BdhError::InvalidStages(
                "training needs either a multi-token answer or at least one latent step".into(),
            ));
        }

        let device = all_logits.device();
        let criterion = CrossEntropyLossConfig::new()
            .with_pad_tokens(self.ignore_token.map(|token| vec![token]))
            .with_weights(class_weights)
            .init(&device);
        Ok(criterion.forward(
            all_logits.reshape([batch * positions, vocabulary]),
            labels.reshape([batch * positions]),
        ))
    }

    /// Run the supplied stages, then decode an answer autoregressively.
    ///
    /// As in upstream, generation is intentionally limited to batch size one:
    /// each sampled scalar token is synchronously read back to the host.
    pub fn generate(
        &self,
        stages: &[Stage<B>],
        memory: Option<Memory<B>>,
        options: GenerateOptions,
    ) -> Result<(Vec<usize>, Memory<B>), BdhError> {
        if options.max_new_tokens.is_none() && options.stop_token.is_none() {
            return Err(BdhError::InvalidGeneration(
                "max_new_tokens or stop_token must provide a stopping condition".into(),
            ));
        }
        if options.temperature < 0.0 {
            return Err(BdhError::InvalidGeneration(
                "temperature cannot be negative".into(),
            ));
        }
        if !(0.0..=1.0).contains(&options.filter_threshold) {
            return Err(BdhError::InvalidGeneration(
                "filter_threshold must lie in [0, 1]".into(),
            ));
        }

        let output = self.forward(
            stages,
            memory,
            ReasoningForwardOptions {
                update_memory: options.update_memory,
                update_latent_memory: options.update_latent_memory,
                update_memory_per_stage: options.update_memory_per_stage,
                ..Default::default()
            },
        )?;
        let mut memory = output.memory;
        if memory.embeds.dims()[0] != 1 {
            return Err(BdhError::InvalidGeneration(
                "generation supports batch size one".into(),
            ));
        }

        let [_, sequence, dim] = memory.embeds.dims();
        let seed = memory
            .embeds
            .clone()
            .slice([0..1, sequence - 1..sequence, 0..dim]);
        let mut logits = self.bdh.project_logits(seed);
        let mut generated = Vec::new();

        while options
            .max_new_tokens
            .is_none_or(|limit| generated.len() < limit)
        {
            let token = sample_last_token(
                logits,
                options.temperature,
                options.filter_threshold,
                self.bdh.num_tokens(),
            );
            generated.push(token);
            if options.stop_token == Some(token) {
                break;
            }

            // Match upstream's exact feedback path: raw embedding lookup,
            // followed by the continuous-input branch (no post-embed norm).
            let device = memory.embeds.device();
            let ids = Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![token as i64], [1, 1]),
                &device,
            );
            let embedding = self.bdh.embed_tokens_raw(ids);
            let output = self.forward(
                &[Stage::Embeddings(embedding)],
                Some(memory),
                ReasoningForwardOptions {
                    update_memory: options.update_memory,
                    update_latent_memory: options.update_latent_memory,
                    ..Default::default()
                },
            )?;
            logits = output.logits.expect("embedding stages produce logits");
            memory = output.memory;
        }

        Ok((generated, memory))
    }
}

/// Greedy or temperature/top-k sampling for a `[1,N,V]` tensor.
fn sample_last_token<B: Backend>(
    logits: Tensor<B, 3>,
    temperature: f64,
    filter_threshold: f64,
    vocabulary: usize,
) -> usize {
    let [_, sequence, _] = logits.dims();
    let values = logits
        .slice([0..1, sequence - 1..sequence, 0..vocabulary])
        .reshape([vocabulary])
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .expect("F32 conversion requested");

    if temperature == 0.0 {
        return values
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .unwrap();
    }

    // Upstream treats exactly 1.0 as “do not filter”, even though thresholds
    // just below 1.0 retain one token. Preserve that endpoint behavior.
    let keep = if filter_threshold == 1.0 {
        vocabulary
    } else {
        ((1.0 - filter_threshold) * vocabulary as f64) as usize
    };
    let keep = keep.max(1).min(vocabulary);
    let mut ranked = values.clone();
    ranked.sort_by(|left, right| right.total_cmp(left));
    let cutoff = ranked[keep - 1];

    let max = values
        .iter()
        .copied()
        .filter(|value| *value >= cutoff)
        .fold(f32::NEG_INFINITY, f32::max);
    let probabilities: Vec<f64> = values
        .iter()
        .map(|value| {
            if *value < cutoff {
                0.0
            } else {
                (((*value - max) as f64) / temperature).exp()
            }
        })
        .collect();
    let total: f64 = probabilities.iter().sum();
    let mut draw = rand::rng().random::<f64>() * total;
    for (index, probability) in probabilities.into_iter().enumerate() {
        draw -= probability;
        if draw <= 0.0 {
            return index;
        }
    }
    vocabulary - 1
}
