//! The tensor-level BDH model and its fixed-size associative memory.
//!
//! This module is the closest Rust counterpart to upstream `bdh_cq.py`.
//! The implementation is deliberately explicit about tensor layouts.  Burn
//! has no Einstein-summation strings in this path, so each contraction appears
//! as a reshape/transpose/matrix multiplication that can be followed with a
//! debugger.

use burn::{
    config::Config,
    module::{Initializer, Module, Param},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{Int, Tensor, activation, backend::Backend},
};

use crate::{error::BdhError, rope::apply_rotary};

/// A single layer's associative fast-weight matrix.
///
/// Layout: `[batch, head, qk_feature, model_feature]`, or `[B, H, Q, D]`.
/// The `None` case is observable when memory writes are disabled before that
/// layer has ever received a token.
pub type FastWeight<B> = Option<Tensor<B, 4>>;

/// Persistent state carried between token chunks and latent-reasoning passes.
///
/// The paper's contextual state `S_t` corresponds primarily to
/// [`fast_weights`](Self::fast_weights).  [`embeds`](Self::embeds) is different:
/// it is the most recent model output and seeds the latent workspace `H_0`.
#[derive(Clone, Debug)]
pub struct Memory<B: Backend> {
    /// Number of sequence positions already processed, including latent steps.
    /// It supplies the offset for rotary positions in the next pass.
    pub tokens_seen: usize,
    /// Normalized output of the most recent pass, shaped `[B, N, D]`.
    pub embeds: Tensor<B, 3>,
    /// One independent `[B, H, Q, D]` state matrix for every recurrent depth.
    pub fast_weights: Vec<FastWeight<B>>,
}

impl<B: Backend> Memory<B> {
    /// Cut autograd history while preserving the numerical contextual state.
    ///
    /// Stateful language-model training carries this value across token
    /// chunks. Detaching periodically implements truncated BPTT: later chunks
    /// can still read every accumulated `K^T V` association, while the
    /// backward graph is bounded independently from the document length.
    pub fn detach(self) -> Self {
        Self {
            tokens_seen: self.tokens_seen,
            embeds: self.embeds.detach(),
            fast_weights: self
                .fast_weights
                .into_iter()
                .map(|weight| weight.map(Tensor::detach))
                .collect(),
        }
    }
}

/// Either discrete token ids or an already-continuous latent workspace.
///
/// This enum makes the Python source's dtype-based branch explicit.  Discrete
/// ids pass through the embedding table and post-embedding normalization;
/// latent embeddings enter the shared BDH block directly.
#[derive(Clone, Debug)]
pub enum ModelInput<B: Backend> {
    /// Integer token ids with shape `[B, N]`.
    TokenIds(Tensor<B, 2, Int>),
    /// Continuous states with shape `[B, N, D]`.
    Embeddings(Tensor<B, 3>),
}

/// Controls one call to [`Bdh::forward`].
///
/// Attention history is owned and returned in [`BdhOutput`].  That ownership
/// is the Rust equivalent of the mutable `all_block_outputs` Python list.
#[derive(Clone, Debug)]
pub struct BdhForwardOptions<B: Backend> {
    /// Whether this pass adds `K^T V` into every fast-weight matrix.
    pub update_memory: bool,
    /// Skip the vocabulary projection during purely latent computation.
    pub return_logits: bool,
    /// Keep each depth's residual output for inspection or recycling.
    pub collect_per_pass_hiddens: bool,
    /// Prior depth/reasoning outputs consumed by the attention residual.
    pub attention_history: Option<Vec<Tensor<B, 3>>>,
    /// Total latent iterations in the current wrapper call; used only by the
    /// optional distance-to-the-end bias.
    pub total_reasoning_iterations: usize,
    /// Number of leading positions that contain real tokens.
    ///
    /// The remaining positions, when any, are shape-stabilizing padding. They
    /// still flow through the fixed-size local computation, but are excluded
    /// from the associative `K^T V` write and from the rotary-position cursor.
    /// This is primarily useful to keep a streaming trainer's GPU shape set
    /// bounded when document boundaries split a physical chunk. The returned
    /// logits and [`Memory::embeds`] retain the physical padded length; callers
    /// must mask padded labels or ignore those trailing positions.
    pub valid_sequence_length: Option<usize>,
}

impl<B: Backend> Default for BdhForwardOptions<B> {
    fn default() -> Self {
        Self {
            update_memory: true,
            return_logits: true,
            collect_per_pass_hiddens: false,
            attention_history: None,
            total_reasoning_iterations: 1,
            valid_sequence_length: None,
        }
    }
}

/// Structured result of one token or latent pass.
#[derive(Clone, Debug)]
pub struct BdhOutput<B: Backend> {
    /// `[B, N, vocabulary]`, or `None` when projection was disabled.
    pub logits: Option<Tensor<B, 3>>,
    /// Updated recurrent state and the pass's final embeddings.
    pub memory: Memory<B>,
    /// Residual states after each recurrent depth, when requested.
    pub per_pass_hiddens: Vec<Tensor<B, 3>>,
    /// Updated cross-depth history when attention residuals are enabled.
    pub attention_history: Option<Vec<Tensor<B, 3>>>,
}

/// Hyperparameters for the public BDH-CQ reconstruction.
///
/// `dim_qk_heads` is the total high-dimensional positive feature space `H*Q`;
/// it is analogous to the original BDH paper's neuron dimension `n`.  The
/// model dimension `dim` is the much smaller communication/value dimension
/// analogous to `d`.
#[derive(Config, Debug)]
pub struct BdhConfig {
    /// Vocabulary size.
    pub num_tokens: usize,
    /// Token/value model dimension `D`.
    pub dim: usize,
    /// Number of recurrent applications of the one shared block.
    #[config(default = 8)]
    pub depth: usize,
    /// Number of independent Q/K feature groups.
    #[config(default = 4)]
    pub heads: usize,
    /// Total Q/K feature count across all heads (`H * Q`).
    #[config(default = 32_768)]
    pub dim_qk_heads: usize,
    /// Number of leading Q/K features per head carrying rotary position.
    ///
    /// Zero preserves the original port's `Q / 2` rule. Wide v2 models set
    /// this explicitly to 64 so most positive neuron features remain purely
    /// semantic rather than being rotated into signed coordinates.
    #[config(default = 0)]
    pub rotary_dim: usize,
    /// Reuse the input embedding matrix for vocabulary projection.
    #[config(default = false)]
    pub tie_embeddings: bool,
    /// Replace identity residual addition with attention across depth/history.
    #[config(default = false)]
    pub attn_residual: bool,
    /// Share one attention-residual pseudo-query across recurrent depths.
    #[config(default = true)]
    pub attn_residual_tied: bool,
    /// Number of learnable cycle-distance bias values; zero disables the bias.
    #[config(default = 0)]
    pub attn_residual_depth_bias_distance: usize,
    /// Carry the full `[B,H,N,Q]` positive workspace across recurrent depth.
    #[config(default = false)]
    pub gated_neuron_state: bool,
    /// Exponentially retain old CQ fast weights before adding the new write.
    #[config(default = false)]
    pub cq_memory_decay: bool,
    /// Initial per-head CQ retention when decay is enabled.
    #[config(default = 0.995)]
    pub cq_memory_retention: f64,
}

impl BdhConfig {
    /// Validate dimensions and initialize a model on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<Bdh<B>, BdhError> {
        self.validate()?;

        let qk_per_head = self.dim_qk_heads / self.heads;
        let rotary_dim = if self.rotary_dim == 0 {
            qk_per_head / 2
        } else {
            self.rotary_dim
        };
        let block = BdhBlock::new(
            self.dim,
            self.heads,
            qk_per_head,
            rotary_dim,
            self.gated_neuron_state,
            device,
        );
        let attention_residual = self.attn_residual.then(|| {
            let pseudo_queries = if self.attn_residual_tied {
                1
            } else {
                self.depth
            };
            AttentionResidual::new(
                self.dim,
                pseudo_queries,
                self.attn_residual_depth_bias_distance,
                device,
            )
        });

        Ok(Bdh {
            token_embed: EmbeddingConfig::new(self.num_tokens, self.dim).init(device),
            block,
            to_logits: (!self.tie_embeddings).then(|| {
                LinearConfig::new(self.dim, self.num_tokens)
                    .with_bias(false)
                    .init(device)
            }),
            attention_residual,
            memory_retention_logit: self.cq_memory_decay.then(|| {
                let probability = self.cq_memory_retention;
                let logit = (probability / (1.0 - probability)).ln();
                Initializer::Constant { value: logit }.init([self.heads], device)
            }),
            dim: self.dim,
            num_tokens: self.num_tokens,
            depth: self.depth,
            heads: self.heads,
            qk_per_head,
            rotary_dim,
            attn_residual_tied: self.attn_residual_tied,
            tie_embeddings: self.tie_embeddings,
        })
    }

    fn validate(&self) -> Result<(), BdhError> {
        if self.num_tokens < 2 {
            return Err(BdhError::InvalidConfig(
                "num_tokens must be at least 2".into(),
            ));
        }
        if self.dim < 2 {
            return Err(BdhError::InvalidConfig(
                "dim must be at least 2 for normalization".into(),
            ));
        }
        if self.depth == 0 {
            return Err(BdhError::InvalidConfig("depth must be non-zero".into()));
        }
        if self.heads == 0 || !self.dim_qk_heads.is_multiple_of(self.heads) {
            return Err(BdhError::InvalidConfig(
                "dim_qk_heads must be divisible by a non-zero heads count".into(),
            ));
        }
        let qk_per_head = self.dim_qk_heads / self.heads;
        if qk_per_head < 4 || !qk_per_head.is_multiple_of(4) {
            return Err(BdhError::InvalidConfig(
                "Q/K features per head must be a multiple of 4 (half is pairwise RoPE)".into(),
            ));
        }
        let rotary_dim = if self.rotary_dim == 0 {
            qk_per_head / 2
        } else {
            self.rotary_dim
        };
        if rotary_dim > qk_per_head || !rotary_dim.is_multiple_of(2) {
            return Err(BdhError::InvalidConfig(format!(
                "rotary_dim must be even and no greater than Q={qk_per_head}, got {rotary_dim}"
            )));
        }
        if !(0.0 < self.cq_memory_retention && self.cq_memory_retention < 1.0) {
            return Err(BdhError::InvalidConfig(
                "cq_memory_retention must be strictly between zero and one".into(),
            ));
        }
        Ok(())
    }
}

/// One set of parameters reused at every model depth and latent iteration.
///
/// Weight tying is central to BDH: `depth` counts recurrent computation, not a
/// vector of separately parameterized Transformer blocks.
#[derive(Module, Debug)]
struct BdhBlock<B: Backend> {
    /// `D -> H*Q`; after ReLU, this is simultaneously Q, K, and the FF gate.
    to_qk: Linear<B>,
    /// Per-head `D -> Q` lift for the attention result.
    proj_up: Param<Tensor<B, 3>>,
    /// `H*Q -> D` low-rank communication back to token/value space.
    proj_out: Linear<B>,
    /// Cheap input-dependent update gate, one scalar per token and head.
    state_update: Option<Linear<B>>,
    /// Per-head strength of the direct neuron-state input; starts at zero.
    state_injection: Option<Param<Tensor<B, 1>>>,
    heads: usize,
    qk_per_head: usize,
    rotary_dim: usize,
}

impl<B: Backend> BdhBlock<B> {
    fn new(
        dim: usize,
        heads: usize,
        qk_per_head: usize,
        rotary_dim: usize,
        gated_neuron_state: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            to_qk: LinearConfig::new(dim, heads * qk_per_head)
                .with_bias(false)
                .init(device),
            proj_up: Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            }
            .init([heads, dim, qk_per_head], device),
            proj_out: LinearConfig::new(heads * qk_per_head, dim)
                .with_bias(false)
                .init(device),
            state_update: gated_neuron_state.then(|| {
                LinearConfig::new(dim, heads)
                    .with_initializer(Initializer::Zeros)
                    .init(device)
            }),
            state_injection: gated_neuron_state.then(|| Initializer::Zeros.init([heads], device)),
            heads,
            qk_per_head,
            rotary_dim,
        }
    }

    /// Execute the public reconstruction's complete linear-attention + gated
    /// low-rank block.
    fn forward(
        &self,
        tokens: Tensor<B, 3>,
        previous_memory: Option<&Tensor<B, 4>>,
        neuron_state: Option<Tensor<B, 4>>,
        tokens_seen: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 4>, Option<Tensor<B, 4>>) {
        let [batch, sequence, dim] = tokens.dims();

        // [B,N,D] -> [B,N,H*Q] -> [B,H,N,Q].  ReLU makes the neuron-like
        // features positive and tends to make them sparse after training.
        let sparse = activation::relu(self.to_qk.forward(tokens.clone()));
        let mut gates = sparse
            .reshape([batch, sequence, self.heads, self.qk_per_head])
            .permute([0, 2, 1, 3]);
        if let (Some(state), Some(injection)) = (&neuron_state, &self.state_injection) {
            let strength = injection.val().reshape([1, self.heads, 1, 1]);
            gates = activation::relu(gates + rms_norm_no_params(state.clone()) * strength);
        }

        // The gate remains unrotated.  Only Q and K receive positional phase.
        let q = apply_rotary(gates.clone(), tokens_seen, self.rotary_dim);
        let k = apply_rotary(gates.clone(), tokens_seen, self.rotary_dim);

        // Current-chunk causal linear attention.  There is deliberately no
        // softmax or 1/sqrt(Q) scaling: Q K^T is an unnormalized affinity.
        // The diagonal is removed, so a position cannot retrieve its own V.
        let similarity = q.clone().matmul(k.clone().transpose()).tril(-1);
        let values_by_head = tokens
            .clone()
            .unsqueeze_dim::<4>(1)
            .repeat_dim(1, self.heads);
        let mut aggregate = similarity.matmul(values_by_head.clone());

        // Previous chunks are compressed into M = sum(K^T V).  Retrieval
        // qM is algebraically the same contraction as attention over every
        // old position, without retaining a growing token cache.
        if let Some(memory) = previous_memory {
            aggregate = aggregate + q.matmul(memory.clone());
        }

        let attention_out = layer_norm_no_params(aggregate);

        // [B,H,N,D] @ [B,H,D,Q] -> [B,H,N,Q].  Multiplication by the original
        // sparse Q/K features is the BDH multiplicative gate.  Since gates are
        // nonnegative, relu(projected * gates) equals gates * relu(projected).
        let projection = self
            .proj_up
            .val()
            .unsqueeze_dim::<4>(0)
            .repeat_dim(0, batch);
        let lifted = activation::relu(attention_out.matmul(projection) * gates);

        // The full positive workspace is a bounded, input-dependent EMA. A
        // single scalar gate is broadcast over Q, avoiding an infeasible
        // `(H*Q)^2` transition while keeping every neuron coordinate alive.
        let next_neuron_state = self.state_update.as_ref().map(|update| {
            // A +logit(0.99) offset makes the first depth almost identical to
            // the old stateless path: with an all-zero previous state the
            // output is 0.99 * `lifted`, not 0.5 * `lifted`. The zero-initial
            // weights can then learn when retaining an older wide state is
            // useful without imposing a large scale discontinuity at step 0.
            let write_gate = activation::sigmoid(update.forward(tokens.clone()) + 4.595_12)
                .permute([0, 2, 1])
                .unsqueeze_dim::<4>(3);
            let previous = neuron_state.unwrap_or_else(|| {
                Tensor::zeros(
                    [batch, self.heads, sequence, self.qk_per_head],
                    &tokens.device(),
                )
            });
            previous * (1.0 - write_gate.clone()) + lifted.clone() * write_gate
        });
        let output_neurons = next_neuron_state.clone().unwrap_or(lifted);

        let block_out = output_neurons.permute([0, 2, 1, 3]).reshape([
            batch,
            sequence,
            self.heads * self.qk_per_head,
        ]);
        let block_out = layer_norm_no_params(self.proj_out.forward(block_out));

        // New write for this chunk: [B,H,Q,N] @ [B,H,N,D] -> [B,H,Q,D].
        // A physically padded tail was zeroed once before entering the shared
        // recurrent block. The bias-free projections and residual path keep
        // those positions exactly zero through every depth, so their K rows
        // contribute zero here without constructing six separate masks.
        let memory_write = k.transpose().matmul(values_by_head);
        debug_assert_eq!(block_out.dims(), [batch, sequence, dim]);

        (block_out, memory_write, next_neuron_state)
    }
}

/// Attention over earlier depth and latent states, replacing `x + block(x)`.
///
/// This is an optional stabilization extension in the public reconstruction,
/// inspired by the separate Attention Residuals paper; it is not specified by
/// the public BDH-CQ paper.  A learned pseudo-query chooses a convex mixture of
/// all saved states independently at every batch/sequence location.
#[derive(Module, Debug)]
pub struct AttentionResidual<B: Backend> {
    query: Param<Tensor<B, 2>>,
    key_norm: RmsNorm<B>,
    depth_bias: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> AttentionResidual<B> {
    fn new(
        dim: usize,
        pseudo_queries: usize,
        depth_bias_distance: usize,
        device: &B::Device,
    ) -> Self {
        let normal = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        Self {
            query: normal.clone().init([pseudo_queries, dim], device),
            // PyTorch's `RMSNorm(dim)` defaults to the machine epsilon of the
            // input dtype. Upstream trains in float32, so use f32 epsilon
            // rather than Burn's otherwise slightly larger 1e-5 default.
            key_norm: RmsNormConfig::new(dim)
                .with_epsilon(f32::EPSILON as f64)
                .init(device),
            depth_bias: (depth_bias_distance > 0)
                .then(|| normal.init([depth_bias_distance], device)),
        }
    }

    fn forward(
        &self,
        expected_shape: [usize; 3],
        keys_values: &[Tensor<B, 3>],
        query_index: usize,
        depth: usize,
        total_reasoning_iterations: usize,
    ) -> Result<Tensor<B, 3>, BdhError> {
        if keys_values.is_empty()
            || keys_values
                .iter()
                .any(|value| value.dims() != expected_shape)
        {
            return Err(BdhError::InvalidStages(format!(
                "attention-residual history entries must all have shape {expected_shape:?}"
            )));
        }

        let [batch, sequence, dim] = expected_shape;
        let layers = keys_values.len();
        let past = Tensor::stack::<4>(keys_values.to_vec(), 0).permute([1, 2, 0, 3]);
        let normalized = self.key_norm.forward(past.clone());
        let query = self
            .query
            .val()
            .slice([query_index..query_index + 1, 0..dim])
            .reshape([1, 1, 1, dim]);
        let mut similarity = (normalized * query).sum_dim(3).squeeze_dim::<3>(3);

        if let Some(schedule) = &self.depth_bias {
            let bias = compute_attn_residual_depth_bias(
                layers,
                schedule.val(),
                depth,
                total_reasoning_iterations,
            )
            .reshape([1, 1, layers]);
            similarity = similarity + bias;
        }

        let weights = activation::softmax(similarity, 2).unsqueeze_dim::<4>(3);
        let readout = (weights * past).sum_dim(2).squeeze_dim::<3>(2);
        debug_assert_eq!(readout.dims(), [batch, sequence, dim]);
        Ok(readout)
    }
}

/// Construct the optional attention-residual distance bias.
///
/// The first key is always the seed workspace and receives zero.  Later keys
/// are grouped by recurrent depth within latent cycles.  Learnable schedule
/// values apply nearest the end of reasoning; earlier cycles are zero-padded.
/// This is public mainly so the indexing rule can be unit-tested in isolation.
pub fn compute_attn_residual_depth_bias<B: Backend>(
    num_keys: usize,
    mut bias_schedule: Tensor<B, 1>,
    depth: usize,
    total_reasoning_iterations: usize,
) -> Tensor<B, 1> {
    assert!(num_keys > 0, "attention history must contain its seed key");
    let device = bias_schedule.device();
    let total_latents = depth * total_reasoning_iterations;
    if total_latents == 0 {
        return Tensor::zeros([num_keys], &device);
    }

    let [mut schedule_len] = bias_schedule.dims();
    if schedule_len > total_reasoning_iterations {
        bias_schedule =
            bias_schedule.slice(schedule_len - total_reasoning_iterations..schedule_len);
        schedule_len = total_reasoning_iterations;
    }

    let mut schedule = bias_schedule
        .reshape([schedule_len, 1])
        .repeat_dim(1, depth)
        .reshape([schedule_len * depth]);
    if schedule_len * depth < total_latents {
        schedule = Tensor::cat(
            vec![
                Tensor::zeros([total_latents - schedule_len * depth], &device),
                schedule,
            ],
            0,
        );
    }

    let num_latents = num_keys - 1;
    let schedule = if num_latents > total_latents {
        let tail = schedule
            .clone()
            .slice(total_latents - 1..total_latents)
            .repeat_dim(0, num_latents - total_latents);
        Tensor::cat(vec![schedule, tail], 0)
    } else {
        schedule.slice(0..num_latents)
    };

    Tensor::cat(vec![Tensor::zeros([1], &device), schedule], 0)
}

/// The public BDH-CQ model.
///
/// Its one `BdhBlock` is shared across `depth`, matching the recurrent-depth
/// design of the original BDH-GPU paper and the upstream Python implementation.
#[derive(Module, Debug)]
pub struct Bdh<B: Backend> {
    token_embed: Embedding<B>,
    block: BdhBlock<B>,
    to_logits: Option<Linear<B>>,
    attention_residual: Option<AttentionResidual<B>>,
    memory_retention_logit: Option<Param<Tensor<B, 1>>>,
    dim: usize,
    num_tokens: usize,
    depth: usize,
    heads: usize,
    qk_per_head: usize,
    rotary_dim: usize,
    attn_residual_tied: bool,
    tie_embeddings: bool,
}

impl<B: Backend> Bdh<B> {
    /// Model/value dimension `D`.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Vocabulary size used by embedding and output projection.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Number of recurrent shared-block applications per pass.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Number of independent positive-feature heads `H`.
    pub fn heads(&self) -> usize {
        self.heads
    }

    /// Positive Q/K features in each head, `Q`.
    pub fn qk_features_per_head(&self) -> usize {
        self.qk_per_head
    }

    /// Total positive Q/K width `H*Q`, analogous to BDH's neuron count `n`.
    pub fn total_qk_features(&self) -> usize {
        self.heads * self.qk_per_head
    }

    /// Number of each head's leading features transformed by RoPE.
    pub fn rotary_features_per_head(&self) -> usize {
        self.rotary_dim
    }

    /// Whether input and output vocabulary projections share one matrix.
    pub fn has_tied_embeddings(&self) -> bool {
        self.tie_embeddings
    }

    /// Whether old CQ matrices use learned exponential retention.
    pub fn has_cq_memory_decay(&self) -> bool {
        self.memory_retention_logit.is_some()
    }

    /// Device on which model parameters live.
    pub fn device(&self) -> B::Device {
        self.token_embed.weight.device()
    }

    /// Project continuous `[B,N,D]` states into vocabulary logits.
    pub fn project_logits(&self, embeddings: Tensor<B, 3>) -> Tensor<B, 3> {
        if let Some(projection) = &self.to_logits {
            projection.forward(embeddings)
        } else {
            let [batch, sequence, dim] = embeddings.dims();
            // Flatten batch/sequence so the shared [V,D] embedding is used by
            // one ordinary GEMM. Repeating [D,V] across B would materialize
            // hundreds of MiB in the production microbatch.
            (embeddings
                .reshape([batch * sequence, dim])
                .matmul(self.token_embed.weight.val().transpose())
                / (dim as f64).sqrt())
            .reshape([batch, sequence, self.num_tokens])
        }
    }

    /// Embed ids and apply the same parameter-free normalization used at the
    /// beginning of a normal token pass.
    pub fn embed_tokens(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        layer_norm_no_params(self.token_embed.forward(ids))
    }

    /// Look up token embeddings without the normal token-input LayerNorm.
    ///
    /// Autoregressive generation in the upstream wrapper feeds a sampled token
    /// back as a *continuous* stage.  Keeping this separate from
    /// [`embed_tokens`](Self::embed_tokens) preserves that small but observable
    /// behavior.
    pub(crate) fn embed_tokens_raw(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.token_embed.forward(ids)
    }

    /// Process a parallel token chunk or a continuous latent state.
    ///
    /// The memory update is the public reconstruction's additive special case
    /// of BDH-CQ paper equation (1): `M_l <- M_l + K_l^T V_l`.
    pub fn forward(
        &self,
        input: ModelInput<B>,
        memory: Option<Memory<B>>,
        options: BdhForwardOptions<B>,
    ) -> Result<BdhOutput<B>, BdhError> {
        let mut tokens = match input {
            ModelInput::TokenIds(ids) => self.embed_tokens(ids),
            ModelInput::Embeddings(embeddings) => {
                if embeddings.dims()[2] != self.dim {
                    return Err(BdhError::InvalidStages(format!(
                        "latent embedding width must be {}, got {}",
                        self.dim,
                        embeddings.dims()[2]
                    )));
                }
                embeddings
            }
        };
        let [batch, sequence, _] = tokens.dims();
        if sequence == 0 {
            return Err(BdhError::InvalidStages(
                "a model pass cannot contain an empty sequence".into(),
            ));
        }
        let valid_sequence_length = options.valid_sequence_length.unwrap_or(sequence);
        if valid_sequence_length == 0 || valid_sequence_length > sequence {
            return Err(BdhError::InvalidStages(format!(
                "valid sequence length must be in 1..={sequence}, got {valid_sequence_length}"
            )));
        }
        if valid_sequence_length < sequence {
            let device = tokens.device();
            let mask = (0..sequence)
                .map(|position| {
                    if position < valid_sequence_length {
                        1.0_f32
                    } else {
                        0.0_f32
                    }
                })
                .collect::<Vec<_>>();
            let mask =
                Tensor::<B, 1>::from_floats(mask.as_slice(), &device).reshape([1, sequence, 1]);
            tokens = tokens * mask;
        }

        let (tokens_seen, previous_weights) = match memory {
            Some(memory) => {
                let [memory_batch, memory_sequence, memory_dim] = memory.embeds.dims();
                if memory_batch != batch {
                    return Err(BdhError::IncompatibleMemory(format!(
                        "batch changed from {} to {batch}",
                        memory_batch
                    )));
                }
                if memory_sequence == 0 || memory_dim != self.dim {
                    return Err(BdhError::IncompatibleMemory(format!(
                        "previous embeddings must have shape [B, nonzero N, {}], got {:?}",
                        self.dim,
                        memory.embeds.dims()
                    )));
                }
                if memory.fast_weights.len() != self.depth {
                    return Err(BdhError::IncompatibleMemory(format!(
                        "expected {} depth states, got {}",
                        self.depth,
                        memory.fast_weights.len()
                    )));
                }
                let expected = [batch, self.heads, self.qk_per_head, self.dim];
                for (depth, state) in memory.fast_weights.iter().enumerate() {
                    if let Some(state) = state
                        && state.dims() != expected
                    {
                        return Err(BdhError::IncompatibleMemory(format!(
                            "depth {depth} fast weights must have shape {expected:?}, got {:?}",
                            state.dims()
                        )));
                    }
                }
                (memory.tokens_seen, memory.fast_weights)
            }
            None => (0, vec![None; self.depth]),
        };

        let mut history = if self.attention_residual.is_some() {
            Some(
                options
                    .attention_history
                    .unwrap_or_else(|| vec![tokens.clone()]),
            )
        } else {
            None
        };
        let mut next_weights = Vec::with_capacity(self.depth);
        // Deliberately local to this call: CQ is the only state crossing token
        // chunks. The wide workspace exists solely across recurrent depth.
        let mut neuron_state = None;
        let mut per_pass_hiddens = Vec::with_capacity(if options.collect_per_pass_hiddens {
            self.depth
        } else {
            0
        });

        for (layer_index, previous) in previous_weights.into_iter().enumerate() {
            let (block_out, memory_write, next_neuron_state) =
                self.block
                    .forward(tokens.clone(), previous.as_ref(), neuron_state, tokens_seen);
            neuron_state = next_neuron_state;

            tokens = if let (Some(residual), Some(states)) =
                (&self.attention_residual, history.as_mut())
            {
                states.push(block_out);
                let query_index = if self.attn_residual_tied {
                    0
                } else {
                    layer_index
                };
                residual.forward(
                    [batch, sequence, self.dim],
                    states,
                    query_index,
                    self.depth,
                    options.total_reasoning_iterations,
                )?
            } else {
                tokens + block_out
            };

            if options.collect_per_pass_hiddens {
                per_pass_hiddens.push(tokens.clone());
            }

            let next = if options.update_memory {
                Some(match previous {
                    Some(old) => {
                        let retained = if let Some(logit) = &self.memory_retention_logit {
                            let retention =
                                activation::sigmoid(logit.val()).reshape([1, self.heads, 1, 1]);
                            old * retention
                        } else {
                            old
                        };
                        retained + memory_write
                    }
                    None => memory_write,
                })
            } else {
                previous
            };
            next_weights.push(next);
        }

        // Unlike a Transformer pre-norm block, upstream applies this only once
        // after all recurrent depths.
        let embeddings = layer_norm_no_params(tokens);
        let logits = options
            .return_logits
            .then(|| self.project_logits(embeddings.clone()));

        Ok(BdhOutput {
            logits,
            memory: Memory {
                tokens_seen: tokens_seen + valid_sequence_length,
                embeds: embeddings,
                fast_weights: next_weights,
            },
            per_pass_hiddens,
            attention_history: history,
        })
    }
}

/// PyTorch-compatible LayerNorm without trainable scale or bias.
fn layer_norm_no_params<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    let mean = input.clone().mean_dim(D - 1);
    let centered = input - mean;
    let variance = centered.clone().square().mean_dim(D - 1);
    centered / (variance + 1e-5).sqrt()
}

/// RMS normalization over the last axis without affine parameters.
///
/// Unlike LayerNorm this preserves the sign of every coordinate; therefore a
/// positive neuron-state stays positive before it is injected into the next
/// recurrent block.
fn rms_norm_no_params<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    let rms = (input.clone().square().mean_dim(D - 1) + f32::EPSILON as f64).sqrt();
    input / rms
}
