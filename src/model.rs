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

/// Wide neuron workspace carried only by consecutive latent reasoning steps.
///
/// This state has layout `[B,H,1,Q]`.  It is intentionally separate from
/// [`Memory`]: token-aligned `[B,H,N,Q]` activations must not cross ordinary
/// chunk boundaries, while a one-position `Think` chain can safely reuse its
/// wide computational state across outer reasoning iterations.
#[derive(Clone, Debug, Default)]
pub struct LatentWorkspace<B: Backend> {
    neuron_state: Option<Tensor<B, 4>>,
}

impl<B: Backend> LatentWorkspace<B> {
    /// Start an independent latent reasoning chain with no wide state.
    pub fn new() -> Self {
        Self { neuron_state: None }
    }

    /// Whether a completed latent iteration has populated the wide state.
    pub fn has_neuron_state(&self) -> bool {
        self.neuron_state.is_some()
    }

    /// Shape of the current wide state, when initialized.
    pub fn neuron_state_dims(&self) -> Option<[usize; 4]> {
        self.neuron_state.as_ref().map(Tensor::dims)
    }

    /// Read the wide state without making it part of persistent CQ memory.
    pub fn neuron_state(&self) -> Option<&Tensor<B, 4>> {
        self.neuron_state.as_ref()
    }

    /// Cut autograd history at an explicit caller-selected reasoning boundary.
    pub fn detach(self) -> Self {
        Self {
            neuron_state: self.neuron_state.map(Tensor::detach),
        }
    }
}

/// Persistent state carried between token chunks and latent-reasoning passes.
///
/// The paper's contextual state corresponds primarily to
/// [`fast_weights`](Self::fast_weights), denoted `M` in this crate to avoid
/// confusion with v2's wide neuron state `S`. [`embeds`](Self::embeds) is
/// different: it is the most recent narrow output and seeds latent `H_0`.
#[derive(Clone, Debug)]
pub struct Memory<B: Backend> {
    /// Sequence positions processed independently by every batch row.
    ///
    /// Stateful TBPTT lanes reset at different document boundaries, so a
    /// single scalar cursor would assign incorrect RoPE phases to all but one
    /// lane. In ordinary single-stream inference this vector has length one.
    pub position_offsets: Vec<usize>,
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
            position_offsets: self.position_offsets,
            embeds: self.embeds.detach(),
            fast_weights: self
                .fast_weights
                .into_iter()
                .map(|weight| weight.map(Tensor::detach))
                .collect(),
        }
    }

    /// Reset selected batch rows without disturbing the other TBPTT lanes.
    ///
    /// Multiplying the stored tensors by a constant keep-mask also blocks
    /// gradients from the new document into the reset row's previous
    /// document. Rows whose mask entry is `false` keep both values and graph.
    pub fn reset_rows(mut self, reset: &[bool]) -> Result<Self, BdhError> {
        let [batch, _, _] = self.embeds.dims();
        if reset.len() != batch || self.position_offsets.len() != batch {
            return Err(BdhError::IncompatibleMemory(format!(
                "row reset mask, position offsets and memory batch must agree: mask={}, offsets={}, batch={batch}",
                reset.len(),
                self.position_offsets.len()
            )));
        }
        if !reset.iter().any(|value| *value) {
            return Ok(self);
        }

        let keep = reset
            .iter()
            .map(|value| if *value { 0.0_f32 } else { 1.0_f32 })
            .collect::<Vec<_>>();
        let device = self.embeds.device();
        let embeds_keep =
            Tensor::<B, 1>::from_floats(keep.as_slice(), &device).reshape([batch, 1, 1]);
        self.embeds = self.embeds * embeds_keep;
        for state in self.fast_weights.iter_mut().flatten() {
            let state_keep =
                Tensor::<B, 1>::from_floats(keep.as_slice(), &device).reshape([batch, 1, 1, 1]);
            *state = state.clone() * state_keep;
        }
        for (offset, should_reset) in self.position_offsets.iter_mut().zip(reset) {
            if *should_reset {
                *offset = 0;
            }
        }
        Ok(self)
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
    /// Fraction of the previous CQ state exposed to the current pass.
    ///
    /// Training uses this scalar for a short memory curriculum: writes and
    /// retention proceed normally while the read path grows smoothly from
    /// zero to one. Keeping the scale on the read avoids changing the
    /// recurrent update `M' = rho * M + K^T V` or corrupting stored memory.
    pub memory_read_scale: f64,
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
    /// Row-major `[B * N]` flags marking tokens that begin a new document.
    ///
    /// A flagged token receives RoPE position zero, cannot read CQ state from
    /// the preceding document, and starts a new local-attention segment. The
    /// returned memory retains only tokens at or after the row's final flag.
    /// This host metadata enables exact independently-reset stateful lanes in
    /// one physical GPU batch.
    pub document_starts: Option<Vec<bool>>,
}

impl<B: Backend> Default for BdhForwardOptions<B> {
    fn default() -> Self {
        Self {
            update_memory: true,
            memory_read_scale: 1.0,
            return_logits: true,
            collect_per_pass_hiddens: false,
            attention_history: None,
            total_reasoning_iterations: 1,
            valid_sequence_length: None,
            document_starts: None,
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
    /// Independent feature-subspace routing heads used by MHAR.
    #[config(default = 1)]
    pub attn_residual_heads: usize,
    /// Number of learnable cycle-distance bias values; zero disables the bias.
    #[config(default = 0)]
    pub attn_residual_depth_bias_distance: usize,
    /// Carry a distinct full `[B,H,N,Q]` delta-state across recurrent depth.
    #[config(default = false)]
    pub gated_neuron_state: bool,
    /// Initial fraction used to move an existing state toward a new candidate.
    #[config(default = 0.2)]
    pub gated_neuron_state_initial_update: f64,
    /// Initial bounded strength of direct wide-state injection into Q/K gates.
    ///
    /// Legacy configurations default to zero. Architecture-v2 configs set a
    /// small non-zero value so the path is active without dominating `W_qk X`.
    #[config(default = 0.0)]
    pub gated_neuron_state_initial_injection: f64,
    /// Normalize the narrow `[B,N,D]` stream after every recurrent depth.
    ///
    /// This is enabled by architecture v2. The default is false so the public
    /// reconstruction retains its original single final normalization.
    #[config(default = false)]
    pub normalize_each_depth: bool,
    /// Exponentially retain old CQ fast weights before adding the new write.
    #[config(default = false)]
    pub cq_memory_decay: bool,
    /// Initial per-neuron CQ retention when decay is enabled.
    #[config(default = 0.995)]
    pub cq_memory_initial_rho: f64,
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
            self.gated_neuron_state_initial_update,
            self.gated_neuron_state_initial_injection,
            device,
        );
        let attention_residual = self.attn_residual.then(|| {
            let pseudo_queries = if self.attn_residual_tied {
                1
            } else {
                self.depth
            };
            MultiHeadAttentionResidual::new(
                self.dim,
                pseudo_queries,
                self.attn_residual_heads,
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
            raw_rho: self.cq_memory_decay.then(|| {
                let probability = self.cq_memory_initial_rho;
                let logit = (probability / (1.0 - probability)).ln();
                Initializer::Constant { value: logit }.init([self.heads, qk_per_head], device)
            }),
            dim: self.dim,
            num_tokens: self.num_tokens,
            depth: self.depth,
            heads: self.heads,
            qk_per_head,
            rotary_dim,
            attn_residual_tied: self.attn_residual_tied,
            attn_residual_heads: self.attn_residual_heads,
            tie_embeddings: self.tie_embeddings,
            normalize_each_depth: self.normalize_each_depth,
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
        if self.attn_residual
            && (self.attn_residual_heads == 0 || !self.dim.is_multiple_of(self.attn_residual_heads))
        {
            return Err(BdhError::InvalidConfig(
                "dim must be divisible by a non-zero attn_residual_heads count".into(),
            ));
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
        if !(0.0 < self.gated_neuron_state_initial_update
            && self.gated_neuron_state_initial_update < 1.0)
        {
            return Err(BdhError::InvalidConfig(
                "gated_neuron_state_initial_update must be strictly between zero and one".into(),
            ));
        }
        if !(-1.0 < self.gated_neuron_state_initial_injection
            && self.gated_neuron_state_initial_injection < 1.0)
        {
            return Err(BdhError::InvalidConfig(
                "gated_neuron_state_initial_injection must be strictly between -1 and 1".into(),
            ));
        }
        if !(0.0 < self.cq_memory_initial_rho && self.cq_memory_initial_rho < 1.0) {
            return Err(BdhError::InvalidConfig(
                "cq_memory_initial_rho must be strictly between zero and one".into(),
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
    /// Per-neuron unconstrained base logit for the full-state update fraction.
    raw_state_update: Option<Param<Tensor<B, 2>>>,
    /// Per-neuron unconstrained direct-state injection, bounded with `tanh`.
    raw_state_injection: Option<Param<Tensor<B, 2>>>,
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
        gated_neuron_state_initial_update: f64,
        gated_neuron_state_initial_injection: f64,
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
                    .with_bias(false)
                    .with_initializer(Initializer::Zeros)
                    .init(device)
            }),
            raw_state_update: gated_neuron_state.then(|| {
                let probability = gated_neuron_state_initial_update;
                let logit = (probability / (1.0 - probability)).ln();
                Initializer::Constant { value: logit }.init([heads, qk_per_head], device)
            }),
            raw_state_injection: gated_neuron_state.then(|| {
                let raw = gated_neuron_state_initial_injection.atanh();
                Initializer::Constant { value: raw }.init([heads, qk_per_head], device)
            }),
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
        metadata: &SequenceMetadata<B>,
        memory_read_scale: f64,
    ) -> BdhBlockOutput<B> {
        let [batch, sequence, dim] = tokens.dims();

        // [B,N,D] -> [B,N,H*Q] -> [B,H,N,Q].  When a previous wide state is
        // available it is injected into the raw projection before the single
        // ReLU, exactly as G = ReLU(W_qk X + alpha * RMS(S_prev)).  The
        // state-free/legacy path therefore remains G = ReLU(W_qk X).
        let projected = self
            .to_qk
            .forward(tokens.clone())
            .reshape([batch, sequence, self.heads, self.qk_per_head])
            .permute([0, 2, 1, 3]);
        let gates = if let (Some(state), Some(raw_injection)) =
            (&neuron_state, &self.raw_state_injection)
        {
            inject_wide_state(
                projected,
                state.clone(),
                raw_injection.val(),
                self.heads,
                self.qk_per_head,
            )
        } else {
            activation::relu(projected)
        };

        // The gate remains unrotated.  Only Q and K receive positional phase.
        let q = apply_rotary(gates.clone(), &metadata.position_ids, self.rotary_dim);
        // This reconstruction deliberately shares the same projection for Q
        // and K, so their rotated tensors are identical. Reusing the result
        // avoids a second host phase build and pair of trigonometric kernels.
        let k = q.clone();

        // Current-chunk causal linear attention.  There is deliberately no
        // softmax or 1/sqrt(Q) scaling: Q K^T is an unnormalized affinity.
        // The diagonal is removed, so a position cannot retrieve its own V.
        let mut similarity = q.clone().matmul(k.clone().transpose()).tril(-1);
        if let Some(mask) = &metadata.local_attention_mask {
            similarity = similarity * mask.clone();
        }
        let values_by_head = tokens
            .clone()
            .unsqueeze_dim::<4>(1)
            .repeat_dim(1, self.heads);
        let mut aggregate = similarity.matmul(values_by_head.clone());

        // Previous chunks are compressed into M = sum(K^T V).  Retrieval
        // qM is algebraically the same contraction as attention over every
        // old position, without retaining a growing token cache.
        if let Some(memory) = previous_memory {
            let mut memory_read = q.matmul(memory.clone()) * memory_read_scale;
            if let Some(mask) = &metadata.previous_memory_mask {
                memory_read = memory_read * mask.clone();
            }
            aggregate = aggregate + memory_read;
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

        // The persistent wide state and the information communicated through
        // narrow D are distinct. With no previous state the first candidate
        // is accepted whole. Later depths form the bounded delta
        //   DeltaS = u * (Z - S_prev); S_next = S_prev + DeltaS.
        // Only DeltaS is projected by W_out, so an old S is never re-added by
        // the ordinary residual or by Attention Residual history.
        let (neuron_delta, next_neuron_state) = if let Some(update) = &self.state_update {
            match neuron_state {
                None => {
                    let (delta, state) = wide_state_transition(lifted, None, None);
                    (delta, Some(state))
                }
                Some(previous) => {
                    let token_bias = update
                        .forward(tokens.clone())
                        .permute([0, 2, 1])
                        .unsqueeze_dim::<4>(3);
                    let base = self
                        .raw_state_update
                        .as_ref()
                        .expect("state gate parameters are initialized together")
                        .val();
                    let write_gate =
                        per_neuron_update_gate(token_bias, base, self.heads, self.qk_per_head);
                    let (delta, state) =
                        wide_state_transition(lifted, Some(previous), Some(write_gate));
                    (delta, Some(state))
                }
            }
        } else {
            (lifted, None)
        };

        let block_out = self.project_neuron_delta(neuron_delta.clone());

        // New write for this chunk: [B,H,Q,N] @ [B,H,N,D] -> [B,H,Q,D].
        // The optional mask keeps only the final document represented in this
        // chunk. Without document boundaries, a physically padded tail was
        // already zeroed before the shared recurrent block and contributes no
        // write because every projection on this path is bias-free.
        let values_for_memory = if let Some(mask) = &metadata.memory_write_mask {
            values_by_head * mask.clone()
        } else {
            values_by_head
        };
        let memory_write = k.transpose().matmul(values_for_memory);
        debug_assert_eq!(block_out.dims(), [batch, sequence, dim]);

        BdhBlockOutput {
            block_out,
            memory_write,
            neuron_state: next_neuron_state,
            #[cfg(test)]
            neuron_delta,
        }
    }

    /// Compress only this depth's state change into the narrow D channel.
    fn project_neuron_delta(&self, delta: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, _, sequence, _] = delta.dims();
        let delta =
            delta
                .permute([0, 2, 1, 3])
                .reshape([batch, sequence, self.heads * self.qk_per_head]);
        layer_norm_no_params(self.proj_out.forward(delta))
    }
}

/// Internal result of one application of the shared block.
///
/// Keeping `neuron_delta` separate from `neuron_state` makes the architectural
/// invariant explicit: only the former may flow through `proj_out`.
struct BdhBlockOutput<B: Backend> {
    block_out: Tensor<B, 3>,
    memory_write: Tensor<B, 4>,
    neuron_state: Option<Tensor<B, 4>>,
    #[cfg(test)]
    neuron_delta: Tensor<B, 4>,
}

/// Apply the v2 delta-state recurrence to synthetic or model-produced tensors.
fn wide_state_transition<B: Backend>(
    candidate: Tensor<B, 4>,
    previous: Option<Tensor<B, 4>>,
    update: Option<Tensor<B, 4>>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    match previous {
        None => (candidate.clone(), candidate),
        Some(previous) => {
            let update = update.expect("an existing wide state requires an update gate");
            let delta = update * (candidate - previous.clone());
            let state = previous + delta.clone();
            (delta, state)
        }
    }
}

/// Combine cheap `[B,H,N,1]` input bias with a learned `[H,Q]` baseline.
fn per_neuron_update_gate<B: Backend>(
    token_bias: Tensor<B, 4>,
    raw_base: Tensor<B, 2>,
    heads: usize,
    qk_per_head: usize,
) -> Tensor<B, 4> {
    activation::sigmoid(token_bias + raw_base.reshape([1, heads, 1, qk_per_head]))
}

/// Inject the previous full wide state without an `H*Q -> H*Q` transition.
fn inject_wide_state<B: Backend>(
    projected: Tensor<B, 4>,
    state: Tensor<B, 4>,
    raw_strength: Tensor<B, 2>,
    heads: usize,
    qk_per_head: usize,
) -> Tensor<B, 4> {
    let strength = raw_strength.tanh().reshape([1, heads, 1, qk_per_head]);
    activation::relu(projected + rms_norm_no_params(state) * strength)
}

/// Multi-head attention over earlier depth and latent states.
///
/// This is an optional stabilization extension in the public reconstruction,
/// inspired by the separate Attention Residuals paper; it is not specified by
/// the public BDH-CQ paper.  A learned pseudo-query chooses a convex mixture of
/// all saved states independently at every batch/sequence location.
#[derive(Module, Debug)]
pub struct MultiHeadAttentionResidual<B: Backend> {
    query: Param<Tensor<B, 2>>,
    key_norm: RmsNorm<B>,
    depth_bias: Option<Param<Tensor<B, 1>>>,
    routing_heads: usize,
}

impl<B: Backend> MultiHeadAttentionResidual<B> {
    /// Initialize zero-query multi-head depth routing.
    pub fn new(
        dim: usize,
        pseudo_queries: usize,
        routing_heads: usize,
        depth_bias_distance: usize,
        device: &B::Device,
    ) -> Self {
        assert!(
            routing_heads > 0 && dim.is_multiple_of(routing_heads),
            "MHAR requires a non-zero routing-head count that divides D"
        );
        assert!(pseudo_queries > 0, "MHAR needs at least one pseudo-query");
        let normal = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        Self {
            // Zero queries make every head read a uniform depth mixture at
            // initialization, matching the corrected MHAR recipe.
            query: Initializer::Zeros.init([pseudo_queries, dim], device),
            // PyTorch's `RMSNorm(dim)` defaults to the machine epsilon of the
            // input dtype. Upstream trains in float32, so use f32 epsilon
            // rather than Burn's otherwise slightly larger 1e-5 default.
            key_norm: RmsNormConfig::new(dim)
                .with_epsilon(f32::EPSILON as f64)
                .init(device),
            depth_bias: (depth_bias_distance > 0)
                .then(|| normal.init([depth_bias_distance], device)),
            routing_heads,
        }
    }

    /// Read a block-diagonal, per-head softmax mixture of `keys_values`.
    pub fn forward(
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
        // RMSNorm is deliberately computed across the full D row before the
        // contiguous feature slices are separated into routing heads.
        let normalized = self.key_norm.forward(past.clone());
        let head_dim = dim / self.routing_heads;
        let normalized =
            normalized.reshape([batch, sequence, layers, self.routing_heads, head_dim]);
        let query = self
            .query
            .val()
            .slice([query_index..query_index + 1, 0..dim])
            .reshape([1, 1, 1, self.routing_heads, head_dim]);
        let mut similarity = (normalized * query).sum_dim(4).squeeze_dim::<4>(4);

        if let Some(schedule) = &self.depth_bias {
            let bias = compute_attn_residual_depth_bias(
                layers,
                schedule.val(),
                depth,
                total_reasoning_iterations,
            )
            .reshape([1, 1, layers, 1]);
            similarity = similarity + bias;
        }

        let values = past.reshape([batch, sequence, layers, self.routing_heads, head_dim]);
        let weights = activation::softmax(similarity, 2).unsqueeze_dim::<5>(4);
        let readout = (weights * values)
            .sum_dim(2)
            .squeeze_dim::<4>(2)
            .reshape([batch, sequence, dim]);
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
    attention_residual: Option<MultiHeadAttentionResidual<B>>,
    raw_rho: Option<Param<Tensor<B, 2>>>,
    dim: usize,
    num_tokens: usize,
    depth: usize,
    heads: usize,
    qk_per_head: usize,
    rotary_dim: usize,
    attn_residual_tied: bool,
    attn_residual_heads: usize,
    tie_embeddings: bool,
    normalize_each_depth: bool,
}

/// Per-token masks derived once and reused by every recurrent depth.
struct SequenceMetadata<B: Backend> {
    position_ids: Vec<usize>,
    local_attention_mask: Option<Tensor<B, 4>>,
    previous_memory_mask: Option<Tensor<B, 4>>,
    memory_write_mask: Option<Tensor<B, 4>>,
    old_memory_keep: Option<Tensor<B, 4>>,
    next_position_offsets: Vec<usize>,
}

#[allow(clippy::too_many_arguments)]
fn sequence_metadata<B: Backend>(
    batch: usize,
    sequence: usize,
    valid_sequence_length: usize,
    position_offsets: &[usize],
    document_starts: Option<&[bool]>,
    device: &B::Device,
) -> Result<SequenceMetadata<B>, BdhError> {
    if position_offsets.len() != batch {
        return Err(BdhError::IncompatibleMemory(format!(
            "expected {batch} position offsets, got {}",
            position_offsets.len()
        )));
    }
    if let Some(starts) = document_starts
        && starts.len() != batch * sequence
    {
        return Err(BdhError::InvalidStages(format!(
            "document_starts must contain B*N={} flags, got {}",
            batch * sequence,
            starts.len()
        )));
    }

    let mut position_ids = Vec::with_capacity(batch * sequence);
    let mut next_position_offsets = Vec::with_capacity(batch);
    let mut document_ids = document_starts.map(|_| Vec::with_capacity(batch * sequence));
    let mut previous_memory = document_starts.map(|_| Vec::with_capacity(batch * sequence));
    let mut memory_write = document_starts.map(|_| Vec::with_capacity(batch * sequence));
    let mut keep_old = document_starts.map(|_| Vec::with_capacity(batch));

    for row in 0..batch {
        let mut position = position_offsets[row];
        let mut document = 0_usize;
        let mut reset_seen = false;
        let mut last_reset = None;
        for column in 0..sequence {
            let logical = column < valid_sequence_length;
            let starts_document =
                logical && document_starts.is_some_and(|starts| starts[row * sequence + column]);
            if starts_document {
                position = 0;
                document += 1;
                reset_seen = true;
                last_reset = Some(column);
            }
            position_ids.push(position);
            if logical {
                position += 1;
            }
            if let Some(ids) = &mut document_ids {
                ids.push(document);
            }
            if let Some(mask) = &mut previous_memory {
                mask.push(if reset_seen { 0.0_f32 } else { 1.0_f32 });
            }
        }

        if let Some(mask) = &mut memory_write {
            let keep_from = last_reset.unwrap_or(0);
            mask.extend((0..sequence).map(|column| {
                if column < valid_sequence_length && column >= keep_from {
                    1.0_f32
                } else {
                    0.0_f32
                }
            }));
        }
        if let Some(mask) = &mut keep_old {
            mask.push(if last_reset.is_some() {
                0.0_f32
            } else {
                1.0_f32
            });
        }
        next_position_offsets.push(match last_reset {
            Some(column) => valid_sequence_length - column,
            None => position_offsets[row] + valid_sequence_length,
        });
    }

    let local_attention_mask = document_ids.map(|ids| {
        let mut values = Vec::with_capacity(batch * sequence * sequence);
        for row in 0..batch {
            let row_ids = &ids[row * sequence..(row + 1) * sequence];
            for query in 0..sequence {
                for key in 0..sequence {
                    values.push(if row_ids[query] == row_ids[key] {
                        1.0_f32
                    } else {
                        0.0_f32
                    });
                }
            }
        }
        Tensor::<B, 1>::from_floats(values.as_slice(), device)
            .reshape([batch, 1, sequence, sequence])
    });
    let previous_memory_mask = previous_memory.map(|values| {
        Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([batch, 1, sequence, 1])
    });
    let memory_write_mask = memory_write.map(|values| {
        Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([batch, 1, sequence, 1])
    });
    let old_memory_keep = keep_old.map(|values| {
        Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([batch, 1, 1, 1])
    });

    Ok(SequenceMetadata {
        position_ids,
        local_attention_mask,
        previous_memory_mask,
        memory_write_mask,
        old_memory_keep,
        next_position_offsets,
    })
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
        self.raw_rho.is_some()
    }

    /// Number of independent feature-subspace depth routers.
    ///
    /// Zero means attention residuals are disabled. One is classic AttnRes;
    /// values above one use multi-head attention residuals.
    pub fn attention_residual_heads(&self) -> usize {
        self.attention_residual
            .as_ref()
            .map_or(0, |_| self.attn_residual_heads)
    }

    /// Learned per-neuron CQ retention probabilities `sigmoid(raw_rho)`.
    ///
    /// The returned layout is `[H,Q]`; logs should summarize it rather than
    /// printing every neuron coordinate.
    pub fn cq_retention_probabilities(&self) -> Option<Tensor<B, 2>> {
        self.raw_rho
            .as_ref()
            .map(|raw_rho| activation::sigmoid(raw_rho.val()))
    }

    /// Learned per-neuron base update probabilities `sigmoid(raw_u)`.
    ///
    /// The token-dependent projection is not included: this diagnostic tracks
    /// whether the learned baseline drifts or saturates during a long run.
    pub fn base_state_update_probabilities(&self) -> Option<Tensor<B, 2>> {
        self.block
            .raw_state_update
            .as_ref()
            .map(|raw_update| activation::sigmoid(raw_update.val()))
    }

    /// Bounded per-neuron wide-state injection strengths `tanh(raw_alpha)`.
    ///
    /// The returned layout is `[H,Q]` and matches the wide workspace's final
    /// two axes exactly.
    pub fn state_injection_strengths(&self) -> Option<Tensor<B, 2>> {
        self.block
            .raw_state_injection
            .as_ref()
            .map(|raw_injection| raw_injection.val().tanh())
    }

    /// Whether the narrow stream is normalized after each recurrent depth.
    pub fn normalizes_each_depth(&self) -> bool {
        self.normalize_each_depth
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
        let (output, _) = self.forward_internal(input, memory, None, options)?;
        Ok(output)
    }

    /// Process one `[B,1,D]` latent iteration while carrying its wide state.
    ///
    /// This is deliberately separate from [`forward`](Self::forward). Normal
    /// token passes always reset the token-aligned wide workspace; only a
    /// caller that is explicitly executing one latent reasoning chain can
    /// return the `[B,H,1,Q]` state to the next outer iteration.
    pub fn forward_latent(
        &self,
        latent: Tensor<B, 3>,
        memory: Option<Memory<B>>,
        workspace: LatentWorkspace<B>,
        options: BdhForwardOptions<B>,
    ) -> Result<(BdhOutput<B>, LatentWorkspace<B>), BdhError> {
        let [_, sequence, _] = latent.dims();
        if sequence != 1 {
            return Err(BdhError::InvalidStages(format!(
                "latent wide-state recurrence requires N=1, got N={sequence}"
            )));
        }
        let (output, neuron_state) = self.forward_internal(
            ModelInput::Embeddings(latent),
            memory,
            workspace.neuron_state,
            options,
        )?;
        Ok((output, LatentWorkspace { neuron_state }))
    }

    fn forward_internal(
        &self,
        input: ModelInput<B>,
        memory: Option<Memory<B>>,
        initial_neuron_state: Option<Tensor<B, 4>>,
        options: BdhForwardOptions<B>,
    ) -> Result<(BdhOutput<B>, Option<Tensor<B, 4>>), BdhError> {
        if !options.memory_read_scale.is_finite()
            || !(0.0..=1.0).contains(&options.memory_read_scale)
        {
            return Err(BdhError::InvalidConfig(format!(
                "memory_read_scale must be finite and in [0, 1], got {}",
                options.memory_read_scale
            )));
        }
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
        if let Some(state) = &initial_neuron_state {
            let expected = [batch, self.heads, sequence, self.qk_per_head];
            if state.dims() != expected {
                return Err(BdhError::InvalidStages(format!(
                    "latent neuron workspace must have shape {expected:?}, got {:?}",
                    state.dims()
                )));
            }
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

        let (position_offsets, previous_weights) = match memory {
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
                if memory.position_offsets.len() != batch {
                    return Err(BdhError::IncompatibleMemory(format!(
                        "expected {batch} position offsets, got {}",
                        memory.position_offsets.len()
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
                (memory.position_offsets, memory.fast_weights)
            }
            None => (vec![0; batch], vec![None; self.depth]),
        };
        let metadata = sequence_metadata(
            batch,
            sequence,
            valid_sequence_length,
            &position_offsets,
            options.document_starts.as_deref(),
            &tokens.device(),
        )?;

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
        // Ordinary `forward` supplies None, so token-aligned wide state never
        // crosses chunks. `forward_latent` may explicitly supply `[B,H,1,Q]`
        // from the preceding iteration of the same Think chain.
        let mut neuron_state = initial_neuron_state;
        let mut per_pass_hiddens = Vec::with_capacity(if options.collect_per_pass_hiddens {
            self.depth
        } else {
            0
        });

        for (layer_index, previous) in previous_weights.into_iter().enumerate() {
            let BdhBlockOutput {
                block_out,
                memory_write,
                neuron_state: next_neuron_state,
                ..
            } = self.block.forward(
                tokens.clone(),
                previous.as_ref(),
                neuron_state,
                &metadata,
                options.memory_read_scale,
            );
            neuron_state = next_neuron_state;

            let next_tokens = if let (Some(residual), Some(states)) =
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
            tokens = if self.normalize_each_depth {
                layer_norm_no_params(next_tokens)
            } else {
                next_tokens
            };

            if options.collect_per_pass_hiddens {
                per_pass_hiddens.push(tokens.clone());
            }

            let next = if options.update_memory {
                Some(match previous {
                    Some(old) => {
                        let old = if let Some(keep) = &metadata.old_memory_keep {
                            old * keep.clone()
                        } else {
                            old
                        };
                        let retained = if let Some(raw_rho) = &self.raw_rho {
                            retain_cq_per_neuron(
                                old,
                                activation::sigmoid(raw_rho.val()),
                                self.heads,
                                self.qk_per_head,
                            )
                        } else {
                            old
                        };
                        retained + memory_write
                    }
                    None => memory_write,
                })
            } else {
                previous.map(|old| {
                    if let Some(keep) = &metadata.old_memory_keep {
                        old * keep.clone()
                    } else {
                        old
                    }
                })
            };
            next_weights.push(next);
        }

        // The public reconstruction always applies this final normalization.
        // V2 additionally normalizes each intermediate recurrent transition;
        // retaining the final pass keeps logits and Memory.embeds on the same
        // stable contract in both modes.
        let embeddings = layer_norm_no_params(tokens);
        let logits = options
            .return_logits
            .then(|| self.project_logits(embeddings.clone()));

        let output = BdhOutput {
            logits,
            memory: Memory {
                position_offsets: metadata.next_position_offsets,
                embeds: embeddings,
                fast_weights: next_weights,
            },
            per_pass_hiddens,
            attention_history: history,
        };
        Ok((output, neuron_state))
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

/// Apply `[H,Q]` retention independently to the matching CQ coordinates.
fn retain_cq_per_neuron<B: Backend>(
    memory: Tensor<B, 4>,
    retention: Tensor<B, 2>,
    heads: usize,
    qk_per_head: usize,
) -> Tensor<B, 4> {
    memory * retention.reshape([1, heads, qk_per_head, 1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::{Autodiff, NdArray},
        optim::GradientsParams,
        tensor::{Int, TensorData, Tolerance},
    };

    type TestBackend = NdArray<f32>;
    type TrainBackend = Autodiff<TestBackend>;

    #[test]
    fn wide_delta_transition_matches_the_v2_equations() {
        let device = Default::default();
        let candidate = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![2.0_f32, 4.0, 8.0], [1, 1, 1, 3]),
            &device,
        );
        let previous = Tensor::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 10.0], [1, 1, 1, 3]),
            &device,
        );
        let update = Tensor::from_data(
            TensorData::new(vec![0.5_f32, 0.25, 0.1], [1, 1, 1, 3]),
            &device,
        );

        let (delta, state) = wide_state_transition(candidate, Some(previous), Some(update));
        delta.into_data().assert_approx_eq::<f32>(
            &TensorData::new(vec![0.5_f32, 0.5, -0.2], [1, 1, 1, 3]),
            Tolerance::absolute(1e-6),
        );
        state.into_data().assert_approx_eq::<f32>(
            &TensorData::new(vec![1.5_f32, 2.5, 9.8], [1, 1, 1, 3]),
            Tolerance::absolute(1e-6),
        );
    }

    #[test]
    fn first_wide_depth_accepts_the_complete_candidate() {
        let device = Default::default();
        let candidate = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![2.0_f32, 4.0, 8.0], [1, 1, 1, 3]),
            &device,
        );
        let (delta, state) = wide_state_transition(candidate.clone(), None, None);
        delta
            .into_data()
            .assert_approx_eq::<f32>(&candidate.to_data(), Tolerance::absolute(0.0));
        state
            .into_data()
            .assert_approx_eq::<f32>(&candidate.into_data(), Tolerance::absolute(0.0));
    }

    #[test]
    fn actual_first_block_state_and_delta_are_identical() {
        let device = Default::default();
        TestBackend::seed(&device, 92);
        let block = BdhBlock::<TestBackend>::new(8, 2, 4, 2, true, 0.2, 0.05, &device);
        let tokens = Tensor::from_data(
            TensorData::new(
                (0..16).map(|value| value as f32 / 11.0).collect(),
                [1, 2, 8],
            ),
            &device,
        );
        let metadata = sequence_metadata(1, 2, 2, &[0], None, &device).unwrap();
        let output = block.forward(tokens, None, None, &metadata, 1.0);
        output.neuron_delta.into_data().assert_approx_eq::<f32>(
            &output
                .neuron_state
                .expect("gated state is enabled")
                .into_data(),
            Tolerance::absolute(0.0),
        );
    }

    #[test]
    fn block_output_projects_delta_instead_of_accumulated_state() {
        let device = Default::default();
        TestBackend::seed(&device, 91);
        let block = BdhBlock::<TestBackend>::new(8, 2, 4, 2, true, 0.2, 0.05, &device);
        let tokens = Tensor::from_data(
            TensorData::new(
                (0..16).map(|value| value as f32 / 7.0 - 1.0).collect(),
                [1, 2, 8],
            ),
            &device,
        );
        let previous = Tensor::ones([1, 2, 2, 4], &device);
        let metadata = sequence_metadata(1, 2, 2, &[0], None, &device).unwrap();
        let output = block.forward(tokens, None, Some(previous), &metadata, 1.0);

        let projected_delta = block.project_neuron_delta(output.neuron_delta.clone());
        output
            .block_out
            .clone()
            .into_data()
            .assert_approx_eq::<f32>(&projected_delta.into_data(), Tolerance::absolute(0.0));
        let projected_state =
            block.project_neuron_delta(output.neuron_state.expect("gated state is enabled"));
        let delta_values = output.block_out.into_data().to_vec::<f32>().unwrap();
        let state_values = projected_state.into_data().to_vec::<f32>().unwrap();
        assert!(
            delta_values
                .iter()
                .zip(state_values)
                .any(|(delta, state)| (*delta - state).abs() > 1e-4),
            "projecting S_next unexpectedly matched projecting DeltaS"
        );
    }

    #[test]
    fn cq_retention_broadcasts_independently_over_every_neuron() {
        let device = Default::default();
        let memory = Tensor::<TestBackend, 4>::ones([1, 2, 3, 2], &device);
        let retention = Tensor::from_data(
            TensorData::new(vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3]),
            &device,
        );
        let retained = retain_cq_per_neuron(memory, retention, 2, 3);
        retained.into_data().assert_approx_eq::<f32>(
            &TensorData::new(
                vec![
                    0.1_f32, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6,
                ],
                [1, 2, 3, 2],
            ),
            Tolerance::absolute(1e-6),
        );
    }

    #[test]
    fn update_and_injection_broadcast_per_neuron_across_tokens() {
        let device = Default::default();
        let probabilities = [0.1_f32, 0.2, 0.3, 0.6, 0.7, 0.8];
        let raw_update = probabilities
            .iter()
            .map(|probability| (probability / (1.0 - probability)).ln())
            .collect::<Vec<_>>();
        let gate = per_neuron_update_gate(
            Tensor::<TestBackend, 4>::zeros([1, 2, 2, 1], &device),
            Tensor::from_data(TensorData::new(raw_update, [2, 3]), &device),
            2,
            3,
        );
        assert_eq!(gate.dims(), [1, 2, 2, 3]);
        gate.into_data().assert_approx_eq::<f32>(
            &TensorData::new(
                vec![
                    0.1_f32, 0.2, 0.3, 0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.6, 0.7, 0.8,
                ],
                [1, 2, 2, 3],
            ),
            Tolerance::absolute(1e-6),
        );

        let strengths = [0.01_f32, 0.02, 0.03, 0.04, 0.05, 0.06];
        let raw_strengths = strengths
            .iter()
            .map(|value| value.atanh())
            .collect::<Vec<_>>();
        let injected = inject_wide_state(
            Tensor::<TestBackend, 4>::zeros([1, 2, 2, 3], &device),
            Tensor::ones([1, 2, 2, 3], &device),
            Tensor::from_data(TensorData::new(raw_strengths, [2, 3]), &device),
            2,
            3,
        );
        injected.into_data().assert_approx_eq::<f32>(
            &TensorData::new(
                vec![
                    0.01_f32, 0.02, 0.03, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06,
                ],
                [1, 2, 2, 3],
            ),
            Tolerance::absolute(1e-6),
        );
    }

    #[test]
    fn zero_cq_read_scale_hides_old_fast_weights_without_changing_offsets() {
        let device = Default::default();
        TestBackend::seed(&device, 707);
        let model = BdhConfig::new(16, 16)
            .with_depth(2)
            .with_heads(2)
            .with_dim_qk_heads(32)
            .with_cq_memory_decay(true)
            .init::<TestBackend>(&device)
            .unwrap();
        let prefix = model
            .forward(
                ModelInput::TokenIds(Tensor::<TestBackend, 2, Int>::from_data(
                    TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
                    &device,
                )),
                None,
                Default::default(),
            )
            .unwrap()
            .memory;
        let without_fast_weights = Memory {
            position_offsets: prefix.position_offsets.clone(),
            embeds: prefix.embeds.clone(),
            fast_weights: (0..prefix.fast_weights.len()).map(|_| None).collect(),
        };
        let continuation = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![5_i64, 6, 7, 8], [1, 4]),
            &device,
        );
        let hidden = model
            .forward(
                ModelInput::TokenIds(continuation.clone()),
                Some(prefix.clone()),
                BdhForwardOptions {
                    memory_read_scale: 0.0,
                    ..Default::default()
                },
            )
            .unwrap()
            .logits
            .unwrap();
        let absent = model
            .forward(
                ModelInput::TokenIds(continuation.clone()),
                Some(without_fast_weights),
                Default::default(),
            )
            .unwrap()
            .logits
            .unwrap();
        let absent_data = absent.into_data();
        hidden
            .into_data()
            .assert_approx_eq::<f32>(&absent_data, Tolerance::absolute(1e-6));

        let visible = model
            .forward(
                ModelInput::TokenIds(continuation),
                Some(prefix),
                Default::default(),
            )
            .unwrap()
            .logits
            .unwrap();
        let visible_values = visible.into_data().to_vec::<f32>().unwrap();
        let absent_values = absent_data.to_vec::<f32>().unwrap();
        assert!(
            visible_values
                .iter()
                .zip(absent_values)
                .any(|(with_memory, without_memory)| (with_memory - without_memory).abs() > 1e-6),
            "a full CQ read unexpectedly had no effect"
        );
    }

    #[test]
    fn two_chunk_tbptt_exposes_exact_cq_decay_to_the_loss() {
        let device = Default::default();
        TrainBackend::seed(&device, 708);
        let model = BdhConfig::new(16, 16)
            .with_depth(2)
            .with_heads(2)
            .with_dim_qk_heads(32)
            .with_cq_memory_decay(true)
            .with_cq_memory_initial_rho(0.995)
            .init::<TrainBackend>(&device)
            .unwrap();
        let ids = |values: [i64; 4]| {
            Tensor::<TrainBackend, 2, Int>::from_data(
                TensorData::new(values.to_vec(), [1, 4]),
                &device,
            )
        };

        // The detached prefix is the state entering a real TBPTT window. The
        // first chunk constructs rho*M + write; the second reads that exact
        // result, so its loss must have a path to raw_rho.
        let prefix = model
            .forward(
                ModelInput::TokenIds(ids([1, 2, 3, 4])),
                None,
                Default::default(),
            )
            .unwrap()
            .memory
            .detach();
        let first = model
            .forward(
                ModelInput::TokenIds(ids([5, 6, 7, 8])),
                Some(prefix),
                Default::default(),
            )
            .unwrap();
        let second = model
            .forward(
                ModelInput::TokenIds(ids([9, 10, 11, 12])),
                Some(first.memory),
                Default::default(),
            )
            .unwrap();
        let loss = second.logits.unwrap().square().mean();
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        let raw_rho = model.raw_rho.as_ref().expect("CQ decay is enabled");
        let gradient = gradients
            .get::<TestBackend, 2>(raw_rho.id)
            .expect("two-chunk loss must reach raw_rho")
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(gradient.iter().all(|value| value.is_finite()));
        assert!(
            gradient.iter().any(|value| value.abs() > 1e-12),
            "raw_rho received only zero gradients"
        );
    }
}
