//! ARC-style in-context-query codec and end-to-end protocol helpers.
//!
//! Grids serialize over colors `0..=9` plus four structural tokens.  A prompt
//! consists of demonstration input/output pairs followed by the query input.
//! The prompt updates recurrent memory, the query's last hidden state undergoes
//! a chosen number of latent steps, and an output grid is decoded
//! autoregressively.

use burn::{
    nn::loss::CrossEntropyLossConfig,
    tensor::{Int, Tensor, TensorData, backend::Backend},
};

use crate::{
    error::BdhError,
    model::{Bdh, BdhConfig, Memory},
    reasoning::{
        GenerateOptions, ReasoningForwardOptions, ReasoningWrapper, ReasoningWrapperConfig, Stage,
    },
    tasks::{Grid, TaskData},
};

/// ARC color and marker vocabulary size.
pub const NUM_TOKENS: usize = 14;
/// Separator between serialized rows.
pub const ROW: usize = 10;
/// Marker starting an input grid.
pub const INPUT: usize = 11;
/// Marker starting an output grid.
pub const OUTPUT: usize = 12;
/// End-of-output marker.
pub const EOS: usize = 13;
/// Upstream's default number of prompt tokens per recurrent chunk.
pub const CHUNK_SIZE: usize = 128;

/// Class weights used by the small ARC replication.
///
/// Background zero is downweighted; colors and structural answer tokens are
/// upweighted to discourage an all-black prediction collapse.
pub const CLASS_WEIGHTS: [f32; NUM_TOKENS] = [
    0.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 1.0, 2.0, 3.0,
];

/// Default model shape from upstream `icq.py`.
pub fn model_config() -> BdhConfig {
    BdhConfig::new(NUM_TOKENS, 384)
        .with_depth(4)
        .with_heads(4)
        .with_dim_qk_heads(2048)
}

/// Initialize the default ARC model.
pub fn make_model<B: Backend>(device: &B::Device) -> Result<Bdh<B>, BdhError> {
    model_config().init(device)
}

/// Initialize the default ARC model and its reasoning wrapper.
pub fn make_wrapper<B: Backend>(
    device: &B::Device,
    latent_step_embedding: bool,
) -> Result<ReasoningWrapper<B>, BdhError> {
    let model = make_model(device)?;
    Ok(ReasoningWrapperConfig::new()
        .with_latent_step_embedding(latent_step_embedding)
        .init(model, device))
}

/// Serialize a grid as `marker, row0, ROW, row1, ...`.
pub fn encode_grid(grid: &Grid, marker: usize) -> Vec<usize> {
    let mut tokens = Vec::with_capacity(grid.len() + grid.height());
    tokens.push(marker);
    for row in 0..grid.height() {
        if row > 0 {
            tokens.push(ROW);
        }
        for column in 0..grid.width() {
            tokens.push(grid.get(row, column) as usize);
        }
    }
    tokens
}

/// Serialize a target grid and append [`EOS`].
pub fn encode_output(grid: &Grid) -> Vec<usize> {
    let mut tokens = encode_grid(grid, OUTPUT);
    tokens.push(EOS);
    tokens
}

/// Decode colors and row separators into a grid.
///
/// Input/output markers are ignored, decoding stops at [`EOS`], and ragged
/// generated rows are padded with black cells exactly as in upstream.
pub fn decode_grid(tokens: &[usize]) -> Result<Grid, BdhError> {
    let mut rows: Vec<Vec<u8>> = Vec::new();
    let mut row = Vec::new();
    for &token in tokens {
        if token == EOS {
            break;
        }
        if token < 10 {
            row.push(token as u8);
        } else if token == ROW && (!row.is_empty() || !rows.is_empty()) {
            rows.push(std::mem::take(&mut row));
        }
    }
    if !row.is_empty() || rows.is_empty() {
        rows.push(row);
    }
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    if width == 0 {
        // The Python code returns a shape (1, 0) NumPy array.  `Grid` rejects
        // empty dimensions, so surface malformed generation as an error.
        return Err(BdhError::InvalidGrid(
            "generated tokens contained no color cells".into(),
        ));
    }
    for row in &mut rows {
        row.resize(width, 0);
    }
    Grid::from_rows(rows)
}

/// Serialize up to `num_demonstrations`, then the first held-out input.
pub fn task_prompt(task: &TaskData, num_demonstrations: usize) -> Vec<usize> {
    let mut tokens = Vec::new();
    for example in task.train.iter().take(num_demonstrations) {
        tokens.extend(encode_grid(&example.input, INPUT));
        tokens.extend(encode_output(&example.output));
    }
    tokens.extend(encode_grid(&task.test[0].input, INPUT));
    tokens
}

/// Serialize the first held-out target.
pub fn task_answer(task: &TaskData) -> Vec<usize> {
    encode_output(&task.test[0].output)
}

/// Exact autoregressive length implied by the query grid dimensions.
pub fn answer_length(task: &TaskData) -> usize {
    // output marker + all cells + row separators + EOS
    1 + task.test[0].input.len() + task.test[0].input.height() - 1 + 1
}

/// Run a serialized prompt through the model in bounded-size chunks.
pub fn ingest<B: Backend>(
    wrapper: &ReasoningWrapper<B>,
    ids: &[usize],
    mut memory: Option<Memory<B>>,
    chunk_size: usize,
    update_memory: bool,
) -> Result<Memory<B>, BdhError> {
    if chunk_size == 0 {
        return Err(BdhError::InvalidStages(
            "ingest chunk_size must be non-zero".into(),
        ));
    }
    let device = wrapper.model().device();
    for chunk in ids.chunks(chunk_size) {
        let tensor = ids_tensor::<B>(chunk, &device);
        let output = wrapper.forward(
            &[Stage::Tokens(tensor)],
            memory,
            ReasoningForwardOptions {
                update_memory,
                ..Default::default()
            },
        )?;
        memory = Some(output.memory);
    }
    memory.ok_or_else(|| BdhError::InvalidStages("cannot ingest an empty id sequence".into()))
}

/// Result of recurrent ingestion with concatenated intermediate outputs.
///
/// The tuple contains `(final memory, hiddens, logits)` with both tensors
/// concatenated along their sequence axis.
pub type IngestedHiddens<B> = (Memory<B>, Tensor<B, 3>, Tensor<B, 3>);

/// Ingest while collecting every chunk's hidden states and logits.
pub fn ingest_hiddens<B: Backend>(
    wrapper: &ReasoningWrapper<B>,
    ids: &[usize],
    chunk_size: usize,
    update_memory: bool,
) -> Result<IngestedHiddens<B>, BdhError> {
    if ids.is_empty() || chunk_size == 0 {
        return Err(BdhError::InvalidStages(
            "ingest_hiddens needs non-empty ids and chunk_size".into(),
        ));
    }
    let device = wrapper.model().device();
    let mut memory = None;
    let mut hiddens = Vec::new();
    let mut logits = Vec::new();
    for chunk in ids.chunks(chunk_size) {
        let output = wrapper.forward(
            &[Stage::Tokens(ids_tensor::<B>(chunk, &device))],
            memory,
            ReasoningForwardOptions {
                update_memory,
                ..Default::default()
            },
        )?;
        hiddens.push(output.memory.embeds.clone());
        logits.push(output.logits.expect("token stages produce logits"));
        memory = Some(output.memory);
    }
    Ok((
        memory.unwrap(),
        Tensor::cat(hiddens, 1),
        Tensor::cat(logits, 1),
    ))
}

/// Full upstream ARC training objective.
///
/// This adds next-token loss across the prompt (excluding `<input>` targets)
/// to the wrapper's latent-step and answer loss.
pub fn train_loss<B: Backend>(
    wrapper: &ReasoningWrapper<B>,
    task: &TaskData,
    reasoning_steps: usize,
    class_weights: Option<Vec<f32>>,
    update_memory: bool,
    update_latent_memory: bool,
) -> Result<Tensor<B, 1>, BdhError> {
    let prompt = task_prompt(task, 3);
    let answer = task_answer(task);
    let (memory, _, prompt_logits) = ingest_hiddens(wrapper, &prompt, CHUNK_SIZE, update_memory)?;
    let device = prompt_logits.device();

    let answer_tensor = ids_tensor::<B>(&answer, &device);
    let output = wrapper.forward(
        &[Stage::Think(reasoning_steps), Stage::Tokens(answer_tensor)],
        Some(memory),
        ReasoningForwardOptions {
            compute_loss: true,
            update_latent_memory,
            class_weights: class_weights.clone(),
            ..Default::default()
        },
    )?;

    // Every prompt position predicts its successor; the last query-input
    // position predicts the `<output>` marker.  Targets that start a new input
    // grid are excluded, matching upstream's mask.
    let mut targets = prompt[1..].to_vec();
    targets.push(OUTPUT);
    let valid_positions: Vec<usize> = targets
        .iter()
        .enumerate()
        .filter_map(|(index, token)| (*token != INPUT).then_some(index))
        .collect();
    let valid_targets: Vec<usize> = valid_positions
        .iter()
        .map(|&index| targets[index])
        .collect();

    let indices = int_vector::<B>(&valid_positions, &device);
    let target_tensor = int_vector::<B>(&valid_targets, &device);
    let selected_logits = prompt_logits.squeeze_dim::<2>(0).select(0, indices);
    let criterion = CrossEntropyLossConfig::new()
        .with_weights(class_weights)
        .init(&device);
    let prompt_loss = criterion.forward(selected_logits, target_tensor);

    Ok(prompt_loss + output.loss.expect("compute_loss requested"))
}

/// Generate the first query's answer tokens at a chosen latent effort.
pub fn generate_answer<B: Backend>(
    wrapper: &ReasoningWrapper<B>,
    task: &TaskData,
    reasoning_steps: usize,
    memory: Option<Memory<B>>,
    update_memory: bool,
    update_latent_memory: bool,
    temperature: f64,
) -> Result<Vec<usize>, BdhError> {
    let memory = match memory {
        Some(memory) => memory,
        None => ingest(wrapper, &task_prompt(task, 3), None, CHUNK_SIZE, true)?,
    };
    let (mut tokens, _) = wrapper.generate(
        &[Stage::Think(reasoning_steps)],
        Some(memory),
        GenerateOptions {
            max_new_tokens: Some(answer_length(task)),
            stop_token: Some(EOS),
            temperature,
            update_memory,
            update_latent_memory,
            ..Default::default()
        },
    )?;
    if tokens.last() != Some(&EOS) {
        tokens.pop();
    }
    Ok(tokens)
}

/// Generate and decode the first held-out output grid.
pub fn solve<B: Backend>(
    wrapper: &ReasoningWrapper<B>,
    task: &TaskData,
    reasoning_steps: usize,
    memory: Option<Memory<B>>,
) -> Result<Grid, BdhError> {
    decode_grid(&generate_answer(
        wrapper,
        task,
        reasoning_steps,
        memory,
        true,
        true,
        0.0,
    )?)
}

/// Cell-level comparison, meaningful only when grid dimensions agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellStats {
    /// Number of exactly matching cells, or zero for a shape mismatch.
    pub correct: usize,
    /// Target cell count, or zero for a shape mismatch.
    pub total: usize,
    /// Whether prediction and target dimensions agree.
    pub dimensions_valid: bool,
}

/// Compare two grids using upstream's shape-sensitive convention.
pub fn cell_stats(predicted: &Grid, target: &Grid) -> CellStats {
    if predicted.height() != target.height() || predicted.width() != target.width() {
        return CellStats {
            correct: 0,
            total: 0,
            dimensions_valid: false,
        };
    }
    CellStats {
        correct: predicted
            .cells()
            .iter()
            .zip(target.cells())
            .filter(|(left, right)| left == right)
            .count(),
        total: target.len(),
        dimensions_valid: true,
    }
}

fn ids_tensor<B: Backend>(ids: &[usize], device: &B::Device) -> Tensor<B, 2, Int> {
    let values: Vec<i64> = ids.iter().map(|token| *token as i64).collect();
    Tensor::from_data(TensorData::new(values, [1, ids.len()]), device)
}

fn int_vector<B: Backend>(values: &[usize], device: &B::Device) -> Tensor<B, 1, Int> {
    let values: Vec<i64> = values.iter().map(|value| *value as i64).collect();
    let length = values.len();
    Tensor::from_data(TensorData::new(values, [length]), device)
}
