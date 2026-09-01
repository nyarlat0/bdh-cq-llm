//! Inspect the two BDH-CQ state recurrences with a tiny CPU model.
//!
//! Run with:
//!
//! ```console
//! cargo run --offline --example architecture_walkthrough
//! ```

use bdh_cq_llm::{
    BdhConfig, ReasoningForwardOptions, ReasoningWrapperConfig, Stage,
    icq::{CHUNK_SIZE, ingest, task_answer, task_prompt},
    tasks::TaskFamily,
};
use burn::{
    backend::NdArray,
    tensor::{Int, Tensor, TensorData, backend::Backend},
};

type Cpu = NdArray<f32>;

fn ids<B: Backend>(values: &[usize], device: &B::Device) -> Tensor<B, 2, Int> {
    let values: Vec<i64> = values.iter().map(|value| *value as i64).collect();
    let length = values.len();
    Tensor::from_data(TensorData::new(values, [1, length]), device)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();
    Cpu::seed(&device, 7);

    // Small enough to inspect on a CPU. The single block is reused twice, but
    // the output Memory contains two independent fast-weight matrices.
    let model = BdhConfig::new(14, 32)
        .with_depth(2)
        .with_heads(2)
        .with_dim_qk_heads(128)
        .init::<Cpu>(&device)?;
    let wrapper = ReasoningWrapperConfig::new()
        .with_latent_step_embedding(true)
        .init(model, &device);

    let task = TaskFamily::Propagation { size: Some(3) }.generate(11)?;
    let prompt = task_prompt(&task, 3);
    let answer = task_answer(&task);
    println!(
        "task: {}, prompt tokens: {}, answer tokens: {}",
        task.name,
        prompt.len(),
        answer.len()
    );
    println!("ingest chunk size: {CHUNK_SIZE}");

    let prompt_memory = ingest(&wrapper, &prompt, None, CHUNK_SIZE, true)?;
    println!("\nafter prompt:");
    print_memory(&prompt_memory);

    // Freeze fast-weight writes to make the state distinction observable:
    // embeds and position_offsets change, while every contextual matrix stays
    // byte-for-byte the same. The unit suite checks that equality directly.
    let reasoned = wrapper.forward(
        &[Stage::Think(3)],
        Some(prompt_memory),
        ReasoningForwardOptions {
            update_latent_memory: false,
            ..Default::default()
        },
    )?;
    println!("\nafter three continuous thought steps (writes frozen):");
    print_memory(&reasoned.memory);

    // Teacher forcing an answer resumes memory writes and returns one logit
    // distribution per supplied answer position.
    let decoded = wrapper.forward(
        &[Stage::Tokens(ids(&answer, &device))],
        Some(reasoned.memory),
        Default::default(),
    )?;
    println!("\nafter answer teacher forcing:");
    print_memory(&decoded.memory);
    println!("answer logits shape: {:?}", decoded.logits.unwrap().dims());

    Ok(())
}

fn print_memory(memory: &bdh_cq_llm::Memory<Cpu>) {
    println!("  positions seen: {:?}", memory.position_offsets);
    println!("  latest embeddings: {:?}", memory.embeds.dims());
    for (depth, matrix) in memory.fast_weights.iter().enumerate() {
        println!(
            "  depth {depth} fast weights: {:?}",
            matrix.as_ref().map(Tensor::dims)
        );
    }
}
