//! Small executable counterpart to `examples/architecture_walkthrough.rs`.
//!
//! `cargo run` deliberately uses tiny dimensions. The upstream default has
//! 32,768 Q/K features and is meant for serious training hardware.

use bdh_cq_llm::{BdhConfig, ModelInput};
use burn::{
    backend::NdArray,
    tensor::{Int, Tensor, TensorData, backend::Backend},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    type Backend = NdArray<f32>;

    let device = Default::default();
    Backend::seed(&device, 42);

    let config = BdhConfig::new(14, 64)
        .with_depth(2)
        .with_heads(4)
        .with_dim_qk_heads(512);
    let model = config.init::<Backend>(&device)?;

    // [batch=1, sequence=6]. In the ARC codec, 11 is <input> and 10 is
    // <row>; here the values merely demonstrate a normal token pass.
    let ids =
        Tensor::<Backend, 2, Int>::from_data(TensorData::from([[11_i64, 1, 2, 10, 3, 4]]), &device);
    let output = model.forward(ModelInput::TokenIds(ids), None, Default::default())?;

    println!("logits: {:?}", output.logits.unwrap().dims());
    println!("tokens seen: {}", output.memory.tokens_seen);
    println!(
        "fast-weight matrices: {} layers of {:?}",
        output.memory.fast_weights.len(),
        output.memory.fast_weights[0].as_ref().unwrap().dims()
    );
    println!("\nRead docs/architecture.md, then src/model.rs.");

    Ok(())
}
