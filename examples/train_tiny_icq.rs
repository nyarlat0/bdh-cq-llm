//! A minimal end-to-end ARC-style optimization loop.
//!
//! This demonstrates mechanics, not convergence or a paper result. The first
//! command-line argument controls the number of updates (default: 5):
//!
//! ```console
//! cargo run --offline --example train_tiny_icq -- 10
//! ```

use bdh_cq_llm::{
    BdhConfig, ReasoningWrapperConfig,
    icq::{CLASS_WEIGHTS, train_loss},
    tasks::TaskFamily,
};
use burn::{
    backend::{Autodiff, NdArray},
    grad_clipping::GradientClippingConfig,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    tensor::backend::Backend,
};

type Inner = NdArray<f32>;
type Train = Autodiff<Inner>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let steps = std::env::args()
        .nth(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(5);
    let device = Default::default();
    Train::seed(&device, 17);

    // These dimensions are intentionally tiny. The upstream ARC experiment
    // uses D=384, depth=4, and H*Q=2048.
    let bdh = BdhConfig::new(14, 24)
        .with_depth(2)
        .with_heads(2)
        .with_dim_qk_heads(96)
        .init::<Train>(&device)?;
    let mut model = ReasoningWrapperConfig::new()
        .with_latent_step_embedding(true)
        .init(bdh, &device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.01)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    for step in 0..steps {
        // A fresh deterministic task per update. `train_loss` performs prompt
        // chunking, latent supervision, answer supervision, and the additional
        // masked prompt next-token loss.
        let task = TaskFamily::Propagation { size: Some(3) }.generate(step as u64)?;
        let loss = train_loss(&model, &task, 2, Some(CLASS_WEIGHTS.to_vec()), true, true)?;
        let scalar = loss.to_data().to_vec::<f32>()?[0];
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(1e-3, model, gradients);
        println!("step {step:03}: loss={scalar:.5}");
    }

    Ok(())
}
