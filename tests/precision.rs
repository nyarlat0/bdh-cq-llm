//! Projection extension tests. CUDA tests are explicit/ignored: CI and AMD
//! machines must never initialize a CUDA context just to run the test suite.
use bdh_cq_llm::precision::projection_matmul;
use burn::{
    backend::{Autodiff, NdArray},
    tensor::{Tensor, TensorData},
};

#[test]
fn v100_config_preserves_architecture_and_batch_sweep_budget() {
    use bdh_cq_llm::pretrain::{PretrainConfig, TrainingSchedule};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let original = PretrainConfig::from_path(root.join("configs/rx6700-v2.json")).unwrap();
    let config = PretrainConfig::from_path(root.join("configs/v100-v2.json")).unwrap();
    assert_eq!(original.model, config.model);
    assert_eq!(config.memory.chunks_per_detach, 2);
    let expected = TrainingSchedule::build(&config).unwrap();
    for batch in [4, 8, 16] {
        let mut candidate = config.clone();
        candidate.optimizer.micro_batch_size = batch;
        candidate.optimizer.gradient_accumulation = 64 / batch;
        let schedule = TrainingSchedule::build(&candidate).unwrap();
        assert_eq!(schedule.effective_tokens, expected.effective_tokens);
        assert_eq!(schedule.phase_one_tokens, expected.phase_one_tokens);
    }
}

#[test]
fn v100_size_sweep_configs_validate() {
    use bdh_cq_llm::pretrain::{PretrainConfig, TrainingSchedule};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let base = PretrainConfig::from_path(root.join("configs/v100-v2.json")).unwrap();
    let mut expected = None;
    for dim in [512, 640, 768, 1024] {
        for batch in [1, 2, 4, 8, 16, 32] {
            let mut cfg = base.clone();
            cfg.model.dim = dim;
            cfg.model.dim_qk_heads = 12 * dim;
            cfg.model.rotary_dim = 3 * dim / 4;
            cfg.schedule_batch_multiple = Some(32);
            cfg.optimizer.micro_batch_size = batch;
            cfg.optimizer.gradient_accumulation = 64 / batch;
            cfg.memory.stateful_after_tokens = 0;
            cfg.memory.memory_read_ramp_tokens = 0;
            cfg.validate().unwrap();
            let schedule = TrainingSchedule::build(&cfg).unwrap();
            let budget = (schedule.effective_tokens, schedule.phase_one_tokens);
            assert_eq!(*expected.get_or_insert(budget), budget);
        }
    }
}

#[test]
fn grouped_projection_matches_native_forward_and_both_gradients() {
    type B = Autodiff<NdArray<f32>>;
    let device = Default::default();
    let left = Tensor::<B, 3>::from_data(
        TensorData::new(
            (0..24).map(|i| i as f32 * 0.03 - 0.3).collect::<Vec<_>>(),
            [2, 3, 4],
        ),
        &device,
    )
    .require_grad();
    let right = Tensor::<B, 3>::from_data(
        TensorData::new(
            (0..40).map(|i| i as f32 * 0.02 - 0.2).collect::<Vec<_>>(),
            [2, 4, 5],
        ),
        &device,
    )
    .require_grad();
    let actual = projection_matmul(left.clone(), right.clone());
    let expected = left.clone().matmul(right.clone());
    let ga = actual.clone().powf_scalar(2.0).mean().backward();
    let ge = expected.clone().powf_scalar(2.0).mean().backward();
    for (a, b) in [
        (actual.into_data(), expected.into_data()),
        (
            left.grad(&ga).unwrap().into_data(),
            left.grad(&ge).unwrap().into_data(),
        ),
        (
            right.grad(&ga).unwrap().into_data(),
            right.grad(&ge).unwrap().into_data(),
        ),
    ] {
        for (x, y) in a
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .zip(b.to_vec::<f32>().unwrap())
        {
            assert!((x - y).abs() < 1e-6);
        }
    }
}

#[test]
#[should_panic(expected = "projection batch axes must match")]
fn projection_rejects_implicit_batch_broadcast() {
    let device = Default::default();
    projection_matmul(
        Tensor::<NdArray<f32>, 3>::zeros([2, 3, 4], &device),
        Tensor::zeros([1, 4, 5], &device),
    );
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use bdh_cq_llm::{BdhConfig, BdhForwardOptions, ModelInput};
    use burn::{
        backend::Cuda,
        module::{Module, ModuleVisitor, Param},
        optim::GradientsParams,
        tensor::{DType, Int},
    };
    type B = Autodiff<Cuda<f32, i32>>;

    #[test]
    #[ignore = "requires NVIDIA GPU and CUDA 12.x; run on V100"]
    fn cuda_grouped_projection_matches_fp32_reference() {
        let device = Default::default();
        let left = Tensor::<B, 3>::from_data(
            TensorData::new(
                (0..512)
                    .map(|i| ((i % 31) as f32 - 15.0) / 17.0)
                    .collect::<Vec<_>>(),
                [2, 16, 16],
            ),
            &device,
        )
        .require_grad();
        let right = Tensor::<B, 3>::from_data(
            TensorData::new(
                (0..512)
                    .map(|i| ((i % 23) as f32 - 11.0) / 19.0)
                    .collect::<Vec<_>>(),
                [2, 16, 16],
            ),
            &device,
        )
        .transpose()
        .require_grad();
        let actual = projection_matmul(left.clone(), right.clone());
        let reference = left.clone().matmul(right.clone());
        let ga = actual.clone().sum().backward();
        let gr = reference.clone().sum().backward();
        for (a, b) in actual
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .zip(reference.into_data().to_vec::<f32>().unwrap())
        {
            assert!((a - b).abs() < 0.01, "{a} vs {b}");
        }
        for (a, b) in [
            (left.grad(&ga).unwrap(), left.grad(&gr).unwrap()),
            (right.grad(&ga).unwrap(), right.grad(&gr).unwrap()),
        ] {
            for (x, y) in a
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .zip(b.into_data().to_vec::<f32>().unwrap())
            {
                assert!((x - y).abs() < 1e-5);
            }
        }
    }

    #[test]
    #[ignore = "requires NVIDIA GPU and CUDA 12.x; run on V100"]
    fn cuda_projection_keeps_large_outputs_and_small_gradients_in_fp32() {
        let device = Default::default();
        let x = Tensor::<B, 2>::ones([16, 16], &device)
            .mul_scalar(100.0)
            .require_grad();
        let w = Tensor::<B, 2>::ones([16, 16], &device)
            .mul_scalar(100.0)
            .require_grad();
        let y = projection_matmul(x.clone(), w.clone());
        assert_eq!(y.dtype(), DType::F32);
        // 160000 exceeds FP16's maximum finite value (65504).
        for value in y.to_data().to_vec::<f32>().unwrap() {
            assert!((value - 160000.0).abs() < 16.0);
        }
        let grads = (y.sum() * 1e-12).backward();
        for gradient in [x.grad(&grads).unwrap(), w.grad(&grads).unwrap()] {
            assert_eq!(gradient.dtype(), DType::F32);
            assert!(
                gradient
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .all(|g| g.is_finite() && *g > 1e-10 && *g < 1e-8)
            );
        }
    }

    struct Check<'a> {
        grads: &'a GradientsParams,
        count: usize,
    }
    impl ModuleVisitor<B> for Check<'_> {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
            assert_eq!(p.val().dtype(), DType::F32);
            if let Some(g) = self.grads.get::<Cuda<f32, i32>, D>(p.id) {
                assert_eq!(g.dtype(), DType::F32);
                assert!(
                    g.into_data()
                        .to_vec::<f32>()
                        .unwrap()
                        .iter()
                        .all(|v| v.is_finite())
                );
                self.count += 1;
            }
        }
    }

    #[test]
    #[ignore = "requires NVIDIA GPU and CUDA 12.x; run on V100"]
    fn cuda_v2_two_chunk_forward_backward() {
        let device = Default::default();
        let model = BdhConfig::new(128, 64)
            .with_depth(3)
            .with_heads(4)
            .with_dim_qk_heads(256)
            .with_rotary_dim(32)
            .with_tie_embeddings(true)
            .with_attn_residual(true)
            .with_attn_residual_heads(4)
            .with_gated_neuron_state(true)
            .with_gated_neuron_state_initial_update(0.2)
            .with_gated_neuron_state_initial_injection(0.05)
            .with_normalize_each_depth(true)
            .with_cq_memory_decay(true)
            .init::<B>(&device)
            .unwrap();
        let ids = || {
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(
                    (0..64).map(|i| (i % 127) as i32).collect::<Vec<_>>(),
                    [2, 32],
                ),
                &device,
            )
        };
        let first = model
            .forward(
                ModelInput::TokenIds(ids()),
                None,
                BdhForwardOptions::default(),
            )
            .unwrap();
        let second = model
            .forward(
                ModelInput::TokenIds(ids()),
                Some(first.memory),
                BdhForwardOptions::default(),
            )
            .unwrap();
        for memory in second.memory.fast_weights.iter().flatten() {
            assert_eq!(memory.dtype(), DType::F32);
            assert!(
                memory
                    .to_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .all(|v| v.is_finite())
            );
        }
        let loss = second.logits.unwrap().powf_scalar(2.0).mean();
        assert!(loss.to_data().to_vec::<f32>().unwrap()[0].is_finite());
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let mut check = Check {
            grads: &grads,
            count: 0,
        };
        model.visit(&mut check);
        assert!(check.count >= 8);
    }
}
