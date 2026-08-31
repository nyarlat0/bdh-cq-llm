use bdh_cq_llm::{
    BdhConfig, BdhForwardOptions, ModelInput, ReasoningForwardOptions, ReasoningWrapperConfig,
    Stage, compute_attn_residual_depth_bias,
};
use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    tensor::{Int, Tensor, TensorData, Tolerance, backend::Backend},
};

type TestBackend = NdArray<f32>;
type TrainBackend = Autodiff<TestBackend>;

fn tiny_config() -> BdhConfig {
    BdhConfig::new(16, 32)
        .with_depth(2)
        .with_heads(2)
        .with_dim_qk_heads(128)
}

fn ids<B: Backend>(batch: usize, sequence: usize, device: &B::Device) -> Tensor<B, 2, Int> {
    let values = (0..batch * sequence)
        .map(|index| (index % 16) as i64)
        .collect();
    Tensor::from_data(TensorData::new(values, [batch, sequence]), device)
}

#[test]
fn core_model_has_fixed_size_per_depth_memory() {
    let device = Default::default();
    let model = tiny_config().init::<TestBackend>(&device).unwrap();
    let first = model
        .forward(
            ModelInput::TokenIds(ids(2, 7, &device)),
            None,
            Default::default(),
        )
        .unwrap();

    assert_eq!(first.logits.unwrap().dims(), [2, 7, 16]);
    assert_eq!(first.memory.tokens_seen, 7);
    assert_eq!(first.memory.fast_weights.len(), 2);
    assert!(
        first
            .memory
            .fast_weights
            .iter()
            .all(|state| state.as_ref().unwrap().dims() == [2, 2, 64, 32])
    );

    let second = model
        .forward(
            ModelInput::TokenIds(ids(2, 3, &device)),
            Some(first.memory),
            Default::default(),
        )
        .unwrap();
    assert_eq!(second.memory.tokens_seen, 10);
    assert_eq!(second.logits.unwrap().dims(), [2, 3, 16]);
}

#[test]
fn chunked_recurrence_matches_one_causal_pass() {
    let device = Default::default();
    TestBackend::seed(&device, 123);
    let model = tiny_config().init::<TestBackend>(&device).unwrap();
    let all_ids = ids(1, 8, &device);

    let whole = model
        .forward(
            ModelInput::TokenIds(all_ids.clone()),
            None,
            Default::default(),
        )
        .unwrap();
    let first = model
        .forward(
            ModelInput::TokenIds(all_ids.clone().slice([0..1, 0..5])),
            None,
            Default::default(),
        )
        .unwrap();
    let second = model
        .forward(
            ModelInput::TokenIds(all_ids.slice([0..1, 5..8])),
            Some(first.memory),
            Default::default(),
        )
        .unwrap();

    // The two contractions associate their multiplications differently, so
    // floating-point results need a small tolerance rather than bit equality.
    whole
        .logits
        .unwrap()
        .slice([0..1, 5..8, 0..16])
        .into_data()
        .assert_approx_eq::<f32>(
            &second.logits.unwrap().into_data(),
            Tolerance::absolute(1e-4),
        );
    for (whole_state, chunked_state) in whole
        .memory
        .fast_weights
        .into_iter()
        .zip(second.memory.fast_weights)
    {
        whole_state.unwrap().into_data().assert_approx_eq::<f32>(
            &chunked_state.unwrap().into_data(),
            Tolerance::absolute(1e-4),
        );
    }
}

#[test]
fn padded_tail_does_not_change_real_logits_or_cq_memory() {
    let device = Default::default();
    TestBackend::seed(&device, 456);
    let model = tiny_config().init::<TestBackend>(&device).unwrap();
    let real_ids = Tensor::from_data(TensorData::new(vec![4_i64, 5, 6], [1, 3]), &device);
    let mut padded_values = vec![0_i64; 16];
    padded_values[..3].copy_from_slice(&[4, 5, 6]);
    let padded_ids = Tensor::from_data(TensorData::new(padded_values, [1, 16]), &device);

    let compact = model
        .forward(ModelInput::TokenIds(real_ids), None, Default::default())
        .unwrap();
    let padded = model
        .forward(
            ModelInput::TokenIds(padded_ids),
            None,
            BdhForwardOptions {
                valid_sequence_length: Some(3),
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(compact.memory.tokens_seen, 3);
    assert_eq!(padded.memory.tokens_seen, 3);
    assert_eq!(padded.memory.embeds.dims(), [1, 16, 32]);
    padded
        .logits
        .unwrap()
        .slice([0..1, 0..3, 0..16])
        .into_data()
        .assert_approx_eq::<f32>(
            &compact.logits.unwrap().into_data(),
            Tolerance::absolute(1e-5),
        );
    for (compact_state, padded_state) in compact
        .memory
        .fast_weights
        .into_iter()
        .zip(padded.memory.fast_weights)
    {
        padded_state.unwrap().into_data().assert_approx_eq::<f32>(
            &compact_state.unwrap().into_data(),
            Tolerance::absolute(1e-5),
        );
    }
}

#[test]
fn wrapper_interleaves_tokens_and_latent_iterations() {
    let device = Default::default();
    let wrapper = ReasoningWrapperConfig::new()
        .init(tiny_config().init::<TestBackend>(&device).unwrap(), &device);
    let output = wrapper
        .forward(
            &[
                Stage::Tokens(ids(1, 5, &device)),
                Stage::Think(3),
                Stage::Tokens(ids(1, 4, &device)),
            ],
            None,
            Default::default(),
        )
        .unwrap();

    assert_eq!(output.logits.unwrap().dims(), [1, 4, 16]);
    assert_eq!(output.memory.tokens_seen, 5 + 3 + 4);
}

#[test]
fn frozen_latent_steps_preserve_fast_weights() {
    let device = Default::default();
    let wrapper = ReasoningWrapperConfig::new()
        .init(tiny_config().init::<TestBackend>(&device).unwrap(), &device);
    let base = wrapper
        .forward(
            &[Stage::Tokens(ids(1, 5, &device))],
            None,
            Default::default(),
        )
        .unwrap()
        .memory;
    let frozen = wrapper
        .forward(
            &[Stage::Think(3)],
            Some(base.clone()),
            ReasoningForwardOptions {
                update_latent_memory: false,
                ..Default::default()
            },
        )
        .unwrap()
        .memory;

    for (before, after) in base.fast_weights.iter().zip(&frozen.fast_weights) {
        let before = before.as_ref().unwrap().to_data().to_vec::<f32>().unwrap();
        let after = after.as_ref().unwrap().to_data().to_vec::<f32>().unwrap();
        assert_eq!(before, after);
    }
}

#[test]
fn latent_and_answer_loss_backpropagates() {
    let device = Default::default();
    let wrapper = ReasoningWrapperConfig::new()
        .with_latent_step_embedding(true)
        .init(
            tiny_config().init::<TrainBackend>(&device).unwrap(),
            &device,
        );
    let output = wrapper
        .forward(
            &[
                Stage::Tokens(ids(2, 5, &device)),
                Stage::Think(2),
                Stage::Tokens(ids(2, 4, &device)),
            ],
            None,
            ReasoningForwardOptions {
                compute_loss: true,
                ..Default::default()
            },
        )
        .unwrap();
    let loss = output.loss.unwrap();
    let value = loss.to_data().to_vec::<f32>().unwrap()[0];
    assert!(value.is_finite() && value >= 0.0);
    let _gradients = loss.backward();
}

#[test]
fn attention_residual_supports_recycling() {
    let device = Default::default();
    let config = tiny_config().with_attn_residual(true);
    let model = config.init::<TestBackend>(&device).unwrap();
    let first = model
        .forward(
            ModelInput::TokenIds(ids(1, 5, &device)),
            None,
            BdhForwardOptions {
                collect_per_pass_hiddens: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(first.per_pass_hiddens.len(), 2);

    let recycled = model
        .forward(
            ModelInput::TokenIds(ids(1, 5, &device)),
            None,
            BdhForwardOptions {
                attention_history: Some(first.per_pass_hiddens),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(recycled.logits.unwrap().dims(), [1, 5, 16]);
}

#[test]
fn depth_bias_matches_upstream_indexing_examples() {
    let device = Default::default();
    let schedule = Tensor::<TestBackend, 1>::from_floats([0.5, 1.0], &device);
    let bias = compute_attn_residual_depth_bias(13, schedule, 4, 3)
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(
        bias,
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0
        ]
    );

    let no_reasoning = compute_attn_residual_depth_bias(
        5,
        Tensor::<TestBackend, 1>::from_floats([0.5, 1.0], &device),
        4,
        0,
    );
    assert_eq!(
        no_reasoning.to_data().to_vec::<f32>().unwrap(),
        vec![0.0; 5]
    );
}

#[test]
fn architecture_v2_features_compose_and_backpropagate() {
    let device = Default::default();
    let model = BdhConfig::new(24, 32)
        .with_depth(3)
        .with_heads(4)
        .with_dim_qk_heads(256)
        .with_rotary_dim(16)
        .with_tie_embeddings(true)
        .with_attn_residual(true)
        .with_attn_residual_tied(true)
        .with_gated_neuron_state(true)
        .with_cq_memory_decay(true)
        .with_cq_memory_retention(0.99)
        .init::<TrainBackend>(&device)
        .unwrap();

    assert_eq!(model.rotary_features_per_head(), 16);
    assert!(model.has_tied_embeddings());
    assert!(model.has_cq_memory_decay());
    let output = model
        .forward(
            ModelInput::TokenIds(ids(2, 6, &device)),
            None,
            BdhForwardOptions {
                collect_per_pass_hiddens: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(output.logits.as_ref().unwrap().dims(), [2, 6, 24]);
    assert_eq!(output.per_pass_hiddens.len(), 3);
    assert!(
        output
            .memory
            .fast_weights
            .iter()
            .all(|state| state.as_ref().unwrap().dims() == [2, 4, 64, 32])
    );
    let loss = output.logits.unwrap().powf_scalar(2.0).mean();
    assert!(loss.to_data().to_vec::<f32>().unwrap()[0].is_finite());
    let _gradients = loss.backward();
}

#[test]
fn tying_embeddings_removes_the_separate_vocabulary_matrix() {
    let device = Default::default();
    let untied = tiny_config().init::<TestBackend>(&device).unwrap();
    let tied = tiny_config()
        .with_tie_embeddings(true)
        .init::<TestBackend>(&device)
        .unwrap();
    assert_eq!(untied.num_params() - tied.num_params(), 16 * 32);
    assert_eq!(
        tied.project_logits(Tensor::zeros([2, 5, 32], &device))
            .dims(),
        [2, 5, 16]
    );
}

#[test]
fn tied_logits_have_variance_preserving_initial_scale() {
    let device = Default::default();
    TestBackend::seed(&device, 999);
    let model = BdhConfig::new(4_096, 64)
        .with_depth(1)
        .with_heads(4)
        .with_dim_qk_heads(256)
        .with_tie_embeddings(true)
        .init::<TestBackend>(&device)
        .unwrap();
    let maximum = model
        .forward(
            ModelInput::TokenIds(ids(1, 8, &device)),
            None,
            Default::default(),
        )
        .unwrap()
        .logits
        .unwrap()
        .abs()
        .max()
        .to_data()
        .to_vec::<f32>()
        .unwrap()[0];
    assert!(maximum < 10.0, "initial tied logit magnitude {maximum}");
}
