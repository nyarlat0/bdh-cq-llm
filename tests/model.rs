use bdh_cq_llm::{
    BdhConfig, BdhForwardOptions, LatentWorkspace, ModelInput, MultiHeadAttentionResidual,
    ReasoningForwardOptions, ReasoningWrapperConfig, Stage, compute_attn_residual_depth_bias,
};
use burn::{
    backend::{Autodiff, NdArray},
    module::{Module, ModuleVisitor, Param},
    optim::GradientsParams,
    tensor::{Int, Tensor, TensorData, Tolerance, backend::Backend},
};

type TestBackend = NdArray<f32>;
type TrainBackend = Autodiff<TestBackend>;

struct FiniteGradientVisitor<'a> {
    gradients: &'a GradientsParams,
    checked: usize,
}

impl ModuleVisitor<TrainBackend> for FiniteGradientVisitor<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<TrainBackend, D>>) {
        if let Some(gradient) = self.gradients.get::<TestBackend, D>(param.id) {
            let values = gradient.into_data().to_vec::<f32>().unwrap();
            assert!(
                values.iter().all(|value| value.is_finite()),
                "parameter {:?} has a non-finite gradient",
                param.id
            );
            self.checked += 1;
        }
    }
}

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
    assert!(!model.normalizes_each_depth());
    assert!(model.base_state_update_probabilities().is_none());
    assert!(model.state_injection_strengths().is_none());
    let first = model
        .forward(
            ModelInput::TokenIds(ids(2, 7, &device)),
            None,
            Default::default(),
        )
        .unwrap();

    assert_eq!(first.logits.unwrap().dims(), [2, 7, 16]);
    assert_eq!(first.memory.position_offsets, vec![7, 7]);
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
    assert_eq!(second.memory.position_offsets, vec![10, 10]);
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

    assert_eq!(compact.memory.position_offsets, vec![3]);
    assert_eq!(padded.memory.position_offsets, vec![3]);
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
    assert_eq!(output.memory.position_offsets, vec![5 + 3 + 4]);
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
fn mhar_h1_and_h2_match_at_uniform_zero_query_initialization() {
    let device = Default::default();
    let single = MultiHeadAttentionResidual::<TestBackend>::new(32, 1, 1, 0, &device);
    let multi = MultiHeadAttentionResidual::<TestBackend>::new(32, 1, 2, 0, &device);

    assert_eq!(single.num_params(), multi.num_params());
    let first = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            (0..64).map(|value| value as f32 / 10.0).collect(),
            [1, 2, 32],
        ),
        &device,
    );
    let second = first.clone() * 0.5 + 1.0;
    let sources = vec![first, second];
    let single_read = single.forward([1, 2, 32], &sources, 0, 2, 1).unwrap();
    let multi_read = multi.forward([1, 2, 32], &sources, 0, 2, 1).unwrap();
    single_read
        .into_data()
        .assert_approx_eq::<f32>(&multi_read.into_data(), Tolerance::absolute(1e-5));
}

#[test]
fn zero_query_mhar_uniform_invariant_survives_per_depth_normalization() {
    let device = Default::default();
    let h1 = MultiHeadAttentionResidual::<TestBackend>::new(32, 1, 1, 0, &device);
    let h2 = MultiHeadAttentionResidual::<TestBackend>::new(32, 1, 2, 0, &device);
    let first = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            (0..64).map(|value| value as f32 / 13.0).collect(),
            [1, 2, 32],
        ),
        &device,
    );
    let sources = vec![first.clone(), first * -0.3 + 2.0];
    let normalize = |input: Tensor<TestBackend, 3>| {
        let centered = input.clone() - input.mean_dim(2);
        let variance = centered.clone().square().mean_dim(2);
        centered / (variance + 1e-5).sqrt()
    };
    let h1 = normalize(h1.forward([1, 2, 32], &sources, 0, 2, 1).unwrap());
    let h2 = normalize(h2.forward([1, 2, 32], &sources, 0, 2, 1).unwrap());
    h1.into_data()
        .assert_approx_eq::<f32>(&h2.into_data(), Tolerance::absolute(1e-5));
}

#[test]
fn learned_probability_parameters_start_at_requested_values() {
    let device = Default::default();
    let model = tiny_config()
        .with_gated_neuron_state(true)
        .with_gated_neuron_state_initial_update(0.2)
        .with_gated_neuron_state_initial_injection(0.05)
        .with_cq_memory_decay(true)
        .with_cq_memory_initial_rho(0.995)
        .init::<TestBackend>(&device)
        .unwrap();
    let base_update = model.base_state_update_probabilities().unwrap();
    assert_eq!(base_update.dims(), [2, 64]);
    for value in base_update.to_data().to_vec::<f32>().unwrap() {
        assert!((value - 0.2).abs() < 1e-6);
    }
    let injection = model.state_injection_strengths().unwrap();
    assert_eq!(injection.dims(), [2, 64]);
    for value in injection.to_data().to_vec::<f32>().unwrap() {
        assert!((value - 0.05).abs() < 1e-6);
    }
    let retention = model.cq_retention_probabilities().unwrap();
    assert_eq!(retention.dims(), [2, 64]);
    for value in retention.to_data().to_vec::<f32>().unwrap() {
        assert!((value - 0.995).abs() < 1e-6);
    }
}

#[test]
fn selective_memory_reset_preserves_other_batch_rows() {
    let device = Default::default();
    let model = tiny_config().init::<TestBackend>(&device).unwrap();
    let memory = model
        .forward(
            ModelInput::TokenIds(ids(2, 3, &device)),
            None,
            Default::default(),
        )
        .unwrap()
        .memory;
    let kept_embed = memory.embeds.clone().slice([1..2, 0..3, 0..32]);
    let kept_weights =
        memory.fast_weights[0]
            .as_ref()
            .unwrap()
            .clone()
            .slice([1..2, 0..2, 0..64, 0..32]);
    let reset = memory.reset_rows(&[true, false]).unwrap();

    assert_eq!(reset.position_offsets, vec![0, 3]);
    assert_eq!(
        reset
            .embeds
            .clone()
            .slice([0..1, 0..3, 0..32])
            .abs()
            .max()
            .into_scalar(),
        0.0
    );
    reset
        .embeds
        .clone()
        .slice([1..2, 0..3, 0..32])
        .into_data()
        .assert_approx_eq::<f32>(&kept_embed.into_data(), Tolerance::absolute(0.0));
    reset.fast_weights[0]
        .as_ref()
        .unwrap()
        .clone()
        .slice([1..2, 0..2, 0..64, 0..32])
        .into_data()
        .assert_approx_eq::<f32>(&kept_weights.into_data(), Tolerance::absolute(0.0));

    let continued = model
        .forward(
            ModelInput::TokenIds(ids(2, 2, &device)),
            Some(reset),
            Default::default(),
        )
        .unwrap();
    assert_eq!(continued.memory.position_offsets, vec![2, 5]);
}

#[test]
fn in_chunk_document_boundaries_match_independent_documents() {
    let device = Default::default();
    TestBackend::seed(&device, 1_337);
    let model = tiny_config()
        .with_attn_residual(true)
        .with_attn_residual_heads(2)
        .with_gated_neuron_state(true)
        .with_gated_neuron_state_initial_update(0.2)
        .with_gated_neuron_state_initial_injection(0.05)
        .with_normalize_each_depth(true)
        .with_cq_memory_decay(true)
        .with_cq_memory_initial_rho(0.995)
        .init::<TestBackend>(&device)
        .unwrap();

    let prefix = Tensor::from_data(TensorData::new(vec![1_i64, 2, 3], [1, 3]), &device);
    let old_memory = model
        .forward(ModelInput::TokenIds(prefix), None, Default::default())
        .unwrap()
        .memory;
    let chunk = Tensor::from_data(
        TensorData::new(vec![4_i64, 5, 6, 7, 8, 9, 10], [1, 7]),
        &device,
    );
    let combined = model
        .forward(
            ModelInput::TokenIds(chunk.clone()),
            Some(old_memory.clone()),
            BdhForwardOptions {
                document_starts: Some(vec![false, false, true, false, false, true, false]),
                ..Default::default()
            },
        )
        .unwrap();

    let before_boundary = model
        .forward(
            ModelInput::TokenIds(chunk.clone().slice([0..1, 0..2])),
            Some(old_memory),
            Default::default(),
        )
        .unwrap();
    let middle_document = model
        .forward(
            ModelInput::TokenIds(chunk.clone().slice([0..1, 2..5])),
            None,
            Default::default(),
        )
        .unwrap();
    let final_document = model
        .forward(
            ModelInput::TokenIds(chunk.slice([0..1, 5..7])),
            None,
            Default::default(),
        )
        .unwrap();
    let reference_logits = Tensor::cat(
        vec![
            before_boundary.logits.unwrap(),
            middle_document.logits.unwrap(),
            final_document.logits.as_ref().unwrap().clone(),
        ],
        1,
    );

    assert_eq!(combined.memory.position_offsets, vec![2]);
    combined
        .logits
        .unwrap()
        .into_data()
        .assert_approx_eq::<f32>(&reference_logits.into_data(), Tolerance::absolute(1e-4));
    for (combined_state, reference_state) in combined
        .memory
        .fast_weights
        .into_iter()
        .zip(final_document.memory.fast_weights)
    {
        combined_state.unwrap().into_data().assert_approx_eq::<f32>(
            &reference_state.unwrap().into_data(),
            Tolerance::absolute(1e-4),
        );
    }
}

#[test]
fn batched_tbptt_matches_independent_lanes_with_different_boundaries() {
    let device = Default::default();
    TestBackend::seed(&device, 7_331);
    let model = tiny_config()
        .with_attn_residual(true)
        .with_attn_residual_heads(2)
        .with_gated_neuron_state(true)
        .with_gated_neuron_state_initial_injection(0.05)
        .with_normalize_each_depth(true)
        .with_cq_memory_decay(true)
        .init::<TestBackend>(&device)
        .unwrap();

    let prefix_values = vec![1_i64, 2, 3, 4, 5, 6];
    let batched_prefix = Tensor::from_data(TensorData::new(prefix_values.clone(), [2, 3]), &device);
    let batched_memory = model
        .forward(
            ModelInput::TokenIds(batched_prefix),
            None,
            Default::default(),
        )
        .unwrap()
        .memory;
    let lane_memories = (0..2)
        .map(|row| {
            let prefix = Tensor::from_data(
                TensorData::new(prefix_values[row * 3..row * 3 + 3].to_vec(), [1, 3]),
                &device,
            );
            model
                .forward(ModelInput::TokenIds(prefix), None, Default::default())
                .unwrap()
                .memory
        })
        .collect::<Vec<_>>();

    let chunk_values = vec![
        7_i64, 8, 9, 10, 11, 12, // lane 0 resets before token 9
        13, 14, 15, 1, 2, 3, // lane 1 resets before token 2
    ];
    let starts = vec![
        false, false, true, false, false, false, false, false, false, false, true, false,
    ];
    let batched = model
        .forward(
            ModelInput::TokenIds(Tensor::from_data(
                TensorData::new(chunk_values.clone(), [2, 6]),
                &device,
            )),
            Some(batched_memory),
            BdhForwardOptions {
                document_starts: Some(starts.clone()),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(batched.memory.position_offsets, vec![4, 2]);

    let batched_logits = batched.logits.unwrap();
    for row in 0..2 {
        let lane = model
            .forward(
                ModelInput::TokenIds(Tensor::from_data(
                    TensorData::new(chunk_values[row * 6..row * 6 + 6].to_vec(), [1, 6]),
                    &device,
                )),
                Some(lane_memories[row].clone()),
                BdhForwardOptions {
                    document_starts: Some(starts[row * 6..row * 6 + 6].to_vec()),
                    ..Default::default()
                },
            )
            .unwrap();
        batched_logits
            .clone()
            .slice([row..row + 1, 0..6, 0..16])
            .into_data()
            .assert_approx_eq::<f32>(&lane.logits.unwrap().into_data(), Tolerance::absolute(1e-4));
        for (batched_state, lane_state) in batched
            .memory
            .fast_weights
            .iter()
            .zip(lane.memory.fast_weights)
        {
            batched_state
                .as_ref()
                .unwrap()
                .clone()
                .slice([row..row + 1, 0..2, 0..64, 0..32])
                .into_data()
                .assert_approx_eq::<f32>(
                    &lane_state.unwrap().into_data(),
                    Tolerance::absolute(1e-4),
                );
        }
    }
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
fn latent_iterations_carry_wide_workspace_until_the_chain_resets() {
    let device = Default::default();
    TestBackend::seed(&device, 8_080);
    let model = tiny_config()
        .with_gated_neuron_state(true)
        .with_gated_neuron_state_initial_update(0.2)
        .with_gated_neuron_state_initial_injection(0.05)
        .with_normalize_each_depth(true)
        .init::<TestBackend>(&device)
        .unwrap();
    let memory = model
        .forward(
            ModelInput::TokenIds(ids(1, 5, &device)),
            None,
            Default::default(),
        )
        .unwrap()
        .memory;
    let [_, sequence, dim] = memory.embeds.dims();
    let latent = memory
        .embeds
        .clone()
        .slice([0..1, sequence - 1..sequence, 0..dim]);
    let frozen = BdhForwardOptions {
        update_memory: false,
        return_logits: false,
        ..Default::default()
    };
    let (first, workspace) = model
        .forward_latent(latent, Some(memory), LatentWorkspace::new(), frozen.clone())
        .unwrap();
    assert!(workspace.has_neuron_state());
    assert_eq!(workspace.neuron_state_dims(), Some([1, 2, 1, 64]));

    let second_latent = first.memory.embeds.clone();
    let second_memory = first.memory;
    let (carried, _) = model
        .forward_latent(
            second_latent.clone(),
            Some(second_memory.clone()),
            workspace,
            frozen.clone(),
        )
        .unwrap();
    let (reset, _) = model
        .forward_latent(
            second_latent,
            Some(second_memory),
            LatentWorkspace::new(),
            frozen,
        )
        .unwrap();
    let carried = carried.memory.embeds.into_data().to_vec::<f32>().unwrap();
    let reset = reset.memory.embeds.into_data().to_vec::<f32>().unwrap();
    assert!(
        carried
            .iter()
            .zip(reset)
            .any(|(with_state, without_state)| (*with_state - without_state).abs() > 1e-5),
        "carrying the latent wide workspace had no observable effect"
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
        .with_attn_residual_heads(2)
        .with_gated_neuron_state(true)
        .with_gated_neuron_state_initial_update(0.2)
        .with_gated_neuron_state_initial_injection(0.05)
        .with_normalize_each_depth(true)
        .with_cq_memory_decay(true)
        .with_cq_memory_initial_rho(0.99)
        .init::<TrainBackend>(&device)
        .unwrap();

    assert_eq!(model.rotary_features_per_head(), 16);
    assert!(model.has_tied_embeddings());
    assert!(model.has_cq_memory_decay());
    assert!(model.normalizes_each_depth());
    let first = model
        .forward(
            ModelInput::TokenIds(ids(2, 6, &device)),
            None,
            BdhForwardOptions {
                collect_per_pass_hiddens: true,
                ..Default::default()
            },
        )
        .unwrap();
    let output = model
        .forward(
            ModelInput::TokenIds(ids(2, 6, &device)),
            Some(first.memory),
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
    let gradients = GradientsParams::from_grads(loss.backward(), &model);
    let mut visitor = FiniteGradientVisitor {
        gradients: &gradients,
        checked: 0,
    };
    model.visit(&mut visitor);
    assert!(visitor.checked > 0);
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

#[test]
fn production_v2_parameter_growth_is_only_per_neuron_linear_storage() {
    let device = Default::default();
    let common = BdhConfig::new(24_576, 512)
        .with_depth(8)
        .with_heads(8)
        .with_dim_qk_heads(6_144)
        .with_rotary_dim(64)
        .with_tie_embeddings(true)
        .with_attn_residual(true)
        .with_attn_residual_tied(true)
        .with_attn_residual_heads(8)
        .with_normalize_each_depth(true);
    let baseline = common.clone().init::<TestBackend>(&device).unwrap();
    let production = common
        .with_gated_neuron_state(true)
        .with_gated_neuron_state_initial_update(0.2)
        .with_gated_neuron_state_initial_injection(0.05)
        .with_cq_memory_decay(true)
        .with_cq_memory_initial_rho(0.995)
        .init::<TestBackend>(&device)
        .unwrap();

    assert_eq!(production.num_params(), 22_043_648);
    assert_eq!(
        production.num_params() - baseline.num_params(),
        512 * 8 + 3 * 6_144,
        "v2 state parameters must be D*H plus three H*Q arrays, never (H*Q)^2"
    );
}
