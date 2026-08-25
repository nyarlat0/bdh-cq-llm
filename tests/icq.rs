use bdh_cq_llm::{
    BdhConfig, ReasoningWrapperConfig,
    icq::{
        CHUNK_SIZE, EOS, INPUT, OUTPUT, ROW, answer_length, cell_stats, decode_grid, encode_grid,
        encode_output, generate_answer, ingest, task_answer, task_prompt,
    },
    tasks::TaskFamily,
};
use burn::backend::NdArray;

type TestBackend = NdArray<f32>;

#[test]
fn every_synthetic_family_round_trips_through_the_codec() {
    for family in TaskFamily::all() {
        let task = family.with_size(match family.name() {
            "propagation" => 3,
            "copy" => 2,
            "order" => 4,
            "nesting" => 3,
            _ => unreachable!(),
        });
        let task = task.generate(0).unwrap();
        for example in &task.train {
            assert_eq!(
                decode_grid(&encode_grid(&example.input, INPUT)).unwrap(),
                example.input
            );
            let encoded = encode_output(&example.output);
            assert_eq!(encoded[0], OUTPUT);
            assert_eq!(encoded.last(), Some(&EOS));
            assert!(encoded.contains(&ROW));
            assert_eq!(decode_grid(&encoded).unwrap(), example.output);
        }
    }
}

#[test]
fn prompt_is_demos_then_query_and_answer_is_terminated() {
    let task = TaskFamily::Order { size: Some(4) }.generate(4).unwrap();
    let prompt = task_prompt(&task, 3);
    assert_eq!(prompt.iter().filter(|token| **token == INPUT).count(), 4);
    assert_ne!(prompt.last(), Some(&EOS));

    let answer = task_answer(&task);
    assert_eq!(answer.last(), Some(&EOS));
    assert_eq!(answer.len(), answer_length(&task));
}

#[test]
fn task_levels_and_cell_stats_follow_upstream_conventions() {
    let task = TaskFamily::Nesting { size: Some(3) }
        .at_level(1, 3, 3, 1)
        .unwrap();
    assert!(task.train.iter().all(|example| example.level <= 2));
    assert_eq!(task.test[0].level, 3);

    let exact = cell_stats(&task.test[0].output, &task.test[0].output);
    assert!(exact.dimensions_valid);
    assert_eq!(exact.correct, exact.total);
}

#[test]
fn ingest_and_generate_exercise_the_complete_protocol() {
    let device = Default::default();
    let model = BdhConfig::new(14, 16)
        .with_depth(1)
        .with_heads(1)
        .with_dim_qk_heads(32)
        .init::<TestBackend>(&device)
        .unwrap();
    let wrapper = ReasoningWrapperConfig::new().init(model, &device);
    let task = TaskFamily::Propagation { size: Some(3) }
        .generate(3)
        .unwrap();
    let prompt = task_prompt(&task, 3);
    let memory = ingest(&wrapper, &prompt, None, CHUNK_SIZE, true).unwrap();
    assert_eq!(memory.tokens_seen, prompt.len());

    let generated = generate_answer(&wrapper, &task, 2, Some(memory), true, true, 0.0).unwrap();
    assert!(generated.len() <= answer_length(&task));
    assert!(generated.iter().all(|token| *token < 14));
}
