//! Partial rotary positional embedding used by the public reconstruction.
//!
//! The PyTorch source constructs `RotaryEmbedding(dim_qk / 2)`, so its default
//! rotates only the first half of every Q/K head. The Rust configuration may
//! choose a narrower even prefix. This module mirrors lucidrains' pairwise
//! `(x0, x1) -> (-x1, x0)` convention and leaves the remaining features alone.

use burn::tensor::{Tensor, backend::Backend};

/// Apply RoPE to `[batch, heads, sequence, qk_per_head]`.
pub(crate) fn apply_rotary<B: Backend>(
    input: Tensor<B, 4>,
    position_ids: &[usize],
    rotary_dim: usize,
) -> Tensor<B, 4> {
    let [batch, heads, sequence, qk_dim] = input.dims();
    assert_eq!(
        position_ids.len(),
        batch * sequence,
        "RoPE needs one explicit position id per batch/sequence element"
    );
    debug_assert!(rotary_dim <= qk_dim);
    debug_assert_eq!(rotary_dim % 2, 0);

    let device = input.device();
    let pairs = rotary_dim / 2;

    // rotary-embedding-torch uses
    // inv_freq[p] = 1 / 10000^(2p / rotary_dim), then repeats each
    // frequency for the two coordinates of a pair.
    let mut phases = Vec::with_capacity(batch * sequence * rotary_dim);
    for position in position_ids {
        for pair in 0..pairs {
            let inv_freq = 1.0_f32 / 10_000.0_f32.powf((2 * pair) as f32 / rotary_dim as f32);
            let phase = *position as f32 * inv_freq;
            phases.push(phase);
            phases.push(phase);
        }
    }

    let phase = Tensor::<B, 1>::from_floats(phases.as_slice(), &device)
        .reshape([batch, 1, sequence, rotary_dim]);
    let cos = phase.clone().cos();
    let sin = phase.sin();

    let middle = input
        .clone()
        .slice([0..batch, 0..heads, 0..sequence, 0..rotary_dim]);
    let paired = middle.clone().reshape([batch, heads, sequence, pairs, 2]);
    let first = paired
        .clone()
        .slice([0..batch, 0..heads, 0..sequence, 0..pairs, 0..1])
        .squeeze_dim::<4>(4);
    let second = paired
        .slice([0..batch, 0..heads, 0..sequence, 0..pairs, 1..2])
        .squeeze_dim::<4>(4);
    let rotated_half =
        Tensor::stack::<5>(vec![-second, first], 4).reshape([batch, heads, sequence, rotary_dim]);
    let transformed = middle * cos + rotated_half * sin;

    if rotary_dim == qk_dim {
        transformed
    } else {
        let untouched = input.slice([0..batch, 0..heads, 0..sequence, rotary_dim..qk_dim]);
        Tensor::cat(vec![transformed, untouched], 3)
    }
}

#[cfg(test)]
mod tests {
    use burn::{backend::NdArray, tensor::TensorData};

    use super::*;

    type TestBackend = NdArray<f32>;

    #[test]
    fn position_zero_is_the_identity() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 4>::from_data(
            TensorData::new((0..8).map(|value| value as f32).collect(), [1, 1, 1, 8]),
            &device,
        );
        assert_eq!(
            apply_rotary(input.clone(), &[0], 4)
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            input.into_data().to_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn partial_rope_never_changes_the_trailing_features() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 4>::from_data(
            TensorData::new((0..16).map(|value| value as f32).collect(), [1, 1, 2, 8]),
            &device,
        );
        let rotated = apply_rotary(input.clone(), &[5, 6], 4);
        assert_eq!(
            rotated
                .slice([0..1, 0..1, 0..2, 4..8])
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            input
                .slice([0..1, 0..1, 0..2, 4..8])
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn every_batch_row_uses_its_own_position_offset() {
        let device = Default::default();
        let row = Tensor::<TestBackend, 4>::from_data(
            TensorData::new((0..16).map(|value| value as f32).collect(), [1, 1, 2, 8]),
            &device,
        );
        let batched = apply_rotary(
            Tensor::cat(vec![row.clone(), row.clone()], 0),
            &[3, 4, 11, 12],
            4,
        );
        let separate = Tensor::cat(
            vec![
                apply_rotary(row.clone(), &[3, 4], 4),
                apply_rotary(row, &[11, 12], 4),
            ],
            0,
        );
        assert_eq!(
            batched.into_data().to_vec::<f32>().unwrap(),
            separate.into_data().to_vec::<f32>().unwrap()
        );
    }
}
