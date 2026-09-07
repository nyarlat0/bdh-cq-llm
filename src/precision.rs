//! Selective projection precision, independent of the BDH recurrence.
//!
//! Master parameters, CQ, gates, normalization, losses and **all backward
//! operations** stay FP32 on CUDA. The opt-in `cuda-fp16` build rounds only
//! projection GEMM inputs to FP16; GEMM output is allocated in FP32. This is
//! not full AMP: no FP16 gradient buffers exist, so loss scaling is not needed.
//! It deliberately leaves unnormalized attention/CQ contractions in FP32.
//! Throughput and numerical quality must be measured on the target GPU.

#[cfg(feature = "cuda")]
use burn::tensor::ops::FloatTensorOps;
use burn::tensor::{Tensor, TensorPrimitive, backend::Backend};
use burn_autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};

/// Backend extension for non-broadcast projection GEMMs.
pub trait ProjectionBackend: Backend {
    /// Multiply equally ranked matrices with identical batch axes.
    fn projection_matmul(
        lhs: Self::FloatTensorPrimitive,
        rhs: Self::FloatTensorPrimitive,
    ) -> Self::FloatTensorPrimitive {
        Self::float_matmul(lhs, rhs)
    }
}

impl ProjectionBackend for burn::backend::NdArray<f32> {}
impl ProjectionBackend for burn::backend::Vulkan<f32, i32> {}

#[cfg(feature = "cuda")]
impl ProjectionBackend for burn::backend::Cuda<f32, i32> {
    fn projection_matmul(
        lhs: Self::FloatTensorPrimitive,
        rhs: Self::FloatTensorPrimitive,
    ) -> Self::FloatTensorPrimitive {
        #[cfg(feature = "cuda-fp16")]
        {
            use burn::tensor::{DType, FloatDType};
            use burn_cubecl::kernel::matmul::{MatmulStrategy, matmul};
            matmul(
                Self::float_cast(lhs, FloatDType::F16),
                Self::float_cast(rhs, FloatDType::F16),
                None,
                MatmulStrategy::default(),
                DType::F32,
            )
            .expect("CUDA FP16-input/FP32-output projection failed")
        }
        #[cfg(not(feature = "cuda-fp16"))]
        Self::float_matmul(lhs, rhs)
    }
}

impl<B: ProjectionBackend, C: CheckpointStrategy> ProjectionBackend for Autodiff<B, C> {
    fn projection_matmul(
        lhs: Self::FloatTensorPrimitive,
        rhs: Self::FloatTensorPrimitive,
    ) -> Self::FloatTensorPrimitive {
        #[derive(Debug)]
        struct Projection;

        impl<B: Backend> Backward<B, 2> for Projection {
            type State = (B::FloatTensorPrimitive, B::FloatTensorPrimitive);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (lhs, rhs) = ops.state;
                let grad = grads.consume::<B>(&ops.node);
                // FP32 straight-through derivative of the rounded forward.
                // Do not call projection_matmul here: small gradients must
                // never be cast to FP16 without an AMP loss-scaling protocol.
                if let Some(parent) = &ops.parents[0] {
                    grads.register::<B>(
                        parent.id,
                        B::float_matmul(grad.clone(), B::float_transpose(rhs)),
                    );
                }
                if let Some(parent) = &ops.parents[1] {
                    grads.register::<B>(parent.id, B::float_matmul(B::float_transpose(lhs), grad));
                }
            }
        }

        match Projection
            .prepare::<C>([lhs.node.clone(), rhs.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let state = (lhs.primitive.clone(), rhs.primitive.clone());
                prep.finish(state, B::projection_matmul(lhs.primitive, rhs.primitive))
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::projection_matmul(lhs.primitive, rhs.primitive))
            }
        }
    }
}

/// Projection GEMM. Explicit equal batch axes avoid implicit broadcast-gradient
/// reductions; shared weights must first fold B*N into their matrix row axis.
pub fn projection_matmul<B: ProjectionBackend, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
) -> Tensor<B, D> {
    assert!(D >= 2);
    assert_eq!(
        &lhs.dims()[..D - 2],
        &rhs.dims()[..D - 2],
        "projection batch axes must match"
    );
    Tensor::from_primitive(TensorPrimitive::Float(B::projection_matmul(
        lhs.into_primitive().tensor(),
        rhs.into_primitive().tensor(),
    )))
}

/// Bias-free D->O projection with one shared weight table, without repeating it
/// across the batch. All BDH large linear projections are bias-free.
pub fn linear<B: ProjectionBackend>(input: Tensor<B, 3>, weight: Tensor<B, 2>) -> Tensor<B, 3> {
    let [batch, sequence, dim] = input.dims();
    let output_dim = weight.dims()[1];
    projection_matmul(input.reshape([batch * sequence, dim]), weight)
        .reshape([batch, sequence, output_dim])
}
