#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <THC/THC.h>
#include "THC/THCDeviceUtils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <assert.h>

#include <cmath>

namespace gpu {

template <typename scalar_t>
__device__ float clamp(scalar_t x, scalar_t a, scalar_t b)
{
  return max(a, min(b, x));
}

template <
    typename scalar_t,
    typename std::enable_if<std::is_same<c10::Half, scalar_t>::value>::type* =
        nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(
    scalar_t* tensor,
    size_t index,
    const size_t numel,
    scalar_t value) {
#if (                         \
    (CUDA_VERSION < 10000) || \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  atomicAdd(
      reinterpret_cast<at::Half*>(tensor) + index,
      static_cast<at::Half>(value));
#else
  bool low_bit = (index % 2 == 0) &&
      (reinterpret_cast<std::uintptr_t>(tensor) % sizeof(__half2) == 0);

  if (low_bit && index < (numel - 1)) {
    __half2 value2;
    value2.x = value;
    value2.y = __int2half_rz(0);
    atomicAdd(reinterpret_cast<__half2*>(tensor) + index / 2, value2);

  } else if (!low_bit && index > 0) {
    __half2 value2;
    value2.x = __int2half_rz(0);
    value2.y = value;
    atomicAdd(reinterpret_cast<__half2*>(tensor) + index / 2, value2);

  } else {
    atomicAdd(
        reinterpret_cast<__half*>(tensor) + index, static_cast<__half>(value));
  }
#endif
}

template <
    typename scalar_t,
    typename std::enable_if<!std::is_same<c10::Half, scalar_t>::value>::type* =
        nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(
    scalar_t* tensor,
    size_t index,
    const size_t numel,
    scalar_t value) {
  atomicAdd(tensor + index, value);
}

template <class scalar_t>
__device__ __forceinline__ void fastAtomicAdd(
    scalar_t* tensor,
    size_t index,
    const size_t numel,
    scalar_t value,
    bool fast_atomics) {
  if (fast_atomics) {
    fastSpecializedAtomicAdd(tensor, index, numel, value);
  } else {
    atomicAdd(tensor + index, value);
  }
}

/**
input: [T, B, H, R]
offset_left: [T, B, H]
offset_right: [T, B, H]
output: [T, B, H, R]
**/
template <typename scalar_t>
__global__ void TaLKConvEncoderKernel(const at::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input, 
    const at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> offset_left,
    const at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> offset_right,
    const scalar_t __restrict__ max_left,
    const scalar_t __restrict__ max_right,
    at::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output){

    const int length = input.size(0);
    const int batchSize = input.size(1);
    const int r_dim = input.size(2);

    const int index = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int rIdx = index % r_dim;
    const int batchIdx = (index / r_dim) % batchSize;
    const int tokenIdx = (index / r_dim) / batchSize;


    if (batchIdx < batchSize and tokenIdx < length and rIdx < r_dim) {
        const scalar_t left_off = static_cast<scalar_t>(offset_left[tokenIdx][batchIdx]);
        const scalar_t right_off = static_cast<scalar_t>(offset_right[tokenIdx][batchIdx]);

        const scalar_t true_left_off = clamp(tokenIdx - left_off * max_left, static_cast<scalar_t>(0.0), static_cast<scalar_t>(length-1));
        const scalar_t true_right_off = clamp(tokenIdx + right_off * max_right, static_cast<scalar_t>(0.0), static_cast<scalar_t>(length-1));

        const int32_t ind_floor_left = clamp(static_cast<int32_t>(floor(true_left_off)), 0, length-1);
        const int32_t ind_ceil_left = clamp(static_cast<int32_t>(ceil(true_left_off)), 0, length-1);

        const int32_t ind_floor_right = clamp(static_cast<int32_t>(floor(true_right_off)), 0, length-1);
        const int32_t ind_ceil_right = clamp(static_cast<int32_t>(ceil(true_right_off)), 0, length-1);

        const scalar_t alpha_left = ind_ceil_left - true_left_off;
       	const scalar_t alpha_right = true_right_off - ind_floor_right;


        const scalar_t S_output = ((1.0 - alpha_right)*input[ind_floor_right][batchIdx][rIdx] + 
        	alpha_right*input[ind_ceil_right][batchIdx][rIdx]) - 
        	(alpha_left*((ind_floor_left-1 < 0)?static_cast<scalar_t>(0.0):input[ind_floor_left-1][batchIdx][rIdx]) + 
        	(1.0 - alpha_left)*((ind_ceil_left-1 < 0)?static_cast<scalar_t>(0.0):input[ind_ceil_left-1][batchIdx][rIdx]));

        output[tokenIdx][batchIdx][rIdx] = S_output;
    }
}

void TaLKConvEncoder(at::Tensor & input,
    at::Tensor & offset_left, at::Tensor & offset_right, 
    int max_left, int max_right,
    at::Tensor & output) {

    const int length = input.size(0);
    const int batchSize = input.size(1);
    const int r_dim = input.size(2);


    const dim3 blockSize(128);
    const dim3 gridSize((length*batchSize*r_dim + blockSize.x - 1) / blockSize.x);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.scalar_type(), "gpu::TaLKConvEncoder", ([&] {
        
        auto inputAcsr = input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto offsetLeftAcsr = offset_left.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
        auto offsetRightAcsr = offset_right.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
        auto outputAcsr = output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();

        scalar_t max_left_f = static_cast<scalar_t>(max_left);
        scalar_t max_right_f = static_cast<scalar_t>(max_right);

        TaLKConvEncoderKernel<scalar_t><<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>> (
                inputAcsr, offsetLeftAcsr, offsetRightAcsr, max_left_f, max_right_f, outputAcsr);

    }));

    AT_CUDA_CHECK(cudaGetLastError());
}


/**
input: [T, B, H, R]
offset_left: [T, B, H]
offset_right: [T, B, H]
output: [T, B, H, R]
**/
template <typename scalar_t>
__global__ void TaLKConvEncoderGradKernel(const at::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input, 
    const at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> offset_left,
    const at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> offset_right,
    const scalar_t __restrict__ max_left,
    const scalar_t __restrict__ max_right,
    const at::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output_grad,
    scalar_t* __restrict__ input_grad,
    scalar_t* __restrict__ offset_left_grad,
    scalar_t* __restrict__ offset_right_grad){

    const int length = input.size(0);
    const int batchSize = input.size(1);
    const int r_dim = input.size(2);

    const int index = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int rIdx = index % r_dim;
    const int batchIdx = (index / r_dim) % batchSize;
    const int tokenIdx = (index / r_dim) / batchSize;


    if (batchIdx < batchSize and tokenIdx < length and rIdx < r_dim) {
        const scalar_t left_off = static_cast<scalar_t>(offset_left[tokenIdx][batchIdx]);
        const scalar_t right_off = static_cast<scalar_t>(offset_right[tokenIdx][batchIdx]);

        const scalar_t true_left_off = clamp(tokenIdx - left_off * max_left, static_cast<scalar_t>(0.0), static_cast<scalar_t>(length-1));
        const scalar_t true_right_off = clamp(tokenIdx + right_off * max_right, static_cast<scalar_t>(0.0), static_cast<scalar_t>(length-1));

        const int32_t ind_floor_left = clamp(static_cast<int32_t>(floor(true_left_off)), 0, length-1);
        const int32_t ind_ceil_left = clamp(static_cast<int32_t>(ceil(true_left_off)), 0, length-1);

        const int32_t ind_floor_right = clamp(static_cast<int32_t>(floor(true_right_off)), 0, length-1);
        const int32_t ind_ceil_right = clamp(static_cast<int32_t>(ceil(true_right_off)), 0, length-1);

        const scalar_t alpha_left = ind_ceil_left - true_left_off;
        const scalar_t alpha_right = true_right_off - ind_floor_right;

        const scalar_t gradOutValue = output_grad[tokenIdx][batchIdx][rIdx];

        if (ind_floor_left-1 >= 0) {
            fastAtomicAdd(input_grad, (((ind_floor_left-1) * batchSize + batchIdx) * r_dim + rIdx), batchSize*length*r_dim, static_cast<scalar_t>(-alpha_left * gradOutValue), true);
        }

        if (ind_ceil_left-1 >= 0){
            fastAtomicAdd(input_grad, (((ind_ceil_left-1) * batchSize + batchIdx) * r_dim + rIdx), batchSize*length*r_dim, static_cast<scalar_t>(-(1.0 - alpha_left) * gradOutValue), true);
        }

        fastAtomicAdd(input_grad, ((ind_floor_right * batchSize + batchIdx) * r_dim + rIdx), batchSize*length*r_dim, static_cast<scalar_t>((1.0 - alpha_right) * gradOutValue), true);
        fastAtomicAdd(input_grad, ((ind_ceil_right * batchSize + batchIdx) * r_dim + rIdx), batchSize*length*r_dim, static_cast<scalar_t>(alpha_right * gradOutValue), true);


        const scalar_t gradOffset_left_floor = ((ind_floor_left-1 < 0)?static_cast<scalar_t>(0.0):input[ind_floor_left-1][batchIdx][rIdx]) * max_left;
        const scalar_t gradOffset_left_ceil = ((ind_ceil_left-1 < 0)?static_cast<scalar_t>(0.0):input[ind_ceil_left-1][batchIdx][rIdx]) * (-max_left);

        const scalar_t gradOffset_right_floor = input[ind_floor_right][batchIdx][rIdx] * (-max_right);
        const scalar_t gradOffset_right_ceil = input[ind_ceil_right][batchIdx][rIdx] * max_right;


        const scalar_t grad_Offset_left = gradOffset_left_floor + gradOffset_left_ceil;
        const scalar_t grad_Offset_right = gradOffset_right_floor + gradOffset_right_ceil;

        fastAtomicAdd(offset_left_grad, (tokenIdx * batchSize + batchIdx), batchSize*length, static_cast<scalar_t>(-grad_Offset_left * gradOutValue), true);
        fastAtomicAdd(offset_right_grad, (tokenIdx * batchSize + batchIdx), batchSize*length, static_cast<scalar_t>(grad_Offset_right * gradOutValue), true);
    }
}

void TaLKConvEncoderGrad(at::Tensor & input,
    at::Tensor & offset_left, at::Tensor & offset_right, 
    int max_left, int max_right,
    at::Tensor & output_grad, at::Tensor & input_grad,
    at::Tensor & offset_left_grad, at::Tensor & offset_right_grad) {

    const int length = input.size(0);
    const int batchSize = input.size(1);
    const int r_dim = input.size(2);


    const dim3 blockSize(128);
    const dim3 gridSize((length*batchSize*r_dim + blockSize.x - 1) / blockSize.x);


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_grad.scalar_type(), "gpu::TaLKConvEncoderGrad", ([&] {
        
        auto inputAcsr = input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto offsetLeftAcsr = offset_left.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
        auto offsetRightAcsr = offset_right.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();

        auto outputGradAcsr = output_grad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();

        scalar_t max_left_f = static_cast<scalar_t>(max_left);
        scalar_t max_right_f = static_cast<scalar_t>(max_right);

        TaLKConvEncoderGradKernel<scalar_t><<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>> (
                inputAcsr, offsetLeftAcsr, offsetRightAcsr, max_left_f, max_right_f, outputGradAcsr,
                input_grad.data<scalar_t>(), offset_left_grad.data<scalar_t>(), offset_right_grad.data<scalar_t>());

    }));

    AT_CUDA_CHECK(cudaGetLastError());
}



/**
input: [T, B, H, R]
offset_left: [T, B, H]
offset_right: [T, B, H]
output: [T, B, H, R]
**/
template <typename scalar_t>
__global__ void TaLKConvDecoderKernel(const at::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input, 
    const at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> offset_left,
    const scalar_t __restrict__ max_left,
    at::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output){

    const int length = input.size(0);
    const int batchSize = input.size(1);
    const int r_dim = input.size(2);

    const int index = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int rIdx = index % r_dim;
    const int batchIdx = (index / r_dim) % batchSize;
    const int tokenIdx = (index / r_dim) / batchSize;

    if (batchIdx < batchSize and tokenIdx < length and rIdx < r_dim) {
        const scalar_t left_off = static_cast<scalar_t>(offset_left[tokenIdx][batchIdx]);

        const scalar_t true_left_off = clamp(tokenIdx - left_off * max_left, static_cast<scalar_t>(0.0), static_cast<scalar_t>(length-1));

        const int32_t ind_floor_left = clamp(static_cast<int32_t>(floor(true_left_off)), 0, length-1);
        const int32_t ind_ceil_left = clamp(static_cast<int32_t>(ceil(true_left_off)), 0, length-1);

        const scalar_t alpha_left = ind_ceil_left - true_left_off;

        const scalar_t S_output = input[tokenIdx][batchIdx][rIdx] - 
            (alpha_left*((ind_floor_left-1 < 0)?static_cast<scalar_t>(0.0):input[ind_floor_left-1][batchIdx][rIdx]) + 
            (1.0 - alpha_left)*((ind_ceil_left-1 < 0)?static_cast<scalar_t>(0.0):input[ind_ceil_left-1][batchIdx][rIdx]));

        output[tokenIdx][batchIdx][rIdx] = S_output;
    }
}

void TaLKConvDecoder(at::Tensor & input,
    at::Tensor & offset_left, 
    int max_left,
    at::Tensor & output) {

    const int length = input.size(0);
    const int batchSize = input.size(1);
    const int r_dim = input.size(2);


    const dim3 blockSize(128);
    const dim3 gridSize((length*batchSize*r_dim + blockSize.x - 1) / blockSize.x);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.scalar_type(), "gpu::TaLKConvDecoder", ([&] {
        
        auto inputAcsr = input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto offsetLeftAcsr = offset_left.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
        auto outputAcsr = output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();

        scalar_t max_left_f = static_cast<scalar_t>(max_left);

        TaLKConvDecoderKernel<scalar_t><<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>> (
                inputAcsr, offsetLeftAcsr, max_left_f, outputAcsr);

    }));

    AT_CUDA_CHECK(cudaGetLastError());
}


/**
input: [T, B, H, R]
offset_left: [T, B, H]
offset_right: [T, B, H]
output: [T, B, H, R]
**/
template <typename scalar_t>
__global__ void TaLKConvDecoderGradKernel(const at::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input, 
    const at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> offset_left,
    const scalar_t __restrict__ max_left,
    const at::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output_grad,
    scalar_t* __restrict__ input_grad,
    scalar_t* __restrict__ offset_left_grad){

    const int length = input.size(0);
    const int batchSize = input.size(1);
    const int r_dim = input.size(2);

    const int index = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int rIdx = index % r_dim;
    const int batchIdx = (index / r_dim) % batchSize;
    const int tokenIdx = (index / r_dim) / batchSize;

    if (batchIdx < batchSize and tokenIdx < length and rIdx < r_dim) {
        const scalar_t left_off = static_cast<scalar_t>(offset_left[tokenIdx][batchIdx]);

        const scalar_t true_left_off = clamp(tokenIdx - left_off * max_left, static_cast<scalar_t>(0.0), static_cast<scalar_t>(length-1));

        const int32_t ind_floor_left = clamp(static_cast<int32_t>(floor(true_left_off)), 0, length-1);
        const int32_t ind_ceil_left = clamp(static_cast<int32_t>(ceil(true_left_off)), 0, length-1);

        const scalar_t alpha_left = ind_ceil_left - true_left_off;

        const scalar_t gradOutValue = output_grad[tokenIdx][batchIdx][rIdx];


        if (ind_floor_left-1 >= 0) {
            fastAtomicAdd(input_grad, (((ind_floor_left-1) * batchSize + batchIdx) * r_dim + rIdx), batchSize*length*r_dim, static_cast<scalar_t>(-alpha_left * gradOutValue), true);
        }

        if (ind_ceil_left-1 >= 0){
            fastAtomicAdd(input_grad, (((ind_ceil_left-1) * batchSize + batchIdx) * r_dim + rIdx), batchSize*length*r_dim, static_cast<scalar_t>(-(1.0 - alpha_left) * gradOutValue), true);
        }

        fastAtomicAdd(input_grad, ((tokenIdx * batchSize + batchIdx) * r_dim + rIdx), batchSize*length*r_dim, gradOutValue, true);


        const scalar_t gradOffset_left_floor = ((ind_floor_left-1 < 0)?static_cast<scalar_t>(0.0):input[ind_floor_left-1][batchIdx][rIdx]) * max_left;
        const scalar_t gradOffset_left_ceil = ((ind_ceil_left-1 < 0)?static_cast<scalar_t>(0.0):input[ind_ceil_left-1][batchIdx][rIdx]) * (-max_left);

        const scalar_t grad_Offset_left = gradOffset_left_floor + gradOffset_left_ceil;

        fastAtomicAdd(offset_left_grad, (tokenIdx * batchSize + batchIdx), batchSize*length, static_cast<scalar_t>(-grad_Offset_left * gradOutValue), true);
    }
}

void TaLKConvDecoderGrad(at::Tensor & input,
    at::Tensor & offset_left, 
    int max_left,
    at::Tensor & output_grad, at::Tensor & input_grad,
    at::Tensor & offset_left_grad) {

    const int length = input.size(0);
    const int batchSize = input.size(1);
    const int r_dim = input.size(2);


    const dim3 blockSize(128);
    const dim3 gridSize((length*batchSize*r_dim + blockSize.x - 1) / blockSize.x);


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_grad.scalar_type(), "gpu::TaLKConvDecoderGrad", ([&] {
        
        auto inputAcsr = input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto offsetLeftAcsr = offset_left.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();

        auto outputGradAcsr = output_grad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        
        scalar_t max_left_f = static_cast<scalar_t>(max_left);

        TaLKConvDecoderGradKernel<scalar_t><<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>> (
                inputAcsr, offsetLeftAcsr, max_left_f, outputGradAcsr,
                input_grad.data<scalar_t>(), offset_left_grad.data<scalar_t>());

    }));

    AT_CUDA_CHECK(cudaGetLastError());
}


/**
input: [T, B, H, R]
offset_left: [T, B, H]
offset_right: [T, B, H]
output: [T, B, H, R]
**/
template <typename scalar_t>
__global__ void TaLKConvDecoderInferenceKernel(const at::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input, 
    const at::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> offset_left,
    const scalar_t __restrict__ max_left,
    at::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output){

    const int length = offset_left.size(0);
    const int batchSize = input.size(1);
    const int r_dim = input.size(2);

    const int index = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int rIdx = index % r_dim;
    const int batchIdx = (index / r_dim) % batchSize;
    const int tokenIdx = (index / r_dim) / batchSize;

    if (batchIdx < batchSize and tokenIdx < length and rIdx < r_dim) {
        const int last_id = input.size(0)-1;

        const scalar_t left_off = static_cast<scalar_t>(offset_left[tokenIdx][batchIdx]);

        const scalar_t true_left_off = clamp(last_id - left_off * max_left, static_cast<scalar_t>(0.0), static_cast<scalar_t>(last_id)); // - max_left - static_cast<scalar_t>(1.0))

        const int32_t ind_floor_left = clamp(static_cast<int32_t>(floor(true_left_off)), 0, last_id);
        const int32_t ind_ceil_left = clamp(static_cast<int32_t>(ceil(true_left_off)), 0, last_id);

        const scalar_t alpha_left = ind_ceil_left - true_left_off;

        const scalar_t S_output = input[last_id][batchIdx][rIdx] - 
            (alpha_left*((ind_floor_left-1 < 0)?static_cast<scalar_t>(0.0):input[ind_floor_left-1][batchIdx][rIdx]) + 
            (1.0 - alpha_left)*((ind_ceil_left-1 < 0)?static_cast<scalar_t>(0.0):input[ind_ceil_left-1][batchIdx][rIdx]));

        output[tokenIdx][batchIdx][rIdx] = S_output;
    }
}

void TaLKConvDecoderInference(at::Tensor & input,
    at::Tensor & offset_left, 
    int max_left,
    at::Tensor & output) {

    const int length = offset_left.size(0);
    const int batchSize = input.size(1);
    const int r_dim = input.size(2);


    const dim3 blockSize(128);
    const dim3 gridSize((length*batchSize*r_dim + blockSize.x - 1) / blockSize.x);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.scalar_type(), "gpu::TaLKConvDecoderInference", ([&] {
        
        auto inputAcsr = input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto offsetLeftAcsr = offset_left.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();
        auto outputAcsr = output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();

        scalar_t max_left_f = static_cast<scalar_t>(max_left);

        TaLKConvDecoderInferenceKernel<scalar_t><<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>> (
                inputAcsr, offsetLeftAcsr, max_left_f, outputAcsr);

    }));

    AT_CUDA_CHECK(cudaGetLastError());
}


}
