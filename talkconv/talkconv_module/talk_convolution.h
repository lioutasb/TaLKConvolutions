#include <ciso646> // && -> and, || -> or etc.

namespace cpu {

}

namespace gpu {

void TaLKConvEncoder(at::Tensor & input_integrated, at::Tensor & offset_left, at::Tensor & offset_right, int max_left, int max_right, at::Tensor & output);
void TaLKConvDecoder(at::Tensor & input_integrated, at::Tensor & offset_left, int max_left, at::Tensor & output);

void TaLKConvEncoderGrad(at::Tensor & input_integrated, at::Tensor & offset_left, at::Tensor & offset_right, int max_left, int max_right, at::Tensor & output_grad, at::Tensor & input_grad, at::Tensor & offset_left_grad, at::Tensor & offset_right_grad);
void TaLKConvDecoderGrad(at::Tensor & input_integrated, at::Tensor & offset_left, int max_left, at::Tensor & output_grad, at::Tensor & input_grad, at::Tensor & offset_left_grad);

void TaLKConvDecoderInference(at::Tensor & input_integrated, at::Tensor & offset_left, int max_left, at::Tensor & output);

}
