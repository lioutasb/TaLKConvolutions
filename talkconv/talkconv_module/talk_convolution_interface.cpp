/* Functions actually called from Python. Registered in torch module in `bind.cpp` */

#include <torch/extension.h>
#include <ATen/AccumulateType.h>
#include <TH/THGeneral.h>

#include "talk_convolution.h"

#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")

at::Tensor talk_convolution_encoder_forward(
    at::Tensor input_integrated,
    at::Tensor offset_left,
    at::Tensor offset_right,
    int max_left, int max_right) {

    input_integrated = input_integrated.contiguous(); 
    

    //const int length = input_integrated.size(0);
    //const int batchSize = input_integrated.size(1);
    //const int r_dim = input_integrated.size(2);

    auto output = at::zeros_like(input_integrated); //at::zeros({length, batchSize, r_dim}, input_integrated.options());

    if (input_integrated.is_cuda()) {
        gpu::TaLKConvEncoder(input_integrated, offset_left, offset_right, max_left, max_right, output);
    }

    return output;
}


std::vector<at::Tensor> talk_convolution_encoder_backward(
    at::Tensor input_integrated,
    at::Tensor offset_left,
    at::Tensor offset_right,
    int max_left, int max_right,
    at::Tensor output_grad) {

    //AT_CHECK(input_integrated.device() == offset_left.device(),"BoxConv1d: input and parameters are on different devices")

    input_integrated = input_integrated.contiguous(); // TODO support noncontiguous too
    //AT_CHECK(input_integrated.dim() == 3, "BoxConv1d: input must have 3 dimensions");
    //AT_CHECK(offsets.dim() == 3, "BoxConv1d: offsets must have 3 dimensions");
    //AT_CHECK(offsets.size(1) == input_integrated.size(1)-1, "BoxConv1d: offsets must have the same length as input");
    //AT_CHECK(offsets.size(2) == 2, "BoxConv1d: offsets must be two");

    //const int length = input_integrated.size(0);
    //const int batchSize = input_integrated.size(1);
    //const int r_dim = input_integrated.size(2);

    auto input_grad = at::zeros_like(input_integrated);
    auto offset_left_grad = at::zeros_like(offset_left);
    auto offset_right_grad = at::zeros_like(offset_right);

    if (input_integrated.is_cuda()) {
        gpu::TaLKConvEncoderGrad(input_integrated, offset_left, offset_right, max_left, max_right, output_grad, input_grad, offset_left_grad, offset_right_grad);
    } 
    // else {
    //     cpu::boxConvUpdateOutput(input_integrated, offsets, output);
    // }

    return {input_grad, offset_left_grad, offset_right_grad};
}

at::Tensor talk_convolution_decoder_forward(
    at::Tensor input_integrated,
    at::Tensor offset_left,
    int max_left) {

    //AT_CHECK(input_integrated.device() == offset_left.device(),"BoxConv1d: input and parameters are on different devices")

    input_integrated = input_integrated.contiguous(); // TODO support noncontiguous too
    //AT_CHECK(input_integrated.dim() == 3, "BoxConv1d: input must have 3 dimensions");
    //AT_CHECK(offsets.dim() == 3, "BoxConv1d: offsets must have 3 dimensions");
    //AT_CHECK(offsets.size(1) == input_integrated.size(1)-1, "BoxConv1d: offsets must have the same length as input");
    //AT_CHECK(offsets.size(2) == 2, "BoxConv1d: offsets must be two");

    //const int length = input_integrated.size(0);
    //const int batchSize = input_integrated.size(1);
    //const int r_dim = input_integrated.size(2);

    auto output = at::zeros_like(input_integrated); 

    if (input_integrated.is_cuda()) {
        gpu::TaLKConvDecoder(input_integrated, offset_left, max_left, output);
    } 
    // else {
    //     cpu::boxConvUpdateOutput(input_integrated, offsets, output);
    // }

    return output;
}

std::vector<at::Tensor> talk_convolution_decoder_backward(
    at::Tensor input_integrated,
    at::Tensor offset_left,
    int max_left,
    at::Tensor output_grad) {

    //AT_CHECK(input_integrated.device() == offset_left.device(),"BoxConv1d: input and parameters are on different devices")

    input_integrated = input_integrated.contiguous(); // TODO support noncontiguous too
    //AT_CHECK(input_integrated.dim() == 3, "BoxConv1d: input must have 3 dimensions");
    //AT_CHECK(offsets.dim() == 3, "BoxConv1d: offsets must have 3 dimensions");
    //AT_CHECK(offsets.size(1) == input_integrated.size(1)-1, "BoxConv1d: offsets must have the same length as input");
    //AT_CHECK(offsets.size(2) == 2, "BoxConv1d: offsets must be two");

    //const int length = input_integrated.size(0);
    //const int batchSize = input_integrated.size(1);
    //const int r_dim = input_integrated.size(2);

    auto input_grad = at::zeros_like(input_integrated);
    auto offset_left_grad = at::zeros_like(offset_left);

    if (input_integrated.is_cuda()) {
        gpu::TaLKConvDecoderGrad(input_integrated, offset_left, max_left, output_grad, input_grad, offset_left_grad);
    } 
    // else {
    //     cpu::boxConvUpdateOutput(input_integrated, offsets, output);
    // }

    return {input_grad, offset_left_grad};
}

at::Tensor talk_convolution_decoder_inference_forward(
    at::Tensor input_integrated,
    at::Tensor offset_left,
    int max_left) {

    //AT_CHECK(input_integrated.device() == offset_left.device(),"BoxConv1d: input and parameters are on different devices")

    input_integrated = input_integrated.contiguous(); // TODO support noncontiguous too
    //AT_CHECK(input_integrated.dim() == 3, "BoxConv1d: input must have 3 dimensions");
    //AT_CHECK(offsets.dim() == 3, "BoxConv1d: offsets must have 3 dimensions");
    //AT_CHECK(offsets.size(1) == input_integrated.size(1)-1, "BoxConv1d: offsets must have the same length as input");
    //AT_CHECK(offsets.size(2) == 2, "BoxConv1d: offsets must be two");

    const int length = offset_left.size(0);
    const int batchSize = input_integrated.size(1);
    const int r_dim = input_integrated.size(2);

    auto output = at::zeros({length, batchSize, r_dim}, input_integrated.options());

    if (input_integrated.is_cuda()) {
        gpu::TaLKConvDecoderInference(input_integrated, offset_left, max_left, output);
    } 
    // else {
    //     cpu::boxConvUpdateOutput(input_integrated, offsets, output);
    // }

    return output;
}
