#include <torch/extension.h>


at::Tensor talk_convolution_encoder_forward(
    at::Tensor input_integrated,
    at::Tensor offset_left,
    at::Tensor offset_right,
    int max_left,
    int max_right);

std::vector<at::Tensor> talk_convolution_encoder_backward(
    at::Tensor input_integrated,
    at::Tensor offset_left,
    at::Tensor offset_right,
    int max_left,
    int max_right,
    at::Tensor output_grad);

at::Tensor talk_convolution_decoder_forward(
    at::Tensor input_integrated,
    at::Tensor offset_left,
    int max_left);

std::vector<at::Tensor> talk_convolution_decoder_backward(
    at::Tensor input_integrated,
    at::Tensor offset_left,
    int max_left,
    at::Tensor output_grad);

at::Tensor talk_convolution_decoder_inference_forward(
    at::Tensor input_integrated,
    at::Tensor offset_left, 
    int max_left);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("talk_convolution_encoder_forward", &talk_convolution_encoder_forward, "TaLK Convolution Encoder Forward");
    m.def("talk_convolution_encoder_backward", &talk_convolution_encoder_backward, "TaLK Convolution Encoder Backward");
    m.def("talk_convolution_decoder_forward", &talk_convolution_decoder_forward, "TaLK Convolution Decoder Forward");
    m.def("talk_convolution_decoder_backward", &talk_convolution_decoder_backward, "TaLK Convolution Decoder Backward");
    m.def("talk_convolution_decoder_inference_forward", &talk_convolution_decoder_inference_forward, "TaLK Convolution Decoder Inference Forward");
}

