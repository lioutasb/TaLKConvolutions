import torch
import talkconv_cuda

try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class TaLKConvolutionEncoderFunction(torch.autograd.Function):
  @staticmethod
  @amp.float_function
  def forward(ctx, input_x, offset_left, offset_right, max_left, max_right):
    output = talkconv_cuda.talk_convolution_encoder_forward(input_x, offset_left, offset_right, max_left, max_right)

    ctx.save_for_backward(input_x, offset_left, offset_right)
    ctx.max_left = max_left
    ctx.max_right = max_right

    return output

  @staticmethod
  @amp.float_function
  def backward(ctx, grad_output):
    input_x, offset_left, offset_right = ctx.saved_tensors
    max_left = ctx.max_left
    max_right = ctx.max_right

    retval = talkconv_cuda.talk_convolution_encoder_backward(input_x, offset_left, offset_right, max_left, max_right, grad_output.contiguous())

    return tuple([retval[0], retval[1], retval[2], None, None])


class TaLKConvolutionDecoderFunction(torch.autograd.Function):
  @staticmethod
  @amp.float_function
  def forward(ctx, input_x, offset_left, max_left):
    output = talkconv_cuda.talk_convolution_decoder_forward(input_x, offset_left, max_left)

    ctx.save_for_backward(input_x, offset_left)
    ctx.max_left = max_left

    return output

  @staticmethod
  @amp.float_function
  def backward(ctx, grad_output):
    input_x, offset_left = ctx.saved_tensors
    max_left = ctx.max_left

    retval = talkconv_cuda.talk_convolution_decoder_backward(input_x, offset_left, max_left, grad_output.contiguous())

    return tuple([retval[0], retval[1], None])
