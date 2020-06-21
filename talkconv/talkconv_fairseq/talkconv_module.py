import torch
import torch.nn as nn
import torch.nn.functional as F


from fairseq import utils

from .talkconv_function import TaLKConvolutionEncoderFunction, TaLKConvolutionDecoderFunction

import talkconv_cuda

class TaLKConv(nn.Module):
  def __init__(self, hid_dim, offsets_dropout=0.0, decode=False, num_heads=1, min_len_left=1, min_len_right=1):
    super().__init__()

    self.hid_dim = hid_dim
    self.decode = decode

    self.num_heads = num_heads
    self.R = self.hid_dim // self.num_heads

    self.min_len_left = min_len_left
    self.min_len_right = min_len_right

    if not self.decode:
        self.offsets = nn.Linear(self.hid_dim, self.num_heads * 2, bias=True)
    else:
        self.offsets = nn.Linear(self.hid_dim, self.num_heads, bias=True)

    self.do = nn.Dropout(offsets_dropout)


  def forward(self, x, incremental_state=None, mask=None):

    _, B, C = x.size()
    H = self.num_heads
    R = C // H
    K = self.min_len_left + self.min_len_right + 1


    if incremental_state is not None:
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is None:
            input_buffer = x * (1/K)
        else:
            input_buffer = torch.cat([input_buffer, (x * (1/K)) + input_buffer[-1:]], dim=0)

        self._set_input_buffer(incremental_state, input_buffer)
        x_sum = input_buffer.view(-1, B*H, R)

        T = x.shape[0]
    else:
        T = x.shape[0]

        x_sum = torch.cumsum(x.view(T, B*H, R)*(1/K), 0)


    x_offsets = torch.sigmoid(self.offsets(x))
    x_offsets = self.do(x_offsets)

    if not self.decode:
        x_offset_left, x_offset_right = x_offsets[:,:,:H].contiguous().view(T, B*H), x_offsets[:,:,H:].contiguous().view(T, B*H)
    else:
        x_offset_left = x_offsets.view(T, B*H)


    if incremental_state is not None:
        x_output = talkconv_cuda.talk_convolution_decoder_inference_forward(x_sum, x_offset_left.squeeze(-1), self.min_len_left)
    else:
        if not self.decode:
            x_output = TaLKConvolutionEncoderFunction.apply(x_sum, x_offset_left.squeeze(-1), x_offset_right.squeeze(-1), self.min_len_left, self.min_len_right)
        else:
            x_output = TaLKConvolutionDecoderFunction.apply(x_sum, x_offset_left.squeeze(-1), self.min_len_left)


    x_output = x_output.view(T, B, C)

    return x_output

  def _get_input_buffer(self, incremental_state):
    return utils.get_incremental_state(self, incremental_state, 'input_buffer')

  def _set_input_buffer(self, incremental_state, new_buffer):
    return utils.set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

  def reorder_incremental_state(self, incremental_state, new_order):
    input_buffer = self._get_input_buffer(incremental_state)
    if input_buffer is not None:
      input_buffer = input_buffer.index_select(1, new_order)
      self._set_input_buffer(incremental_state, input_buffer)
