import numpy as np

from typing import Tuple, Union

from extra.datasets.librispeech import feature_extract, splice
from tinygrad import Tensor, dtypes


def _stack_subsample_frames(x:Tensor, x_lens:Tensor, stacking:int, subsampling:int) -> Tuple[Tensor, Tensor]:
  seq = [x]
  for n in range(1, stacking):
    tmp = Tensor.zeros(x.shape, dtype=x.dtype, device=x.device)
    tmp[:, :, :-n] = x[:, :, n:]
    seq.append(tmp)

  x = Tensor.cat(*seq, dim=1)[:, :, ::subsampling]

  if subsampling > 1:
    x_lens = (x_lens.float() / subsampling).ceil().cast(dtypes.int)
    x_lens_max = x_lens.max().unsqueeze(0).item()
    if x.shape[2] > x_lens_max:
      assert abs(x.shape[2] - x_lens_max) <= 1
      x = x[:, :, :x_lens_max]

  return x, x_lens

def _tilde(x:Tensor) -> Tensor:
  if x.dtype == dtypes.bool: return (1 - x).cast(dtypes.bool)
  return (x + 1) * -1  # this seems to be what the ~ operator does in pytorch for non bool


class FilterbankOp:
  def __call__(self, x:Tensor, x_lens:Tensor) -> Tuple[Tensor, Tensor]:
    x, x_lens = feature_extract(x.numpy(), x_lens.numpy(), apply_splicing=False)
    return Tensor(x.transpose(1, 2, 0)), Tensor(x_lens, dtype=dtypes.int)


class SpecAugmentOp:
  def __init__(
    self,
    freq_masks:int = 0,
    min_freq:int = 0,
    max_freq:Union[int, float] = 10,
    time_masks: Union[int, float] = 0,
    min_time:int = 0,
    max_time:Union[int, float] = 10,
    noise_magnitude:float = 0
  ):
    assert 0 <= min_freq <= max_freq, "invalid values for min_freq and/or max_freq"
    assert 0 <= min_time <= max_time, "invalid values for min_time and/or max_time"

    self.freq_masks = freq_masks
    self.min_freq = min_freq
    self.max_freq = max_freq
    self.time_masks = time_masks
    self.min_time = min_time
    self.max_time = max_time
    self.noise_magnitude = noise_magnitude

  def __call__(self, x:Tensor, x_lens:Tensor) -> Tuple[Tensor, Tensor]:
    sh = x.shape
    mask = np.zeros_like(x.numpy(), dtype=np.bool_)

    for i in range(sh[0]):
      for _ in range(self.freq_masks):
        w = Tensor.randint(1, low=self.min_freq, high=self.max_freq + 1).item()
        f0 = Tensor.randint(1, low=0, high=max(1, sh[1] - w + 1)).item()
        mask[i, f0:f0+w] = 1

      time_masks = self.time_masks
      if 0 < time_masks < 1.0: time_masks = int(round(x_lens[i].item() * time_masks))

      max_time = self.max_time
      if 0 < max_time < 1.0: max_time = int(round(x_lens[i].item() * max_time))

      for _ in range(time_masks):
        w = Tensor.randint(1, low=self.min_time, high=max_time + 1).item()
        t0 = Tensor.randint(1, low=0, high=max(1, sh[2] - w + 1)).item()
        mask[i, :, t0:t0+w] = 1

    mask = Tensor(mask, dtype=dtypes.bool)

    if self.noise_magnitude > 0:
      mean = Tensor.zeros(x.shape[0], x.shape[1], 1, device=x.device)
      std = Tensor.zeros(x.shape[0], x.shape[1], 1, device=x.device)
      for i in range(sh[0]):
        mean[i, :, 0] = x[i, :, :x_lens[i]].mean(axis=1)
        std[i, :, 0] = x[i, :, :x_lens[i]].mean(axis=1)

      std *= self.noise_magnitude
      # TODO: verify this
      noise = _tilde(mask).where(0, mean + Tensor.normal(x.shape, dtype=x.dtype, device=x.device) * std)
    else:
      noise = 0

    return mask.where(0, x) + noise, x_lens
  

class FrameSplicingOp:
  def __call__(self, x:Tensor, x_lens:Tensor) -> Tuple[Tensor, Tensor]:
    return Tensor(splice(x.numpy())), x_lens


class PermuteAudioOp:
  def __call__(self, x:Tensor, x_lens:Tensor) -> Tuple[Tensor, Tensor]:
    return x.permute(2, 0, 1), x_lens
  

class ComposeOp:
  def __init__(self, ops):
    self.ops = ops

  def __call__(self, x:Tensor, x_lens:Tensor) -> Tuple[Tensor, Tensor]:
    for op in self.ops: x, x_lens = op(x, x_lens)
    return x, x_lens
