import librosa
import math
import torch

from typing import Optional, Tuple, Union

from tinygrad import Tensor
from tinygrad.helpers import dtypes


def _normalize_batch(x:Tensor, x_lens:Tensor, normalize_type:str):
  if normalize_type == "per_feature":
    mean = Tensor.zeros(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
    std = Tensor.zeros(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)

    for i in range(x.shape[0]):
      mean[i, :] = x[i, :, :x_lens[i].item()].mean(axis=1)
      std[i, :] = x[i, :, :x_lens[i].item()].std(axis=1)

    return (x - mean.unsqueeze(2)) / (std.unsqueeze(2) + 1e-5)
  elif normalize_type == "all_features":
    mean = Tensor.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    std = Tensor.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    for i in range(x.shape[0]):
      mean[i] = x[i, :, :x_lens[i].item()].mean()
      std[i] = x[i, :, :x_lens[i].item()].std()

    return (x - mean.reshape(-1, 1, 1)) / (std.reshape(-1, 1, 1) + 1e-5)

  return x

def _stack_subsample_frames(x:Tensor, x_lens:Tensor, stacking:int, subsampling:int) -> Tuple[Tensor, Tensor]:
  seq = [x]
  for n in range(1, stacking):
    tmp = Tensor.zeros(x.shape, dtype=x.dtype, device=x.device)
    tmp[:, :, :-n] = x[:, :, n:]
    seq.append(tmp)

  x = Tensor.cat(*seq, dim=1)[:, :, ::subsampling]

  if subsampling > 1:
    x_lens = (x_lens.float() / subsampling).ceil().cast(dtypes.int)

    if x.shape[2] > x_lens.max().item():
      assert abs(x.shape[2] - x_lens.max().item()) <= 1
      x = x[:, :, :x_lens.max().item()]

  return x, x_lens

def _tilde(x:Tensor) -> Tensor:
  if x.dtype == dtypes.bool: return (1 - x).cast(dtypes.bool)
  return (x + 1) * -1  # this seems to be what the ~ operator does in pytorch for non bool


class FilterbankOp:
  def __init__(
    self,
    dither:float,
    sample_rate:int,
    window_size:float,
    window_stride:float,
    n_filt:int,
    normalize:str,
    preemph:Optional[float] = 0.97,
    lowfreq:float = 0,
    highfreq:Optional[float] = None,
    log:bool = True,
    n_fft:Optional[int] = None,
    window:Optional[str] = "hann"
  ):
    self.dither = dither
    self.preemph = preemph
    self.log = log
    self.normalize = normalize
    self.win_length = int(sample_rate * window_size)
    self.hop_length = int(sample_rate * window_stride)
    self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
    self.fb = Tensor(librosa.filters.mel(sr=sample_rate, n_fft=self.n_fft, n_mels=n_filt, fmin=lowfreq, fmax=highfreq), dtype=dtypes.float).unsqueeze(0)

    torch_windows = {
      'hann': torch.hann_window,
      'hamming': torch.hamming_window,
      'blackman': torch.blackman_window,
      'bartlett': torch.bartlett_window,
      'none': None,
    }
    window_fn = torch_windows.get(window, None)
    self.window = window_fn(self.win_length, periodic=False) if window_fn is not None else None

  def __call__(self, x:Tensor, x_lens:Tensor) -> Tuple[Tensor, Tensor]:
    Tensor.no_grad = True

    if self.dither > 0:
      x += self.dither * Tensor.randn(*x.shape, dtype=x.dtype, device=x.device)

    if self.preemph is not None:
      x = Tensor.cat(x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1], dim=1)

    x = self._stft(x)
    x_lens = (x_lens.float() / self.hop_length).ceil().cast(dtypes.int)
    x = x.pow(2).sum(-1)
    x = self.fb.cast(x.dtype) @ x

    if self.log: x = (x + 1e-20).log()

    x = _normalize_batch(x, x_lens, normalize_type=self.normalize)

    Tensor.no_grad = False

    return x, x_lens

  def _stft(self, x:Tensor) -> Tensor:
    stft = torch.stft(
      torch.tensor(x.numpy()),
      n_fft=self.n_fft,
      hop_length=self.hop_length,
      win_length=self.win_length,
      window=self.window,
      return_complex=False
    )
    return Tensor(stft.numpy())


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
    Tensor.no_grad = False

    sh = x.shape
    mask = Tensor.zeros(x.shape, dtype=dtypes.bool, device=x.device)

    for i in range(sh[0]):
      for _ in range(self.freq_masks):
        w = Tensor.randint((1,), low=self.min_freq, high=self.max_freq + 1).item()
        f0 = Tensor.randint((1,), low=0, high=max(1, sh[1] - w + 1))
        mask[i, f0:f0+w] = 1

      time_masks = self.time_masks
      if 0 < time_masks < 1.0: time_masks = int(round(x_lens[i].item() * time_masks))

      max_time = self.max_time
      if 0 < max_time < 1.0: max_time = int(round(x_lens[i].item() * max_time))

      for _ in range(time_masks):
        w = Tensor.randint((1,), low=self.min_time, high=max_time + 1).item()
        t0 = Tensor.randint((1,), low=0, high=max(1, sh[2] - w + 1))
        mask[i, :, t0:t0+w] = 1

    if self.noise_magnitude > 0:
      mean = Tensor.zeros(x.shape[0], x.shape[1], 1, device=x.device)
      std = Tensor.zeros(x.shape[0], x.shape[1], 1, device=x.device)
      for i in range(sh[0]):
        mean[i, :, 0] = x[i, :, :x_lens[i]].mean(axis=1)
        std[i, :, 0] = x[i, :, :x_lens[i]].mean(axis=1)

      std *= self.noise_magnitude
      # TODO: verify this
      noise = _tilde(mask).where(0, mean + Tensor.randn(x.shape, dtype=x.dtype, device=x.device) * std)
    else:
      noise = 0

    Tensor.no_grad = True

    return mask.where(0, x) + noise, x_lens
  

class FrameSplicingOp:
  def __init__(self, frame_stacking:int = 1, frame_subsampling:int = 1):
    self.frame_stacking = frame_stacking
    self.frame_subsampling = 1

  def __call__(self, x:Tensor, x_lens:Tensor) -> Tuple[Tensor, Tensor]:
    if self.frame_stacking > 1 or self.frame_subsampling > 1:
      x, x_lens = _stack_subsample_frames(x, x_lens, self.frame_stacking, self.frame_subsampling)
    return x, x_lens
  

class ComposeOp:
  def __init__(self, ops):
    self.ops = ops

  def __call__(self, x:Tensor, x_lens:Tensor) -> Tuple[Tensor, Tensor]:
    for op in self.ops: x, x_lens = op(x, x_lens)
    return x, x_lens
