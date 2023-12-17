import librosa
import numpy as np
import random
import sox
import soundfile as sf

from pathlib import Path
from typing import Optional

class AudioSegment:
  def __init__(
    self,
    filename:Path,
    target_sr:int = None,
    int_values:bool = False,
    offset:int = 0,
    duration:float = 0.0,
    trim:bool = False,
    trim_db:int = 60
  ):
    with sf.SoundFile(filename, "r") as f:
      dtype = "int32" if int_values else "float32"
      sample_rate = f.samplerate
      if offset > 0: f.seek(int(offset * sample_rate))
      if duration > 0: samples = f.read(int(duration * sample_rate), dtype=dtype)
      else: samples = f.read(dtype=dtype)

    samples = samples.transpose()
    samples = self._convert_samples_to_float32(samples)

    if target_sr is not None and target_sr != sample_rate:
      samples = librosa.core.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
      sample_rate = target_sr
    
    if trim: samples, _ = librosa.effects.trim(samples, trim_db)

    self._samples = samples
    self._sample_rate = sample_rate
    if self._samples.ndim >= 2: self._samples = np.mean(self._samples, 1)

  def __eq__(self, other):
    if type(other) is not type(self): return False
    if self._sample_rate != other._sample_rate: return False
    if self._samples.shape != other._samples.shape: return False
    if np.any(self.samples != other._samples): return False
    return True
  
  def __ne__(self, other): return not self.__eq__(other)

  @staticmethod
  def _convert_samples_to_float32(samples):
    float32_samples = samples.astype("float32")
    if samples.dtype in np.sctypes["int"]:
      bits = np.iinfo(samples.dtype).bits
      float32_samples *= (1. / 2 ** (bits - 1))
    elif samples.dtype in np.sctypes["float"]: pass
    else: raise TypeError(f"unsupported sample type: {samples.dtype}.")
    return float32_samples
  
  @property
  def samples(self): return self._samples.copy()

  @property
  def sample_rate(self): return self._sample_rate

  @property
  def num_samples(self): return self._samples.shape[0]

  @property
  def duration(self): return self._samples.shape[0] / float(self._sample_rate)

  @property
  def rms_db(self): return 10 * np.log10(np.mean(self._samples ** 2))

  def gain_db(self, gain):
    self._samples *= 10. ** (gain / 20.)

  def pad(self, pad_size, symmetric:bool = False):
    self._samples = np.pad(self._samples, (pad_size if symmetric else 0, pad_size), mode="constant")

  def subsegment(self, start_time:Optional[float] = None, end_time:Optional[float] = None):
    start_time = 0.0 if start_time is None else start_time
    end_time = self.duration if end_time is None else end_time
    if start_time < 0.0: start_time = self.duration + start_time
    if end_time < 0.0: end_time = self.duration + end_time
    if start_time < 0.0:
      raise ValueError("The slice start position (%f s) is out of "
                       "bounds." % start_time)
    if end_time < 0.0:
      raise ValueError("The slice end position (%f s) is out of bounds." %
                       end_time)
    if start_time > end_time:
      raise ValueError("The slice start position (%f s) is later than "
                       "the end position (%f s)." % (start_time, end_time))
    if end_time > self.duration:
      raise ValueError("The slice end position (%f s) is out of bounds "
                       "(> %f s)" % (end_time, self.duration))
    start_sample = int(round(start_time * self._sample_rate))
    end_sample = int(round(end_time * self._sample_rate))
    self._samples = self._samples[start_sample:end_sample]
