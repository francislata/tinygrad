import json
import librosa
import numpy as np
import soundfile as sf

from pathlib import Path
from typing import Optional

from tinygrad.helpers import DEBUG, dtypes
from tinygrad import Tensor


class AudioSegment:
  def __init__(self, filename:Path, target_sr:int = None, int_values:bool = False, offset:int = 0, duration:float = 0.0, trim:bool = False, trim_db:int = 60):
    with sf.SoundFile(filename, mode="r") as f:
      dtype = "int32" if int_values else "float32"
      sample_rate = f.samplerate
      if offset > 0: f.seek(int(offset * sample_rate))
      if duration > 0: samples = f.read(int(duration * sample_rate), dtype=dtype)
      else: samples = f.read(dtype=dtype)

    samples = samples.transpose()
    samples = self._convert_samples_to_float32(samples)

    if target_sr is not None and target_sr != sample_rate:
      samples = librosa.core.resample(samples, sample_rate, target_sr)
      sample_rate = target_sr
    
    if trim: samples, _ = librosa.effects.trim(samples, trim_db)

    self._samples = samples
    self._sample_rate = sample_rate
    if self._samples.ndim >= 2: self._samples = np.mean(self._samples, 1)

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

class AudioDataset:
  def __init__(self, data_dir:Path, manifest_names:str, min_duration:float = 0.1, max_duration:float = float("inf"), sample_rate:int = 16000, trim_silence:bool = False):
    self.data_dir = data_dir
    self.min_duration = min_duration
    self.max_duration = max_duration
    self.sample_rate = sample_rate
    self.trim_silence = trim_silence
    self.samples = []
    self.duration = 0.0
    self.duration_filtered = 0.0

    manifest_paths = [data_dir / Path(name) for name in manifest_names.split(",") if len(name) > 0]
    for path in manifest_paths: self._load_manifest(path)

  def __getitem__(self, index):
    s = self.samples[index]
    rn_index = np.random.randint(len(s["audio_filepath"]))
    duration = s["audio_duration"][rn_index] if "audio_duration" in s else 0
    offset = s.get("offset", 0)

    if DEBUG >= 1: print(f"audio filepath: {s['audio_filepath']}")

    segment = AudioSegment(s["audio_filepath"][rn_index], target_sr=self.sample_rate, offset=offset, duration=duration, trim=self.trim_silence)

    # TODO: apply pertubations?
    segment = Tensor(segment.samples)
    # TODO: need to tokenize "transcript" still
    return segment, Tensor(segment.shape[0], dtype=dtypes.int), Tensor(s["transcript"]), Tensor(len(s["transcript"]), dtype=dtypes.int)
    
  def __len__(self):
    return len(self.samples)
  
  def _load_manifest(self, path:Path):
    with path.open(mode="r", encoding="utf-8") as fp: j = json.load(fp)
    for i, s in enumerate(j):
      s_max_duration = s["original_duration"]

      s["duration"] = s.pop("original_duration")
      if self.max_duration is not None and not (self.min_duration <= s_max_duration <= self.max_duration):
        self.duration_filtered += s["duration"]

      tr = s.get("transcript", None) or self._load_transcript(Path(s["text_filepath"]))
      # TODO: should we tokenize here?

      files = s.pop("files")
      # TODO: do speed pertubation

      s["audio_duration"] = [f["duration"] for f in files]
      s["audio_filepath"] = [str(self.data_dir / Path(f["fname"])) for f in files]

      self.samples.append(s)
      self.duration += s["duration"]

    if DEBUG >= 1: print(f"audio duration: {self.duration:.2f} / audio filtered duration: {self.duration_filtered:.2f}")

  def _load_transcript(self, path:Path):
    with path.open(mode="r", encoding="utf-8") as fp: return fp.read().replace("\n", "")


class AudioPipeline:#
  def __init__(self, file_path:Path):
    self.file_path = file_path


class RnntIterator:
  def __init__(self, batch_size):
    self.batch_size = batch_size

  def __iter__(self):
    return self

class LibriTTSDataset:
  def __init__(self, root_dir:Path):
    self.root_dir = root_dir

  def __getitem__(self, idx):
    raise NotImplementedError
  

class BatchSampler:
  ...
  

class AudioDataLoader:
  def __init__(self, dataset:LibriTTSDataset):
    self.dataset = dataset

  def __next__(self):
    raise NotImplementedError
  
  def __iter__(self):
    return self

  