import json
import numpy as np
from tqdm import tqdm

from pathlib import Path
from typing import Optional, Dict, Tuple

from examples.mlperf.rnnt.data.audio import AudioSegment
from examples.mlperf.rnnt.text.tokenizer import Tokenizer
from examples.mlperf.rnnt.text import clean_text, punctuation_map
from tinygrad.helpers import DEBUG, dtypes
from tinygrad import Tensor


def normalize_string(s, charset, punct_map):
  """Normalizes string.

  Example:
    'call me at 8:00 pm!' -> 'call me at eight zero pm'
  """
  charset = set(charset)
  try:
    text = clean_text(s, ["english_cleaners"], punct_map).strip()
    return ''.join([tok for tok in text if all(t in charset for t in tok)])
  except:
    print(f"WARNING: Normalizing failed: {s}")
    return None


class AudioDataset:
  def __init__(
    self,
    data_dir:Path,
    manifest_names:str,
    tokenizer:Tokenizer,
    normalize_transcripts:bool = True,
    min_duration:float = 0.1,
    max_duration:float = float("inf"),
    sample_rate:int = 16000,
    trim_silence:bool = False,
    speed_perturbations:Optional[Dict] = None,
    ignore_offline_speed_perturbation:bool = False
  ):
    self.data_dir = data_dir
    self.tokenizer = tokenizer
    self.normalize_transcripts = normalize_transcripts
    self.min_duration = min_duration
    self.max_duration = max_duration
    self.sample_rate = sample_rate
    self.speed_perturbations = speed_perturbations
    self.trim_silence = trim_silence
    self.ignore_offline_speed_perturbation = ignore_offline_speed_perturbation
    self.samples = []
    self.duration = 0.0
    self.duration_filtered = 0.0
    self.punctuation_map = punctuation_map(self.tokenizer.charset)

    manifest_paths = [data_dir / Path(name) for name in manifest_names.split(",") if len(name) > 0]
    for path in manifest_paths: self._load_manifest(path)

  def __getitem__(self, index):
    s = self.samples[index]
    rn_index = np.random.randint(len(s["audio_filepath"]))
    duration = s["audio_duration"][rn_index] if "audio_duration" in s else 0
    offset = s.get("offset", 0)
    if DEBUG >= 2: print(f"audio: {s['audio_filepath']}")
    if self.speed_perturbations is not None:
      if DEBUG >= 2: print("applying speed perturbation")
      speed_perturbation_coeffs = Tensor.uniform(
        (1,),
        low=self.speed_perturbations["min_rate"],
        high=self.speed_perturbations["max_rate"]
      )
      resample_coeffs = speed_perturbation_coeffs.item() * self.sample_rate
    else:
      if DEBUG >= 2: print("no speed perturbation")
      resample_coeffs = self.sample_rate
    segment = AudioSegment(s["audio_filepath"][rn_index], target_sr=resample_coeffs, offset=offset, duration=duration, trim=self.trim_silence)
    return segment.samples, segment.num_samples, s["transcript"], len(s["transcript"])
    
  def __len__(self): return len(self.samples)
  
  def _load_manifest(self, path:Path):
    if DEBUG >= 1: print(f"loading manifest file: {path}")

    with path.open(mode="r", encoding="utf-8") as fp: j = json.load(fp)
    for s in tqdm(j, desc="loading manifest"):
      s_max_duration = s["original_duration"]

      s["duration"] = s.pop("original_duration")
      if self.max_duration is not None and not (self.min_duration <= s_max_duration <= self.max_duration):
        self.duration_filtered += s["duration"]

      tr = s.get("transcript", None) or self._load_transcript(Path(s["text_filepath"]))

      if not isinstance(tr, str):
        if DEBUG >= 1: print("skipping sample as transcript is not str")
        self.duration_filtered += s["duration"]

      if self.normalize_transcripts: tr = normalize_string(tr, self.tokenizer.charset, self.punctuation_map)

      s["transcript"] = self.tokenizer.tokenize(tr)

      files = s.pop("files")
      if self.ignore_offline_speed_perturbation:
        files = [f for f in files if f["speed"] == 1.0]

      s["audio_duration"] = [f["duration"] for f in files]
      s["audio_filepath"] = [str(self.data_dir / Path(f["fname"])) for f in files]

      self.samples.append(s)
      self.duration += s["duration"]

    if DEBUG >= 2: print(f"audio duration: {self.duration:.2f} / audio filtered duration: {self.duration_filtered:.2f}")

  def _load_transcript(self, path:Path):
    with path.open(mode="r", encoding="utf-8") as fp: return fp.read().replace("\n", "")

  def _prepare_item(self, s:Dict) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    rn_index = np.random.randint(len(s["audio_filepath"]))
    duration = s["audio_duration"][rn_index] if "audio_duration" in s else 0
    offset = s.get("offset", 0)
    if DEBUG >= 2: print(f"audio: {s['audio_filepath']}")
    if self.speed_perturbations is not None:
      if DEBUG >= 2: print("applying speed perturbation")
      speed_perturbation_coeffs = Tensor.uniform(
        (1,),
        low=self.speed_perturbations["min_rate"],
        high=self.speed_perturbations["max_rate"]
      )
      resample_coeffs = speed_perturbation_coeffs.item() * self.sample_rate
    else:
      if DEBUG >= 2: print("no speed perturbation")
      resample_coeffs = self.sample_rate
    segment = AudioSegment(s["audio_filepath"][rn_index], target_sr=resample_coeffs, offset=offset, duration=duration, trim=self.trim_silence)
    segment = Tensor(segment.samples)
    return segment, Tensor(segment.shape[0], dtype=dtypes.int), Tensor(s["transcript"]), Tensor(len(s["transcript"]), dtype=dtypes.int)
