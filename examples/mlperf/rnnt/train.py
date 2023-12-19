from pathlib import Path
from typing import List

from examples.mlperf.rnnt.config import load
from examples.mlperf.rnnt.data.dataset import AudioDataset
from examples.mlperf.rnnt.ops import ComposeOp, FilterbankOp
from examples.mlperf.rnnt.data.sampler import BucketingSampler
from examples.mlperf.rnnt.text.tokenizer import Tokenizer
from tinygrad import Tensor

class RNNTTrainer:
  def __init__(self, config_filepath:Path, data_dir:Path, manifest_names:List[str], batch_size:int = 128, num_buckets:int = 1):
    self.config = load(config_filepath)

    tokenizer = Tokenizer(self.config["tokenizer"]["labels"], sentpiece_model=self.config["tokenizer"]["sentpiece_model"])
    self.dataset = AudioDataset(
      data_dir,
      manifest_names,
      tokenizer,
      max_duration=self.config["input_train"]["audio_dataset"]["max_duration"],
      speed_perturbations=self.config["input_train"]["audio_dataset"]["speed_perturbation"]
    )
    self.sampler = BucketingSampler(self.dataset, num_buckets, batch_size)
    self.ops = ComposeOp([FilterbankOp(**self.config["input_train"]["filterbank_features"])])

    # TODO: make train ops here

  def train_step(self, x:Tensor, x_lens:Tensor, y:Tensor, y_lens:Tensor, epoch:int):
    import pdb; pdb.set_trace
    x, x_lens = self.ops(x.unsqueeze(0), x_lens.unsqueeze(0))
    import pdb; pdb.set_trace
