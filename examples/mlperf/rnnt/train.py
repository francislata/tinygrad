from pathlib import Path
from typing import List

from examples.mlperf.rnnt.config import load
from examples.mlperf.rnnt.data.dataset import AudioDataset
from examples.mlperf.rnnt.ops import ComposeOp, FilterbankOp, SpecAugmentOp, FrameSplicingOp, PermuteAudioOp
from examples.mlperf.rnnt.data.sampler import BucketingSampler
from examples.mlperf.rnnt.loss import TransducerLoss
from examples.mlperf.rnnt.text.tokenizer import Tokenizer
from extra.models.rnnt import RNNT
from tinygrad import Tensor

class RNNTTrainer:
  def __init__(self, config_filepath:Path, data_dir:Path, manifest_names:List[str], batch_size:int = 128, num_buckets:int = 1):
    self.config = load(config_filepath)

    self.tokenizer = Tokenizer(self.config["tokenizer"]["labels"], sentpiece_model=self.config["tokenizer"]["sentpiece_model"])
    self.dataset = AudioDataset(
      data_dir,
      manifest_names,
      self.tokenizer,
      max_duration=self.config["input_train"]["audio_dataset"]["max_duration"],
      speed_perturbations=self.config["input_train"]["audio_dataset"]["speed_perturbation"]
    )
    self.sampler = BucketingSampler(self.dataset, num_buckets, batch_size)
    self.train_ops = ComposeOp([
      FilterbankOp(),
      SpecAugmentOp(**self.config["input_train"]["spec_augment"]),
      FrameSplicingOp(),
      PermuteAudioOp()
    ])
    self.model = RNNT(vocab_size=self.tokenizer.num_labels + 1, pred_hidden_size=512) # TODO: take RNNT config from config yaml
    self.loss_fn = TransducerLoss(self.tokenizer.num_labels)
    # TODO: setup optimizer

  def train_step(self, x:Tensor, x_lens:Tensor, y:Tensor, y_lens:Tensor):
    x, x_lens = self.train_ops(x, x_lens)
    y_, y_lens_ = self.model(x, y, x_lens=x_lens)
    
    # TODO: reset optimizer
    loss = self.loss_fn(y_, y_lens_, y, y_lens).realize()
    loss.backward()
    # TODO: optimizer step
