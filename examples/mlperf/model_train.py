from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv

def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  # TODO: Unet3d
  pass

def train_rnnt():
  from pathlib import Path

  from examples.mlperf.rnnt.data.dataloader import audio_dataloader
  from examples.mlperf.rnnt.ops import FilterbankOp
  from examples.mlperf.rnnt.train import RNNTTrainer

  batch_size = getenv("BS", 1)
  config_filepath = Path(getenv("CONFIG", ""))
  data_dir = Path(getenv("DATA_DIR", ""))
  manifest_names = getenv("MANIFEST_NAMES", "")
  num_buckets = getenv("NUM_BUCKETS", 1)
  num_epochs = getenv("NUM_EPOCHS", 1)

  trainer = RNNTTrainer(config_filepath, data_dir, manifest_names, batch_size=batch_size, num_buckets=num_buckets)
  for epoch in range(1, num_epochs + 1):
    trainer.sampler.set_epoch(epoch)

    for batch in audio_dataloader(trainer.dataset, trainer.sampler): pass

def train_bert():
  # TODO: BERT
  pass

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()


