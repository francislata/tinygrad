from examples.mlperf.rnnt.data.dataset import AudioDataset
from examples.mlperf.rnnt.data.sampler import BucketingSampler, BatchSampler

def audio_dataloader(dataset:AudioDataset, sampler: BucketingSampler, batch_size:int = 1):
  assert batch_size > 0, "batch_size must be greater than 0"
  # TODO: implement data bucketing
  # batch_sampler = BatchSampler(sampler, batch_size)
  # for idxs in batch_sampler:
  #   import pdb; pdb.set_trace()
  #   yield dataset[idxs]

  for i in range(0, len(dataset), batch_size):
    yield dataset[i:i + batch_size]
