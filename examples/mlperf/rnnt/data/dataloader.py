from examples.mlperf.rnnt.data.dataset import AudioDataset
from examples.mlperf.rnnt.data.sampler import BucketingSampler

def audio_dataloader(dataset:AudioDataset, sampler: BucketingSampler):
  # TODO: need to implement collate
  for idxs in sampler: yield dataset[idxs]
