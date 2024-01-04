import numpy as np

from examples.mlperf.rnnt.data.dataset import AudioDataset
from tinygrad import Tensor, dtypes


class BucketingSampler:
  def __init__(self, dataset:AudioDataset, num_buckets:int, batch_size:int):
    self.num_buckets = num_buckets
    self.batch_size = batch_size
    self.epoch = 0

    len_ids = np.argsort([sample["duration"] for sample in dataset.samples])
    self.buckets = [Tensor(t) for t in np.array_split(len_ids, num_buckets)]

  def __iter__(self):
    # use current epoch to set the seed
    orig_seed = Tensor._seed
    Tensor.manual_seed(self.epoch)

    indices = []
    for bid in range(self.num_buckets):
      perm = Tensor.randint(self.buckets[bid].shape[0])
      bucket_indices = self.buckets[bid][perm]
      indices.append(bucket_indices)

    indices = Tensor.cat(*indices)

    length = indices.shape[0] // self.batch_size * self.batch_size
    indices = indices[:length]

    assert indices.shape[0] % self.batch_size == 0 

    indices = self._reshuffle_batches(indices)
    indices = indices.cast(dtypes.int32).numpy().tolist()

    # restore original seed
    Tensor.manual_seed(orig_seed)

    return iter(indices)
  
  def set_epoch(self, epoch):
    self.epoch = epoch

  def _reshuffle_batches(self, indices:Tensor) -> Tensor:
    indices = indices.reshape(-1, self.batch_size)
    num_batches = indices.shape[0]
    order = Tensor.randint(num_batches)
    indices = indices[order, :]
    indices = indices.reshape(-1)
    return indices
  

class BatchSampler:
  def __init__(self, sampler, batch_size:int):
    self.sampler = sampler
    self.batch_size = batch_size

  def __iter__(self):
    batch = [0] * self.batch_size
    idx_in_batch = 0
    for idx in self.sampler:
      batch[idx_in_batch] = idx
      idx_in_batch += 1
      if idx_in_batch == self.batch_size:
        yield batch
        idx_in_batch = 0
        batch = [0] * self.batch_size
    if idx_in_batch > 0: yield batch[:idx_in_batch]
