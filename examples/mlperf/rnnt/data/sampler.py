import numpy as np

from examples.mlperf.rnnt.data.dataset import AudioDataset
from tinygrad import Tensor
from tinygrad.helpers import dtypes


class BucketingSampler:
  def __init__(self, dataset:AudioDataset, num_buckets:int):
    self.num_buckets = num_buckets

    len_ids = np.argsort([sample["duration"] for sample in dataset.samples])
    self.buckets = [Tensor(t) for t in np.array_split(len_ids, num_buckets)]

  def __iter__(self):
    indices = []
    for bid in range(self.num_buckets):
      perm = Tensor.randint(self.buckets[bid].shape[0])
      bucket_indices = self.buckets[bid][perm]
      indices.append(bucket_indices)

    indices = Tensor.cat(*indices)
    indices = self._reshuffle_batches(indices)
    return iter(indices.cast(dtypes.int32))

  def _reshuffle_batches(self, indices:Tensor) -> Tensor:
    indices = indices.unsqueeze(-1) # 1 in this reshape corresponds to global_batch_size
    num_batches = indices.shape[0]
    order = Tensor.randint(num_batches)
    indices = indices[order, :]
    indices = indices.squeeze(-1)
    return indices
  

class BatchSampler:
  def __init__(self, sampler, batch_size:int):
    self.sampler = sampler
    self.batch_size = batch_size

  def __iter__(self):
    batch = [0] * self.batch_size
    idx_in_batch = 0
    for idx in self.sampler:
      batch[idx_in_batch] = idx.item()
      idx_in_batch += 1
      if idx_in_batch == self.batch_size:
        yield batch
        idx_in_batch = 0
        batch = [0] * self.batch_size
    if idx_in_batch > 0: yield batch[:idx_in_batch]
