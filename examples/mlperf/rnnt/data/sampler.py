import numpy as np

from examples.mlperf.rnnt.data.dataset import AudioDataset
from tinygrad import Tensor
from tinygrad.helpers import dtypes


class BucketingSampler:
  def __init__(self, dataset:AudioDataset, num_buckets:int, num_epochs:int, num_workers:int, batch_size:int, rng):
    self.num_buckets = num_buckets
    self.num_epochs = num_epochs
    self.num_workers = num_workers
    self.batch_size = batch_size
    self.rng = rng

    len_ids = np.argsort([sample["duration"] for sample in dataset.samples])
    # self.buckets = [Tensor(t) for t in np.array_split(len_ids, num_buckets)]
    buckets = np.array_split(len_ids, num_buckets)

    shuffled_buckets = np.array([
      perm
      for _ in range(num_epochs)
      for bucket in buckets
      for perm in self.rng.permutation(bucket)
    ])

    gbs = batch_size * num_workers
    epochs = np.reshape(shuffled_buckets, [num_epochs, -1])
    to_drop = epochs.shape[1] - (epochs.shape[1] // gbs * gbs)
    for epoch in epochs:
      dropped_idxs = self.rng.choice(epochs.shape[1], to_drop, replace=False)
      if dropped_idxs is not None:
        epoch[dropped_idxs] = -1
    epochs = epochs[epochs != -1].reshape(self.num_epochs, -1)
    self.dataset_size = epochs.shape[1]

    epochs_iters_batch = np.reshape(epochs, [num_epochs, -1, gbs])

    for epoch in epochs_iters_batch:
      self.rng.shuffle(epoch, axis=0)

    epochs_iters_batch_worker = np.reshape(epochs_iters_batch, [num_epochs, -1, batch_size, num_workers])
    workers_epochs_iters_batch = np.moveaxis(epochs_iters_batch_worker, -1, 0)

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
