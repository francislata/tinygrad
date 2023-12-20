from typing import Tuple, List

from examples.mlperf.rnnt.data.dataset import AudioDataset
from examples.mlperf.rnnt.data.sampler import BucketingSampler, BatchSampler
from tinygrad import Tensor
from tinygrad.helpers import dtypes


def _collate_fn(batch:List[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
  max_len = lambda l, idx: max(len(el[idx]) for el in l)
  audio, audio_lens = [], []
  transcript, transcript_lens = [], []

  for sample in batch:
    audio_t = Tensor(sample[0])
    audio.append(audio_t.pad(((0, max_len(batch, 0) - audio_t.shape[0]),)))
    audio_lens.append(Tensor(sample[1], dtype=dtypes.int))

    transcript_t = Tensor(sample[2])
    transcript.append(transcript_t.pad(((0, max_len(batch, 2) - transcript_t.shape[0]),)))
    transcript_lens.append(Tensor(sample[3], dtype=dtypes.int))
  
  audio, audio_lens = Tensor.stack(audio), Tensor.stack(audio_lens)
  transcript, transcript_lens = Tensor.stack(transcript), Tensor.stack(transcript_lens)
  return audio, audio_lens, transcript, transcript_lens


def audio_dataloader(dataset:AudioDataset, sampler: BucketingSampler, batch_size:int):
  assert batch_size > 0, "batch_size must be greater than 0"
  batch_sampler = BatchSampler(sampler, batch_size)
  for batch_idxs in batch_sampler:
    batch = []
    for batch_idx in batch_idxs: batch.append(dataset[batch_idx])
    yield _collate_fn(batch)
