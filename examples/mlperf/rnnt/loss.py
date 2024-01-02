import torch
import torchaudio.functional as F

from dataclasses import dataclass
from tinygrad import Tensor


@dataclass(frozen=True)
class TransducerLoss:
  blank_idx: int

  def __call__(self, y_:Tensor, y_lens_:Tensor, y:Tensor, y_lens:Tensor):
    y_, y_lens_ = torch.Tensor(y_.numpy(), dtype=torch.float), torch.Tensor(y_lens_.numpy(), dtype=torch.int)
    y, y_lens = torch.Tensor(y.numpy(), dtype=torch.int), torch.Tensor(y_lens.numpy(), dtype=torch.int)
    loss = F.rnnt_loss(y_, y, y_lens_, y_lens, blank=self.blank_idx)
    return Tensor(loss.numpy(), requires_grad=True)
