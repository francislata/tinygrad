from tinygrad import Tensor

def sigmoid_focal_loss(inputs:Tensor, targets:Tensor, alpha:float = 0.25, gamma:float = 2, reduction:str = "none") -> Tensor:
  p = inputs.sigmoid()
  # NOTE: from binary_crossentropy_logits implementation without mean reduction
  ce_loss = inputs.maximum(0) - targets * inputs + (1 + inputs.abs().neg().exp()).log()
  p_t = p * targets + (1 - p) * (1 - targets)
  loss = ce_loss * ((1 - p_t) ** gamma)

  if alpha >= 0:
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss *= alpha_t

  if reduction == "mean": loss = loss.mean()
  elif reduction == "sum": loss = loss.sum()
  return loss

def l1_loss(inputs:Tensor, targets:Tensor, reduction:str="mean") -> Tensor:
  loss = (inputs - targets).abs()
  if reduction == "mean": return loss.mean()
  elif reduction == "sum": return loss.sum()
