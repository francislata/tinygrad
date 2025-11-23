from examples.sdxl import FirstStage
from tinygrad import Tensor, nn, dtypes
from extra.models.clip import FrozenClosedClipEmbedder

import math
from typing import List, Optional, Union, Tuple, Callable
from dataclasses import dataclass

def tensor_identity(x:Tensor) -> Tensor: return x

class AutoEncoder:
  def __init__(self, scale_factor:float, shift_factor:float):
    self.decoder = FirstStage.Decoder(128, 3, 3, 16, [1, 2, 4, 4], 2, 256)
    self.scale_factor = scale_factor
    self.shift_factor = shift_factor

  def decode(self, z:Tensor) -> Tensor:
    z = z / self.scale_factor + self.shift_factor
    return self.decoder(z)

# Conditioner
class ClipEmbedder(FrozenClosedClipEmbedder):
  def __call__(self, texts:Union[str, List[str], Tensor]) -> Tensor:
    if isinstance(texts, str): texts = [texts]
    assert isinstance(texts, (list,tuple)), f"expected list of strings, got {type(texts).__name__}"
    tokens = Tensor.cat(*[Tensor(self.tokenizer.encode(text)) for text in texts], dim=0)
    return self.transformer.text_model(tokens.reshape(len(texts),-1))[:, tokens.argmax(-1)]

# https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py
def attention(q:Tensor, k:Tensor, v:Tensor, pe:Tensor) -> Tensor:
  q, k = apply_rope(q, k, pe)
  x = Tensor.scaled_dot_product_attention(q, k, v)
  return x.rearrange("B H L D -> B L (H D)")

def rope(pos:Tensor, dim:int, theta:int) -> Tensor:
  assert dim % 2 == 0
  scale = Tensor.arange(0, dim, 2, dtype=dtypes.float32, device=pos.device) / dim # NOTE: this is torch.float64 in reference implementation
  omega = 1.0 / (theta**scale)
  out = Tensor.einsum("...n,d->...nd", pos, omega)
  out = Tensor.stack(Tensor.cos(out), -Tensor.sin(out), Tensor.sin(out), Tensor.cos(out), dim=-1)
  out = out.rearrange("b n d (i j) -> b n d i j", i=2, j=2)
  return out.float()

def apply_rope(xq:Tensor, xk:Tensor, freqs_cis:Tensor) -> Tuple[Tensor, Tensor]:
  xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
  return xq_out.reshape(*xq.shape).cast(xq.dtype), xk_out.reshape(*xk.shape).cast(xk.dtype)


# https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py
class EmbedND:
  def __init__(self, dim:int, theta:int, axes_dim:List[int]):
    self.dim = dim
    self.theta = theta
    self.axes_dim = axes_dim

  def __call__(self, ids:Tensor) -> Tensor:
    n_axes = ids.shape[-1]
    emb = Tensor.cat(*[rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
    return emb.unsqueeze(1)

class MLPEmbedder:
  def __init__(self, in_dim:int, hidden_dim:int):
    self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
    self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

  def __call__(self, x:Tensor) -> Tensor:
    return self.out_layer(self.in_layer(x).silu())

class QKNorm:
  def __init__(self, dim:int):
    self.query_norm = nn.RMSNorm(dim)
    self.key_norm = nn.RMSNorm(dim)

  def __call__(self, q:Tensor, k:Tensor) -> Tuple[Tensor, Tensor]:
    return self.query_norm(q), self.key_norm(k)

class SelfAttention:
  def __init__(self, dim:int, num_heads:int = 8, qkv_bias:bool = False):
    self.num_heads = num_heads
    head_dim = dim // num_heads

    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.norm = QKNorm(head_dim)
    self.proj = nn.Linear(dim, dim)

  def __call__(self, x:Tensor, pe:Tensor) -> Tensor:
    qkv = self.qkv(x)
    q, k, v = qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    q, k = self.norm(q, k)
    x = attention(q, k, v, pe=pe)
    return self.proj(x)

@dataclass
class ModulationOut:
  shift:Tensor
  scale:Tensor
  gate:Tensor

class Modulation:
  def __init__(self, dim:int, double:bool):
    self.is_double = double
    self.multiplier = 6 if double else 3
    self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

  def __call__(self, vec:Tensor) -> Tuple[ModulationOut, Optional[ModulationOut]]:
    out = self.lin(vec.silu())[:, None, :].chunk(self.multiplier, dim=-1)
    return ModulationOut(*out[:3]), ModulationOut(*out[3:]) if self.is_double else None

class DoubleStreamBlock:
  def __init__(self, hidden_size:int, num_heads:int, mlp_ratio:float, qkv_bias:bool = False):
    mlp_hidden_dim = int(hidden_size * mlp_ratio)
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.img_mod = Modulation(hidden_size, double=True)
    self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

    self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_mlp = [nn.Linear(hidden_size, mlp_hidden_dim, bias=True), Tensor.gelu, nn.Linear(mlp_hidden_dim, hidden_size, bias=True)]

    self.txt_mod = Modulation(hidden_size, double=True)
    self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

    self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_mlp = [nn.Linear(hidden_size, mlp_hidden_dim, bias=True), Tensor.gelu, nn.Linear(mlp_hidden_dim, hidden_size, bias=True)]

  def __call__(self, img:Tensor, txt:Tensor, vec:Tensor, pe:Tensor) -> tuple[Tensor, Tensor]:
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)
    assert img_mod2 is not None and txt_mod2 is not None
    # prepare image for attention
    img_modulated = self.img_norm1(img)
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
    img_qkv = self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = img_qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    img_q, img_k = self.img_attn.norm(img_q, img_k)

    # prepare txt for attention
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = txt_qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k)

    # run actual attention
    q = Tensor.cat(txt_q, img_q, dim=2)
    k = Tensor.cat(txt_k, img_k, dim=2)
    v = Tensor.cat(txt_v, img_v, dim=2)

    attn = attention(q, k, v, pe=pe)
    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

    # calculate the img bloks
    img = img + img_mod1.gate * self.img_attn.proj(img_attn)
    img = img + img_mod2.gate * ((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift).sequential(self.img_mlp)

    # calculate the txt bloks
    txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
    txt = txt + txt_mod2.gate * ((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift).sequential(self.txt_mlp)
    return img, txt


class SingleStreamBlock:
  """
  A DiT block with parallel linear layers as described in
  https://arxiv.org/abs/2302.05442 and adapted modulation interface.
  """

  def __init__(self,hidden_size:int, num_heads:int, mlp_ratio:float=4.0, qk_scale:Optional[float]=None):
    self.hidden_dim = hidden_size
    self.num_heads = num_heads
    head_dim = hidden_size // num_heads
    self.scale = qk_scale or head_dim**-0.5

    self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
    # qkv and mlp_in
    self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
    # proj and mlp_out
    self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

    self.norm = QKNorm(head_dim)

    self.hidden_size = hidden_size
    self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    self.mlp_act = Tensor.gelu
    self.modulation = Modulation(hidden_size, double=False)

  def __call__(self, x:Tensor, vec:Tensor, pe:Tensor) -> Tensor:
    mod, _ = self.modulation(vec)
    x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
    qkv, mlp = Tensor.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
    q, k, v = qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    q, k = self.norm(q, k)

    # compute attention
    attn = attention(q, k, v, pe=pe)
    # compute activation in mlp stream, cat again and run second linear layer
    output = self.linear2(Tensor.cat(attn, self.mlp_act(mlp), dim=2))
    return x + mod.gate * output


class LastLayer:
  def __init__(self, hidden_size:int, patch_size:int, out_channels:int):
    self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
    self.adaLN_modulation:List[Callable[[Tensor], Tensor]] = [Tensor.silu, nn.Linear(hidden_size, 2 * hidden_size, bias=True)]

  def __call__(self, x:Tensor, vec:Tensor) -> Tensor:
    shift, scale = vec.sequential(self.adaLN_modulation).chunk(2, dim=1)
    x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
    return self.linear(x)

def timestep_embedding(t:Tensor, dim:int, max_period:int=10000, time_factor:float=1000.0) -> Tensor:
  """
  Create sinusoidal timestep embeddings.
  :param t: a 1-D Tensor of N indices, one per batch element.
                    These may be fractional.
  :param dim: the dimension of the output.
  :param max_period: controls the minimum frequency of the embeddings.
  :return: an (N, D) Tensor of positional embeddings.
  """
  t = time_factor * t
  half = dim // 2
  freqs = Tensor.exp(-math.log(max_period) * Tensor.arange(0, stop=half, dtype=dtypes.float32) / half).to(t.device)

  args = t[:, None].float() * freqs[None]
  embedding = Tensor.cat(Tensor.cos(args), Tensor.sin(args), dim=-1)
  if dim % 2:  embedding = Tensor.cat(*[embedding, Tensor.zeros_like(embedding[:, :1])], dim=-1)
  if Tensor.is_floating_point(t):  embedding = embedding.cast(t.dtype)
  return embedding

# https://github.com/black-forest-labs/flux/blob/main/src/flux/model.py
class Flux:
  """
  Transformer model for flow matching on sequences.
  """

  def __init__(
      self,
      guidance_embed:bool,
      in_channels:int = 64,
      vec_in_dim:int = 768,
      context_in_dim:int = 4096,
      hidden_size:int = 3072,
      mlp_ratio:float = 4.0,
      num_heads:int = 24,
      depth:int = 19,
      depth_single_blocks:int = 38,
      axes_dim:Optional[List[int]] = None,
      theta:int = 10_000,
      qkv_bias:bool = True,
      ):

    axes_dim = axes_dim or [16, 56, 56]
    self.guidance_embed = guidance_embed
    self.in_channels = in_channels
    self.out_channels = self.in_channels
    if hidden_size % num_heads != 0:
      raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")
    pe_dim = hidden_size // num_heads
    if sum(axes_dim) != pe_dim:
      raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
    self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
    self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
    self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)
    self.guidance_in:Callable[[Tensor], Tensor] = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if guidance_embed else tensor_identity
    self.txt_in = nn.Linear(context_in_dim, self.hidden_size)

    self.double_blocks = [DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias) for _ in range(depth)]
    self.single_blocks = [SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio) for _ in range(depth_single_blocks)]
    self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

  def __call__(self, img:Tensor, img_ids:Tensor, txt:Tensor, txt_ids:Tensor, timesteps:Tensor, y:Tensor, guidance:Optional[Tensor] = None) -> Tensor:
    if img.ndim != 3 or txt.ndim != 3:
      raise ValueError("Input img and txt tensors must have 3 dimensions.")
    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256))
    if self.guidance_embed:
      if guidance is None:
        raise ValueError("Didn't get guidance strength for guidance distilled model.")
      vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
    vec = vec + self.vector_in(y)
    txt = self.txt_in(txt)
    ids = Tensor.cat(txt_ids, img_ids, dim=1)
    pe = self.pe_embedder(ids)
    for double_block in self.double_blocks:
      img, txt = double_block(img=img, txt=txt, vec=vec, pe=pe)

    img = Tensor.cat(txt, img, dim=1)
    for single_block in self.single_blocks:
      img = single_block(img, vec=vec, pe=pe)

    img = img[:, txt.shape[1] :, ...]

    return self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
