# pip3 install sentencepiece

# This file incorporates code from the following:
# Github Name                    | License | Link
# black-forest-labs/flux         | Apache  | https://github.com/black-forest-labs/flux/tree/main/model_licenses

from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.helpers import fetch, tqdm, colored
from extra.models.flux import ClipEmbedder, AutoEncoder, Flux
from extra.models.t5 import T5Embedder
import numpy as np

import math, time, argparse, tempfile
from typing import List, Dict, Union
from pathlib import Path
from PIL import Image

urls:dict = {
  "flux-schnell": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors",
  "flux-dev": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev.sft",
  "ae": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors",
  "T5_1_of_2": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder_2/model-00001-of-00002.safetensors",
  "T5_2_of_2": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder_2/model-00002-of-00002.safetensors",
  "T5_tokenizer": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/tokenizer_2/spiece.model",
  "clip": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder/model.safetensors"
}

# https://github.com/black-forest-labs/flux/blob/main/src/flux/util.py
def load_flow_model(name:str, model_path:str):
  # Loading Flux
  print("Init model")
  model = Flux(guidance_embed=(name != "flux-schnell"))
  if not model_path: model_path = fetch(urls[name])
  state_dict = {k.replace("scale", "weight"): v for k, v in safe_load(model_path).items()}
  load_state_dict(model, state_dict)
  return model

def load_T5(max_length:int=512):
  # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
  print("Init T5")
  T5 = T5Embedder(max_length, fetch(urls["T5_tokenizer"]))
  pt_1 = fetch(urls["T5_1_of_2"])
  pt_2 = fetch(urls["T5_2_of_2"])
  load_state_dict(T5.encoder, safe_load(pt_1) | safe_load(pt_2), strict=False)
  return T5

def load_clip():
  print("Init Clip")
  clip = ClipEmbedder()
  load_state_dict(clip.transformer, safe_load(fetch(urls["clip"])))
  return clip

def load_ae() -> AutoEncoder:
  # Loading the autoencoder
  print("Init AE")
  ae = AutoEncoder(0.3611, 0.1159)
  load_state_dict(ae, safe_load(fetch(urls["ae"])))
  return ae

# https://github.com/black-forest-labs/flux/blob/main/src/flux/sampling.py
def prepare(T5:T5Embedder, clip:ClipEmbedder, img:Tensor, prompt:Union[str, List[str]]) -> Dict[str, Tensor]:
  bs, _, h, w = img.shape
  if bs == 1 and not isinstance(prompt, str):
    bs = len(prompt)

  img = img.rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
  if img.shape[0] == 1 and bs > 1:
    img = img.expand((bs, *img.shape[1:]))

  img_ids = Tensor.zeros(h // 2, w // 2, 3).contiguous()
  img_ids[..., 1] = img_ids[..., 1] + Tensor.arange(h // 2)[:, None]
  img_ids[..., 2] = img_ids[..., 2] + Tensor.arange(w // 2)[None, :]
  img_ids = img_ids.rearrange("h w c -> 1 (h w) c")
  img_ids = img_ids.expand((bs, *img_ids.shape[1:]))

  if isinstance(prompt, str):
    prompt = [prompt]
  txt = T5(prompt).realize()
  if txt.shape[0] == 1 and bs > 1:
    txt = txt.expand((bs, *txt.shape[1:]))
  txt_ids = Tensor.zeros(bs, txt.shape[1], 3)

  vec = clip(prompt).realize()
  if vec.shape[0] == 1 and bs > 1:
    vec = vec.expand((bs, *vec.shape[1:]))

  return {"img": img, "img_ids": img_ids.to(img.device), "txt": txt.to(img.device), "txt_ids": txt_ids.to(img.device), "vec": vec.to(img.device)}


def get_schedule(num_steps:int, image_seq_len:int, base_shift:float=0.5, max_shift:float=1.15, shift:bool=True) -> List[float]:
  # extra step for zero
  step_size = -1.0 / num_steps
  timesteps = Tensor.arange(1, 0 + step_size, step_size)

  # shifting the schedule to favor high timesteps for higher signal images
  if shift:
    # estimate mu based on linear estimation between two points
    mu = 0.5 + (max_shift - base_shift) * (image_seq_len - 256) / (4096 - 256)
    timesteps = math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1))
  return timesteps.tolist()

@TinyJit
def run(model, *args): return model(*args).realize()

def denoise(model, img:Tensor, img_ids:Tensor, txt:Tensor, txt_ids:Tensor, vec:Tensor, timesteps:List[float], guidance:float=4.0) -> Tensor:
  # this is ignored for schnell
  guidance_vec = Tensor((guidance,), device=img.device, dtype=img.dtype).expand((img.shape[0],))
  for t_curr, t_prev in tqdm(list(zip(timesteps[:-1], timesteps[1:])), "Denoising"):
    t_vec = Tensor((t_curr,), device=img.device, dtype=img.dtype).expand((img.shape[0],))
    pred = run(model, img, img_ids, txt, txt_ids, t_vec, vec, guidance_vec)
    img = img + (t_prev - t_curr) * pred

  return img

def unpack(x:Tensor, height:int, width:int) -> Tensor:
  return x.rearrange("b (h w) (c ph pw) -> b c (h ph) (w pw)", h=math.ceil(height / 16), w=math.ceil(width / 16), ph=2, pw=2)

# https://github.com/black-forest-labs/flux/blob/main/src/flux/cli.py
if __name__ == "__main__":
  default_prompt = "bananas and a can of coke"
  parser = argparse.ArgumentParser(description="Run Flux.1", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--name",       type=str,   default="flux-schnell", help="Name of the model to load")
  parser.add_argument("--model_path", type=str,   default="",             help="path of the model file")
  parser.add_argument("--width",      type=int,   default=512,            help="width of the sample in pixels (should be a multiple of 16)")
  parser.add_argument("--height",     type=int,   default=512,            help="height of the sample in pixels (should be a multiple of 16)")
  parser.add_argument("--seed",       type=int,   default=None,           help="Set a seed for sampling")
  parser.add_argument("--prompt",     type=str,   default=default_prompt, help="Prompt used for sampling")
  parser.add_argument('--out',        type=str,   default=Path(tempfile.gettempdir()) / "rendered.png", help="Output filename")
  parser.add_argument("--num_steps",  type=int,   default=None,           help="number of sampling steps (default 4 for schnell, 50 for guidance distilled)") #noqa:E501
  parser.add_argument("--guidance",   type=float, default=3.5,            help="guidance value used for guidance distillation")
  parser.add_argument("--output_dir", type=str,   default="output",       help="output directory")
  args = parser.parse_args()

  if args.name not in ["flux-schnell", "flux-dev"]:
    raise ValueError(f"Got unknown model name: {args.name}, chose from flux-schnell and flux-dev")

  if args.num_steps is None:
    args.num_steps = 4 if args.name == "flux-schnell" else 50

  # allow for packing and conversion to latent space
  height = 16 * (args.height // 16)
  width = 16 * (args.width // 16)

  if args.seed is None: args.seed = Tensor._seed
  else: Tensor.manual_seed(args.seed)

  print(f"Generating with seed {args.seed}:\n{args.prompt}")
  t0 = time.perf_counter()

  # prepare input noise
  x = Tensor.randn(1, 16, 2 * math.ceil(height / 16), 2 * math.ceil(width / 16), dtype="bfloat16")

  # load text embedders
  T5 = load_T5(max_length=256 if args.name == "flux-schnell" else 512)
  clip = load_clip()

  # embed text to get inputs for model
  inp = prepare(T5, clip, x, prompt=args.prompt)
  timesteps = get_schedule(args.num_steps, inp["img"].shape[1], shift=(args.name != "flux-schnell"))

  # done with text embedders
  del T5, clip

  # load model
  model = load_flow_model(args.name, args.model_path)

  # denoise initial noise
  x = denoise(model, **inp, timesteps=timesteps, guidance=args.guidance)

  # done with model
  del model, run

  # load autoencoder
  ae = load_ae()

  # decode latents to pixel space
  x = unpack(x.float(), height, width)
  x = ae.decode(x).realize()

  t1 = time.perf_counter()
  print(f"Done in {t1 - t0:.1f}s. Saving {args.out}")

  # bring into PIL format and save
  x = x.clamp(-1, 1)
  x = x[0].rearrange("c h w -> h w c")
  x = (127.5 * (x + 1.0)).cast("uint8")

  img = Image.fromarray(x.numpy())

  img.save(args.out)

  # validation!
  if args.prompt == default_prompt and args.name=="flux-schnell" and args.seed == 0 and args.width == args.height == 512:
    ref_image = Tensor(np.array(Image.open("examples/flux1_seed0.png")))
    distance = (((x.cast(dtypes.float) - ref_image.cast(dtypes.float)) / ref_image.max())**2).mean().item()
    assert distance < 4e-3, colored(f"validation failed with {distance=}", "red")
    print(colored(f"output validated with {distance=}", "green"))