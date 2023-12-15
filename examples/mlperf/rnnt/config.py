from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Union, List


@dataclass(frozen=True)
class Config:
  tokenizer: Dict[str, Union[str, List[str]]]


rnnt_config = Config(
  tokenizer=dict(
    sentpiece_model="/datasets/sentencepieces/librispeech1023.model",
    labels=[" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
  )
)
