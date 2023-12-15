from typing import Optional, List

from examples.mlperf.rnnt.config import Config

import sentencepiece as spm


class Tokenizer:
  def __init__(self, config:Config):
    self.charset = config.tokenizer.get("labels")
    self.use_sentpiece = config.tokenizer.get("sentpiece_model") is not None
    if self.use_sentpiece:
      self.sentpiece = spm.SentencePieceProcessor(model_file=config.tokenizer.get("sentpiece_model"))
      self.num_labels = len(self.sentpiece)
    else:
      self.num_labels = len(self.charset)
      self.label2ind = {lab: i for i, lab in enumerate(self.charset)}

  def tokenize(self, transcript):
    if self.use_sentpiece:
      inds = self.sentpiece.encode(transcript, out_type=int)
      assert 0 not in inds, "<unk> found during tokenization (OOV?)"
    else: inds = [self.label2ind[x] for x in transcript if x in self.label2ind]
    return inds
  
  def detokenize(self, inds):
    if self.use_sentpiece: return self.sentpiece.decode(inds)
    else: ''.join(self.charset[i] for i in inds)
