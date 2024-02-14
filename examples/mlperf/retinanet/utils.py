from tinygrad import Tensor, dtypes

class Matcher:
  BELOW_LOW_THRESHOLD = -1
  BETWEEN_THRESHOLDS = -2

  def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
    assert low_threshold <= high_threshold
    self.high_threshold = high_threshold
    self.low_threshold = low_threshold
    self.allow_low_quality_matches = allow_low_quality_matches

  def __call__(self, match_quality_matrix):
    if match_quality_matrix.numel() == 0:
      if match_quality_matrix.shape[0] == 0:
        raise ValueError("No ground-truth boxes available for one of the images during training")
      else:
        raise ValueError("No proposal boxes vailable for one of the images during training")
      
    matched_vals = match_quality_matrix.max(axis=0)
    matches = match_quality_matrix.argmax(axis=0)
    if self.allow_low_quality_matches: all_matches = Tensor.zeros_like(matches).assign(matches)
    else: all_matches = None

    below_threshold = (matched_vals < self.low_threshold).cast(dtypes.int)
    between_thresholds = ((matched_vals >= self.low_threshold) * (matched_vals < self.high_threshold)).cast(dtypes.int)

    matches = (below_threshold[Tensor.arange(0, below_threshold.shape[0])] == 1).where(self.BELOW_LOW_THRESHOLD, matches)
    matches = (between_thresholds[Tensor.arange(0, between_thresholds.shape[0])] == 1).where(self.BETWEEN_THRESHOLDS, matches)

    if self.allow_low_quality_matches:
      assert all_matches is not None
      matches = self.set_low_quality_matches(matches, all_matches, match_quality_matrix)

    return matches
  
  def set_low_quality_matches(self, matches, all_matches, match_quality_matrix):
    highest_quality_foreach_gt = match_quality_matrix.max(axis=1)
    matches = (match_quality_matrix == highest_quality_foreach_gt[:, None]).where(all_matches, matches).squeeze(0)
    return matches
