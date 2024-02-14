def _box_area(boxes):
  return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
  area1 = _box_area(boxes1)
  area2 = _box_area(boxes2)

  lt = boxes1[:, None, :2].maximum(boxes2[:, :2])
  rb = boxes1[:, None, 2:].minimum(boxes2[:, 2:])

  wh = rb - lt
  wh = (wh > 0).where(wh, 0)

  inter = wh[:, :, 0] * wh[:, :, 1]
  union = area1[:, None] + area2 - inter
  return inter / union
