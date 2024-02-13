from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv

def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  from extra.models.resnet import ResNeXt50_32X4D
  from extra.models.retinanet import RetinaNet
  from extra.datasets.openimages import openimages, iterate
  from pycocotools.coco import COCO
  from pycocotools.cocoeval import COCOeval
  import numpy as np

  BS = getenv("BS", 1)

  coco = COCO(openimages()[0])
  coco_eval = COCOeval(coco, iouType="bbox")

  mdl = RetinaNet(ResNeXt50_32X4D())
  mdl.load_from_pretrained()

  input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
  input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

  def input_target_fixup(x, y=None):
    # normalize image
    x = x.permute([0,3,1,2]) / 255.0
    x -= input_mean
    x /= input_std

    if y is not None:
      # resize bbox
      for i in range(x.shape[0]):
        bbox = y[i]["boxes"]
        ratio_height, ratio_width = [s / s_orig for s, s_orig in zip(x.shape[-2:], y[i]["image_size"])]
        xmin, ymin, xmax, ymax = bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]
        xmin, ymin, xmax, ymax = np.split(bbox, 4, axis=1)

        xmin *= ratio_width
        xmax *= ratio_width
        ymin *= ratio_height
        ymax *= ratio_height

        y[i]["boxes"] = np.concatenate([xmin, ymin, xmax, ymax], axis=1)

    return x, y

  for x, targets in iterate(coco, BS, val=False):
    dat = Tensor(x.astype(np.float32))
    outs =  mdl(*input_target_fixup(dat, y=targets)).numpy()

def train_unet3d():
  # TODO: Unet3d
  pass

def train_rnnt():
  # TODO: RNN-T
  pass

def train_bert():
  # TODO: BERT
  pass

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()


