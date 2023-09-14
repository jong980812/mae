import torch
class ThresholdTransform(object):
  def __init__(self, thr_255):
    self.thr = thr_255 /255.0  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)  # do not change the data type
  
class AddNoise(object):
  def __init__(self, scale):
    self.scl = scale
    self.noise = torch.rand_like(x)/ self.scl
  def __call__(self, x):
    return (x -self.noise )

class DetachWhite(object):
  def __init__(self, pixel):
    self.pixel = pixel
  def __call__(self, x):
    self.detach_pixel=x[:,:self.pixel]
    return (self.detach_pixel)