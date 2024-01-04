import numpy as np

from tinygrad import Tensor
from tinygrad.tensor import Function


def _logsumexp(a:Tensor, b:Tensor) -> Tensor:
  mx = Tensor.maximum(a,b).maximum(-1e10)
  s = (a-mx).exp() + (b-mx).exp()
  return s.log() + mx

inf = float('inf')

def _shear(d:Tensor, value:int = 0) -> Tensor:
  B,X,Y,C = d.shape
  d = d.pad(((0,0),(0,Y),(0,0),(0,0)),value=value)
  d = d.transpose(1,2).reshape((B,-1,C))
  d = d[:,:(X+Y-1)*Y,:].realize()
  return d.reshape((B,Y,X+Y-1,C)).transpose(1,2)

def _unshear(x:Tensor) -> Tensor:
  B,X,Y = x.shape
  x = x.reshape((B,-1,))
  x = x.pad(((0,0),(0,X),))
  x = x.reshape((B,X,Y+1))
  return x.shrink(((0,B),(0,X),(0,Y+1-X)))

  
class TransducerLoss(Function):
  def forward(self, d:Tensor, labels:Tensor):
    self.B,self.X,self.Y,self.C = d.shape

    self.labels = Tensor(labels).pad(((0,0),(0,1)))
    self.lattice = _shear(Tensor(d))
    self.X = self.X+self.Y-1
    assert self.lattice.shape == (self.B,self.X,self.Y,self.C), f"{self.lattice.shape}"

    self.skip = self.lattice[:,:,:,-1].log()
    self.p = self.lattice[
      Tensor(np.arange(self.B).reshape((-1,1,1))),
      Tensor(np.arange(self.X).reshape((1,-1,1))),
      Tensor(np.arange(self.Y).reshape((1,1,-1))),
      self.labels.reshape((self.B,1,-1))
    ].log()

    assert self.p.shape == (self.B, self.X, self.Y)
    self.a = [Tensor([0]*self.B).reshape(-1,1).pad(((0,0),(0,self.Y-1),),-inf).realize()]

    for x in range(0,self.X-1):
      self.a.append(_logsumexp((self.a[-1] + self.skip[:,x,:]).realize(), (self.a[-1][:,:-1].pad(((0,0),(1,0),),-inf).realize() + self.p[:,x,:-1].pad(((0,0),(1,0),),-inf)).realize()))

    return (-self.a[-1][:,-1] - self.skip[:,-1,-1]).lazydata

  def backward(self, g):
    self.b = [None] * (self.X-1) + [Tensor([0]*self.B).reshape(-1,1).pad(((0,0),(self.Y-1,0),),-inf).realize()]
    for x in range(self.X-2,-1,-1):
      self.b[x] = (
        _logsumexp(
        self.b[x+1] + self.skip[:,x,:],
        self.b[x+1][:,1:].pad(((0,0),(0,1),),-inf).realize() + self.p[:,x,:].realize()
      )).realize()

    self.skg, self.p_grad = None, None

    for a,b in zip(self.a[:-1], self.b[1:]):
      sg = (a + b).reshape(self.B, 1,-1)
      self.skg = sg if self.skg is None else self.skg.cat(sg,dim=1).realize()
      pg = a.unsqueeze(1) + b[:,1:].pad(((0,0),(0,1),),-inf).unsqueeze(1)
      self.p_grad = pg if self.p_grad is None else self.p_grad.cat(pg,dim=1).realize()

    self.skg = (_unshear(Tensor.cat(self.skg,(self.a[-1] + self.b[-1]).reshape(self.B, 1,-1),dim=1).realize().transpose(1,2)) - self.b[0][:,0].unsqueeze(1).unsqueeze(1)).exp().realize()
    self.p_grad = (_unshear(self.p_grad.pad(((0,0),(0,1),(0,0))).transpose(1,2)) + Tensor([1]*(self.Y-1) + [-inf]).unsqueeze(-1) - self.b[0][:,0].realize().unsqueeze(1).unsqueeze(1)).exp().realize()
    self.p_grad = self.p_grad.unsqueeze(-1).mul(Tensor.eye(self.C-1)[self.labels].unsqueeze(2))

    return (-Tensor.cat(self.p_grad,self.skg.unsqueeze(-1), dim=-1)).transpose(1,2).realize().lazydata,None
