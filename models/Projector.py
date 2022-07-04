import torch


class Projector(torch.nn.Module):

    def __init__(self, c):
        super().__init__()
        self.c = torch.tensor(c, dtype=torch.float)
        self.z = torch.zeros(3, dtype=torch.float)
        self.mf = torch.nn.Linear(3, 3)
        self.register_buffer('mb', torch.zeros((3,3), dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.view(x, self.z)

    def view(self, t: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:
        """ Computes projection of the target point on a z-focused viewport surface. """
        # target as seen from the surface
        ts = self.mf.forward(t)
        # camera as seen from the surface
        cs = self.mf.forward(self.c)
        # projection of target on the focused surface, as seen by camera
        k = cs[2].sub(focus[2]).divide(ts[2] - cs[2]).neg_()
        return ts.sub(cs).mul_(k).add_(cs)

    def scene(self, s: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:
        """ Computes projection of the viewport surface point on z-focused target. """
        # viewport point as it seen on the scene
        ts = self.mb.matmul(s.sub(self.mf.bias))
        # camera as it seen on the scene
        cs = self.c
        # projection of target on the focused surface, as seen by camera
        k = cs[2].sub(focus[2]).divide(ts[2] - cs[2]).neg_()
        return ts.sub(cs).mul_(k).add_(cs)

    def lock(self):
        torch.inverse(self.mf.weight.data.detach(), out=self.mb.data)
