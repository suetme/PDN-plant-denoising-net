import torch
import pytorch3d.loss
import pytorch3d.structures
from torch_cluster import fps


class FCLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation=None):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

        if activation is None:
            self.activation = torch.nn.Identity()
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'elu':
            self.activation = torch.nn.ELU(alpha=1.0)
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU(0.1)
        else:
            raise ValueError()

    def forward(self, x):
        return self.activation(self.linear(x))



def normalize_sphere(pc, radius=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
    ## Center
    p_max = pc.max(dim=-2, keepdim=True)[0]
    p_min = pc.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2    # (B, 1, 3)
    pc = pc - center
    ## Scale
    scale = (pc ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / radius  # (B, N, 1)
    pc = pc / scale
    return pc, center, scale


def normalize_pcl(pc, center, scale):
    return (pc - center) / scale


def chamfer_distance_unit_sphere(gen, ref, batch_reduction='mean', point_reduction='mean'):
    ref, center, scale = normalize_sphere(ref)
    gen = normalize_pcl(gen, center, scale)
    return pytorch3d.loss.chamfer_distance(gen, ref, batch_reduction=batch_reduction, point_reduction=point_reduction)


def farthest_point_sampling(pcls, num_pnts):
    """
    Args:
        pcls:  A batch of point clouds, (B, N, 3).
        num_pnts:  Target number of points.
    """
    ratio = 0.01 + num_pnts / pcls.size(1)
    sampled = []
    indices = []
    for i in range(pcls.size(0)):
        idx = fps(pcls[i], ratio=ratio, random_start=False)[:num_pnts]
        sampled.append(pcls[i:i+1, idx, :])
        indices.append(idx)
    sampled = torch.cat(sampled, dim=0)
    return sampled, indices





