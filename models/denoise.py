import torch
from torch import nn
import pytorch3d.ops
import numpy as np
from .feature import FeatureExtraction
from .score import ScoreNet


def get_random_indices(n, m):
    assert m < n
    return np.random.permutation(n)[:m]

class DenoiseNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.frame_knn = args.frame_knn
        self.num_train_points = args.num_train_points
        self.num_clean_nbs = args.num_clean_nbs
        # score-matching
        self.dsm_sigma = args.dsm_sigma
        # networks
        self.feature_net = FeatureExtraction()
        self.score_net = ScoreNet(
            z_dim=self.feature_net.out_channels,
            dim=3, 
            out_dim=3,
            hidden_size=args.score_net_hidden_dim,
            num_blocks=args.score_net_num_blocks,
        )

    def get_supervised_loss(self, pcl_noisy, pcl_clean):

        N_noisy = pcl_noisy.size(1)
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)

        # PDF
        feat = self.feature_net(pcl_noisy) # (B, N, F)
        feat = feat[:, pnt_idx, :] # (B, n, F)

        # UOF
        _, _, frames = pytorch3d.ops.knn_points(pcl_noisy[:,pnt_idx,:], pcl_noisy, K=self.frame_knn, return_nn=True) # (B, n, K, 3)
        frames_centered = frames - pcl_noisy[:,pnt_idx,:].unsqueeze(dim=2) # (B, n, K, 3)
        frames_centered = frames_centered.mean(dim=2) # (B, n, 3)

        # ground truth
        _, _, clean_nbs = pytorch3d.ops.knn_points(
            pcl_noisy[:,pnt_idx,:],
            pcl_clean,
            K=self.num_clean_nbs,
            return_nn=True,
        )   # (B , n ,  C, 3)
        noise_vecs = pcl_noisy[:,pnt_idx,:].unsqueeze(dim=2) - clean_nbs  # (B, n, C, 3)
        noise_vecs = noise_vecs.mean(dim=2)  # (B, n, 3)

        # DG
        grad_pred = self.score_net(
            x = frames_centered,
            c = feat
        )   # (B, n, 3)

        grad_target = - 1 * noise_vecs   # (B, n, 3)
        loss = 0.5 * ((grad_target - grad_pred) ** 2.0 * (1.0 / self.dsm_sigma)).sum(dim=-1).mean()
        
        return loss

    def denoise_langevin_dynamics(self, pcl_noisy, step_size, denoise_knn=16, num_steps=1):
        """
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
        """
        traj = [pcl_noisy.clone().cpu()]
        pcl_next = pcl_noisy.clone()

        with torch.no_grad():
            for step in range(num_steps):
                # PDF
                self.feature_net.eval()
                feat = self.feature_net(pcl_next) # (B, N, F)

                # UOF
                _, _, frames = pytorch3d.ops.knn_points(pcl_next, pcl_next, K=denoise_knn, return_nn=True)# (B, N, K, 3)
                frames_centered = frames - pcl_next.unsqueeze(dim=2)  # (B, N, K, 3)
                frames_centered = frames_centered.mean(dim=2)  # (B, N , 3)

                # DG
                self.score_net.eval()
                grad_pred = self.score_net(
                    x=frames_centered,
                    c=feat
                ) # (B, N, 3)

                pcl_next += step_size*grad_pred
                traj.append(pcl_next.clone().cpu())
            
        return pcl_next, traj
