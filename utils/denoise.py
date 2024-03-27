import math
import torch
import pytorch3d.ops
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from models.utils import farthest_point_sampling
from .transforms import NormalizeUnitSphere
from models.utils import chamfer_distance_unit_sphere

#for training use
def patch_based_denoise(model, pcl_noisy, pcl_clean, ld_step_size=1, ld_num_steps=1, patch_size=1000, seed_k=3, denoise_knn=16, get_traj=False):
    """
    Args:
        pcl_noisy:  Input point cloud, (N, 3)
    """
    assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be (N, 3).'
    N, d = pcl_noisy.size()

    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    seed_pnts, _idx = farthest_point_sampling(pcl_noisy, int(seed_k * N / patch_size))
    _, _, patches = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
    patches = patches[0]  # (N, K, 3)

    with torch.no_grad():
        model.eval()
        patches_denoised, traj = model.denoise_langevin_dynamics(patches, step_size=ld_step_size, denoise_knn=denoise_knn, num_steps=ld_num_steps)

    pcl_denoised, fps_idx = farthest_point_sampling(patches_denoised.view(1, -1, d), N)
    pcl_denoised = pcl_denoised[0]
    fps_idx = fps_idx[0]

    pcl_clean = pcl_clean.unsqueeze(0)
    pcl_denoised = pcl_denoised.unsqueeze(0)
    chamfer = chamfer_distance_unit_sphere(pcl_denoised, pcl_clean, batch_reduction='mean')[0].item()
    pcl_denoised = pcl_denoised.squeeze(0)
    if get_traj:
        for i in range(len(traj)):
            traj[i] = traj[i].view(-1, d)[fps_idx, :]
        return pcl_denoised, traj
    else:
        return pcl_denoised,chamfer

#for testing use
def patch_based_denoise_2(model, pcl_noisy, ld_step_size=1, ld_num_steps=1, patch_size=1024, seed_k=3, denoise_knn=18, get_traj=False):
    """
    Args:
        pcl_noisy:  Input point cloud, (N, 3)
    """
    assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be (N, 3).'
    N, d = pcl_noisy.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    seed_pnts, _ = farthest_point_sampling(pcl_noisy, int(seed_k * N / patch_size))
    _, _, patches = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
    patches = patches[0]    # (N, K, 3)

    with torch.no_grad():
        model.eval()
        patches_denoised, traj = model.denoise_langevin_dynamics(patches, step_size=ld_step_size, denoise_knn=denoise_knn, num_steps=ld_num_steps)

    pcl_denoised, fps_idx = farthest_point_sampling(patches_denoised.view(1, -1, d), N)
    pcl_denoised = pcl_denoised[0]
    fps_idx = fps_idx[0]
    if get_traj:
        for i in range(len(traj)):
            traj[i] = traj[i].view(-1, d)[fps_idx, :]
        return pcl_denoised, traj
    else:
        return pcl_denoised

#for testing large use
def denoise_large_pointcloud(model, pcl, cluster_size, seed=0):
    device = pcl.device
    pcl = pcl.cpu().numpy()

    print('Running KMeans to construct clusters...')
    n_clusters = math.ceil(pcl.shape[0] / cluster_size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(pcl)
    pcl_parts = []
    for i in tqdm(range(n_clusters), desc='Denoise Clusters'):
        pts_idx = kmeans.labels_ == i

        pcl_part_noisy = torch.FloatTensor(pcl[pts_idx]).to(device)
        pcl_part_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_part_noisy)
        pcl_part_denoised = patch_based_denoise_2(
            model,
            pcl_part_noisy,
        )
        pcl_part_denoised = pcl_part_denoised * scale + center
        pcl_parts.append(pcl_part_denoised)

    return torch.cat(pcl_parts, dim=0)
