import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from models.utils import *
from .misc import BlackHole


def load_xyz(xyz_dir):
    all_pcls = {}
    for fn in tqdm(os.listdir(xyz_dir), desc='Loading'):
        if fn[-3:] != 'xyz':
            continue
        name = fn[:-4]
        path = os.path.join(xyz_dir, fn)
        all_pcls[name] = torch.FloatTensor(np.loadtxt(path, dtype=np.float32))
    return all_pcls

class Evaluator(object):

    def __init__(self, output_pcl_dir, dataset_root, dataset, summary_dir, experiment_name, device='cuda', res_gts='8192_poisson', logger=BlackHole()):
        super().__init__()
        self.output_pcl_dir = output_pcl_dir
        self.dataset_root = dataset_root
        self.dataset = dataset
        self.summary_dir = summary_dir
        self.experiment_name = experiment_name
        self.gts_pcl_dir = os.path.join(dataset_root, dataset, 'pointclouds', 'test', res_gts)
        self.res_gts = res_gts
        self.device = device
        self.logger = logger
        self.load_data()

    def load_data(self):
        self.pcls_up = load_xyz(self.output_pcl_dir)
        self.pcls_high = load_xyz(self.gts_pcl_dir)
        self.pcls_name = list(self.pcls_up.keys())

    def run(self):
        pcls_up, pcls_high, pcls_name = self.pcls_up, self.pcls_high, self.pcls_name
        results = {}
        for name in tqdm(pcls_name, desc='Evaluate'):
            pcl_up = pcls_up[name][:,:3].unsqueeze(0).to(self.device)
            if name not in pcls_high:
                self.logger.warning('Shape `%s` not found, ignored.' % name)
                continue
            pcl_high = pcls_high[name].unsqueeze(0).to(self.device)
            cd_sph = chamfer_distance_unit_sphere(pcl_up, pcl_high)[0].item()
            results[name] = {
                'cd_sph': cd_sph
            }

        results = pd.DataFrame(results).transpose()
        res_mean = results.mean(axis=0)
        self.logger.info("\n" + repr(results))
        self.logger.info("\nMean\n" + '\n'.join([
            '%s\t%.12f' % (k, v) for k, v in res_mean.items()
        ]))

        update_summary(
            os.path.join(self.summary_dir, 'Summary_%s.csv' % self.dataset),
            model=self.experiment_name,
            metrics={
                'cd_sph(mean)': res_mean['cd_sph']
            }
        )

def update_summary(path, model, metrics):
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, sep="\s*,\s*", engine='python')
    else:
        df = pd.DataFrame()
    for metric, value in metrics.items():
        setting = metric
        if setting not in df.columns:
            df[setting] = np.nan
        df.loc[model, setting] = value
    df.to_csv(path, float_format='%.12f')
    return df
