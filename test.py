import argparse
import torch.utils.tensorboard
from utils.misc import *
from utils.denoise import *
from utils.evaluate import *
from models.denoise import *

def input_iter(input_dir):
    for fn in os.listdir(input_dir):
        if fn[-3:] != 'xyz':
            continue
        pcl_noisy = torch.FloatTensor(np.loadtxt(os.path.join(input_dir, fn)))
        pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
        yield {
            'pcl_noisy': pcl_noisy,
            'name': fn[:-4],
            'center': center,
            'scale': scale
        }

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/pretrained.pt')
parser.add_argument('--input_root', type=str, default='./data/noise')
parser.add_argument('--output_root', type=str, default='./output')
parser.add_argument('--dataset_root', type=str, default='./data/clean')
parser.add_argument('--dataset', type=str, default='Maize')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--resolution', type=str, default='10000_poisson')
parser.add_argument('--noise', type=str, default='0.01')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=2020)
# Denoiser parameters
parser.add_argument('--ld_step_size', type=float, default=1)
parser.add_argument('--ld_num_steps', type=int, default=1)
parser.add_argument('--seed_k', type=int, default=3)
parser.add_argument('--denoise_knn', type=int, default=18)
args = parser.parse_args()
seed_all(args.seed)

# Input/Output
input_dir = os.path.join(args.input_root, '%s_%s_%s' % (args.dataset, args.resolution, args.noise))
save_title = '{dataset}_Ours_{tag}_{res}_{noise}_{time}'.format_map({
    'dataset': args.dataset,
    'tag': args.tag,
    'res': args.resolution,
    'noise': args.noise,
    'time': time.strftime('%m-%d-%H-%M-%S', time.localtime())
})
output_dir = os.path.join(args.output_root, save_title)
os.makedirs(output_dir)
os.makedirs(os.path.join(output_dir, 'pcl'))    # Output point clouds
logger = get_logger('test', output_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Model
ckpt = torch.load(args.ckpt, map_location=args.device)
model = DenoiseNet(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

# Denoise
ld_step_size = args.ld_step_size if args.ld_step_size is not None else ckpt['args'].ld_step_size
logger.info('ld_step_size = %.8f' % ld_step_size)
for data in input_iter(input_dir):
    logger.info(data['name'])
    pcl_noisy = data['pcl_noisy'].to(args.device)

    with torch.no_grad():
        model.eval()
        pcl_next = pcl_noisy
        torch.cuda.synchronize()

        pcl_next = patch_based_denoise_2(
                model=model,
                pcl_noisy=pcl_next,
                ld_step_size=ld_step_size,
                ld_num_steps=args.ld_num_steps,
                seed_k=args.seed_k,
                denoise_knn=args.denoise_knn,
        )
        pcl_denoised = pcl_next.cpu()
        # Denormalize
        pcl_denoised = pcl_denoised * data['scale'] + data['center']
        torch.cuda.synchronize()

    save_path = os.path.join(output_dir, 'pcl', data['name'] + '.xyz')
    np.savetxt(save_path, pcl_denoised.numpy(), fmt='%.8f')

# Evaluate
evaluator = Evaluator(
    output_pcl_dir=os.path.join(output_dir, 'pcl'),
    dataset_root=args.dataset_root,
    dataset=args.dataset,
    summary_dir=args.output_root,
    experiment_name=save_title,
    device=args.device,
    res_gts=args.resolution,
    logger=logger
)
evaluator.run()
