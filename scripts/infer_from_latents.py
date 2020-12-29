import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import pickle

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp


def main():
    test_opts = TestOptions().parse()
    if test_opts.resize_factors is not None:
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
                                        'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.cuda()

    with Path('/home/yotam/projects/rewriting/data/wplus/cats/subset/flickr_cat_000008.pickle').open('rb') as fp:
        cat_wplus = pickle.load(fp)

    person_z = np.random.randn(1, 512).astype('float32')
    _, person_w = net(torch.from_numpy(person_z).to("cuda"),
                      input_code=True,
                      return_latents=True)

    base_dir = Path('/home/yotam/projects/pixel2style2pixel/experiments/mixing_people_and_cats/')
    largest = max([int(f.stem) for f in base_dir.iterdir()])
    new = largest + 1
    new_dir = base_dir.joinpath(f'{new}')
    new_dir.mkdir(exist_ok=True)

    for i in tqdm(range(18)):
        mix_at_layer = i

        # mixed_w_plus = torch.cat([cat_wplus[:, :mix_at_layer, :], person_w[:, mix_at_layer:, :]], dim=1)

        mixed_w_plus = torch.cat([person_w[:, :mix_at_layer, :], cat_wplus[:, mix_at_layer:, :]], dim=1)


        mixed_images = net.decoder([mixed_w_plus],
                                   input_is_latent=True,
                                   randomize_noise=False,
                                   return_latents=False)

        mixed_images = tensor2im(mixed_images[0][0])
        Image.fromarray(np.array(mixed_images)).save(str(new_dir.joinpath(f'mixed_at_layer_{mix_at_layer:03}.png')))


if __name__ == '__main__':
    main()
