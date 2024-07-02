#!/usr/bin/env python3

"""Samples from k-diffusion cascaded models."""
"""Here you need to put different checkpoint name"""

import argparse
from pathlib import Path
import os

import accelerate
import safetensors.torch as safetorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm

import k_diffusion as K



def block_noise(ref_x, randn_like=torch.randn_like, block_size=1, device=None):
    """
    build block noise
    """
    g_noise = randn_like(ref_x)
    if block_size == 1:
        return g_noise
    
    blk_noise = torch.zeros_like(ref_x, device=device)
    for px in range(block_size):
        for py in range(block_size):
            blk_noise += torch.roll(g_noise, shifts=(px, py), dims=(-2, -1))
            
    blk_noise = blk_noise / block_size # to maintain the same std on each pixel
    
    return blk_noise

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=64,
                   help='the batch size')
    p.add_argument('--checkpoint1', type=Path, required=True,
                   help='the first checkpoint to use')
    p.add_argument('--checkpoint2', type=Path, required=True,
                   help='the second checkpoint to use')
    p.add_argument('--config', type=Path,
                   help='the model config')
    p.add_argument('-n1', type=int, default=64,
                   help='the number of images to sample in the first model')
    p.add_argument('-n2', type=int, default=64,
                   help='the number of images to sample in the second model')
    p.add_argument('--prefix1', type=str, default='out',
                   help='the first output prefix')
    p.add_argument('--prefix2', type=str, default='out',
                   help='the second output prefix')
    p.add_argument('--output', type=str, default='out',
                   help='the output folder')
    p.add_argument('--steps', type=int, default=50,
                   help='the number of denoising steps')
    args = p.parse_args()

    args.checkpoint1 = Path(f'Checkpoint{args.prefix1}/{args.checkpoint1}')
    args.checkpoint2 = Path(f'Checkpoint{args.prefix2}/{args.checkpoint2}')

    if not os.path.exists(f'Samples{args.output}'):
        os.makedirs(f'Samples{args.output}')
        print(f"New directory created: Samples{args.output}")
    config1 = K.config.load_config(args.config if args.config else args.checkpoint1)
    model_config1 = config1['model']
    print(model_config1['sigma_data'])
    config2 = K.config.load_config(args.config if args.config else args.checkpoint2)
    model_config2 = config2['model']

    # TODO: allow non-square input sizes
    assert len(model_config1['input_size']) == 2 and model_config1['input_size'][0] == model_config1['input_size'][1]
    assert len(model_config2['input_size']) == 2 and model_config2['input_size'][0] == model_config2['input_size'][1]
    size1 = model_config1['input_size']
    size2 = model_config2['input_size']

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

    inner_model1 = K.config.make_model(config1).eval().requires_grad_(False).to(device)
    inner_model1.load_state_dict(safetorch.load_file(args.checkpoint1))
    inner_model2 = K.config.make_model(config2).eval().requires_grad_(False).to(device)
    inner_model2.load_state_dict(safetorch.load_file(args.checkpoint2))

    accelerator.print('Parameters:', K.utils.n_params(inner_model1))
    model1 = K.Denoiser(inner_model1, sigma_data=model_config1['sigma_data'])
    model2 = K.Denoiser(inner_model2, sigma_data=model_config2['sigma_data'])

    sigma_min = model_config1['sigma_min']
    sigma_max = model_config1['sigma_max'] 

    start_step = int(args.steps * 3/4)

    @torch.no_grad()
    @K.utils.eval_mode(model1)
    @K.utils.eval_mode(model2)
    def run():
        if accelerator.is_local_main_process:
            tqdm.write('Sampling...')
        sigmas = K.sampling.get_sigmas_karras(args.steps, sigma_min, sigma_max, rho=7., device=device)
        noise_ratio = sigmas[start_step] / sigma_max
        print(f'start_step is {start_step}')
        print(sigmas[0], sigmas[-1], sigmas[start_step])
        def sample_fn1(n):
            x = torch.randn([n, model_config1['input_channels'], size1[0], size1[1]], device=device) * sigma_max
            x_0 = K.sampling.sample_lms(model1, x, sigmas, disable=not accelerator.is_local_main_process)
            return x_0
        def sample_fn2(n, x):
            noise = torch.randn([n, model_config2['input_channels'], size2[0], size2[1]], device=device) * sigma_max
            image = noise_ratio * noise + (1 - noise_ratio) * x
            x_0 = K.sampling.sample_lms(model2, image, sigmas, disable=not accelerator.is_local_main_process, start_step=start_step)
            return x_0
        x_0 = K.evaluation.compute_features(accelerator, sample_fn1, lambda x: x, args.n1, args.batch_size)

        x_0 = F.interpolate(x_0, scale_factor=2, mode='nearest')
        if accelerator.is_main_process:
            for i, out in enumerate(x_0):
                filename = f'Samples{args.output}/out1_{i:05}.png'
                K.utils.to_pil_image(out).save(filename)
        print(x_0.shape)
        x_0 = K.evaluation.compute_features(accelerator, sample_fn2, lambda x: x, args.n2, args.batch_size, x=x_0)
        
        
        if accelerator.is_main_process:
            for i, out in enumerate(x_0):
                filename = f'Samples{args.output}/out2_{i:05}.png'
                K.utils.to_pil_image(out).save(filename)

    try:
        run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
