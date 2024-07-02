#!/usr/bin/env python3

"""Samples from k-diffusion models."""

import argparse
from pathlib import Path
import os

import accelerate
import safetensors.torch as safetorch
import torch
from tqdm import trange, tqdm

import k_diffusion as K


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=64,
                   help='the batch size')
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='the checkpoint to use')
    p.add_argument('--config', type=Path,
                   help='the model config')
    p.add_argument('-n', type=int, default=64,
                   help='the number of images to sample')
    p.add_argument('--prefix', type=str, default='out',
                   help='the output prefix')
    p.add_argument('--output', type =str, 
                   help='the output folder')
    p.add_argument('--steps', type=int, default=50,
                   help='the number of denoising steps')
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    if checkpoint.suffix == ".safetensors":
        checkpoint = Path(f'Checkpoint{args.prefix}/')
    else:
        args.checkpoint = Path(f'Checkpoint{args.prefix}/{args.checkpoint}')

    if not os.path.exists(f'Samples{args.prefix}'):
        os.makedirs(f'Samples{args.prefix}')
        print(f"New directory created: Samples{args.prefix}")
    config = K.config.load_config(args.config if args.config else args.checkpoint)
    model_config = config['model']
    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

    inner_model = K.config.make_model(config).eval().requires_grad_(False).to(device)
    inner_model.load_state_dict(safetorch.load_file(args.checkpoint))

    accelerator.print('Parameters:', K.utils.n_params(inner_model))
    model = K.Denoiser(inner_model, sigma_data=model_config['sigma_data'])

    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']

    @torch.no_grad()
    @K.utils.eval_mode(model)
    def run():
        if accelerator.is_local_main_process:
            tqdm.write('Sampling...')
        sigmas = K.sampling.get_sigmas_karras(args.steps, sigma_min, sigma_max, rho=7., device=device)
        def sample_fn(n):
            x = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
            x_0 = K.sampling.sample_lms(model, x, sigmas, disable=not accelerator.is_local_main_process)
            return x_0
        x_0 = K.evaluation.compute_features(accelerator, sample_fn, lambda x: x, args.n, args.batch_size)
        if accelerator.is_main_process:
            for i, out in enumerate(x_0):
                filename = f'Samples{args.prefix}/out_{i:05}.png'
                K.utils.to_pil_image(out).save(filename)

    try:
        run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
