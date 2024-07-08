#!/usr/bin/env python3

"""Samples from k-diffusion cascaded models."""
"""Here you need to put different checkpoint name"""

import argparse
from pathlib import Path
import os
import paramiko
from scp import SCPClient

import accelerate
import safetensors.torch as safetorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm

import k_diffusion as K

ip = "192.168.67.20"
port = 2222

def create_ssh_client(server, port, user, password):
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, port, user, password)
    return ssh

def transfer_folder(ssh, remote_path, local_path):
    with SCPClient(ssh.get_transport()) as scp:
        scp.get(remote_path, local_path, recursive=True)

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

def schedule():
    """
    return: list of schedule split"""
    L = []
    return L

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=64,
                   help='the batch size')
    p.add_argument('--checkpoints', type=Path, nargs='+',
                   help='the first checkpoint to use')
    p.add_argument('--config', type=Path,
                   help='the model config')
    p.add_argument('--n', type=int, default=64,
                   help='the number of images to sample')
    p.add_argument('--prefix', type=str, nargs='+',
                   help='the first output prefix')
    p.add_argument('--output', type=str, default='out',
                   help='the output folder')
    p.add_argument('--steps', type=int, default=50,
                   help='the number of denoising steps')
    args = p.parse_args()
    
    if not os.path.exists(f'Samples/{args.output}'):
        os.makedirs(f'Samples/{args.output}')
        print(f"New directory created: Samples/{args.output}")
    configs = []
    model_configs = []
    inner_models = []
    models = []
    sigmas_min = []
    sigmas_max = []
    sizes = []
    start_steps = []
    for i,checkpoint in enumerate(args.checkpoints):
        checkpoint = Path(f'Checkpoint/{args.prefix[i]}/{checkpoint}')

        configs.append(K.config.load_config(checkpoint))
        model_configs.append(configs[i]['model'])

        # TODO: allow non-square input sizes
        assert len(model_configs[i]['input_size']) == 2 and model_configs[i]['input_size'][0] == model_configs[i]['input_size'][1]
        sizes.append(model_configs[i]['input_size'])

        accelerator = accelerate.Accelerator()
        device = accelerator.device
        print('Using device:', device, flush=True)

        inner_models.append(K.config.make_model(configs[i]).eval().requires_grad_(False).to(device))
        inner_models[i].load_state_dict(safetorch.load_file(checkpoint))


        accelerator.print('Parameters:', K.utils.n_params(inner_models[i]))
        models.append(K.Denoiser(inner_models[i], sigma_data=model_configs[i]['sigma_data']))


        sigmas_min.append(model_configs[i]['sigma_min'])
        sigmas_max.append(model_configs[i]['sigma_max'])

        start_step = int(args.steps * 9/10)

    
    for i in range(len(args.checkpoints)):
        K.utils.eval_mode(models[i])
        
    @torch.no_grad()
    def run():
        if accelerator.is_local_main_process:
            tqdm.write('Sampling...') 
        model, model_config, sigma_min, sigma_max = models[0], model_configs[0], sigmas_min[0], sigmas_max[0]
        sigmas = K.sampling.get_sigmas_karras(args.steps, sigma_min, sigma_max, rho=7., device=device)
        def sample_fn1(n):
            x = torch.randn([n, model_config['input_channels'], sizes[0][0], sizes[0][1]], device=device) * sigma_max
            x_0 = K.sampling.sample_lms(model, x, sigmas, disable=not accelerator.is_local_main_process)
            return x_0
        x_0 = K.evaluation.compute_features(accelerator, sample_fn1, lambda x: x, args.n, args.batch_size)

        if accelerator.is_main_process:
            for i, out in enumerate(x_0):
                filename = f'Samples/{args.output}/out0_{i:05}.png'
                K.utils.to_pil_image(out).save(filename)
        j = 1
        if len(args.checkpoints) != 1:
            for model,model_config,sigma_min,sigma_max,size in zip(models[1:],model_configs[1:],sigmas_min[1:],sigmas_max[1:],sizes[1:]):
                sigmas = K.sampling.get_sigmas_karras(args.steps, sigma_min, sigma_max, rho=7., device=device)
                noise_ratio = sigmas[start_step] / sigma_max
                print(sigmas[0], sigmas[-1], sigmas[start_step])


                x_0 = F.interpolate(x_0, scale_factor=2, mode='nearest')

                print(x_0.shape)
                    
                def sample_fn2(n, x):
                    noise = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
                    image = noise_ratio * noise + (1 - noise_ratio) * x
                    x_0 = K.sampling.sample_lms(model, image, sigmas, disable=not accelerator.is_local_main_process, start_step=start_step)
                    return x_0
                x_0 = K.evaluation.compute_features(accelerator, sample_fn2, lambda x: x, args.n, args.batch_size, x=x_0)
                
                if accelerator.is_main_process:
                    for i, out in enumerate(x_0):
                        filename = f'Samples/{args.output}/out{j}_{i:05}.png'
                        K.utils.to_pil_image(out).save(filename)
                j += 1

        


    try:
        run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
