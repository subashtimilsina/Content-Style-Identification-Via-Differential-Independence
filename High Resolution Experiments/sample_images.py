# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import sys

import legacy
from PIL import Image

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--classes', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option(
    '--mode', 'sampling_mode',
    type=click.Choice(['same_content', 'same_style']),
    default='same_content',
    show_default=True,
    help='Latent reuse mode: keep content fixed and vary style, or keep style fixed and vary content.'
)

def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    classes: Optional[List[int]] = None,
    sampling_mode: str = 'same_content',
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)

    device = torch.device('cuda')
    print(f"Using device: {device}")
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    num_class = G.mapping.c_dim
    i_dim = G.mapping.i_dim
    z_dim = G.z_dim

    print(num_class, i_dim, z_dim)


    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print ('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # seeds = [int(seed) for seed in range(11)]

    # Generate images and save as a grid
    classes_list = classes if classes is not None else [i for i in range(num_class)]
    grid_images = []

    same_content = (sampling_mode == 'same_content')
    if same_content:
        # Same content for all styles
        z_c = torch.from_numpy(np.random.RandomState(seeds[0]).randn(1, G.z_dim - i_dim)).to(device)
    else:

        z_s = torch.from_numpy(np.random.RandomState(seeds[0]).randn(1, i_dim)).to(device)
            
    
    # num_class = 5
    for class_idx in range(num_class):
        # if class_idx not in classes_list:
        #     continue
        if class_idx ==1:
            continue
        label = torch.zeros([1, G.c_dim], device=device)
        label[:, class_idx] = 1

        os.makedirs(outdir+'/{}'.format(class_idx), exist_ok=True)
        row_images = []
        for seed_idx, seed in enumerate(seeds):
            # seed = seed*1000 + class_idx*10000+ seed_idx*100
            print(seed_idx, seed)
            sys.stderr.write('\rGenerating image of class %d for seed %d (%d/%d) ...' % (class_idx, seed, seed_idx, len(seeds)))

            if same_content:
                z_s = torch.from_numpy(np.random.RandomState(seed).randn(1, i_dim)).to(device)
                z = torch.cat([z_s, z_c], dim=1)
                img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            else:

                z_c = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim - i_dim)).to(device)
                z = torch.cat([z_s, z_c], dim=1)
                img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)

            
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
            row_images.append(img)
            img.save(f'{outdir}/{class_idx}/seed{seed:04d}.png')

        
        grid_images.append(row_images)

    # Create a grid image
    grid_width = len(seeds)
    grid_height = len(grid_images)
    img_width, img_height = grid_images[0][0].size
    grid = Image.new('RGB', (grid_width * img_width, grid_height * img_height))

    for row_idx, row_images in enumerate(grid_images):
        for col_idx, img in enumerate(row_images):
            grid.paste(img, (col_idx * img_width, row_idx * img_height))

    img_str_name = network_pkl.split('/')[-2][:5]
    grid.save(f'{outdir}{sampling_mode}_'+img_str_name+'_grid.png')
    print(f"Grid image saved at {outdir}{sampling_mode}_"+img_str_name+'_grid.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------