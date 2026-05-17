#!/usr/bin/env python3
"""
Generate MNIST (32x32 RGB) images from trained Generator (.pth) as a grid.

- Rows: classes (in the order provided)
- Cols: seeds   (in the order provided)

Latent sampling matches training code's "dependent content/style":
  if content_dim <= latent_dim//2:
      common_dim = content_dim//2
      z = [z_common(common_dim), z_c(content_dim-common_dim), z_common(common_dim), z_s(style_dim-common_dim)]
  else:
      common_dim = (latent_dim-content_dim)//2
      z = [z_common(common_dim), z_c(content_dim-common_dim), z_common(common_dim), z_s(common_dim)]

Examples:
  SEEDS=111,222,333,444,555,666,777,888,999,121
  CLASSES=2,1,0

  python generate_mnist_grid.py \
    --network /path/to/generator_best.pth \
    --outdir ./results \
    --seeds "$SEEDS" \
    --classes "$CLASSES" \
    --num_class 3 \
    --latent_dim 128 \
    --content_dim 96 \
    --trunc 0.7
"""

import os
from typing import List, Optional

import click
import numpy as np
import torch
import PIL.Image
from PIL import Image

from models import Generator


# -------------------------
# User-specified loader
# -------------------------
def load_generator(model_path: str, device: torch.device, num_classes: int, latent_dim: int, content_dim: int):
    """Load MNIST GAN generator (state_dict) from a .pth file."""
    print(f'Loading generator state_dict from "{model_path}"...')
    G = Generator(num_classes=num_classes, latent_dim=latent_dim, content_dim=content_dim).to(device)
    state = torch.load(model_path, map_location=device)
    # Handle cases where checkpoint might be wrapped
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    G.load_state_dict(state, strict=True)
    G.eval().requires_grad_(False)
    return G


# -------------------------
# Parsing helpers
# -------------------------
def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    """Parse '1,2,3' -> [1,2,3]. Returns None if s is None or empty."""
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


# -------------------------
# Image helpers
# -------------------------
def to_pil(img_t: torch.Tensor) -> PIL.Image.Image:
    """
    img_t: (1,3,H,W) in [-1,1]
    returns PIL RGB image
    """
    img = (img_t.clamp(-1, 1) + 1.0) * 0.5  # [0,1]
    img = (img * 255.0).round().clamp(0, 255).to(torch.uint8)
    img = img.permute(0, 2, 3, 1)[0].cpu().numpy()
    return PIL.Image.fromarray(img, "RGB")



# -------------------------
# CLI
# -------------------------
@click.command()
@click.option("--network", "network_pth", required=True, type=str, help="Path to generator .pth (state_dict).")
@click.option("--outdir", required=True, type=str, help="Output directory.")
@click.option("--seeds", required=True, type=str, help='Comma seeds, e.g. "111,222,333"')
@click.option("--classes", required=True, type=str, help='Comma classes, e.g. "2,1,0" (order preserved)')
@click.option("--num_class", required=True, type=int, help="Total number of classes (domains).")
@click.option("--latent_dim", default=128, show_default=True, type=int)
@click.option("--content_dim", default=96, show_default=True, type=int)
@click.option("--trunc", "truncation_psi", default=0.7, show_default=True, type=float)
@click.option("--device", default="cuda", show_default=True, type=str, help="cuda or cpu")
@click.option(
    "--mode",
    default="random",
    show_default=True,
    type=click.Choice(["random", "same_style", "same_content"], case_sensitive=False),
    help="random: z_c and z_s both vary by seed; same_style: fix z_s, vary z_c; same_content: fix z_c, vary z_s",
)
def main(
    network_pth: str,
    outdir: str,
    seeds: str,
    classes: str,
    num_class: int,
    latent_dim: int,
    content_dim: int,
    truncation_psi: float,
    device: str,
    mode: str,
):
    os.makedirs(outdir, exist_ok=True)

    # Device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    print(f"Using device: {dev}")

    seed_list = parse_int_list(seeds)
    class_list = parse_int_list(classes)

    if seed_list is None or len(seed_list) == 0:
        raise ValueError("No seeds parsed. Provide --seeds like '111,222,333'")
    if class_list is None or len(class_list) == 0:
        raise ValueError("No classes parsed. Provide --classes like '2,1,0'")

    # Validate classes
    for c in class_list:
        if c < 0 or c >= num_class:
            raise ValueError(f"class {c} out of bounds for num_class={num_class}")

    # Load generator using your function
    G = load_generator(network_pth, dev, num_class, latent_dim, content_dim)

    # --- Added: fixed style/content depending on mode (do not change other logic) ---
    fixed_seed = seed_list[0]
    fixed_z_c = None
    fixed_z_s = None
    if mode.lower() == "same_style":
        fixed_z_s = torch.from_numpy(np.random.RandomState(fixed_seed).randn(1, latent_dim-content_dim)).float().to(dev)
    elif mode.lower() == "same_content":
        fixed_z_c = torch.from_numpy(np.random.RandomState(fixed_seed).randn(1, content_dim)).float().to(dev)
    # ------------------------------------------------------------------------------

    # Generate: rows=classes (in given order), cols=seeds (in given order)
    grid_rows: List[List[PIL.Image.Image]] = []

    with torch.no_grad():
        for cls in class_list:
            cls_out = os.path.join(outdir, str(cls))
            os.makedirs(cls_out, exist_ok=True)

            row_imgs: List[PIL.Image.Image] = []
            for j, seed in enumerate(seed_list):
                
                if mode.lower() == "same_style":
                    z_c = torch.from_numpy(np.random.RandomState(seed).randn(1, content_dim)).float().to(dev)
                    z_s = fixed_z_s
                elif mode.lower() == "same_content":
                    z_c = fixed_z_c
                    z_s = torch.from_numpy(np.random.RandomState(seed).randn(1, latent_dim-content_dim)).float().to(dev)
                else:
                    z_c = torch.from_numpy(np.random.RandomState(seed).randn(1, content_dim)).float().to(dev)
                    z_s = torch.from_numpy(np.random.RandomState(seed).randn(1, latent_dim-content_dim)).float().to(dev)


                z = torch.cat([z_c, z_s], dim=1)

                labels = torch.tensor([cls], device=dev, dtype=torch.long)

                img_t = G(z, labels)  # (1,3,32,32) in [-1,1]
                pil = to_pil(img_t)

                pil.save(os.path.join(cls_out, f"seed{seed}.png"))
                row_imgs.append(pil)

                print(f"Generated class={cls} seed={seed} ({j+1}/{len(seed_list)})")

            grid_rows.append(row_imgs)

    # Save combined grid image
    img_w, img_h = grid_rows[0][0].size
    grid_w = len(seed_list) * img_w
    grid_h = len(class_list) * img_h
    grid = Image.new("RGB", (grid_w, grid_h))

    for r, row in enumerate(grid_rows):
        for c, im in enumerate(row):
            grid.paste(im, (c * img_w, r * img_h))

    base = os.path.splitext(os.path.basename(network_pth))[0]
    grid_path = os.path.join(outdir, f"{mode}_{base}_grid.png")
    grid.save(grid_path)
    print(f"Saved grid: {grid_path}")


if __name__ == "__main__":
    main()
