#!/usr/bin/env python3
"""
StyleGAN2-ADA latent translation (content/style) using the SAME projection logic as your original code,
but saving results as a tight content x style grid image per batch.

Added features:
  - Select which content rows and style columns appear in the grid via:
        --content_idx "0,2,5"   or "0:8:2" or "1-4" or "-1"
        --style_idx   "0,1,3"   or "2:10:2" etc.
    (indices are within the CURRENT batch)

  - Optionally synthesize only certain grid positions (sparse fill) via:
        --grid_cells "0-0,0-3,2-1"
    where (i,j) are LOCAL grid coords after applying content_idx/style_idx.
    Unspecified cells remain blank.

Example:
python translate_grid.py \
  --network /path/to/network.pkl \
  --content /path/to/content_dir \
  --style /path/to/style_dir \
  --outdir ./out \
  --source_class 0 --target_class 1 \
  --num-steps 200 --batchsize 32 \
  --grid_n 7 --grid_prefix cs_grid \
  --content_idx "0,2,5" --style_idx "1,3,4" \
  --grid_cells "0-0,0-2,2-1"

If you want the old individual saving behavior:
  add: --save_individual
"""

import os
import random
import pickle

import click
import imageio
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import dnnlib
import legacy
from tqdm import trange


def _sanitize_stem(path: str) -> str:
    """Return a filesystem-safe stem (basename without extension)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = stem.replace(" ", "_")
    stem = "".join(ch if (ch.isalnum() or ch in ["-", "_", "."]) else "_" for ch in stem)
    return stem


def _unique_path(out_path: str) -> str:
    """If out_path exists, append _1, _2, ..."""
    if not os.path.exists(out_path):
        return out_path
    base, ext = os.path.splitext(out_path)
    k = 1
    while True:
        cand = f"{base}_{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1


def postprocess_and_save_named(
    trans_img: torch.Tensor,          # [B,3,H,W] in [-1,1]
    outdir: str,
    src_path: str,
    tgt_path: str = None,
    target_class: int = None,
):
    trans_img = (trans_img + 1.0) * (255 / 2.0)
    trans_img = trans_img.clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()

    src_stem = _sanitize_stem(src_path)

    if tgt_path is not None:
        tgt_stem = _sanitize_stem(tgt_path)
        base_name = f"{src_stem}_{tgt_stem}"
    else:
        # fallback when no explicit target image
        if target_class is None:
            base_name = f"{src_stem}_randomstyle"
        else:
            base_name = f"{src_stem}_class{target_class}"

    os.makedirs(outdir, exist_ok=True)

    # Usually B=1 here, but keep it general
    for j in range(trans_img.shape[0]):
        name = f"{base_name}.png" if trans_img.shape[0] == 1 else f"{base_name}_{j}.png"
        out_path = _unique_path(os.path.join(outdir, name))
        imageio.imwrite(out_path, trans_img[j])


def logprint(*args, verbose=True):
    if verbose:
        print(*args)


# --------------------------
# NEW: index/cell spec parsers
# --------------------------
def _parse_index_spec(spec: str, max_len: int):
    """
    Parse index spec within [0, max_len).

    Supports:
      - comma list: "0,2,5"
      - python slice: "0:8:2" / "3:10" / ":7"
      - inclusive range: "2-6"   (note: "a-b" where a is not negative)
      - negative indices: "-1" (last), "-2", ...

    Returns a de-duplicated list preserving order.
    If spec is empty -> returns None (means "use default").
    """
    spec = (spec or "").strip()
    if spec == "":
        return None

    items = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        # slice form a:b:c
        if ":" in part:
            toks = part.split(":")
            if len(toks) not in (2, 3):
                raise click.BadParameter(f"Bad slice spec: '{part}'")
            start = int(toks[0]) if toks[0] != "" else 0
            stop = int(toks[1]) if toks[1] != "" else max_len
            step = int(toks[2]) if (len(toks) == 3 and toks[2] != "") else 1
            items.extend(list(range(start, stop, step)))
            continue

        # inclusive range a-b (but not negative like "-1")
        if "-" in part and not part.startswith("-"):
            a, b = part.split("-", 1)
            a = int(a)
            b = int(b)
            if b < a:
                raise click.BadParameter(f"Bad range (b<a): '{part}'")
            items.extend(list(range(a, b + 1)))
            continue

        # single int (can be negative)
        items.append(int(part))

    # normalize negatives + bounds check + unique preserve order
    out = []
    seen = set()
    for idx in items:
        if idx < 0:
            idx = max_len + idx
        if not (0 <= idx < max_len):
            raise click.BadParameter(f"Index {idx} out of range for batch length {max_len}")
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def _parse_cell_spec(spec: str, Nc: int, Ns: int):
    """
    Parse sparse cell list like '0-0,0-2,3-1' where i in [0,Nc), j in [0,Ns).
    Accepts i-j or i:j, supports negative indices.
    Returns list of (i,j) unique preserving order.
    If spec is empty -> returns None (means "full cartesian").
    """
    spec = (spec or "").strip()
    if spec == "":
        return None

    pairs = []
    seen = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            i_s, j_s = part.split("-", 1)
        elif ":" in part:
            i_s, j_s = part.split(":", 1)
        else:
            raise click.BadParameter(f"Bad cell '{part}'. Use 'i-j' like '2-3'.")

        i = int(i_s)
        j = int(j_s)
        if i < 0:
            i = Nc + i
        if j < 0:
            j = Ns + j

        if not (0 <= i < Nc) or not (0 <= j < Ns):
            raise click.BadParameter(f"Cell ({i},{j}) out of range for grid {Nc}x{Ns}")

        if (i, j) not in seen:
            seen.add((i, j))
            pairs.append((i, j))
    return pairs


# --------------------------
# Grid helpers (PIL, tight)
# --------------------------
def chw_u8_batch_to_hwc_list(x_u8_chw: np.ndarray):
    """
    x_u8_chw: [B,3,H,W] uint8 -> list of [H,W,3] uint8
    """
    assert x_u8_chw.ndim == 4 and x_u8_chw.shape[1] == 3
    return [x_u8_chw[i].transpose(1, 2, 0) for i in range(x_u8_chw.shape[0])]


def bchw_minus1_1_to_hwc_u8_list(x: torch.Tensor):
    """
    x: [B,3,H,W] float in [-1,1] -> list of [H,W,3] uint8
    """
    x_u8 = ((x + 1.0) * (255.0 / 2.0)).clamp(0, 255).to(torch.uint8)
    x_u8 = x_u8.detach().cpu().permute(0, 2, 3, 1).numpy()
    return [x_u8[i] for i in range(x_u8.shape[0])]


def save_content_style_grid_pil(
    content_refs_hwc_u8,   # list length Nc, each [H,W,3] uint8
    style_refs_hwc_u8,     # list length Ns, each [H,W,3] uint8
    gen_cells_hwc_u8,      # list length Nc*Ns in row-major; may contain None for blank cells
    out_path: str,
    bg=(255, 255, 255),
):
    """
    Creates a tight grid with:
      (0,0) blank
      top row: style refs
      left col: content refs
      inner: generated cells (skip if None)
    """
    Nc = len(content_refs_hwc_u8)
    Ns = len(style_refs_hwc_u8)

    assert len(gen_cells_hwc_u8) == Nc * Ns
    assert Nc > 0 and Ns > 0

    H, W, _ = content_refs_hwc_u8[0].shape
    canvas = Image.new("RGB", ((Ns + 1) * W, (Nc + 1) * H), color=bg)

    # Top row = styles
    for j in range(Ns):
        canvas.paste(Image.fromarray(style_refs_hwc_u8[j]), ((j + 1) * W, 0))

    # Left col = contents + inner generated
    idx = 0
    for i in range(Nc):
        canvas.paste(Image.fromarray(content_refs_hwc_u8[i]), (0, (i + 1) * H))
        for j in range(Ns):
            cell = gen_cells_hwc_u8[idx]
            if cell is not None:
                canvas.paste(Image.fromarray(cell), ((j + 1) * W, (i + 1) * H))
            idx += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)
    print(f"Saved grid: {out_path}")


# --------------------------
# Translation class (same logic)
# --------------------------
class Translation:
    def __init__(
        self,
        generator_path,
        c_dim,
        i_dim,
        num_steps=1000,
        psi=0.7,
        device="cuda",
        num_styles_per_content=1,
        initial_learning_rate=0.1,
        use_noise=True,
        random_start=False,
    ):
        self.device = device
        self.c_dim = c_dim
        self.i_dim = i_dim
        self.num_steps = num_steps
        self.psi = psi
        self.num_styles_per_content = num_styles_per_content
        self.initial_learning_rate = initial_learning_rate
        self.use_noise = use_noise
        self.random_start = random_start

        self._load_networks(generator_path)
        self._compute_stats()

        self.fixed_style = torch.randn(1, self.G.z_dim, device=self.device)

    def _load_networks(self, generator_path):
        print(f'Loading networks from "{generator_path}"...')
        device = torch.device(self.device)
        with dnnlib.util.open_url(generator_path) as fp:
            self.G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)

        # VGG16 used in StyleGAN2-ADA projection
        url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
        with dnnlib.util.open_url(url) as f:
            self.vgg16 = torch.jit.load(f).eval().to(device)

    def _compute_stats(self):
        w_avg_samples = 10000
        z_samples = np.random.RandomState(123).randn(w_avg_samples, self.G.z_dim)

        c = torch.zeros([z_samples.shape[0], self.c_dim], dtype=torch.float32, device=self.device)
        cum_num_samples_per_class = (
            [z_samples.shape[0] // self.c_dim] * (self.c_dim - 1)
            + [z_samples.shape[0] - (z_samples.shape[0] // self.c_dim) * (self.c_dim - 1)]
        )
        idx_class = [0] + list(np.cumsum(cum_num_samples_per_class))
        for i in range(self.c_dim):
            c[idx_class[i] : idx_class[i + 1], i] = 1

        latent_samples = (
            self.G.mapping(torch.from_numpy(z_samples).to(self.device), c)
            .cpu()
            .numpy()
            .astype(np.float32)
        )  # [N, num_ws, C]
        content, style = latent_samples[:, 0], latent_samples[:, -1]
        styles = [style[idx_class[i] : idx_class[i + 1]] for i in range(self.c_dim)]

        self.latent_avg = np.mean(latent_samples, axis=0, keepdims=True)  # [1, num_ws, C]
        self.latent_std = (np.sum((latent_samples - self.latent_avg) ** 2) / w_avg_samples) ** 0.5
        self.content_avg = np.mean(content, axis=0, keepdims=True)  # [1, C]
        self.styles_avg = [np.mean(s, axis=0, keepdims=True) for s in styles]  # [1, C]
        self.content_std = (np.sum((content - self.content_avg) ** 2) / w_avg_samples) ** 0.5
        self.styles_std = [
            (np.sum((s - self.styles_avg[i]) ** 2) / cum_num_samples_per_class[i]) ** 0.5
            for i, s in enumerate(styles)
        ]  # [1, C]

        print("content_std", self.content_std)
        print("styles_std", self.styles_std)
        print("latent_std", self.latent_std)
        print("cum num_samples per class", cum_num_samples_per_class)

    def refresh_style(self):
        self.fixed_style = torch.randn(1, self.G.z_dim, device=self.device)

    def get_ws_from_c_s(self, c, s):
        return np.concatenate(
            [
                np.repeat(np.expand_dims(c, axis=1), self.G.mapping.num_c_res, axis=1),
                np.repeat(np.expand_dims(s, axis=1), self.G.num_ws - self.G.mapping.num_c_res, axis=1),
            ],
            axis=1,
        )

    def get_ws_from_c_s_torch(self, c, s):
        if len(c.shape) > 2:
            return torch.cat(
                [c, s.unsqueeze(1).repeat(1, self.G.num_ws - self.G.mapping.num_c_res, 1)],
                axis=1,
            )
        return torch.cat(
            [
                c.unsqueeze(1).repeat(1, self.G.mapping.num_c_res, 1),
                s.unsqueeze(1).repeat(1, self.G.num_ws - self.G.mapping.num_c_res, 1),
            ],
            axis=1,
        )

    def project(self, img, cls=None):
        """
        Takes a batch of images (uint8 0..255 CHW) and returns the optimized ws latents.
        """
        if cls is not None:
            latent_init = self.get_ws_from_c_s(self.content_avg, self.styles_avg[cls])  # [1, num_ws, C]
        else:
            latent_init = self.latent_avg
        latent_init = np.repeat(latent_init, img.shape[0], axis=0)  # [N, num_ws, C]

        if self.random_start:
            c = torch.zeros([latent_init.shape[0], self.c_dim], dtype=torch.float32, device=self.device)
            c[:, cls] = 1
            latent_init = (
                self.G.mapping(torch.randn(img.shape[0], self.G.z_dim).to(self.device), c)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        img = img.to(self.device).to(torch.float32)

        # VGG features for target image (expects 0..255 float)
        img_features = self.vgg16(img, resize_images=False, return_lpips=True)

        latent = self.optimize_latent(latent_init, img_features, self.latent_std, img)
        return latent

    def postprocess(self, content_hat, target_class, fixed_style=False):
        """
        Combines content with a (random or fixed) style from target_class (used when no style_dir is provided).
        """
        if not fixed_style:
            random_z = torch.randn(content_hat.shape[0], self.G.z_dim, device=self.device)
        else:
            random_z = self.fixed_style.repeat(content_hat.shape[0], 1)
        target_c = torch.zeros([content_hat.shape[0], self.c_dim], dtype=torch.float32, device=self.device)
        target_c[:, target_class] = 1
        latent = self.G.mapping(random_z, target_c)
        style_hat = latent[:, -1]
        if self.psi is not None:
            style_hat = (1 - self.psi) * torch.tensor(self.styles_avg[target_class]).to(self.device).float() + self.psi * style_hat
        return self.get_ws_from_c_s_torch(content_hat, style_hat)

    def optimize_latent(
        self,
        w_avg,
        target_features,
        w_std,
        content_images,
        initial_noise_factor=0.05,
        noise_ramp_length=0.75,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        initial_learning_rate=0.1,
        is_style=False,
    ):
        initial_learning_rate = self.initial_learning_rate

        content = torch.tensor(w_avg[:, 0].copy(), dtype=torch.float32, device=self.device, requires_grad=True)
        style = torch.tensor(w_avg[:, -1].copy(), dtype=torch.float32, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([content, style], betas=(0.9, 0.999), lr=initial_learning_rate)

        for step in trange(self.num_steps):
            w_opt = self.get_ws_from_c_s_torch(content, style)

            # schedules (unchanged)
            t = step / self.num_steps
            w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # synth
            if self.use_noise:
                w_noise = torch.randn_like(w_opt) * w_noise_scale
                w_in = w_opt + w_noise
            else:
                w_in = w_opt

            synth_images = self.G.synthesis(w_in, noise_mode="const")

            # VGG uses 0..255 float
            synth_images_255 = (synth_images + 1) * (255 / 2)
            if synth_images_255.shape[2] > 256:
                synth_images_255 = F.interpolate(synth_images_255, size=(256, 256), mode="area")

            synth_features = self.vgg16(synth_images_255, resize_images=False, return_lpips=True)
            loss = (target_features - synth_features).square().sum() / synth_features.shape[0]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        return w_opt.detach()

    def get_batch(self, filenames):
        images = []
        for target_fname in filenames:
            target_pil = Image.open(target_fname).convert("RGB")
            w, h = target_pil.size
            s = min(w, h)
            target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            target_pil = target_pil.resize((self.G.img_resolution, self.G.img_resolution), Image.LANCZOS)
            target_uint8 = np.array(target_pil, dtype=np.uint8)
            images.append(target_uint8.transpose([2, 0, 1]))  # CHW
        return np.stack(images, axis=0)


def postprocess_and_save(trans_img, outdir, counter):
    """
    Old behavior: save each generated image individually.
    """
    trans_img = (trans_img + 1.0) * (255 / 2.0)
    trans_img = trans_img.clamp(0, 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
    for j in range(len(trans_img)):
        imageio.imwrite(f"{outdir}/{counter}_{j}.png", trans_img[j])


# --------------------------
# CLI: Translation (grid)
# --------------------------
@click.command()
@click.option("--network", "network_pkl", help="Network pickle filename", required=True)
@click.option("--content", "content_dir", help="Content image folder", required=True, metavar="DIR")
@click.option("--style", "style_dir", help="Style image folder", required=False, metavar="DIR")
@click.option("--num-steps", help="Number of optimization steps", type=int, default=200, show_default=True)
@click.option("--seed", help="Random seed", type=int, default=303, show_default=True)
@click.option("--outdir", help="Where to save the output images", required=True, metavar="DIR")
@click.option("--batchsize", help="Batch size for projection", type=int, default=32, show_default=True)
@click.option("--psi", help="Truncation psi (only used when style_dir is None)", type=float, default=None, show_default=True)
@click.option("--source_class", help="Source class for the projection", type=int, default=None, show_default=True)
@click.option("--target_class", help="Target class for the projection", type=int, default=None, show_default=True)
@click.option("--num_styles_per_content", help="(kept for compatibility)", type=int, default=1, show_default=True)
@click.option("--c_dim", type=int, default=3, show_default=True)

@click.option("--from_projection", help="If true, will run from projection", type=bool, default=False)
@click.option("--projection_pkl", help="projection pkl", type=str, default="")

@click.option("--save_img_to_pkl", help="save images to pkl", type=bool, default=False)
@click.option("--img_pkl", help="save images to pkl", type=str, default="")

@click.option("--atob", help="direction of translation", type=bool, default=True)
@click.option("--max_images", help="max images to generate", type=int, default=1000, show_default=True)

# Grid options
@click.option("--grid_n", help="Grid size N (NxN) default selection size", type=int, default=7, show_default=True)
@click.option("--grid_prefix", help="Prefix for saved grid files", type=str, default="cs_grid", show_default=True)
@click.option("--synth_batch", help="Batch size for synthesis (grid cells)", type=int, default=32, show_default=True)
@click.option("--save_grid/--save_individual", help="Save grids (default) or individual images", default=True, show_default=True)

# NEW: selection options
@click.option(
    "--content_idx",
    type=str,
    default="",
    show_default=True,
    help="Content indices within the CURRENT batch. Examples: '0,2,5' or '0:8:2' or '0-3' or '-1'. "
         "Empty -> first min(grid_n, batch).",
)
@click.option(
    "--style_idx",
    type=str,
    default="",
    show_default=True,
    help="Style indices within the CURRENT batch. Examples: '0,1,4' or '1:10:3' or '-1'. "
         "Empty -> first min(grid_n, batch).",
)
@click.option(
    "--grid_cells",
    type=str,
    default="",
    show_default=True,
    help="Optional sparse cells to synthesize using LOCAL grid coords after content_idx/style_idx. "
         "Format: 'i-j,i-j,...' e.g. '0-0,0-2,3-1'. Empty -> synthesize full cartesian product.",
)
def run_translation(
    network_pkl: str,
    content_dir: str,
    style_dir: str,
    num_steps: int,
    seed: int,
    outdir: str,
    batchsize: int,
    psi: float,
    source_class: int,
    target_class: int,
    num_styles_per_content: int,
    c_dim: int,
    from_projection: bool,
    projection_pkl: str,
    save_img_to_pkl: bool,
    img_pkl: str,
    atob: bool,
    max_images: int,
    grid_n: int,
    grid_prefix: str,
    synth_batch: int,
    save_grid: bool,
    content_idx: str,
    style_idx: str,
    grid_cells: str,
):
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    counter = 0

    translation = Translation(
        network_pkl,
        c_dim=c_dim,
        i_dim=16,
        num_steps=num_steps,
        psi=psi,
        num_styles_per_content=num_styles_per_content,
        initial_learning_rate=0.01,
        use_noise=True,
        random_start=False,
    )

    content_files = sorted([os.path.join(content_dir, f) for f in os.listdir(content_dir)])
    style_files = None
    if style_dir is not None:
        style_files = sorted([os.path.join(style_dir, f) for f in os.listdir(style_dir)])
        max_len = min(len(content_files), len(style_files))
        content_files = content_files[:max_len]
        style_files = style_files[:max_len]
        if not save_img_to_pkl:
            random.shuffle(style_files)

    if from_projection:
        print("getting projections")
        with open(projection_pkl, "rb") as f:
            projections = pickle.load(f)

    # pkl saving setup (kept as-is)
    if save_img_to_pkl:
        if os.path.exists(img_pkl):
            with open(img_pkl, "rb") as f:
                save_dict = pickle.load(f)
            fill_dummy = False
        else:
            fill_dummy = True
            save_dict = {"real_A": [], "real_B": [], "fake_A": [], "fake_B": [], "rec_A": [], "rec_B": [], "ref_A": [], "ref_B": []}

        if atob:
            save_dict["real_A"] = []
            save_dict["fake_B"] = []
            save_dict["rec_A"] = []
            save_dict["ref_A"] = []
        else:
            save_dict["real_B"] = []
            save_dict["fake_A"] = []
            save_dict["rec_B"] = []
            save_dict["ref_B"] = []

    if not save_img_to_pkl:
        random.shuffle(content_files)

    # sanity: grid requires styles if saving grid in "content×style" form
    if save_grid and (style_files is None):
        raise RuntimeError("Grid mode requires --style DIR. (If you want random style per content, disable grid or extend logic.)")

    # main loop
    for batch_start in trange(0, min(len(content_files), max_images), batchsize):
        bs = min(batchsize, min(len(content_files), max_images) - batch_start)
        if bs <= 0:
            break

        # -----------------------------
        # Obtain content/style latents (same technique as before)
        # -----------------------------
        if not from_projection:
            # load images as uint8 CHW (0..255)
            content_np = translation.get_batch(content_files[batch_start : batch_start + bs])
            content_images = torch.tensor(content_np, device=translation.device)
            latent_content = translation.project(img=content_images, cls=source_class)
            content_vecs = latent_content[:, 0]  # [B, w_dim]

            if style_files is not None:
                style_np = translation.get_batch(style_files[batch_start : batch_start + bs])
                style_images = torch.tensor(style_np, device=translation.device)
                latent_style = translation.project(img=style_images, cls=target_class)
                style_vecs = latent_style[:, -1]  # [B, w_dim]
        else:
            latent_content = torch.stack(
                [torch.tensor(projections[fname]) for fname in content_files[batch_start : batch_start + bs]]
            ).to(translation.device)
            content_vecs = latent_content[:, 0]
            if style_files is not None:
                latent_style = torch.stack(
                    [torch.tensor(projections[fname]) for fname in style_files[batch_start : batch_start + bs]]
                ).to(translation.device)
                style_vecs = latent_style[:, -1]

            # for headers we still need images
            content_images = None
            style_images = None

        # -----------------------------
        # If saving to PKL: keep your original behavior
        # -----------------------------
        if save_img_to_pkl:
            for i, ct in enumerate(content_vecs):
                ct = ct.unsqueeze(0)
                if style_files is not None:
                    st = style_vecs[i].unsqueeze(0)
                    latent = translation.get_ws_from_c_s_torch(ct, st)
                else:
                    latent = translation.postprocess(ct, target_class)

                with torch.no_grad():
                    trans_img = translation.G.synthesis(latent, noise_mode="const")

                trans_img = (trans_img + 1) / 2.0
                trans_img = trans_img.clamp(0, 1).to(torch.float32).cpu().numpy()

                if atob:
                    # NOTE: content_images/style_images exist only if not from_projection
                    if content_images is None:
                        # load single image for saving
                        tmp = translation.get_batch([content_files[batch_start + i]])
                        save_dict["real_A"].append(torch.tensor(tmp, dtype=torch.float32).unsqueeze(0) / 255.0)
                    else:
                        save_dict["real_A"].append((content_images[i : i + 1] / 255.0).cpu())

                    save_dict["fake_B"].append(trans_img[0:1])

                    if style_files is not None:
                        if style_images is None:
                            tmp = translation.get_batch([style_files[batch_start + i]])
                            save_dict["ref_A"].append(torch.tensor(tmp, dtype=torch.float32).unsqueeze(0) / 255.0)
                        else:
                            save_dict["ref_A"].append((style_images[i : i + 1] / 255.0).cpu())
                else:
                    if content_images is None:
                        tmp = translation.get_batch([content_files[batch_start + i]])
                        save_dict["real_B"].append(torch.tensor(tmp, dtype=torch.float32).unsqueeze(0) / 255.0)
                    else:
                        save_dict["real_B"].append((content_images[i : i + 1] / 255.0).cpu())

                    save_dict["fake_A"].append(trans_img[0:1])

                    if style_files is not None:
                        if style_images is None:
                            tmp = translation.get_batch([style_files[batch_start + i]])
                            save_dict["ref_B"].append(torch.tensor(tmp, dtype=torch.float32).unsqueeze(0) / 255.0)
                        else:
                            save_dict["ref_B"].append((style_images[i : i + 1] / 255.0).cpu())

                counter += 1

            continue  # next batch

        # -----------------------------
        # Otherwise: Save GRID (default) or old individual images
        # -----------------------------
        if save_grid:
            bs_curr = content_vecs.shape[0]  # == bs

            # defaults: first min(grid_n, bs)
            default_c = list(range(min(grid_n, bs_curr)))
            default_s = list(range(min(grid_n, bs_curr)))

            Nc_indices = _parse_index_spec(content_idx, bs_curr) or default_c
            Ns_indices = _parse_index_spec(style_idx, bs_curr) or default_s

            # selected latents
            contents_sel = content_vecs[Nc_indices]  # [Nc, w_dim]
            styles_sel = style_vecs[Ns_indices]      # [Ns, w_dim]
            Nc = contents_sel.shape[0]
            Ns = styles_sel.shape[0]
            print(f"Selected grid size: {Nc}x{Ns} (from batch of {bs_curr})")

            if Nc <= 0 or Ns <= 0:
                break

            # header reference images
            if not from_projection:
                # content_images/style_images are uint8 CHW (0..255)
                content_refs_hwc = chw_u8_batch_to_hwc_list(
                    content_images[Nc_indices].detach().cpu().numpy()
                )
                style_refs_hwc = chw_u8_batch_to_hwc_list(
                    style_images[Ns_indices].detach().cpu().numpy()
                )
            else:
                content_sel_files = [content_files[batch_start + i] for i in Nc_indices]
                style_sel_files = [style_files[batch_start + j] for j in Ns_indices]
                content_refs_np = translation.get_batch(content_sel_files)
                style_refs_np = translation.get_batch(style_sel_files)
                content_refs_hwc = chw_u8_batch_to_hwc_list(content_refs_np)
                style_refs_hwc = chw_u8_batch_to_hwc_list(style_refs_np)

            # optional sparse cells in LOCAL coords after selection
            cell_pairs = _parse_cell_spec(grid_cells, Nc, Ns)

            if cell_pairs is None:
                # full cartesian
                c_flat = contents_sel[:, None, :].repeat(1, Ns, 1).reshape(-1, contents_sel.shape[-1])
                s_flat = styles_sel[None, :, :].repeat(Nc, 1, 1).reshape(-1, styles_sel.shape[-1])
                ws_all = translation.get_ws_from_c_s_torch(c_flat, s_flat)  # [Nc*Ns, num_ws, w_dim]

                gen_cells = []
                with torch.no_grad():
                    for st in range(0, ws_all.shape[0], synth_batch):
                        ws_chunk = ws_all[st : st + synth_batch]
                        imgs = translation.G.synthesis(ws_chunk, noise_mode="const")  # [-1,1]
                        gen_cells.extend(bchw_minus1_1_to_hwc_u8_list(imgs))
            else:
                # sparse synthesis: only requested (i,j)
                c_list, s_list, flat_ids = [], [], []
                for (i, j) in cell_pairs:
                    c_list.append(contents_sel[i])
                    s_list.append(styles_sel[j])
                    flat_ids.append(i * Ns + j)

                c_flat = torch.stack(c_list, dim=0)
                s_flat = torch.stack(s_list, dim=0)
                ws_all = translation.get_ws_from_c_s_torch(c_flat, s_flat)

                gen_cells = [None] * (Nc * Ns)  # leave unspecified cells blank
                out_k = 0
                with torch.no_grad():
                    for st in range(0, ws_all.shape[0], synth_batch):
                        ws_chunk = ws_all[st : st + synth_batch]
                        imgs = translation.G.synthesis(ws_chunk, noise_mode="const")
                        chunk = bchw_minus1_1_to_hwc_u8_list(imgs)
                        for img_hwc in chunk:
                            gen_cells[flat_ids[out_k]] = img_hwc
                            out_k += 1

            grid_path = os.path.join(outdir, f"{grid_prefix}_{batch_start:06d}.png")
            save_content_style_grid_pil(
                content_refs_hwc_u8=content_refs_hwc,
                style_refs_hwc_u8=style_refs_hwc,
                gen_cells_hwc_u8=gen_cells,
                out_path=grid_path,
                bg=(255, 255, 255),
            )

            counter += Nc
        else:
            # old behavior: per-image saving
            for i, ct in enumerate(content_vecs):
                if style_files is not None:
                    st = style_vecs[i].unsqueeze(0)
                    ct1 = ct.unsqueeze(0)
                    latent = translation.get_ws_from_c_s_torch(ct1, st)
                    tgt_path = style_files[batch_start + i]
                else:
                    latent = translation.postprocess(ct.unsqueeze(0), target_class)
                    tgt_path = None

                with torch.no_grad():
                    trans_img = translation.G.synthesis(latent, noise_mode="const")

                src_path = content_files[batch_start + i]
                postprocess_and_save_named(
                    trans_img=trans_img,
                    outdir=outdir,
                    src_path=src_path,
                    tgt_path=tgt_path,
                    target_class=target_class,
                )
                counter += 1

    # finalize pkl saving (unchanged)
    if save_img_to_pkl:
        if atob:
            save_dict["real_A"] = np.concatenate(save_dict["real_A"], axis=0)
            save_dict["fake_B"] = np.concatenate(save_dict["fake_B"], axis=0)
            if len(save_dict["ref_A"]) > 0:
                save_dict["ref_A"] = np.concatenate(save_dict["ref_A"], axis=0)
        else:
            save_dict["real_B"] = np.concatenate(save_dict["real_B"], axis=0)
            save_dict["fake_A"] = np.concatenate(save_dict["fake_A"], axis=0)
            if len(save_dict["ref_B"]) > 0:
                save_dict["ref_B"] = np.concatenate(save_dict["ref_B"], axis=0)

        if fill_dummy:
            if atob:
                save_dict["real_B"] = np.zeros_like(save_dict["real_A"])
                save_dict["fake_A"] = np.zeros_like(save_dict["fake_B"])

        with open(img_pkl, "wb") as f:
            pickle.dump(save_dict, f)
            print("Saved images successfully!")


# ----------------------------------------------------------------------------
# Optional: keep your save_projections command (unchanged) as a separate entry.
# NOTE: This command is not invoked by default; run it by calling this function
# from another __main__ or refactor into a click.Group if you want both CLIs.
# ----------------------------------------------------------------------------
@click.command()
@click.option("--data_path", help="Dataset path", required=True)
@click.option("--out_pkl", help="Where to save the representations", required=True)
@click.option("--network", "network_pkl", help="Network pickle filename", required=True)
@click.option("--data_name", help="name of the dataset: afhq or celebahq", required=True)
@click.option("--num_steps", help="Number of optimization steps", type=int, default=200, show_default=True)
@click.option("--seed", help="Random seed", type=int, default=303, show_default=True)
@click.option("--batchsize", help="Batch size for generator", type=int, default=32, show_default=True)
@click.option("--psi", help="Truncation psi", type=float, default=None, show_default=True)
@click.option("--num_styles_per_content", type=int, default=1, show_default=True)
@click.option("--c_dim", type=int, default=3, show_default=True)
def save_projections(
    data_path: str,
    out_pkl: str,
    network_pkl: str,
    data_name: str,
    seed: int,
    num_steps: int,
    batchsize: int,
    psi: float,
    num_styles_per_content: int,
    c_dim: int,
):
    outdir = os.path.dirname(out_pkl)
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    domains = {"afhq": ("cat", "dog", "wild"), "celebahq": ("male", "female")}
    assert data_name in domains.keys()
    folders = [os.path.join(data_path, f) for f in domains[data_name]]

    translation = Translation(
        network_pkl,
        c_dim=c_dim,
        i_dim=16,
        num_steps=num_steps,
        psi=psi,
        num_styles_per_content=num_styles_per_content,
        initial_learning_rate=0.01,
        use_noise=True,
    )

    projections = {}
    for folder in folders:
        print(folder)
        content_files = sorted([os.path.join(folder, f) for f in os.listdir(folder)])
        for batch_start in trange(0, len(content_files), batchsize):
            bs = min(batchsize, len(content_files) - batch_start)
            imgs = torch.tensor(translation.get_batch(content_files[batch_start : batch_start + bs]), device=translation.device)
            latent = translation.project(img=imgs, cls=None)
            for i, fname in enumerate(content_files[batch_start : batch_start + bs]):
                projections[fname] = latent[i].cpu().numpy()

    with open(out_pkl, "wb") as f:
        pickle.dump(projections, f)
    print(f"Saved projections: {out_pkl}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    run_translation()
