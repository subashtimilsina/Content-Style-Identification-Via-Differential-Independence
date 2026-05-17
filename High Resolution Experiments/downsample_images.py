#!/usr/bin/env python3
"""
Downsample all images under an input root (e.g., AFHQ 256x256) to 128x128,
preserving the directory structure.

Example:
  python downsample_images.py \
    --in_root  /nfs/stak/users/timilsis/hpc-share/content_style/data/afhq \
    --out_root /nfs/stak/users/timilsis/hpc-share/content_style/data/afhq_128 \
    --size 128 --workers 8


    python downsample_images.py \
    --in_root  /nfs/stak/users/timilsis/hpc-share/content_style/data/celeba_hq \
    --out_root /nfs/stak/users/timilsis/hpc-share/content_style/data/celeba_hq_128 \
    --size 128 --workers 8
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _resize_one(src_path: Path, in_root: Path, out_root: Path, size: int, resample: int, jpeg_quality: int):
    rel = src_path.relative_to(in_root)
    dst_path = out_root / rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as im:
        # Keep consistent mode; AFHQ is typically RGB
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")

        im = im.resize((size, size), resample=resample)

        suffix = src_path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            im.save(dst_path, quality=jpeg_quality, subsampling=0, optimize=True)
        else:
            im.save(dst_path)

    return str(dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_root", type=str, required=True, help="Input dataset root")
    parser.add_argument("--out_root", type=str, required=True, help="Output dataset root")
    parser.add_argument("--size", type=int, default=128, help="Target H=W size (default: 128)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    parser.add_argument("--interp", type=str, default="lanczos", choices=["nearest", "bilinear", "bicubic", "lanczos"])
    parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality (default: 95)")
    args = parser.parse_args()

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    interp_map = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    resample = interp_map[args.interp]

    # Collect image files
    files = [p for p in in_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not files:
        raise RuntimeError(f"No images found under: {in_root}")

    # Parallel resize
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(_resize_one, p, in_root, out_root, args.size, resample, args.jpeg_quality)
            for p in files
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downsampling"):
            pass

    print(f"Done. Wrote {len(files)} images to: {out_root}")


if __name__ == "__main__":
    main()
