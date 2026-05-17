"""
Translation Metrics Evaluation Script for Disentangled StyleGAN

Computes LPIPS and FID metrics for image-to-image translation tasks.

This script operates in two phases:
1. Generation Phase: Translate source images and save to pkl file + image folder
2. Metrics Phase: Load translations and compute LPIPS/FID

If the pkl file already exists, generation is skipped (use --force_generation to override).

LPIPS (Learned Perceptual Image Patch Similarity):
  - For each source image, translate to target domain with multiple random styles
  - Compute pairwise LPIPS between translated outputs to measure style diversity
  - Higher LPIPS = more diverse style transfer

FID (Frechet Inception Distance):
  - Compare translated images against real images in target domain
  - Lower FID = better quality and closer distribution to real images

Usage:
    python compute_translation_metrics.py \
        --model_path /path/to/model.pkl \
        --dataset_path /path/to/dataset \
        --task afhq \
        --output_dir ./translation_outputs

Example for AFHQ:
    python compute_translation_metrics.py \
        --model_path ./model.pkl \
        --dataset_path ./afhq_v2/test \
        --task afhq \
        --output_dir ./outputs \
        --num_steps 200 \
        --num_styles 10
"""

import argparse
import json
import os
import pickle
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dnnlib
import legacy
from i2i_metrics.lpips import LPIPS
from i2i_metrics.fid import calculate_fid_given_paths


class TranslationEvaluator:
    """Evaluator class for computing translation metrics."""
    
    def __init__(
        self,
        model_path: str,
        c_dim: int,
        i_dim: int = 16,
        num_steps: int = 200,
        psi: float = 1.0,
        initial_learning_rate: float = 0.01,
        use_noise: bool = True,
        device: str = 'cuda'
    ):
        """
        Initialize the translation evaluator.
        
        Args:
            model_path: Path to the pretrained model .pkl file
            c_dim: Number of domains (conditioning dimension)
            i_dim: Style dimension in the latent space
            num_steps: Number of optimization steps for projection
            psi: Truncation psi for style mixing
            initial_learning_rate: Learning rate for projection optimization
            use_noise: Whether to use noise during optimization
            device: Device to run computations on
        """
        self.device = torch.device(device)
        self.c_dim = c_dim
        self.i_dim = i_dim
        self.num_steps = num_steps
        self.psi = psi
        self.initial_learning_rate = initial_learning_rate
        self.use_noise = use_noise
        
        self._load_networks(model_path)
        self._compute_stats()
        
    def _load_networks(self, model_path: str):
        """Load the generator and VGG networks."""
        print(f'Loading networks from "{model_path}"...')
        with dnnlib.util.open_url(model_path) as fp:
            self.G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(self.device)
        
        # Load VGG for perceptual loss during projection
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            self.vgg16 = torch.jit.load(f).eval().to(self.device)
    
    def _compute_stats(self):
        """Compute latent space statistics for truncation."""
        w_avg_samples = 10000
        z_samples = np.random.RandomState(123).randn(w_avg_samples, self.G.z_dim)

        c = torch.zeros([z_samples.shape[0], self.c_dim], dtype=torch.float32, device=self.device)
        cum_num_samples_per_class = [z_samples.shape[0]//self.c_dim] * (self.c_dim-1) + \
                                     [z_samples.shape[0] - (z_samples.shape[0]//self.c_dim)*(self.c_dim-1)]
        idx_class = [0] + list(np.cumsum(cum_num_samples_per_class))
        for i in range(self.c_dim):
            c[idx_class[i]:idx_class[i+1], i] = 1
        
        
        latent_samples = self.G.mapping(
            torch.from_numpy(z_samples).to(self.device), c
        ).cpu().numpy().astype(np.float32)
        
        content, style = latent_samples[:, 0], latent_samples[:, -1]
        styles = [style[idx_class[i]:idx_class[i+1]] for i in range(self.c_dim)]

        self.latent_avg = np.mean(latent_samples, axis=0, keepdims=True)
        self.latent_std = (np.sum((latent_samples - self.latent_avg) ** 2) / w_avg_samples) ** 0.5
        self.content_avg = np.mean(content, axis=0, keepdims=True)
        self.styles_avg = [np.mean(s, axis=0, keepdims=True) for s in styles]
        self.content_std = (np.sum((content - self.content_avg) ** 2) / w_avg_samples) ** 0.5
        self.styles_std = [(np.sum((s - self.styles_avg[i]) ** 2) / cum_num_samples_per_class[i]) ** 0.5 
                          for i, s in enumerate(styles)]
        
        print(f'Latent stats computed: content_std={self.content_std:.4f}, latent_std={self.latent_std:.4f}')
    
    def get_ws_from_c_s(self, c: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Combine content and style latents into full W latent."""
        return np.concatenate([
            np.repeat(np.expand_dims(c, axis=1), self.G.mapping.num_c_res, axis=1),
            np.repeat(np.expand_dims(s, axis=1), self.G.num_ws - self.G.mapping.num_c_res, axis=1)
        ], axis=1)
    
    def get_ws_from_c_s_torch(self, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Combine content and style latents into full W latent (torch version)."""
        if len(c.shape) > 2:
            return torch.cat([
                c,
                s.unsqueeze(1).repeat(1, self.G.num_ws - self.G.mapping.num_c_res, 1)
            ], dim=1)
        
        return torch.cat([
            c.unsqueeze(1).repeat(1, self.G.mapping.num_c_res, 1),
            s.unsqueeze(1).repeat(1, self.G.num_ws - self.G.mapping.num_c_res, 1)
        ], dim=1)
    
    def load_images(self, filenames: List[str]) -> torch.Tensor:
        """Load and preprocess images from file paths."""
        images = []
        for target_fname in filenames:
            target_pil = PIL.Image.open(target_fname).convert('RGB')
            w, h = target_pil.size
            s = min(w, h)
            target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            target_pil = target_pil.resize((self.G.img_resolution, self.G.img_resolution), PIL.Image.LANCZOS)
            target_uint8 = np.array(target_pil, dtype=np.uint8)
            images.append(target_uint8.transpose([2, 0, 1]))
        return torch.tensor(np.stack(images, axis=0), device=self.device, dtype=torch.float32)
    
    def project(self, img: torch.Tensor, cls: Optional[int] = None) -> torch.Tensor:
        """
        Project images to the latent space.
        
        Args:
            img: Batch of images [N, C, H, W] in [0, 255] range
            cls: Source class index for initialization
            
        Returns:
            Latent codes [N, num_ws, latent_dim]
        """
        if cls is not None:
            latent_init = self.get_ws_from_c_s(self.content_avg, self.styles_avg[cls])
        else:
            latent_init = self.latent_avg
        latent_init = np.repeat(latent_init, img.shape[0], axis=0)

        # Features for target image
        img = img.to(self.device).to(torch.float32)
        # Preprocess for VGG: scale to [0, 255]
        img_vgg = img.clone()
        if img_vgg.shape[2] > 256:
            img_vgg = F.interpolate(img_vgg, size=(256, 256), mode='area')
        img_features = self.vgg16(img_vgg, resize_images=False, return_lpips=True)
        
        latent = self._optimize_latent(latent_init, img_features, self.latent_std, img)
        return latent
    
    def _optimize_latent(
        self,
        w_avg: np.ndarray,
        target_features: torch.Tensor,
        w_std: float,
        content_images: torch.Tensor,
        initial_noise_factor: float = 0.05,
        noise_ramp_length: float = 0.75,
        lr_rampdown_length: float = 0.25,
        lr_rampup_length: float = 0.05
    ) -> torch.Tensor:
        """Optimize latent codes to match target images."""
        initial_learning_rate = self.initial_learning_rate
        
        # Use enable_grad to ensure gradients work even if called from no_grad context
        with torch.enable_grad():
            content = torch.tensor(w_avg[:, 0].copy(), dtype=torch.float32, device=self.device, requires_grad=True)
            style = torch.tensor(w_avg[:, -1].copy(), dtype=torch.float32, device=self.device, requires_grad=True)
            
            optimizer = torch.optim.Adam([content, style], betas=(0.9, 0.999), lr=initial_learning_rate)
            
            for step in range(self.num_steps):
                w_opt = self.get_ws_from_c_s_torch(content, style)
                
                # Learning rate schedule
                t = step / self.num_steps
                w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
                lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
                lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
                lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
                lr = initial_learning_rate * lr_ramp
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # Synthesize images
                if self.use_noise:
                    w_noise = torch.randn_like(w_opt) * w_noise_scale
                    w_in = w_opt + w_noise
                else:
                    w_in = w_opt
                
                synth_images = self.G.synthesis(w_in, noise_mode='const')
                synth_images = (synth_images + 1) * (255/2)
                if synth_images.shape[2] > 256:
                    synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

                # Compute perceptual loss
                synth_features = self.vgg16(synth_images, resize_images=False, return_lpips=True)
                loss = (target_features - synth_features).square().sum() / synth_features.shape[0]
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            
        return w_opt.detach()
    
    @torch.no_grad()
    def translate_with_random_style(
        self,
        content: torch.Tensor,
        target_class: int,
        num_styles: int = 1
    ) -> torch.Tensor:
        """
        Translate content to target domain with random styles.
        
        Args:
            content: Content latent [N, latent_dim]
            target_class: Target domain index
            num_styles: Number of style variations to generate
            
        Returns:
            Translated images [N, num_styles, C, H, W]
        """
        batch_size = content.shape[0]
        images_list = []
        
        for _ in range(num_styles):
            # Sample random style
            random_z = torch.randn(batch_size, self.G.z_dim, device=self.device)
            target_c = torch.zeros([batch_size, self.c_dim], dtype=torch.float32, device=self.device)
            target_c[:, target_class] = 1
            
            latent = self.G.mapping(random_z, target_c)
            style_hat = latent[:, -1]
            
            # Apply truncation
            if self.psi is not None:
                style_avg = torch.tensor(self.styles_avg[target_class], device=self.device, dtype=torch.float32)
                style_hat = (1 - self.psi) * style_avg + self.psi * style_hat
            
            # Combine content and style
            ws = self.get_ws_from_c_s_torch(content, style_hat)
            
            # Synthesize
            img = self.G.synthesis(ws, noise_mode='const')
            images_list.append(img)
        
        return torch.stack(images_list, dim=1)  # [N, num_styles, C, H, W]
    
    @torch.no_grad()
    def synthesize(self, latent: torch.Tensor) -> torch.Tensor:
        """Synthesize images from latent codes."""
        return self.G.synthesis(latent, noise_mode='const')


# =============================================================================
# Generation Functions
# =============================================================================

def save_image(img_tensor: torch.Tensor, path: str):
    """Save a single image tensor to disk."""
    # img_tensor is [C, H, W] in [-1, 1] range
    img = (img_tensor + 1) * (255 / 2)
    img = img.clamp(0, 255).to(torch.uint8).cpu().permute(1, 2, 0).numpy()
    PIL.Image.fromarray(img).save(path)


def generate_translations_for_pair(
    evaluator: TranslationEvaluator,
    source_dir: str,
    source_class: int,
    target_class: int,
    num_styles: int,
    batch_size: int,
    output_pkl_path: str,
    output_img_dir: str,
    num_images: Optional[int] = None
) -> Dict:
    """
    Generate translations for a single source->target pair.
    Saves:
    - PKL file with all style variations (for LPIPS)
    - Image folder with ALL style variations (for FID) - total = num_source * num_styles
    
    Args:
        evaluator: TranslationEvaluator instance
        source_dir: Directory containing source images
        source_class: Source domain index
        target_class: Target domain index
        num_styles: Number of style variations per source image
        batch_size: Batch size for processing
        output_pkl_path: Path to save pkl file
        output_img_dir: Directory to save images for FID
        num_images: Number of images to use (None = use all)
        
    Returns:
        Dictionary with metadata
    """
    source_files = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if num_images is not None and num_images > 0:
        source_files = source_files[:num_images]
    
    if len(source_files) == 0:
        print(f"Warning: No images found in {source_dir}")
        return {'num_images': 0}
    
    # Create output image directory
    os.makedirs(output_img_dir, exist_ok=True)
    
    all_source_images = []
    all_translated_images = []
    img_counter = 0
    
    for batch_start in tqdm(range(0, len(source_files), batch_size), desc="Generating translations"):
        batch_files = source_files[batch_start:batch_start + batch_size]
        
        # Load and project source images
        source_images = evaluator.load_images(batch_files)
        latent = evaluator.project(source_images, cls=source_class)
        content = latent[:, 0]  # Extract content
        
        # Translate with multiple styles
        translated = evaluator.translate_with_random_style(content, target_class, num_styles)
        # translated shape: [N, num_styles, C, H, W] in [-1, 1] range
        
        # Save ALL style variations as images for FID
        for i in range(translated.shape[0]):
            for s in range(num_styles):
                img_path = os.path.join(output_img_dir, f"{img_counter:05d}.png")
                save_image(translated[i, s], img_path)
                img_counter += 1
        
        # Convert to numpy and normalize to [0, 1] for pkl (LPIPS)
        source_np = (source_images / 255.0).cpu().numpy()  # [N, C, H, W]
        translated_np = ((translated + 1) / 2).clamp(0, 1).cpu().numpy()  # [N, num_styles, C, H, W]
        
        all_source_images.append(source_np)
        all_translated_images.append(translated_np)
    
    # Save pkl for LPIPS
    data = {
        'source_images': np.concatenate(all_source_images, axis=0),  # [total_N, C, H, W]
        'translated_images': np.concatenate(all_translated_images, axis=0),  # [total_N, num_styles, C, H, W]
        'num_images': img_counter,
        'num_styles': num_styles
    }
    
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(data, f)
    
    return data


def generate_and_save_translations(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    task: str,
    num_steps: int = 200,
    num_styles: int = 10,
    batch_size: int = 4,
    psi: float = 1.0,
    i_dim: int = 16,
    num_images: Optional[int] = None,
    seed: int = 42,
    force_generation: bool = False
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Generate translations for all pairs and save to pkl files + image folders.
    
    Args:
        model_path: Path to the pretrained model
        dataset_path: Path to the dataset root
        output_dir: Directory to save pkl files and images
        task: Task name ('afhq' or 'celebahq')
        num_steps: Number of optimization steps for projection
        num_styles: Number of style variations per image
        batch_size: Batch size for processing
        psi: Truncation psi
        i_dim: Style dimension
        num_images: Number of images to use (None = use all)
        seed: Random seed
        force_generation: Force regeneration even if files exist
        
    Returns:
        Tuple of (pkl_paths dict, img_dir_paths dict)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Determine c_dim based on task
    c_dim = 3 if task == 'afhq' else 2
    
    # Create output directories
    pkl_dir = os.path.join(output_dir, 'pkl')
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(pkl_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    # Initialize evaluator (only if we need to generate)
    evaluator = None
    
    # Get translation pairs
    translation_pairs = get_translation_pairs(task)
    
    pkl_paths = {}
    img_dir_paths = {}
    
    for src_name, tgt_name, src_class, tgt_class in translation_pairs:
        pair_name = f"{src_name}2{tgt_name}"
        pkl_path = os.path.join(pkl_dir, f"{pair_name}.pkl")
        pair_img_dir = os.path.join(img_dir, pair_name)
        
        pkl_paths[pair_name] = pkl_path
        img_dir_paths[pair_name] = pair_img_dir
        
        print(f"\n{'='*50}")
        print(f"Processing: {pair_name}")
        print(f"{'='*50}")
        
        # Check if files already exist
        pkl_exists = os.path.exists(pkl_path)
        img_dir_exists = os.path.exists(pair_img_dir) and len(os.listdir(pair_img_dir)) > 0
        
        if pkl_exists and img_dir_exists and not force_generation:
            print(f"  Files already exist:")
            print(f"    PKL: {pkl_path}")
            print(f"    Images: {pair_img_dir}")
            print(f"  Skipping generation (use --force_generation to override)")
            continue
        
        # Initialize evaluator if needed
        if evaluator is None:
            print("\nInitializing TranslationEvaluator...")
            evaluator = TranslationEvaluator(
                model_path=model_path,
                c_dim=c_dim,
                i_dim=i_dim,
                num_steps=num_steps,
                psi=psi,
                device=str(device)
            )
        
        source_dir = os.path.join(dataset_path, src_name)
        
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory not found: {source_dir}")
            continue
        
        # Generate translations
        data = generate_translations_for_pair(
            evaluator=evaluator,
            source_dir=source_dir,
            source_class=src_class,
            target_class=tgt_class,
            num_styles=num_styles,
            batch_size=batch_size,
            output_pkl_path=pkl_path,
            output_img_dir=pair_img_dir,
            num_images=num_images
        )
        
        num_source = data['translated_images'].shape[0]
        print(f"  Saved PKL to: {pkl_path}")
        print(f"  Saved {data['num_images']} images to: {pair_img_dir} ({num_source} sources x {num_styles} styles)")
    
    return pkl_paths, img_dir_paths


# =============================================================================
# Metrics Functions
# =============================================================================

@torch.no_grad()
def compute_lpips_from_pkl(pkl_path: str, lpips_model: LPIPS, device: torch.device) -> float:
    """
    Compute LPIPS from saved translations.
    
    Args:
        pkl_path: Path to pkl file with translations
        lpips_model: LPIPS model
        device: Device for computation
        
    Returns:
        Mean LPIPS value
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    translated_images = data['translated_images']  # [N, num_styles, C, H, W]
    num_images, num_styles = translated_images.shape[:2]
    
    lpips_values = []
    
    for i in tqdm(range(num_images), desc="Computing LPIPS"):
        # Get all style variations for this image
        styles = translated_images[i]  # [num_styles, C, H, W]
        
        # Convert to torch and scale to [-1, 1] for LPIPS
        styles_tensor = torch.tensor(styles, device=device, dtype=torch.float32)
        styles_tensor = styles_tensor * 2 - 1  # [0, 1] -> [-1, 1]
        
        # Compute pairwise LPIPS
        pair_lpips = []
        for j in range(num_styles - 1):
            for k in range(j + 1, num_styles):
                lpips_val = lpips_model(styles_tensor[j:j+1], styles_tensor[k:k+1])
                pair_lpips.append(lpips_val.item())
        
        if pair_lpips:
            lpips_values.append(np.mean(pair_lpips))
    
    return np.mean(lpips_values) if lpips_values else 0.0


def compute_fid_from_folders(
    generated_img_dir: str,
    real_img_dir: str,
    img_size: int = 256,
    batch_size: int = 64
) -> float:
    """
    Compute FID using the original calculate_fid_given_paths function.
    
    This ensures consistent preprocessing between generated and real images.
    
    Args:
        generated_img_dir: Directory containing generated images
        real_img_dir: Directory containing real target images
        img_size: Image size for FID computation
        batch_size: Batch size
        
    Returns:
        FID value
    """
    paths = [real_img_dir, generated_img_dir]
    fid_value = calculate_fid_given_paths(paths, img_size, batch_size)
    return fid_value


def compute_metrics_from_files(
    pkl_paths: Dict[str, str],
    img_dir_paths: Dict[str, str],
    dataset_path: str,
    fid_real_path: Optional[str] = None,
    output_path: Optional[str] = None,
    compute_lpips: bool = True,
    compute_fid: bool = True,
    img_size: int = 256,
    batch_size: int = 64
) -> Dict:
    """
    Compute metrics from saved pkl files and image folders.
    
    Args:
        pkl_paths: Dictionary mapping pair names to pkl file paths
        img_dir_paths: Dictionary mapping pair names to image directories
        dataset_path: Path to dataset root (for source images during generation)
        fid_real_path: Path to real images for FID computation (e.g., train set).
                       If None, uses dataset_path. Should contain domain subdirs (cat/, dog/, etc.)
        output_path: Path to save results JSON
        compute_lpips: Whether to compute LPIPS
        compute_fid: Whether to compute FID
        img_size: Image size for FID
        batch_size: Batch size for FID computation
        
    Returns:
        Dictionary with all metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Use fid_real_path if provided, otherwise fall back to dataset_path
    real_images_root = fid_real_path if fid_real_path is not None else dataset_path
    print(f"FID real images path: {real_images_root}")
    
    # Initialize LPIPS model if needed
    lpips_model = None
    if compute_lpips:
        print("\nInitializing LPIPS model...")
        lpips_model = LPIPS().eval().to(device)
    
    results = OrderedDict()
    lpips_values = []
    fid_values = []
    
    for pair_name in pkl_paths.keys():
        pkl_path = pkl_paths[pair_name]
        img_dir = img_dir_paths.get(pair_name)
        
        print(f"\n{'='*50}")
        print(f"Computing metrics for: {pair_name}")
        print(f"{'='*50}")
        
        # Get target domain name from pair name (e.g., "cat2dog" -> "dog")
        target_name = pair_name.split('2')[1]
        # Use real_images_root for FID (train set typically has more images)
        target_dir = os.path.join(real_images_root, target_name)
        
        # Compute LPIPS
        if compute_lpips:
            if not os.path.exists(pkl_path):
                print(f"Warning: PKL file not found: {pkl_path}")
            else:
                lpips_value = compute_lpips_from_pkl(pkl_path, lpips_model, device)
                results[f"LPIPS/{pair_name}"] = lpips_value
                lpips_values.append(lpips_value)
                print(f"  LPIPS/{pair_name}: {lpips_value:.4f}")
        
        # Compute FID using original function (loads from folders)
        # Compares generated images with real images of the TARGET domain
        # e.g., cat2dog generated images are compared with real dog images
        if compute_fid:
            if img_dir is None or not os.path.exists(img_dir):
                print(f"Warning: Generated image directory not found: {img_dir}")
            elif not os.path.exists(target_dir):
                print(f"Warning: Real target directory not found: {target_dir}")
            else:
                print(f"  Comparing {pair_name} with real {target_name} images from {target_dir}")
                fid_value = compute_fid_from_folders(
                    generated_img_dir=img_dir,
                    real_img_dir=target_dir,
                    img_size=img_size,
                    batch_size=batch_size
                )
                results[f"FID/{pair_name}"] = fid_value
                fid_values.append(fid_value)
                print(f"  FID/{pair_name}: {fid_value:.4f}")
    
    # Compute mean metrics
    if compute_lpips and lpips_values:
        results["LPIPS/mean"] = np.mean(lpips_values)
    if compute_fid and fid_values:
        results["FID/mean"] = np.mean(fid_values)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Translation Metrics Summary")
    print("=" * 60)
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 60)
    
    # Save results
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


# =============================================================================
# Helper Functions
# =============================================================================

def get_translation_pairs(task: str) -> List[Tuple[str, str, int, int]]:
    """
    Get list of translation pairs for a given task.
    
    Returns list of tuples: (source_domain_name, target_domain_name, source_class_idx, target_class_idx)
    """
    if task == 'afhq':
        domains = ['cat', 'dog', 'wild']
        pairs = []
        for i, src in enumerate(domains):
            for j, tgt in enumerate(domains):
                if i != j:
                    pairs.append((src, tgt, i, j))
        return pairs
    elif task == 'celebahq':
        return [
            ('male', 'female', 1, 0),
            ('female', 'male', 0, 1)
        ]
    else:
        raise ValueError(f"Unknown task: {task}. Supported: afhq, celebahq")


# =============================================================================
# Main Entry Points
# =============================================================================

def compute_all_metrics(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    task: str,
    num_steps: int = 200,
    num_images: Optional[int] = None,
    num_styles: int = 10,
    batch_size: int = 4,
    psi: float = 1.0,
    i_dim: int = 16,
    output_path: Optional[str] = None,
    compute_lpips: bool = True,
    compute_fid: bool = True,
    img_size: int = 256,
    seed: int = 42,
    skip_generation: bool = False,
    force_generation: bool = False,
    fid_real_path: Optional[str] = None
) -> Dict:
    """
    Main function: Generate translations (if needed) and compute metrics.
    
    Args:
        model_path: Path to the pretrained model
        dataset_path: Path to the dataset root (test set for source images)
        output_dir: Directory to save/load pkl files and images
        task: Task name ('afhq' or 'celebahq')
        num_steps: Number of optimization steps for projection
        num_images: Number of images to use (None = use all available)
        num_styles: Number of style variations for LPIPS
        batch_size: Batch size for processing
        psi: Truncation psi
        i_dim: Style dimension
        output_path: Path to save results JSON
        compute_lpips: Whether to compute LPIPS
        compute_fid: Whether to compute FID
        img_size: Image size for metrics
        seed: Random seed
        skip_generation: If True, only compute metrics from existing files
        force_generation: If True, regenerate even if files exist
        fid_real_path: Path to real images for FID (train set, should have more images).
                       If None, uses dataset_path. Must contain domain subdirs (cat/, dog/, etc.)
        
    Returns:
        Dictionary with all metrics
    """
    print(f"\n{'='*60}")
    print("Translation Metrics Evaluation")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset path (source images): {dataset_path}")
    print(f"FID real path: {fid_real_path if fid_real_path else dataset_path}")
    print(f"Number of images: {'all available' if num_images is None else num_images}")
    print(f"Number of styles: {num_styles}")
    print(f"Projection steps: {num_steps}")
    print(f"Truncation psi: {psi}")
    print(f"Force generation: {force_generation}")
    
    # Get translation pairs
    translation_pairs = get_translation_pairs(task)
    
    # Build file paths
    pkl_dir = os.path.join(output_dir, 'pkl')
    img_dir = os.path.join(output_dir, 'images')
    
    pkl_paths = {f"{src}2{tgt}": os.path.join(pkl_dir, f"{src}2{tgt}.pkl") 
                 for src, tgt, _, _ in translation_pairs}
    img_dir_paths = {f"{src}2{tgt}": os.path.join(img_dir, f"{src}2{tgt}") 
                     for src, tgt, _, _ in translation_pairs}
    
    # Check which files exist
    print(f"\nChecking existing files...")
    for pair_name in pkl_paths:
        pkl_exists = os.path.exists(pkl_paths[pair_name])
        img_exists = os.path.exists(img_dir_paths[pair_name]) and \
                     len(os.listdir(img_dir_paths[pair_name])) > 0 if os.path.exists(img_dir_paths[pair_name]) else False
        status = "EXISTS" if (pkl_exists and img_exists) else "MISSING"
        print(f"  - {pair_name}: {status}")
    
    # Step 1: Generate translations (if needed)
    if not skip_generation:
        print(f"\n{'='*60}")
        print("Step 1: Generating Translations")
        print(f"{'='*60}")
        
        pkl_paths, img_dir_paths = generate_and_save_translations(
            model_path=model_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            task=task,
            num_steps=num_steps,
            num_styles=num_styles,
            batch_size=batch_size,
            psi=psi,
            i_dim=i_dim,
            num_images=num_images,
            seed=seed,
            force_generation=force_generation
        )
    else:
        print("\nSkipping generation (--skip_generation is set)")
    
    # Step 2: Compute metrics
    print(f"\n{'='*60}")
    print("Step 2: Computing Metrics")
    print(f"{'='*60}")
    
    results = compute_metrics_from_files(
        pkl_paths=pkl_paths,
        img_dir_paths=img_dir_paths,
        dataset_path=dataset_path,
        fid_real_path=fid_real_path,
        output_path=output_path,
        compute_lpips=compute_lpips,
        compute_fid=compute_fid,
        img_size=img_size,
        batch_size=batch_size
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute translation metrics (LPIPS and FID) for Disentangled StyleGAN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: generate translations and compute metrics
  # Use test set for source images, train set for FID real images
  python compute_translation_metrics.py \\
      --model_path ./model.pkl \\
      --dataset_path ./afhq_v2/test \\
      --fid_real_path ./afhq_v2/train \\
      --task afhq \\
      --output_dir ./translation_outputs \\
      --num_steps 200 \\
      --num_styles 10

  # Force regeneration even if files exist
  python compute_translation_metrics.py \\
      --model_path ./model.pkl \\
      --dataset_path ./afhq_v2/test \\
      --fid_real_path ./afhq_v2/train \\
      --task afhq \\
      --output_dir ./translation_outputs \\
      --force_generation

  # Only compute metrics from existing files (skip generation)
  python compute_translation_metrics.py \\
      --model_path ./model.pkl \\
      --dataset_path ./afhq_v2/test \\
      --fid_real_path ./afhq_v2/train \\
      --task afhq \\
      --output_dir ./translation_outputs \\
      --skip_generation

  # Compute only FID (skip LPIPS)
  python compute_translation_metrics.py \\
      --model_path ./model.pkl \\
      --dataset_path ./afhq_v2/test \\
      --fid_real_path ./afhq_v2/train \\
      --task afhq \\
      --output_dir ./translation_outputs \\
      --no_lpips

  # Limit to 100 images for faster testing
  python compute_translation_metrics.py \\
      --model_path ./model.pkl \\
      --dataset_path ./afhq_v2/test \\
      --task afhq \\
      --output_dir ./translation_outputs \\
      --num_images 100
        """
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the pretrained model pickle file (.pkl)'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to the dataset root directory for source images (test set, contains domain subdirs)'
    )
    parser.add_argument(
        '--fid_real_path',
        type=str,
        default=None,
        help='Path to real images for FID computation (train set, more images). '
             'Should contain domain subdirs (cat/, dog/, etc.). '
             'If not provided, uses dataset_path. '
             'E.g., for AFHQ: --fid_real_path ./afhq_v2/train'
    )
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['afhq', 'celebahq'],
        help='Task/dataset name for determining translation pairs'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save/load translation files (pkl and images)'
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=200,
        help='Number of optimization steps for GAN inversion (default: 200)'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=None,
        help='Number of source images to use (default: None = use all available)'
    )
    parser.add_argument(
        '--num_styles',
        type=int,
        default=10,
        help='Number of style variations per image for LPIPS (default: 10)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for generation (default: 4)'
    )
    parser.add_argument(
        '--psi',
        type=float,
        default=1.0,
        help='Truncation psi value (default: 1.0)'
    )
    parser.add_argument(
        '--i_dim',
        type=int,
        default=16,
        help='Style dimension in the latent space (default: 16)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='Image size for metrics computation (default: 256)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save results JSON file (optional)'
    )
    parser.add_argument(
        '--no_lpips',
        action='store_true',
        help='Skip LPIPS computation'
    )
    parser.add_argument(
        '--no_fid',
        action='store_true',
        help='Skip FID computation'
    )
    parser.add_argument(
        '--skip_generation',
        action='store_true',
        help='Skip generation, only compute metrics from existing files'
    )
    parser.add_argument(
        '--force_generation',
        action='store_true',
        help='Force regeneration even if files already exist'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.skip_generation and not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path not found: {args.dataset_path}")
        sys.exit(1)
    
    if args.fid_real_path is not None and not os.path.exists(args.fid_real_path):
        print(f"Error: FID real path not found: {args.fid_real_path}")
        sys.exit(1)
    
    # Run evaluation
    compute_all_metrics(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        task=args.task,
        num_steps=args.num_steps,
        num_images=args.num_images,
        num_styles=args.num_styles,
        batch_size=args.batch_size,
        psi=args.psi,
        i_dim=args.i_dim,
        output_path=args.output,
        compute_lpips=not args.no_lpips,
        compute_fid=not args.no_fid,
        img_size=args.img_size,
        seed=args.seed,
        skip_generation=args.skip_generation,
        force_generation=args.force_generation,
        fid_real_path=args.fid_real_path
    )


if __name__ == '__main__':
    main()
