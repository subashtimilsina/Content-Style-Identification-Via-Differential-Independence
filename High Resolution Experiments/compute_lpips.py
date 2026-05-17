"""
LPIPS Evaluation Script for Disentangled StyleGAN

Computes LPIPS (Learned Perceptual Image Patch Similarity) to measure 
style diversity while keeping content fixed.

For each domain:
  - Sample N content latent codes (fixed content part of z)
  - For each content, sample K style latent codes (varying style part of z)
  - Generate K images per content
  - Compute pairwise LPIPS between K images
  - Report per-domain and overall mean LPIPS

Usage:
    python compute_lpips.py --model_path /path/to/model.pkl \
                            --num_content 200 \
                            --num_styles 10 \
                            --domain_names cat dog wild
"""

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dnnlib
import legacy
from lpips import LPIPS


def load_generator(model_path: str, device: torch.device):
    """Load the generator from a pickle file."""
    print(f'Loading generator from "{model_path}"...')
    with dnnlib.util.open_url(model_path) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)
    return G


def get_model_dimensions(G):
    """Extract model dimensions from the generator's mapping network."""
    mapping = G.mapping
    
    z_dim = mapping.z_dim
    c_dim = mapping.c_dim
    
    # Get style dimension (i_dim) - this determines content/style split
    if hasattr(mapping, 'i_dim'):
        print("Found i_dim attribute in mapping network.")
        i_dim = mapping.i_dim
    else:
        # Default fallback if attribute not found
        i_dim = 256
        print(f"Warning: i_dim not found in mapping network, using default {i_dim}")
    
    content_dim = z_dim - i_dim
    
    print(f"Model dimensions: z_dim={z_dim}, c_dim={c_dim}, i_dim (style)={i_dim}, content_dim={content_dim}")
    
    return z_dim, c_dim, i_dim, content_dim


@torch.no_grad()
def calculate_lpips_given_images(group_of_images, lpips_model):
    """Calculate pairwise LPIPS between all images in the group."""
    lpips_values = []
    num_outputs = len(group_of_images)
    
    # Calculate average of pairwise distances among all outputs
    for i in range(num_outputs - 1):
        for j in range(i + 1, num_outputs):
            lpips_value = lpips_model(group_of_images[i], group_of_images[j])
            lpips_values.append(lpips_value)
    
    lpips_mean = torch.mean(torch.stack(lpips_values, dim=0))
    return lpips_mean.item()


@torch.no_grad()
def compute_lpips_for_domain(
    G,
    domain_idx: int,
    num_domains: int,
    num_content: int,
    num_styles: int,
    z_dim: int,
    i_dim: int,
    lpips_model: LPIPS,
    device: torch.device,
    batch_size: int = 1,
):
    """
    Compute LPIPS for a single domain.
    
    Args:
        G: Generator model
        domain_idx: Index of target domain
        num_domains: Total number of domains (c_dim)
        num_content: Number of different content samples
        num_styles: Number of style variations per content
        z_dim: Total latent dimension
        i_dim: Style dimension (first i_dim dims of z)
        lpips_model: LPIPS model for computing perceptual distance
        device: CUDA/CPU device
        batch_size: Batch size for generation
    
    Returns:
        Mean LPIPS value for this domain
    """
    content_dim = z_dim - i_dim
    lpips_values = []
    
    # Create one-hot domain label
    c = torch.zeros(batch_size, num_domains, device=device)
    c[:, domain_idx] = 1.0
    
    for content_idx in tqdm(range(num_content), desc=f"Domain {domain_idx}", leave=False):
        # Sample fixed content for this iteration
        z_content = torch.randn(batch_size, content_dim, device=device)
        
        # Generate images with different styles but same content
        group_of_images = []
        for style_idx in range(num_styles):
            # Sample random style
            z_style = torch.randn(batch_size, i_dim, device=device)
            
            # Combine: z = [style | content] (first i_dim is style, rest is content)
            z = torch.cat([z_style, z_content], dim=1)
            
            # Generate image
            ws = G.mapping(z, c)
            img = G.synthesis(ws)
            
            # Normalize to [-1, 1] range (already in this range from generator)
            group_of_images.append(img)
        
        # Compute LPIPS for this content
        lpips_value = calculate_lpips_given_images(group_of_images, lpips_model)
        lpips_values.append(lpips_value)
    
    return np.mean(lpips_values)


@torch.no_grad()
def compute_lpips(
    model_path: str,
    num_content: int = 200,
    num_styles: int = 10,
    domain_names: list = None,
    output_path: str = None,
    batch_size: int = 1,
):
    """
    Main function to compute LPIPS for all domains.
    
    Args:
        model_path: Path to the model pickle file
        num_content: Number of content samples per domain
        num_styles: Number of style variations per content
        domain_names: Optional list of domain names for labeling
        output_path: Optional path to save results JSON
        batch_size: Batch size for generation
    
    Returns:
        Dictionary with LPIPS values
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load generator
    G = load_generator(model_path, device)
    
    # Get model dimensions
    z_dim, c_dim, i_dim, content_dim = get_model_dimensions(G)
    
    # Set default domain names if not provided
    if domain_names is None:
        domain_names = [f"domain_{i}" for i in range(c_dim)]
    else:
        assert len(domain_names) == c_dim, f"Number of domain names ({len(domain_names)}) must match c_dim ({c_dim})"
    
    print(f"\nConfiguration:")
    print(f"  - Number of domains: {c_dim}")
    print(f"  - Domain names: {domain_names}")
    print(f"  - Content samples per domain: {num_content}")
    print(f"  - Style variations per content: {num_styles}")
    print(f"  - Total images per domain: {num_content * num_styles}")
    
    # Initialize LPIPS model
    print("\nInitializing LPIPS model...")
    lpips_model = LPIPS().eval().to(device)
    
    # Compute LPIPS for each domain
    print("\nComputing LPIPS...")
    lpips_dict = OrderedDict()
    
    for domain_idx, domain_name in enumerate(domain_names):
        print(f"\nProcessing domain: {domain_name} ({domain_idx + 1}/{c_dim})")
        
        lpips_value = compute_lpips_for_domain(
            G=G,
            domain_idx=domain_idx,
            num_domains=c_dim,
            num_content=num_content,
            num_styles=num_styles,
            z_dim=z_dim,
            i_dim=i_dim,
            lpips_model=lpips_model,
            device=device,
            batch_size=batch_size,
        )
        
        lpips_dict[f"LPIPS/{domain_name}"] = lpips_value
        print(f"  LPIPS/{domain_name}: {lpips_value:.4f}")
    
    # Compute overall mean
    lpips_mean = np.mean(list(lpips_dict.values()))
    lpips_dict["LPIPS/mean"] = lpips_mean
    
    # Print summary
    print("\n" + "=" * 50)
    print("LPIPS Results Summary")
    print("=" * 50)
    for key, value in lpips_dict.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 50)
    
    # Save results if output path provided
    if output_path is not None:
        import json
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(lpips_dict, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return lpips_dict


def main():
    parser = argparse.ArgumentParser(
        description="Compute LPIPS for style diversity evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with AFHQ model
  python compute_lpips.py --model_path ./model.pkl --domain_names cat dog wild
  
  # Custom number of samples
  python compute_lpips.py --model_path ./model.pkl \\
                          --num_content 200 \\
                          --num_styles 10 \\
                          --domain_names cat dog wild \\
                          --output lpips_results.json
        """
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help='Path to the model pickle file (.pkl)'
    )
    parser.add_argument(
        '--num_content', 
        type=int, 
        default=200,
        help='Number of content samples per domain (default: 200)'
    )
    parser.add_argument(
        '--num_styles', 
        type=int, 
        default=10,
        help='Number of style variations per content (default: 10)'
    )
    parser.add_argument(
        '--domain_names', 
        type=str, 
        nargs='+',
        default=None,
        help='Names of domains (e.g., cat dog wild). If not provided, uses domain_0, domain_1, etc.'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Path to save results JSON file (optional)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=1,
        help='Batch size for generation (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Run LPIPS computation
    compute_lpips(
        model_path=args.model_path,
        num_content=args.num_content,
        num_styles=args.num_styles,
        domain_names=args.domain_names,
        output_path=args.output,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()

