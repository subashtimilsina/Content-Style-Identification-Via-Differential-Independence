import torch
from torchmetrics.image.fid import FrechetInceptionDistance

def compute_classwise_fid(generator, dataset, device, num_classes, latent_dim, num_samples=1000):
    """
    Compute FID scores for each class separately.
    
    Args:
        generator: The generator model
        dataset: The dataset containing real images
        device: The device to run computations on
        num_classes: Number of classes in the dataset
        latent_dim: Dimension of the latent space
        num_samples: Number of samples to use for FID computation
    
    Returns:
        dict: Dictionary containing FID scores for each class
    """
    generator.eval()
    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid_scores = {}
    
    # Group real images by class
    class_images = {i: [] for i in range(num_classes)}
    for img, label in dataset:
        class_images[label].append(img)
    
    with torch.no_grad():
        for class_idx in range(num_classes):
            # Reset FID for this class
            fid.reset()
            
            # Get real images for this class
            real_imgs = torch.stack(class_images[class_idx][:num_samples]).to(device)
            # Ensure images are in correct format (B, C, H, W) and range [0, 1]
            real_imgs = (real_imgs + 1) / 2
            # FID expects images in range [0, 255]
            real_imgs = (real_imgs * 255).type(torch.uint8)
            
            # Generate fake images for this class
            z = torch.randn(num_samples, latent_dim).to(device)
            labels = torch.full((num_samples,), class_idx, dtype=torch.long).to(device)
            fake_imgs = generator(z, labels)
            fake_imgs = (fake_imgs + 1) / 2
            fake_imgs = (fake_imgs * 255).type(torch.uint8)
            
            # Update FID
            fid.update(real_imgs, real=True)
            fid.update(fake_imgs, real=False)
            
            # Compute FID
            fid_scores[f"fid_class_{class_idx}"] = fid.compute().item()
    
    generator.train()
    return fid_scores
