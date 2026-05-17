import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import Generator, Discriminator, ContentEncoder, StyleEncoder
import os
import wandb
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F
from fid import compute_classwise_fid
from dataset import CombinedMNIST
import argparse
import numpy as np
import time


# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train Disentangled GAN')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--project_name', type=str, default='disentangled_gan', help='Project name for wandb')
    parser.add_argument('--run_name', type=str, default='disent_gan_r1_075', help='Run name for wandb')
    parser.add_argument('--r1_lambda', type=float, default=0.75, help='R1 regularization strength')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--content_dim', type=int, default=96, help='Content dimension')
    parser.add_argument('--data_path', type=str, default='data/mnist32', help='Path to dataset')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--label_embed_dim', type=int, default=None, help='Label embedding dimension')
    parser.add_argument('--domain1', type=str, default='digit_colored', help='Domain 1')
    parser.add_argument('--domain2', type=str, default='rbg_colored', help='Domain 2')
    parser.add_argument('--domain3', type=str, default='null', help='Domain 3')
    parser.add_argument('--truncation', type=float, default=0.7, help='Truncation factor')
    parser.add_argument('--n_discriminator_updates', type=int, default=1, help='Number of discriminator updates per generator update')
    parser.add_argument('--max_channels', type=int, default=512, help='Maximum number of channels')
    parser.add_argument('--orth_reg_lambda', type=float, default=1.0, help='Orthogonal regularization strength')
    parser.add_argument('--noise_probes', type=int, default=8, help='Number of noise probes for jacobian estimation')
    parser.add_argument('--eps', type=float, default=1e-8, help='Small epsilon value')
    parser.add_argument('--other_server', action='store_true', help='Are you running in hpc cluster ?')
    parser.add_argument('--main_path', type=str, default="your/path", help='Path to dataset')
    parser.add_argument('--entity_name', type=str, default="CSDI", help='Path to dataset')

    
    return parser.parse_args()


# Replace hardcoded hyperparameters with args
args = parse_args()
print(args)

# make args dictionary for wandb logging
configs = vars(args)


main_path = args.main_path
print("Running in.....", main_path)


    

# Remove the hardcoded hyperparameters and use args instead
folders = []
for domain in [args.domain1, args.domain2, args.domain3]:
    if domain != 'null':
        folders.append(domain)
num_classes = len(folders)

assert args.content_dim < args.latent_dim, "Content dimension must be less than latent dimension"

run_name_unique_identifier =  time.strftime("%Y-%m-%d_%H-%M-%S")
run_name_ = args.run_name
os.makedirs(f'{main_path}trained_models_mnists/{args.run_name}_{run_name_unique_identifier}', exist_ok=True)

# Initialize wandb
wandb.init(
    #mode='disabled',
    entity=args.entity_name,
    project=args.project_name,
    name= run_name_+"_"+run_name_unique_identifier,
    config=configs,
    dir=f"{main_path}trained_models_mnists/",
)

best_fid = float('inf')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #0, 1 (a40) in kruskal 2 (gtx) 
data_path = f"{main_path}{args.data_path}"

# Load Combined MNIST Dataset
transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For RGB images
])

print('Loading dataset...')
if args.debug:
    dataset = CombinedMNIST(root=data_path, transform=transform, folders=folders, max_samples=1000)
else:
    dataset = CombinedMNIST(root=data_path, transform=transform, folders=folders)
print('Creating dataloader...')
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Setup for content encoder and style encoder for invertibility
e_c_inv = ContentEncoder(content_dim=args.content_dim).to(device)
e_s_inv = StyleEncoder(num_classes=num_classes, style_dim=args.latent_dim - args.content_dim).to(device)
content_encoder_criterion = nn.MSELoss()
encoder_optim = optim.Adam(list(e_c_inv.parameters()) + list(e_s_inv.parameters()), lr=args.lr, betas=(args.beta1, 0.999))

# Initialize networks and optimizers
generator, discriminator = Generator(num_classes, args.latent_dim, args.content_dim), Discriminator(num_classes)    
generator = generator.to(device)
discriminator = discriminator.to(device)

g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

criterion = nn.BCELoss()

# Initialize FID metric
fid = FrechetInceptionDistance(feature=2048).to(device)
step = 0

num_noise_probes = args.noise_probes
eps = args.eps

start_time = time.time()

# Training loop
for epoch in range(args.num_epochs):
    for i, (real_images, labels) in enumerate(dataloader):
        assert labels.max() < num_classes, "Labels are out of bounds"

        # Train Discriminator multiple times with new samples each time
        for _ in range(args.n_discriminator_updates):
            # Get fresh batch of real samples
            try:
                real_images, labels = next(iter(dataloader))
            except StopIteration:
                dataloader_iter = iter(dataloader)
                real_images, labels = next(dataloader_iter)
                
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device)
            
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            d_optimizer.zero_grad()
            
            # R1 regularization
            real_images.requires_grad = True
            real_pred = discriminator(real_images, labels)
            r1_reg = 0
            grad_real = torch.autograd.grad(
                outputs=real_pred.sum(), inputs=real_images,
                create_graph=True, retain_graph=True)[0]
            r1_reg = args.r1_lambda * grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
            
            d_real_loss = criterion(real_pred, real_labels)
            
            # Generate new fake samples
            # For dependent content and style
            if args.content_dim <= (args.latent_dim // 2):
                common_dim = args.content_dim // 2
                z_c = torch.randn(batch_size, common_dim)
                z_s = torch.randn(batch_size, args.latent_dim - args.content_dim - common_dim)
                z_common = torch.randn(batch_size, common_dim)
            else:
                common_dim = (args.latent_dim - args.content_dim) // 2
                z_c = torch.randn(batch_size, args.content_dim - common_dim)
                z_s = torch.randn(batch_size, common_dim)
                z_common = torch.randn(batch_size, common_dim)

            z = torch.cat([z_common, z_c, z_common, z_s], dim=1).to(device)
            
            content_features, style_features, fake_images = generator(z, labels, get_latent=True)
            d_fake_loss = criterion(discriminator(fake_images.detach(), labels), fake_labels)
            
            # Add R1 regularization to discriminator loss
            d_loss = d_real_loss + d_fake_loss + r1_reg
            d_loss.backward()
            d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()

        encoder_optim.zero_grad()


        g_loss = criterion(discriminator(fake_images, labels), real_labels) #+ sparse_loss

        # Inverse content encoder loss
        reconstructed_content_z = e_c_inv(content_features)
        e_c_inv_loss = content_encoder_criterion(reconstructed_content_z, z[:, :args.content_dim])

        reconstructed_style_z = e_s_inv(style_features, labels)
        e_s_inv_loss = content_encoder_criterion(reconstructed_style_z, z[:, args.content_dim:])


        loss_dicts = {}
        content_style_orthogonality_loss = torch.tensor(0.0)
        content_jacobian_loss = torch.tensor(0.0)
        style_jacobian_loss = torch.tensor(0.0)
        content_style_orthogonality_normalized = torch.tensor(0.0)

        # Orthogonal regularization between content and style jacobians
        if args.orth_reg_lambda > 0.0:
            gen_img = fake_images  # Use a subset of generated images for efficiency
            gen_content = content_features
            gen_style = style_features

            vjp_noise = torch.randn(num_noise_probes, *gen_img.shape, device=gen_img.device) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])

            vjp_jacobian_content, vjp_jacobian_style = torch.autograd.grad(
                outputs=gen_img,
                inputs=[content_features, style_features],
                grad_outputs=vjp_noise,
                is_grads_batched=True,
                create_graph=True,
                retain_graph=True,
            )

            # J_c^Tv --> (noise_probes, B, content_dim), J_s^Tv --> (noise_probes, B, style_dim)
            # J_s^T J_c = J_s^T I J_c = J_s^T E_v[v v^T] J_c = E_v[ J_s^Tv (J_c^Tv)^T ] --> (B, style_dim, content_dim)
            content_style_matrix = torch.einsum('kbi,kbj->bij', vjp_jacobian_style, vjp_jacobian_content) / num_noise_probes

            # aggregate over batch and compute squared Frobenius norm
            content_style_orthogonality_loss = content_style_matrix.square().sum(dim=(1, 2))

            
            content_jacobian_loss = vjp_jacobian_content.square().sum(dim=2).mean(dim=0)
            style_jacobian_loss = vjp_jacobian_style.square().sum(dim=2).mean(dim=0)

            content_style_orthogonality_normalized = (content_style_orthogonality_loss / (content_jacobian_loss * style_jacobian_loss + eps)).mean()



        # scale by user-specified weight and add to generator loss
        tot_g_loss = g_loss + args.orth_reg_lambda * content_style_orthogonality_normalized + 0.01 * e_c_inv_loss + 0.01 * e_s_inv_loss

        # Backprop and step
        tot_g_loss.backward()
        g_optimizer.step()

        encoder_optim.step()

        #logging inside no grad
        with torch.no_grad():
            loss_dicts = {
                "content_style_orthogonality_normalized": content_style_orthogonality_normalized.item(),
                "e_c_inv_loss": e_c_inv_loss.item(),
                "e_s_inv_loss": e_s_inv_loss.item(),
            }
            
        
            step += 1
            if (step + 1) % 100 == 0:
                time_elapsed = time.time() - start_time
                print(f'Epoch [{epoch}/{args.num_epochs}], Step [{i+1}/{len(dataloader)}], '
                    f'd_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, '
                    f'r1_reg: {r1_reg.item():.4f}, '
                    f'CSDI Loss: {content_style_orthogonality_normalized.item():.4f}, '
                        f'e_c_inv_loss: {e_c_inv_loss.item():.4f}, '
                        f'e_s_inv_loss: {e_s_inv_loss.item():.4f}'
                    )
                wandb.log({
                    "discriminator_loss": d_loss.item(),
                    "generator_loss": g_loss.item(),
                    "r1_regularization": r1_reg.item(),
                    # "sparse_loss": sparse_loss.item(),
                    **loss_dicts
                }, step=step)
                start_time = time.time()

    # Save generated images
    if (epoch ) % 5 == 0:
        generator.eval()
        with torch.no_grad():
            num_samples = 8
            num_style_samples = 4  # Number of different style variations
            truncation = args.truncation  # Truncation factor for sampling

            # Sample and truncate content vectors using linear interpolation
            z_content = torch.randn(num_samples, args.content_dim).to(device)
            contents = args.truncation * z_content  # Linear interpolation between 0 and z with truncation weight
            
            # Sample and truncate style vectors using linear interpolation
            style_dim = args.latent_dim - args.content_dim
            z_style = torch.randn(num_style_samples, style_dim).to(device)
            base_styles = args.truncation * z_style  # Linear interpolation between 0 and z with truncation weight
            
            styles = base_styles.repeat_interleave(num_classes, dim=0)
            # Create a grid of images (8 rows x (2 + num_style_samples) columns)
            generated_images = []
            for n in range(num_samples):
                # Keep content part fixed, generate multiple style variations
                content = contents[n].unsqueeze(0).repeat(num_classes * num_style_samples, 1)
                combined_z = torch.cat([content, styles], dim=1)
                #Each row has same content but different styles.
                
                # Generate images for all classes using the same content but different styles
                test_labels = torch.arange(num_classes).repeat(num_style_samples).to(device)
                images = (generator(combined_z, test_labels) + 1.0) / 2.0
                generated_images.append(images)
            
            # Stack all images into a grid
            generated_grid = torch.cat(generated_images, dim=0)
            
            # Create the plot
            plt.figure(figsize=(4 * num_style_samples, 16))  # Adjust figure size for wider grid
            for idx, img in enumerate(generated_grid):
                plt.subplot(num_samples, num_classes * num_style_samples, idx + 1)
                plt.imshow(img.cpu().detach().numpy().transpose(1, 2, 0))
                plt.axis('off')
            
            # log to wandb
            wandb.log({
                "generated_images": wandb.Image(generated_grid),
            }, step=step)
            plt.close()

        start_time = time.time()
        generator.train()

    # Add inside the epoch loop, after the batch loop
    if (epoch + 1) % 25 == 0:
        print("Computing FID scores...")
        fid_scores = compute_classwise_fid(generator, dataset, device, num_classes, args.latent_dim)
        fid_mean = sum(fid_scores.values()) / len(fid_scores)

        # Log FID scores to wandb
        wandb.log({
            **fid_scores
        }, step=step)
        print("FID scores:", fid_scores)

        torch.save(generator.state_dict(), f'{main_path}trained_models_mnists/{args.run_name}_{run_name_unique_identifier}/generator_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'{main_path}trained_models_mnists/{args.run_name}_{run_name_unique_identifier}/discriminator_{epoch}.pth')

        # save the generator model every 25 epochs if the fid was better than the previous best fid
        if fid_mean < best_fid:
            torch.save(generator.state_dict(), f'{main_path}trained_models_mnists/{args.run_name}_{run_name_unique_identifier}/generator_best.pth')
            torch.save(discriminator.state_dict(), f'{main_path}trained_models_mnists/{args.run_name}_{run_name_unique_identifier}/discriminator_best.pth')
            best_fid = fid_mean

        start_time = time.time()

# Finish wandb run
wandb.finish()
