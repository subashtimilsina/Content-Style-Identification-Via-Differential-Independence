import torch
import torchvision
import os

# Create dataset class for combined MNIST variants
class CombinedMNIST(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, folders=None, max_samples=100_000):
        self.transform = transform
        self.root = root
        
        # Get list of folders in root directory
        if folders is None:
            self.folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
        else:
            self.folders = folders
        print(self.folders)
        self.folders.sort() # Sort to ensure consistent label assignment
        
        # Create folder to label mapping
        self.folder_to_label = {folder: i for i, folder in enumerate(self.folders)}
        print(self.folder_to_label)
        # Store images and labels
        self.images = []
        self.labels = []
        
        self.max_samples = max_samples
        for folder in self.folders:
            folder_path = os.path.join(root, folder)
            label = self.folder_to_label[folder]
            # print(f"Loading images from folder: {folder} with label: {label}")
            # Load all images in this folder
            for i, img_name in enumerate(os.listdir(folder_path)):
                if i >= self.max_samples:
                    break
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, img_name)
                    img = torchvision.io.read_image(img_path).type(torch.FloatTensor) / 255.0
                    if self.transform:
                        # print(f"Applying transform to image: {img_name}", img.shape)
                        img = self.transform(img)
                    self.images.append(img)
                    self.labels.append(label)
            
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
