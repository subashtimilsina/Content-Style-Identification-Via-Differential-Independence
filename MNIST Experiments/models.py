import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, num_classes, latent_dim, content_dim=5):
        super().__init__()
        assert content_dim < latent_dim, "Content dimension must be less than latent dimension"

        self.content_dim = content_dim
        self.style_dim = latent_dim - content_dim
        self.num_classes = num_classes

        # Content encoder (2 layers FCN)
        self.content_encoder = nn.Sequential(
            nn.Linear(self.content_dim, self.content_dim),
        )

        self.style_encoders = nn.ModuleList()
        for i in range(num_classes):
            self.style_encoders.append(nn.Sequential(
                nn.Linear(self.style_dim, self.style_dim),
            ))
        
        self.init_size = 2  # Initial size before upsampling
        # Modified to accept concatenated content and style features (128 + 128 = 256)
        self.l1 = nn.Sequential(nn.Linear(self.content_dim + self.style_dim, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 4x4
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2),  # 8x8
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2),  # 16x16
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2),  # 32x32
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def get_content_style(self, z, labels):
        # Content encoding (independent of labels)
        content_input = z[:, :self.content_dim]
        style_input = z[:, self.content_dim:]
        content_features = self.content_encoder(content_input)
        
        # Style encoding (includes label information)
        style_features = torch.stack([self.style_encoders[i](style_input) for i in range(self.num_classes)], dim=1)
        idx = torch.LongTensor(range(z.size(0))).to(z.device)
        assert labels.max() < self.num_classes, "label out of bounds"
        assert z.size(0) == labels.size(0), "need same number of elements"
        style_features = style_features[idx, labels]

        return content_features, style_features
    
    def generate_image(self, content_features, style_features):
        # Combine content and style features
        x = torch.cat([content_features, style_features], dim=1)
        x = self.l1(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)

        return self.conv_blocks(x)

    def forward(self, z, labels, get_latent=False):
        
        content_features, style_features = self.get_content_style(z, labels)

        if get_latent:
            return content_features, style_features, self.generate_image(content_features, style_features)
        else:
            return self.generate_image(content_features, style_features)
        
        
        
        


class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, 32*32)
        
        self.conv_blocks = nn.Sequential(
            # input is (1 + num_classes) x 32 x 32
            nn.Conv2d(3 + 1, 64, 3, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 8x8
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 4x4
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(256),
        )
        
        # The height and width of downsampled image
        ds_size = 4
        self.fc = nn.Sequential(
            nn.Linear(256 * ds_size * ds_size, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(label_embedding.size(0), 1, 32, 32)
        x = torch.cat([img, label_embedding], dim=1)
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



# Content Encoder invertible network with respect to content_input
class ContentEncoder(nn.Module):
    def __init__(self, content_dim=64):
        super().__init__()
        self.content_dim = content_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.content_dim, self.content_dim),
        )
        
    def forward(self, content_input):
        return self.net(content_input)
    

# Style Encoder invertible network with respect to style_input also multiple styles according to labels
class StyleEncoder(nn.Module):
    def __init__(self, num_classes, style_dim=64):
        super().__init__()
        self.style_dim = style_dim
        self.num_classes = num_classes
        
        self.style_encoders = nn.ModuleList()
        for i in range(num_classes):
            self.style_encoders.append(nn.Sequential(
                nn.Linear(self.style_dim, self.style_dim),
            ))
        
    def forward(self, style_input, labels):
        style_features = torch.stack([self.style_encoders[i](style_input) for i in range(self.num_classes)], dim=1)
        idx = torch.LongTensor(range(style_input.size(0))).to(style_input.device)
        assert labels.max() < self.num_classes, "label out of bounds"
        assert style_input.size(0) == labels.size(0), "need same number of elements"
        style_features = style_features[idx, labels]
        return style_features
