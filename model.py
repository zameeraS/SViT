import torch
import torch.nn as nn
import torch.nn.functional as F

class FireModule(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(FireModule, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class ModifiedSqueezeNet(nn.Module):
    def __init__(self):
        super(ModifiedSqueezeNet, self).__init__()
        # 1. Initial convolutional layer: 96 filters, 3x3, stride 2
        # Note: Standard SqueezeNet uses 7x7 stride 2, but description says 3x3 stride 2.
        # Description: "A single 3 x 3 convolutional layer with 96 filters and has a stride factor of 2."
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2), # Output: (227-3)/2 + 1 = 113
            nn.ReLU(inplace=True),
            # 2. Max-pooling layer: 3x3, stride 2
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # Output: (113-3)/2 + 1 = 56
            
            # 3. Fire modules (FM): Fire2 to Fire9
            FireModule(96, 16, 64, 64),   # Fire2
            FireModule(128, 16, 64, 64),  # Fire3
            FireModule(128, 32, 128, 128),# Fire4
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # MaxPool after Fire4 (standard SqueezeNet)
            
            FireModule(256, 32, 128, 128),# Fire5
            FireModule(256, 48, 192, 192),# Fire6
            FireModule(384, 48, 192, 192),# Fire7
            FireModule(384, 64, 256, 256),# Fire8
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # MaxPool after Fire8
            
            FireModule(512, 64, 256, 256),# Fire9
        )
        # Final output channels: 512 (from Fire9 expand layers: 256+256)
        # Output spatial dim: 
        # Input 227 -> Conv 113 -> Pool 56 -> Fire2-4 -> Pool 28 -> Fire5-8 -> Pool 14 -> Fire9 -> 14x14
        
        # Wait, standard SqueezeNet output at Fire9 is 13x13 or 14x14 depending on padding.
        # Let's verify dimensions.
        # 227x227
        # Conv3x3s2 -> 113x113
        # MaxPool3x3s2 -> 56x56
        # Fire2 (128) -> 56x56
        # Fire3 (128) -> 56x56
        # Fire4 (256) -> 56x56
        # MaxPool3x3s2 -> 28x28
        # Fire5 (256) -> 28x28
        # Fire6 (384) -> 28x28
        # Fire7 (384) -> 28x28
        # Fire8 (512) -> 28x28
        # MaxPool3x3s2 -> 14x14
        # Fire9 (512) -> 14x14
        
        # The description says: "The architecture of SqueezeNet has been modified to remove the final convolutional layer, global average pooling layers, and softmax layers."
        # So we stop at Fire9 output.

    def forward(self, x):
        x = self.features(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_classes=4, num_layers=3, num_heads=8, dropout=0.1):
        super(VisionTransformer, self).__init__()
        # Input is Feature Map from SqueezeNet: H' x W' x C' (14 x 14 x 512)
        # We treat each spatial position as a patch.
        # Patch size P=1 (effectively, since we are using the feature map pixels as patches)
        # Or we can say the patch size is the receptive field of the feature map pixel.
        # The description says: "Input feature map is divided into non-overlapping patches... Each patch is then linearly embedded... into a 1D vector of length D"
        # If we take the feature map directly, C' is the embedding dimension?
        # Eq 8: xp = Flatten(Patch(Fmap))
        # If P=1 (1x1 spatial patch in feature map), then xp is just the vector at that position, length C'.
        # If we want to project it to dimension D (hidden_dim), we use a linear layer.
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Linear projection to hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional Encoding
        # We have 14x14 = 196 patches
        self.num_patches = 14 * 14 
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: [Batch, C, H, W] -> [Batch, 512, 14, 14]
        b, c, h, w = x.shape
        
        # Flatten spatial dimensions: [Batch, C, H*W] -> [Batch, H*W, C]
        x = x.flatten(2).transpose(1, 2) 
        
        # Linear Embedding
        x = self.embedding(x) # [Batch, 196, hidden_dim]
        
        # Add Positional Encoding
        x = x + self.pos_embedding
        
        # Transformer Encoder
        x = self.transformer_encoder(x) # [Batch, 196, hidden_dim]
        
        # Pooling (Mean Pooling as per Eq 11)
        # "A pooling operation, such as mean pooling... is applied to the sequence... to obtain a single global feature vector"
        x = x.mean(dim=1) # [Batch, hidden_dim]
        
        # Classification
        x = self.classifier(x) # [Batch, num_classes]
        
        return x

class SViT(nn.Module):
    def __init__(self, num_classes=4, vit_layers=3, vit_heads=8, vit_dim=512, dropout=0.1):
        super(SViT, self).__init__()
        self.squeezenet = ModifiedSqueezeNet()
        # SqueezeNet output channels = 512
        self.vit = VisionTransformer(input_dim=512, hidden_dim=vit_dim, num_classes=num_classes, num_layers=vit_layers, num_heads=vit_heads, dropout=dropout)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.squeezenet(x)
        output = self.vit(features)
        return output
