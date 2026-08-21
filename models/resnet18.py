import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18

class DiseaseTrajectoryVAE(nn.Module):
    def __init__(self, latent_dim=128, dropout_p=0.6, in_channels=4):
        super(DiseaseTrajectoryVAE, self).__init__()

        # 1. Use 3D ResNet-18 as the feature extractor
        backbone = r3d_18(weights=None)

        backbone.stem[0] = nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7),
                                     stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        
        # Keep the penultimate AdaptiveAvgPool3d((1, 1, 1)) layer intact,
        # letting the CNN compress 3D spatial features down to a single point.
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.dropout = nn.Dropout(p=dropout_p)
        
        # ==========================================
        # 2. VAE Disease Mapping Head
        # ==========================================
        # Directly receive the highly condensed 512-dimensional features
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # ==========================================
        # 3. Minimalist Decoder (maintaining VAE generative standards)
        # ==========================================
        self.decoder_proj = nn.Sequential(
            nn.Linear(latent_dim, 3 * 4 * 3 * 64),
            nn.ReLU()
        )

        self.decoder_layers = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 4, 2, 1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 4, 2, 1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.ConvTranspose3d(16, 8, 4, 2, 1), nn.BatchNorm3d(8), nn.ReLU(),
            nn.ConvTranspose3d(8, 4, 4, 2, 1), nn.BatchNorm3d(4), nn.ReLU(),
            nn.ConvTranspose3d(4, in_channels, 4, 2, 1)
        )

    def reparameterize(self, mu, logvar):
        """VAE core: reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu # During inference, use the mean directly and disable randomness

    def forward(self, img):
        """
        img: (N, 4, 91, 109, 91) - Clean residual images preprocessed by Nilearn
        """
        # 1. Extract image features: leverage the built-in pooling layer for maximum compression
        raw_feat = self.feature_extractor(img)         # (N, 512, 1, 1, 1)
        img_feat = raw_feat.view(raw_feat.size(0), -1)      # Flatten to (N, 512)

        img_feat = self.dropout(img_feat)
        
        # 2. Generate latent space probability distribution
        mu_raw = self.fc_mu(img_feat)         # (N, latent_dim)
        logvar = self.fc_logvar(img_feat)     # (N, latent_dim)

        # Enforce non-negative constraint: apply Softplus only to dim 0 representing Score
        mu_score = F.softplus(mu_raw[:, 0:1]) # (N, 1), ensures score is strictly > 0
        mu_rest = mu_raw[:, 1:]               # (N, latent_dim - 1)
        mu = torch.cat([mu_score, mu_rest], dim=1) # Recombine back to (N, latent_dim)
        
        # 3. Sample latent vector z
        z = self.reparameterize(mu, logvar) # (N, latent_dim) 
        
        # 4. Get Score
        score = mu[:, 0]
        
        # 5. 3D image reconstruction
        # Reshape to (N, 64, 3, 4, 3) for deconvolution
        x = self.decoder_proj(z).view(z.size(0), 64, 3, 4, 3)
        x = self.decoder_layers(x)

        # Current shape is (N, 1, 96, 128, 96), perform precise center crop back to original resolution
        d_s = (96 - 91) // 2   # 2
        h_s = (128 - 109) // 2 # 9
        w_s = (96 - 91) // 2   # 2
        
        recon_imgs = x[:, :, d_s:d_s+91, h_s:h_s+109, w_s:w_s+91]

        # Return format matches ViT exactly: (score, reconstruction, mean, log-variance)
        return score, recon_imgs, mu, logvar
