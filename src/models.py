"""
models.py

This file defines the neural network components for the Variational Autoencoder (VAE).

The file contains three main classes:
1.  Encoder: A convolutional neural network that compresses an input image into the
    parameters (mu and log_var) of a latent Gaussian distribution.
2.  Decoder: A transposed convolutional neural network that reconstructs an image
    from a latent space vector `z`.
3.  VAE: The main module that encapsulates the Encoder and Decoder, handles the
    reparameterization trick, and defines the full forward pass.

This VAE will serve as the foundational component for the more complex Recurrent
State-Space Model (RSSM) to be built later.
"""

# PyTorch and neural network libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple


class Encoder(nn.Module):
    """
    Encoder module for the Variational Autoencoder (VAE).

    This module takes an input image and encodes it into the parameters of a latent
    Gaussian distribution (mean and log variance).
    """

    def __init__(self, input_channels: int = 3, latent_dim: int = 32):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
    
        self.conv_layers = nn.Sequential(
            # Input: [B, 3, 64, 64]
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1), # -> [B, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> [B, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> [B, 128, 8, 8]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> [B, 256, 4, 4]
            nn.ReLU(),
            nn.Flatten() # -> [B, 256*4*4]
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Encoder.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W].

        Returns:
            mu (torch.Tensor): Mean of the latent Gaussian distribution.
            log_var (torch.Tensor): Log variance of the latent Gaussian distribution.
        """
        x = self.conv_layers(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var
    
class Decoder(nn.Module):
    """
    Decoder module for the Variational Autoencoder (VAE).

    This module takes a latent space vector `z` and reconstructs the original image.
    """

    def __init__(self, latent_dim: int = 32, output_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels

        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        self.deconv_layers = nn.Sequential(
            # Input: [B, 256, 4, 4]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> [B, 128, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # -> [B, 64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # -> [B, 32, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1), # -> [B, C, 64, 64]
            nn.Sigmoid()  # Output pixel values between [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Decoder.

        Args:
            z (torch.Tensor): Latent space tensor of shape [B, latent_dim].

        Returns:
            x_recon (torch.Tensor): Reconstructed image tensor of shape [B, C, H, W].
        """
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        x_recon = self.deconv_layers(x)
        return x_recon
    
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) module.

    This module encapsulates the Encoder and Decoder, handles the reparameterization
    trick, and defines the full forward pass.
    """

    def __init__(self, input_channels: int = 3, latent_dim: int = 32):
        super().__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu (torch.Tensor): Mean of the latent Gaussian distribution.
            log_var (torch.Tensor): Log variance of the latent Gaussian distribution.

        Returns:
            z (torch.Tensor): Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through the VAE.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W].

        Returns:
            x_recon (torch.Tensor): Reconstructed image tensor.
            mu (torch.Tensor): Mean of the latent Gaussian distribution.
            log_var (torch.Tensor): Log variance of the latent Gaussian distribution.
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

# TODO: Implement the RSSM model
