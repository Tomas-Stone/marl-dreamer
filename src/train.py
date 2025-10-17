# train.py

import torch
from torch.nn import functional as F
from tqdm import tqdm # For a nice progress bar

def vae_loss_function(recon_x, x, mu, logvar):
    """
    Calculates the VAE loss, which is a sum of reconstruction loss and KL divergence.
    """
    # Reconstruction loss using Binary Cross-Entropy
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

def train_vae(vae, dataloader, optimizer, epochs, device):
    """
    The main training loop for the VAE.
    """
    vae.train() # Set the model to training mode
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Use tqdm for a progress bar
        for batch_idx, (data,) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            
            # Forward pass
            recon_batch, mu, logvar = vae(data)
            
            # Calculate loss
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader.dataset)
        print(f'===> Epoch: {epoch+1} Average loss: {avg_loss:.4f}')