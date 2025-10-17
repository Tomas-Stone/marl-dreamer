# main.py

import torch
import os
import pickle
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from tqdm import tqdm

# Our custom modules
from env import CarRacingEnv
from models import VAE
from train import train_vae

if __name__ == "__main__":
    # --- 1. Configuration ---
    CONFIG = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_episodes": 500,
        "dataset_path": "data/car_racing_dataset.pkl",
        "image_channels": 1, # Grayscale
        "latent_dim": 32,
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "model_save_path": "checkpoints/vae.pth",
    }
    print(f"Using device: {CONFIG['device']}")

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(CONFIG['dataset_path']), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)

    # --- 2. Data Collection ---
    if not os.path.exists(CONFIG['dataset_path']):
        print("Dataset not found. Collecting data...")
        env = CarRacingEnv(device=CONFIG['device'])
        observations = []
        for episode in tqdm(range(CONFIG['num_episodes']), desc="Collecting Data"):
            _ = env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                action = env.action_space.sample() # Take a random action
                obs, _, terminated, truncated, _ = env.step(action)
                observations.append(obs.cpu()) # Store on CPU to save GPU memory
        
        with open(CONFIG['dataset_path'], 'wb') as f:
            pickle.dump(observations, f)
        env.close()
        print(f"Data collected and saved to {CONFIG['dataset_path']}")
    
    # --- 3. Dataset and DataLoader ---
    print("Loading dataset...")
    with open(CONFIG['dataset_path'], 'rb') as f:
        # Stack all observations into a single tensor
        data = torch.stack(pickle.load(f), dim=0)

    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    print("Dataset loaded successfully.")

    # --- 4. Model, Optimizer, and Training ---
    vae = VAE(
        input_channels=CONFIG['image_channels'],
        latent_dim=CONFIG['latent_dim']
    ).to(CONFIG['device'])
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=CONFIG['learning_rate'])

    print("Starting VAE training...")
    train_vae(
        vae=vae,
        dataloader=dataloader,
        optimizer=optimizer,
        epochs=CONFIG['epochs'],
        device=CONFIG['device']
    )
    print("Training finished.")

    # --- 5. Save the Model and Visualize Results ---
    torch.save(vae.state_dict(), CONFIG['model_save_path'])
    print(f"Model saved to {CONFIG['model_save_path']}")

    # Visualize some reconstructions
    with torch.no_grad():
        # Get a fixed batch of data for visualization
        sample_batch = next(iter(dataloader))[0].to(CONFIG['device'])
        
        # Reconstruct the images
        recon_batch, _, _ = vae(sample_batch)
        
        # Combine original and reconstructed images side-by-side
        comparison = torch.cat([sample_batch, recon_batch])
        
        save_image(comparison.cpu(), 'results/reconstruction.png', nrow=CONFIG['batch_size'])
        print("Saved reconstruction comparison to 'results/reconstruction.png'")