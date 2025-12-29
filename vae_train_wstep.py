import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Dataset Class for flat folder ---
class DrawingDataset(Dataset):
    def __init__(self, root_dir):
        # Read all PNGs directly from folder
        self.img_paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        print("Total images found:", len(self.img_paths))

        # Transform: grayscale image to tensor
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_tensor = self.transform(img)
        return img_tensor

# --- VAE Model (unchanged) ---
class VAE64(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE64, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.GELU()
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 512)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 1, 1)),
            nn.ConvTranspose2d(512, 128, 8, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- Weighted Loss Function ---
def weighted_vae_loss(recon_x, x, mu, logvar, epoch):
    weight = torch.ones_like(x)
    weight[x > 0.5] = 50.0
    mse = torch.sum(weight * (recon_x - x) ** 2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    beta = 0 if epoch < 50 else min(0.0001, (epoch - 50) * 0.000002)
    return mse + (beta * kld)

# --- Trainer ---
class VAETrainer:
    def __init__(self, model, device, lr=5e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self, loader, epoch):
        self.model.train()
        train_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(batch)
            loss = weighted_vae_loss(recon_batch, batch, mu, logvar, epoch)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        return train_loss / len(loader.dataset)

    def visualize_results(self, epoch, test_batch):
        self.model.eval()
        with torch.no_grad():
            test_batch = test_batch.to(self.device)
            recon, _, _ = self.model(test_batch)
            orig = test_batch[0].cpu().squeeze().numpy()
            rec = recon[0].cpu().squeeze().numpy()
            combined = np.hstack((orig, rec))
            plt.imshow(combined, cmap='gray')
            plt.title(f"Epoch {epoch}: Target vs Prediction")
            plt.axis('off')
            plt.savefig(f"epoch_{epoch}_new.png")
            plt.close()

# --- Main Training ---
if __name__ == "__main__":
    LATENT_DIM = 128
    BATCH_SIZE = 16  # smaller batch for small dataset
    EPOCHS = 100      # fewer epochs for 50 images
    DATA_DIR = "images"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DrawingDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = VAE64(latent_dim=LATENT_DIM)
    trainer = VAETrainer(model, DEVICE)

    for epoch in range(1, EPOCHS + 1):
        avg_loss = trainer.train_epoch(loader, epoch)
        print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
        if epoch % 5 == 0:
            sample_batch = next(iter(loader))
            trainer.visualize_results(epoch, sample_batch)

    torch.save(model.state_dict(), "vae_new.pth")
    print("Model saved as vae_flat_images.pth")
