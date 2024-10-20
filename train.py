import torch
import torch.nn as nn
from models.unet import UNet
from diffusion import Diffusion
from datasets import get_dataset



DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'


epochs = 1
batch_size = 1
IMG_SIZE = 64

lr = 3e-4
dataset = get_dataset(IMG_SIZE)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = UNet().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()
diffusion = Diffusion(img_size=64, device=DEVICE)

model.train()
torch.autograd.set_detect_anomaly(True)
for epoch in range(epochs):
  for i, (img, _) in enumerate(dataloader):
    optimizer.zero_grad()
    img = img.to(DEVICE)

    t = diffusion.sample_timestep(img.shape[0]).to(DEVICE)

    x_noisy, noise = diffusion.forward_noise(img, t)

    noise_pred = model(x_noisy, t)


    loss = criterion(noise_pred, noise)
    loss.backward()

    optimizer.step()
    if i % 100 == 0:
      print(f"Epoch {epoch} | Batch {i} | Loss {loss.item()}")
