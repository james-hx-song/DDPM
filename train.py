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
print(f"Using {DEVICE}")
IMG_SIZE = 64
dataset = get_dataset(IMG_SIZE)

epochs = 1
batch_size = 128

num_chn = dataset[0][0].shape[0]
print(f"Number of channels: {num_chn}")

lr = 3e-4

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = UNet(c_in=1, c_out=1, layers=4).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()
diffusion = Diffusion(img_size=IMG_SIZE, device=DEVICE, num_chn=num_chn)



model.train()
torch.autograd.set_detect_anomaly(True)
for epoch in range(epochs):
  print(f"Epoch {epoch}")
  for i, (img, _) in enumerate(dataloader):
    optimizer.zero_grad()
    img = img.to(DEVICE)

    t = diffusion.sample_timestep(img.shape[0]).to(DEVICE)

    x_noisy, noise = diffusion.forward_noise(img, t)

    noise_pred = model(x_noisy, t)


    loss = criterion(noise_pred, noise)
    loss.backward()

    optimizer.step()
    # if i % 100 == 0:
    print(f"Batch {i} | Loss {loss.item()}")


x = diffusion.sample(model, 1)
from utils import plot_image
import matplotlib.pyplot as plt
plot_image(x)
plt.show()

