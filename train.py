import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
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

epochs = 2
batch_size = 512

num_chn = dataset[0][0].shape[0]
print(f"Number of channels: {num_chn}")

lr = 3e-4

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = UNet(c_in=num_chn, c_out=num_chn).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
diffusion = Diffusion(img_size=IMG_SIZE, device=DEVICE, num_chn=num_chn, noise_sch='cosine')



model.train()
torch.autograd.set_detect_anomaly(True)
for epoch in range(epochs):
  print(f"Epoch {epoch}")
  epoch_loss = 0
  for i, (img, _) in enumerate(dataloader):
    optimizer.zero_grad()
    img = img.to(DEVICE)

    t = diffusion.sample_timestep(img.shape[0]).to(DEVICE)

    x_noisy, noise = diffusion.forward_noise(img, t)

    noise_pred = model(x_noisy, t)


    loss = F.l1_loss(noise_pred, noise)
    loss.backward()

    optimizer.step()
    # if i % 100 == 0:
    print(f"Batch {i} | Loss {loss.item()}")
    epoch_loss += loss.detach().item()
  print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss / len(dataloader):.4f}")


checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
}
torch.save(checkpoint, f"checkpoints/model_epoch={epoch}.pth")

x = diffusion.sample(model, 1)
from utils import plot_image
import matplotlib.pyplot as plt
plot_image(x)
plt.show()

