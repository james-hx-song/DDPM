from datasets import get_dataset
from diffusion import Diffusion
from models.unet import UNet
from utils import plot_image
import matplotlib.pyplot as plt
import torch

IMG_SIZE = 64
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'


diffusion = Diffusion(img_size=IMG_SIZE, device=DEVICE, num_chn=1, noise_sch='cosine')

checkpoint = torch.load("checkpoints/model_epoch=1.pth")

model = UNet(c_in=1, c_out=1).to(DEVICE)
model.load_state_dict(checkpoint['model'])

x = diffusion.sample(model, 10)
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i in range(10):
    axes[i].imshow(x[i].permute(1, 2, 0).cpu().numpy(),)
    axes[i].axis('off')
plt.show()

