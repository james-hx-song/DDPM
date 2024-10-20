import torch
import matplotlib.pyplot as plt

def plot_image(x):
    plt.imshow(x.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.axis('off')

def plot_increasing_noise_comparison(x0, diff1, diff2, steps=5):
    fig, axs = plt.subplots(2, steps, figsize=(15, 6))
    
    for i, t in enumerate(torch.linspace(0, 999, steps)):
        # Noise using Diff 1
        x1, eps1 = diff1.forward_noise(x0, int(t))
        axs[0, i].imshow(x1.squeeze(0).permute(1, 2, 0).cpu().numpy())
        axs[0, i].axis('off')
        axs[0, i].set_title(f"Noise Level: {int(t)}")

        # Noise using Diff 2
        x2, eps2 = diff2.forward_noise(x0, int(t))
        axs[1, i].imshow(x2.squeeze(0).permute(1, 2, 0).cpu().numpy())
        axs[1, i].axis('off')

    axs[0, 0].set_ylabel('Linear Schedule')
    axs[1, 0].set_ylabel('Cosine Schedule')

    plt.tight_layout()
    plt.show()

