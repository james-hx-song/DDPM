import torch
import torch.nn as nn

class Diffusion:
    def __init__(self, T=1000, num_chn=3, beta_1=1e-4, beta_T=0.02, img_size=64, noise_sch='linear', device='cuda'):
        assert noise_sch in ['linear', 'cosine'], f"noise_sch must be either 'linear' or 'cosine', got {noise_sch}"

        self.T = T
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.img_size = img_size
        self.noise_sch = noise_sch
        self.device = device
        self.s = 0.008
        self.num_chn = num_chn

        self.beta = self.alpha_bar = self.alpha = None
        self.set_schedule()

    def sample_timestep(self, batch_size):
        return torch.randint(0, self.T, (batch_size,), device=self.device)
    def set_schedule(self,):
        if self.noise_sch == 'cosine':
            t =  torch.linspace(0, self.T, self.T, device=self.device)
            f = torch.cos((t/self.T + self.s) / (1 + self.s) * torch.pi / 2) ** 2
            self.alpha_bar = f / f[0]
            self.alpha = self.alpha_bar.clone()
            self.alpha[1:] = self.alpha_bar[1:] / self.alpha_bar[:-1]
            self.beta = 1 - self.alpha
        else:
            self.beta = torch.linspace(self.beta_1, self.beta_T, self.T, device=self.device)
            self.alpha = 1 - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward_noise(self, x, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1)

        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1)

        noise = torch.randn_like(x, device=self.device)

        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise
    
    def sample(self, model, n):
        print(f"Sampling {n} images")
        model.eval()
        with torch.no_grad():
            x = torch.randn(n, 1, self.img_size, self.img_size, device=self.device)
            for i in reversed(range(1, self.T)):
                z = torch.randn_like(x, device=self.device) if i > 1 else 0
                
                t = (torch.ones(n, device=self.device) * i).long()
                alpha = self.alpha[t][:, None, None, None]
                alpha_bar = self.alpha_bar[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                x = 1 / torch.sqrt(alpha_bar) * (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * model(x, t)) +  torch.sqrt(beta) * z
        model.train()
        x = x.clamp(-1, 1)
        x = (x+1)/2
        x = x*255
        return x

        
if __name__ == "__main__":
    diff = Diffusion(device='cpu')

    import torchvision
    IMG_SIZE = 512
    transforms = torchvision.transforms.Compose(
        [
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.ToTensor(),
        ]
    )
    dataset = torchvision.datasets.MNIST(root='.', download=True, transform=transforms)

    print(dataset[0][0].shape)
    img = dataset[0][0].unsqueeze(0)

    
    diff2 = Diffusion(img_size=IMG_SIZE, device='cpu', noise_sch='cosine')
    from utils import plot_increasing_noise_comparison  
    plot_increasing_noise_comparison(img, diff, diff2, steps=10)
    # diff = Diffusion(device='cpu')
    # from models.unet import UNet
    # model = UNet()

    # x = diff.sample(model, 1)









        


