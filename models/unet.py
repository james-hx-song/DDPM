import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbeddings(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, t):
    device = t.device

    half_dim = self.dim // 2
    # this is following
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
 
    embeddings = t.unsqueeze(-1) * embeddings.unsqueeze(0)
    embeddings = torch.stack((embeddings.sin(), embeddings.cos()), dim=-1)
    embeddings = embeddings.view(t.shape[0], -1)
    return embeddings


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.relu = nn.ReLU()
        self.norm = nn.GroupNorm(8, out_ch)

    def forward(self, x, t):
        h = self.relu(self.norm(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        h = h + time_emb[:, :, None, None]
        h = self.relu(self.norm(self.conv2(h)))
        return h
    
class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):

      B, C, H, W = x.shape

      # Flatten x in height and width
      x = x.view(B, C, H * W).permute(0, 2, 1) # (B, H*W, C)

      qkv = self.qkv(x) # (B, H*W, C*3)

      q, k, v = qkv.chunk(3, dim=2) # (B, H*W, C)

      q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, H*W, hs)
      k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
      v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

      x = F.scaled_dot_product_attention(q, k, v)
      x = x.transpose(1, 2).contiguous().view(B, -1, C)

      x = self.proj(x) # (B, H*W, C)

      x = x.transpose(1, 2).view(B, C, H, W)

      return x
    
class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, layers=4, time_emb_dim=32, num_heads=4):
        super().__init__()

        # downs = [16, 32, 64, 128, 256, 512]
        # ups = [512, 256, 128, 64, 32, 16]

        downs = [16 * 2**i for i in range(layers)]
        ups = [16 * 2**i for i in range(layers-1, -1, -1)]


        self.time_embeddings = PositionalEmbeddings(time_emb_dim)

        self.conv0 = nn.Conv2d(in_channels=c_in, out_channels=16, kernel_size=3, padding=1)
        self.downs = nn.ModuleList([Block(downs[i], downs[i+1], time_emb_dim) for i in range(len(downs)-1)])
        self.ups = nn.ModuleList([Block(ups[i], ups[i+1], time_emb_dim) for i in range(len(ups)-1)])

        self.attention = SelfAttention(downs[-1], num_heads=num_heads)
        self.outconv = nn.Conv2d(ups[-1], c_out, kernel_size=3, padding=1)

    def forward(self, x, t):
        t = self.time_embeddings(t)
        x = self.conv0(x)
        skip_connections = []
        for down in self.downs:
            x = down(x, t)
            skip_connections.append(x)
            x = F.avg_pool2d(x, kernel_size=2)
        # print(x.shape)
        x = self.attention(x)
        for idx, up in enumerate(self.ups):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            skip_connection = skip_connections[-(idx + 1)] 
            x = x + skip_connection 
            x = up(x, t)


        x = self.outconv(x)
        return x

