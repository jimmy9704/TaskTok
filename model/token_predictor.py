import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import math
from model.titok.modeling.modules.blocks import ResidualAttentionBlock
from torch.distributions.normal import Normal


class RestoreEncoder(nn.Module):
    def __init__(self, token_size: int=64, width: int=512, num_latent_tokens: int=256, num_layers: int=6, num_heads: int=8):
        super().__init__()
        self.token_size = token_size
        self.width = width
        self.num_latent_tokens = num_latent_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads

        scale = self.width ** -0.5
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)
        self.conv_in = nn.Conv2d(self.token_size, self.width, kernel_size=1, bias=True)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

    def forward(self, latent_tokens):
        batch_size = latent_tokens.shape[0]

        latent_tokens = latent_tokens.permute(0, 1, 3, 2) # ND1L ->NDL1
        latent_tokens = self.conv_in(latent_tokens)
        latent_tokens = latent_tokens.permute(0, 2, 1, 3).reshape(batch_size, self.num_latent_tokens, self.width) # NDL1 -> NLD

        latent_tokens = latent_tokens + self.latent_token_positional_embedding

        latent_tokens = self.ln_pre(latent_tokens)
        latent_tokens = latent_tokens.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            latent_tokens = self.transformer[i](latent_tokens)
        latent_tokens = latent_tokens.permute(1, 0, 2)  # LND -> NLD
        latent_tokens = self.ln_post(latent_tokens)

        latent_tokens = latent_tokens.reshape(batch_size, self.num_latent_tokens, self.width, 1).permute(0, 2, 1, 3) # NLD ->NDL1
        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.permute(0, 1, 3, 2)  # NDL1 -> NDL1
        return latent_tokens

class TokenPredictor(nn.Module):
    """
    Simple transformer encoder that refines latent tokens.
    Input / output shape: (B, C, H, W) kept identical.
    """

    def __init__(
        self,
        d_token: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 1,
        n_tokens: int = 256,
        n_tasks: int = 3,
        init_scale: float = 0.1,
        T: float = 1.0,
    ):
        super().__init__()

        self.encoder = RestoreEncoder(
            token_size=d_token,
            width=d_model,
            num_latent_tokens=n_tokens,
            num_layers=n_layers,
            num_heads=n_heads
        )

        self.n_tokens = n_tokens
        self.n_tasks = n_tasks
        self.T = float(T)

        # per-task switch logits (length = n_tokens)
        self.token_switch_tasks = nn.ParameterList([
            nn.Parameter(init_scale * torch.randn(n_tokens))
            for _ in range(n_tasks)
        ])

    def init_token_switch(self, order, p_min=0.0, p_max=1.0, noise_std=0.00, task_id=None):
        """Initialize token_switch with order-based probability (order[0]=p_min, order[-1]=p_max)."""
        device = self.token_switch_tasks[0].device
        order = torch.as_tensor(order, dtype=torch.long, device=device)
        
        # Linear interpolation: p_min -> p_max along order (order[0]=p_min, order[-1]=p_max)
        p = torch.linspace(p_min, p_max, self.n_tokens, device=device).clamp(1e-4, 1-1e-4)
        logits = torch.log(p) - torch.log1p(-p)  # logit
        
        init_logits = torch.empty(self.n_tokens, device=device)
        init_logits[order] = logits
        
        with torch.no_grad():
            self.token_switch_tasks[task_id].copy_(init_logits + noise_std * torch.randn_like(init_logits))
        
    def forward(self, z: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        """
        Args:
            z: (B, C, H, W)
        Returns:
            refined z with the same shape
        """
        b, c, h, w = z.shape

        #-----refine z with transformer-----
        z_refined = z + self.encoder(z)

        token_switch = self.token_switch_tasks[task_id]
        prob = torch.sigmoid(token_switch).view(1, 1, h, w)  # (1,1,H,W)

        z_selected_soft = z_refined * prob + z * (1.0 - prob)
        hard = (prob > 0.5).float()  # (1,1,H,W)
        z_selected_hard = z_refined * hard + z * (1.0 - hard)
        z_selected = z_selected_hard.detach() - z_selected_soft.detach() + z_selected_soft

        return z_selected, z_refined, hard.sum(), prob, token_switch