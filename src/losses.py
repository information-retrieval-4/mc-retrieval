"""CLIP-style symmetric contrastive loss (InfoNCE)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    """Symmetric InfoNCE loss with a learnable temperature.

    Given L2-normalised text embeddings T and voxel embeddings V of shape
    (B, D), computes:
        logits = (T @ V^T) / τ          — (B, B) cosine-similarity matrix
        loss   = (CE(logits, labels) + CE(logits^T, labels)) / 2

    where labels = [0, 1, ..., B-1] (each sample matches itself).
    """

    def __init__(self, temperature_init: float = 0.07):
        super().__init__()
        # learnable log-temperature (clamped for stability)
        self.log_temp = nn.Parameter(torch.tensor(temperature_init).log())

    @property
    def temperature(self) -> torch.Tensor:
        # clamp between ~0.01 and ~1.0
        return self.log_temp.exp().clamp(min=0.01, max=1.0)

    def forward(
        self,
        text_emb: torch.Tensor,
        voxel_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_emb:  (B, D) L2-normalised text embeddings
            voxel_emb: (B, D) L2-normalised voxel embeddings
        Returns:
            scalar loss
        """
        # cosine similarity scaled by temperature
        logits = (text_emb @ voxel_emb.T) / self.temperature   # (B, B)

        labels = torch.arange(len(logits), device=logits.device)

        loss_t2v = F.cross_entropy(logits, labels)        # text  → voxel
        loss_v2t = F.cross_entropy(logits.T, labels)      # voxel → text

        return (loss_t2v + loss_v2t) / 2.0


class TriModalCLIPLoss(nn.Module):
    """Symmetric InfoNCE loss on three modality pairs: TI, TV, IV.

    L = λ_TI * L(T,I) + λ_TV * L(T,V) + λ_IV * L(I,V)
    """

    def __init__(
        self,
        temperature_init: float = 0.07,
        lambda_ti: float = 1.0,
        lambda_tv: float = 1.0,
        lambda_iv: float = 1.0,
    ):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(temperature_init).log())
        self.lambda_ti = lambda_ti
        self.lambda_tv = lambda_tv
        self.lambda_iv = lambda_iv

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(min=0.01, max=1.0)

    def _infonce(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        logits = (a @ b.T) / self.temperature
        labels = torch.arange(len(logits), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0

    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        voxel_emb: torch.Tensor,
    ):
        l_ti = self._infonce(text_emb, image_emb)
        l_tv = self._infonce(text_emb, voxel_emb)
        l_iv = self._infonce(image_emb, voxel_emb)
        total = self.lambda_ti * l_ti + self.lambda_tv * l_tv + self.lambda_iv * l_iv
        return total, {
            "loss_TI": l_ti.item(),
            "loss_TV": l_tv.item(),
            "loss_IV": l_iv.item(),
        }


class SimCLRLoss(nn.Module):
    """NT-Xent loss for SimCLR."""
    def __init__(self, temperature_init: float = 0.1):
        super().__init__()
        self.temperature = temperature_init

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: (B, D) L2-normalised embeddings
        Returns:
            scalar loss
        """
        B = z1.size(0)
        # Concat views: (2B, D)
        z = torch.cat([z1, z2], dim=0)
        
        # Sim matrix: (2B, 2B)
        sim = torch.matmul(z, z.T) / self.temperature
        
        # Labels for positive pairs: 
        # z1[i] is positive with z2[i], i.e., index i with i+B, and i+B with i
        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels], dim=0)
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, -float("inf"))
        
        loss = F.cross_entropy(sim, labels)
        return loss
