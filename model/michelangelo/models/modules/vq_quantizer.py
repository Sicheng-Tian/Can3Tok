from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


class VQOutput(NamedTuple):
    quantized: torch.Tensor
    indices: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor


class EmbeddingQuantizer(nn.Module):
    def __init__(
        self,
        n_embed: int,
        embed_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embed = torch.randn(n_embed, embed_dim)
        embed = F.normalize(embed, dim=1)
        if decay > 0.0:
            register_buffer = self.register_buffer
        else:
            register_buffer = lambda name, tensor, persistent=True: setattr(
                self, name, nn.Parameter(tensor.requires_grad_(False))
            )

        register_buffer("embedding", embed)
        register_buffer("cluster_size", torch.zeros(n_embed))
        register_buffer("embed_avg", embed.clone())

    def forward(self, z: torch.Tensor) -> VQOutput:
        B, N, C = z.shape
        z_flat = z.view(-1, C)
        z_flat = F.normalize(z_flat, dim=1)

        d = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.t())
        )

        indices = torch.argmin(d, dim=1)

        quantized_flat = F.embedding(indices, self.embedding)
        quantized = quantized_flat.view(B, N, C)

        if self.training and self.decay > 0.0:
            cluster_size = torch.zeros(self.n_embed, device=z.device)
            cluster_size.scatter_add_(
                0, indices, torch.ones_like(indices, dtype=torch.float)
            )
            cluster_size = cluster_size.view(-1)

            embed_sum = torch.zeros_like(self.embedding)
            embed_sum.index_add_(0, indices, z_flat)

            torch.distributed.all_reduce(
                cluster_size, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(embed_sum, op=torch.distributed.ReduceOp.SUM)

            n = cluster_size.sum()
            cluster_size = (
                (cluster_size + self.epsilon) / (n + self.n_embed * self.epsilon) * n
            )
            embed_normalized = embed_sum / cluster_size.unsqueeze(1)

            self.cluster_size.data.mul_(self.decay).add_(
                cluster_size, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(
                embed_normalized, alpha=1 - self.decay
            )

            embed_normalized = self.embed_avg / self.cluster_size.unsqueeze(1)
            embed_normalized = F.normalize(embed_normalized, dim=1)
            self.embedding.data.copy_(embed_normalized)

        commitment_loss = F.mse_loss(quantized.detach(), z)
        codebook_loss = F.mse_loss(quantized, z.detach())

        quantized = z + (quantized - z).detach()

        return VQOutput(
            quantized=quantized,
            indices=indices,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
        )

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        indices_flat = indices.view(-1)
        quantized_flat = F.embedding(indices_flat, self.embedding)
        return quantized_flat
