import math

import numpy as np
import torch
from einops import repeat
from Michelangelo.michelangelo.models.modules import checkpoint
from Michelangelo.michelangelo.models.modules.embedder import FourierEmbedder
from Michelangelo.michelangelo.models.modules.transformer_blocks import (
    ResidualCrossAttentionBlock,
    Transformer,
)
from Michelangelo.michelangelo.models.modules.vq_quantizer import (
    EmbeddingQuantizer,
)
from torch import nn

from .tsal_base import ShapeAsLatentModule


class CrossAttentionEncoder(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device | None,
        dtype: torch.dtype | None,
        num_latents: int,
        fourier_embedder: FourierEmbedder,
        fourier_embedder_ID: FourierEmbedder,
        point_feats: int,
        width: int,
        heads: int,
        layers: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        flash: bool = False,
        use_ln_post: bool = False,
        use_checkpoint: bool = False,
    ):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents

        voxel_reso = 4
        x_y = np.linspace(-8, 8, voxel_reso)
        z_res = np.linspace(-8, 8, voxel_reso)
        xv, yv, zv = np.meshgrid(x_y, x_y, z_res, indexing="ij")
        voxel_centers = torch.tensor(
            np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T,
            device=device,
            dtype=dtype,
        ).reshape([-1, 3])

        dummy_tensor2 = (
            torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02
        )
        dummy_tensor2[:, :192] = voxel_centers.reshape([-1]) * 0.01
        self.query = nn.Parameter(dummy_tensor2)

        self.point_feats = point_feats
        self.fourier_embedder = fourier_embedder
        self.fourier_embedder_ID = fourier_embedder_ID

        self.input_proj = nn.Linear(
            self.fourier_embedder.out_dim + point_feats,
            width,
            device=device,
            dtype=dtype,
        )

        self.cross_attn = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )

        self.self_attn = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=False,
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width, dtype=dtype, device=device)
        else:
            self.ln_post = None

    def _forward(self, pc, feats):
        bs = pc.shape[0]
        feats = feats[:, :, 7:]

        data = self.fourier_embedder(pc[:, :, 4:7])
        if feats is not None:
            data = torch.cat([data, feats], dim=-1).to(dtype=torch.float32)

        data = self.input_proj(data)
        query = repeat(self.query, "m c -> b m c", b=bs)

        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc

    def forward(self, pc: torch.FloatTensor, feats: torch.FloatTensor | None = None):
        return checkpoint(
            self._forward, (pc, feats), self.parameters(), self.use_checkpoint
        )


class CrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device | None,
        dtype: torch.dtype | None,
        num_latents: int,
        out_channels: int,
        fourier_embedder: FourierEmbedder,
        width: int,
        heads: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        flash: bool = False,
        use_checkpoint: bool = False,
    ):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.fourier_embedder = fourier_embedder

        self.query_proj = nn.Linear(
            self.fourier_embedder.out_dim, width, device=device, dtype=dtype
        )

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )

        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, out_channels, device=device, dtype=dtype)

    def _forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.fourier_embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        return checkpoint(
            self._forward, (queries, latents), self.parameters(), self.use_checkpoint
        )


class GS_decoder(nn.Module):
    def __init__(self, D=3, W=1024, input_ch=4, skip=[4], output_ch=56):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skip
        self.output_ch = output_ch
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)])
        for i in range(D - 1):
            self.pts_linears.append(nn.Linear(W, W))
            self.pts_linears.append(nn.LayerNorm(W))
            self.pts_linears.append(nn.ReLU())

        self.output_linear = nn.Linear(in_features=W, out_features=output_ch)

    def forward(self, x):
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
        x = self.output_linear(x)
        return x


class ShapeAsLatentVQPerceiver(ShapeAsLatentModule):
    def __init__(
        self,
        *,
        device: torch.device | None,
        dtype: torch.dtype | None,
        num_latents: int,
        point_feats: int = 0,
        codebook_size: int = 8192,
        codebook_dim: int = 1024,
        num_freqs: int = 8,
        include_pi: bool = True,
        width: int,
        heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        flash: bool = True,
        use_ln_post: bool = False,
        use_checkpoint: bool = False,
        commitment_cost: float = 0.25,
        vq_decay: float = 0.99,
        gs_decoder_output_dim: int = 40000 * 14,
    ):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.fourier_embedder = FourierEmbedder(
            num_freqs=num_freqs, include_pi=include_pi, input_dim=3
        )
        self.fourier_embedder_ID = FourierEmbedder(
            num_freqs=num_freqs, include_pi=include_pi, input_dim=3
        )

        init_scale = init_scale * math.sqrt(1.0 / width)

        self.encoder = CrossAttentionEncoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            fourier_embedder_ID=self.fourier_embedder_ID,
            num_latents=num_latents,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint,
        )

        self.pre_vq = nn.Linear(width, codebook_dim, device=device, dtype=dtype)
        self.quantizer = EmbeddingQuantizer(
            n_embed=codebook_size,
            embed_dim=codebook_dim,
            commitment_cost=commitment_cost,
            decay=vq_decay,
        )
        self.post_vq = nn.Linear(codebook_dim, width, device=device, dtype=dtype)
        self.latent_shape = (num_latents, codebook_dim)

        self.transformer = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint,
        )

        gs_decoder_input_dim = num_latents * codebook_dim
        self.GS_decoder = GS_decoder(
            3, 1024, gs_decoder_input_dim, [4], gs_decoder_output_dim
        )

        self.geo_decoder = CrossAttentionDecoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint,
        )

    def encode(
        self,
        pc: torch.FloatTensor,
        feats: torch.FloatTensor | None = None,
        return_vq_output: bool = True,
    ):
        latents, center_pos = self.encoder(pc, feats)

        latents_flat = latents.view(latents.shape[0], -1, latents.shape[-1])
        z_pre = self.pre_vq(latents_flat)

        vq_out = None
        quantized = None
        if return_vq_output:
            vq_out = self.quantizer(z_pre)
            quantized = vq_out.quantized
        else:
            quantized = z_pre

        return latents, quantized, center_pos, vq_out

    def decode(self, latents: torch.FloatTensor, volume_queries=None):
        latents = self.post_vq(latents)
        latents = self.transformer(latents)
        return self.GS_decoder(latents.reshape(latents.shape[0], -1))

    def query_geometry(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        logits = self.geo_decoder(queries, latents).squeeze(-1)
        return logits

    def forward(
        self,
        pc: torch.FloatTensor,
        feats: torch.FloatTensor,
        volume_queries: torch.FloatTensor,
        return_vq_output: bool = True,
    ):
        _, quantized, _, vq_out = self.encode(
            pc, feats, return_vq_output=return_vq_output
        )
        decoded = self.decode(quantized)
        logits = self.query_geometry(volume_queries, quantized)
        return decoded, quantized, vq_out


class AlignedShapeLatentVQPerceiver(ShapeAsLatentVQPerceiver):
    def __init__(
        self,
        *,
        device: torch.device | None,
        dtype: torch.dtype | None,
        num_latents: int,
        point_feats: int = 0,
        codebook_size: int = 8192,
        codebook_dim: int = 1024,
        num_freqs: int = 8,
        include_pi: bool = True,
        width: int,
        heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        flash: bool = True,
        use_ln_post: bool = False,
        use_checkpoint: bool = False,
        commitment_cost: float = 0.25,
        vq_decay: float = 0.99,
        gs_decoder_output_dim: int = 40000 * 14,
    ):

        super().__init__(
            device=device,
            dtype=dtype,
            num_latents=1 + num_latents,
            point_feats=point_feats,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            num_freqs=num_freqs,
            include_pi=include_pi,
            width=width,
            heads=heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint,
            commitment_cost=commitment_cost,
            vq_decay=vq_decay,
            gs_decoder_output_dim=gs_decoder_output_dim,
        )

        self.width = width

    def encode(
        self,
        pc: torch.FloatTensor,
        feats: torch.FloatTensor | None = None,
        return_vq_output: bool = True,
    ):
        shape_embed, latents = self.encode_latents(pc, feats)

        z_pre = self.pre_vq(latents)

        vq_out = None
        quantized = None
        if return_vq_output:
            vq_out = self.quantizer(z_pre)
            quantized = vq_out.quantized
        else:
            quantized = z_pre

        quantized_flat = quantized.reshape(quantized.shape[0], -1)
        indices = None
        if vq_out is not None:
            indices = vq_out.indices

        return shape_embed, quantized, quantized_flat, indices, vq_out

    def encode_latents(
        self, pc: torch.FloatTensor, feats: torch.FloatTensor | None = None
    ):
        x, _ = self.encoder(pc, feats)
        shape_embed = x[:, 0]
        latents = x[:, 1:]
        return shape_embed, latents

    def forward(
        self,
        pc: torch.FloatTensor,
        feats: torch.FloatTensor,
        volume_queries: torch.FloatTensor,
        return_vq_output: bool = True,
    ):
        shape_embed, quantized, quantized_flat, indices, vq_out = self.encode(
            pc, feats, return_vq_output=return_vq_output
        )

        quantized_for_decode = self.post_vq(quantized)
        decoded = self.decode(quantized_for_decode)
        logits = self.query_geometry(volume_queries, quantized_for_decode)

        return shape_embed, decoded, quantized_flat, indices, vq_out
