from functools import partial

import pytorch_lightning as pl
import torch
from Michelangelo.michelangelo.utils import instantiate_from_config
from omegaconf import DictConfig
from torch.optim import lr_scheduler

from .inference_utils import extract_geometry
from .tsal_base import (
    Latent2MeshOutput,
    ShapeAsLatentModule,
)


class VQLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0, codebook_weight: float = 1.0):
        super().__init__()
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight

    def forward(self, vq_out, commitment_loss, codebook_loss, **kwargs):
        vq_loss = (
            self.commitment_weight * commitment_loss
            + self.codebook_weight * codebook_loss
        )
        return vq_loss


class AlignedShapeAsLatentVQPLModule(pl.LightningModule):
    def __init__(
        self,
        *,
        shape_module_cfg,
        aligned_module_cfg=None,
        loss_cfg=None,
        optimizer_cfg: DictConfig | None = None,
        ckpt_path: str | None = None,
        ignore_keys: tuple[str] | list[str] = (),
    ):

        super().__init__()

        self.shape_model: ShapeAsLatentModule = instantiate_from_config(
            shape_module_cfg, device=None, dtype=None
        )

        if loss_cfg is not None:
            self.loss = instantiate_from_config(loss_cfg)
        else:
            self.loss = VQLoss(commitment_weight=1.0, codebook_weight=1.0)

        self.optimizer_cfg = optimizer_cfg

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.save_hyperparameters()

    def set_shape_model_only(self):
        pass

    @property
    def latent_shape(self):
        return self.shape_model.latent_shape

    @property
    def zero_rank(self):
        if self._trainer:
            zero_rank = self.trainer.local_rank == 0
        else:
            zero_rank = True
        return zero_rank

    def init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def configure_optimizers(self) -> tuple[list, list]:
        lr = self.learning_rate

        trainable_parameters = list(self.shape_model.parameters())

        if self.optimizer_cfg is None:
            optimizers = [
                torch.optim.AdamW(
                    trainable_parameters, lr=lr, betas=(0.9, 0.99), weight_decay=1e-3
                )
            ]
            schedulers = []
        else:
            optimizer = instantiate_from_config(
                self.optimizer_cfg.optimizer, params=trainable_parameters
            )
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr,
            )
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=scheduler_func.schedule
                ),
                "interval": "step",
                "frequency": 1,
            }
            optimizers = [optimizer]
            schedulers = [scheduler]

        return optimizers, schedulers

    def forward(
        self,
        surface: torch.FloatTensor,
        image: torch.FloatTensor,
        text: torch.FloatTensor,
        volume_queries: torch.FloatTensor,
    ):

        shape_embed, decoded, quantized_flat, indices, vq_out = self.shape_model(
            pc=surface,
            feats=surface,
            volume_queries=volume_queries,
            return_vq_output=True,
        )

        return shape_embed, decoded, quantized_flat, indices, vq_out

    def encode(self, surface: torch.FloatTensor, sample_posterior=True):
        shape_embed, quantized, quantized_flat, indices, vq_out = (
            self.shape_model.encode(pc=surface, feats=surface, return_vq_output=True)
        )
        return shape_embed, quantized, quantized_flat, indices, vq_out

    def decode(self, z_q, bounds=1.1, octree_depth=7, num_chunks=10000):
        latents = self.shape_model.decode(z_q)
        return latents

    def training_step(
        self,
        batch: dict[str, torch.FloatTensor],
        batch_idx: int,
        optimizer_idx: int = 0,
    ) -> torch.FloatTensor:

        surface = batch["surface"]
        image = batch["image"]
        text = batch["text"]
        volume_queries = batch["geo_points"][..., 0:3]

        shape_embed, decoded, quantized_flat, indices, vq_out = self(
            surface, image, text, volume_queries
        )

        commitment_loss = vq_out.commitment_loss
        codebook_loss = vq_out.codebook_loss

        vq_loss = self.loss(vq_out, commitment_loss, codebook_loss)

        self.log(
            "train_vq_commitment_loss",
            commitment_loss.item(),
            prog_bar=True,
            logger=True,
            batch_size=surface.shape[0],
        )
        self.log(
            "train_vq_codebook_loss",
            codebook_loss.item(),
            prog_bar=True,
            logger=True,
            batch_size=surface.shape[0],
        )
        self.log(
            "train_vq_total_loss",
            vq_loss.item(),
            prog_bar=True,
            logger=True,
            batch_size=surface.shape[0],
        )

        return vq_loss

    def validation_step(
        self, batch: dict[str, torch.FloatTensor], batch_idx: int
    ) -> torch.FloatTensor:

        surface = batch["surface"]
        image = batch["image"]
        text = batch["text"]
        volume_queries = batch["geo_points"][..., 0:3]

        shape_embed, decoded, quantized_flat, indices, vq_out = self(
            surface, image, text, volume_queries
        )

        commitment_loss = vq_out.commitment_loss
        codebook_loss = vq_out.codebook_loss

        vq_loss = self.loss(vq_out, commitment_loss, codebook_loss)

        self.log(
            "val_vq_commitment_loss",
            commitment_loss.item(),
            prog_bar=True,
            logger=True,
            batch_size=surface.shape[0],
        )
        self.log(
            "val_vq_codebook_loss",
            codebook_loss.item(),
            prog_bar=True,
            logger=True,
            batch_size=surface.shape[0],
        )
        self.log(
            "val_vq_total_loss",
            vq_loss.item(),
            prog_bar=True,
            logger=True,
            batch_size=surface.shape[0],
        )

        return vq_loss

    def latent2mesh(
        self, latents: torch.FloatTensor, bounds=1.1, octree_depth=7, num_chunks=10000
    ) -> list[Latent2MeshOutput]:
        outputs = []
        geometric_func = partial(self.shape_model.query_geometry, latents=latents)
        device = latents.device
        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=device,
            batch_size=len(latents),
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=not self.zero_rank,
        )

        for i, ((mesh_v, mesh_f), is_surface) in enumerate(zip(mesh_v_f, has_surface)):
            if not is_surface:
                outputs.append(None)
                continue

            out = Latent2MeshOutput()
            out.mesh_v = mesh_v
            out.mesh_f = mesh_f
            outputs.append(out)

        return outputs
