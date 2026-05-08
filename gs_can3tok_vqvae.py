import os
import random
from random import randint

import numpy as np
import torch
import torch.utils.data as Data
import torchvision.transforms as T
from chamferdist import ChamferDistance
from geomloss import SamplesLoss
from scipy.stats import special_ortho_group
from spconv.pytorch.utils import PointToVoxel
from torch import nn
from tqdm import tqdm

from gaussian_renderer import render
from gs_dataset import gs_dataset
from model.michelangelo import *
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
from scene import GaussianModel, Scene
from utils.loss_utils import l1_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_usage = "L1"
random_permute = 0
random_rotation = 1
random_shuffle = 1

resol = 200
data_path = "./output/"

dummy_image_path = "./demo/"

folder_path_each = os.listdir(data_path)
num_epochs = 200000
save_path = "./save/"

bch_size = 200
k_rendering_loss = 1000
enable_rendering_loss = 0
label_gt = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).to(device)
L2 = torch.nn.CrossEntropyLoss()
LBCE = torch.nn.BCELoss()
chamferDist = ChamferDistance()
sinkhorn_eff = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)


class GroupParams:
    pass


def group_extract(param_list, param_value):
    group = GroupParams()
    for idx in range(len(param_list)):
        setattr(group, param_list[idx], param_value[idx])
    return group


model_params_list = [
    "sh_degree",
    "source_path",
    "model_path",
    "images",
    "resolution",
    "white_background",
    "data_device",
    "num_gs_per_scene_end",
    "eval",
]
model_params_value = [0, dummy_image_path, "", "images", -1, False, "cuda", 256, False]
pipeline_params_list = ["convert_SHs_python", "compute_cov3D_python", "debug"]
pipeline_params_value = [False, False, False]
optimization_params_list = [
    "iterations",
    "position_lr_init",
    "position_lr_final",
    "position_lr_delay_mult",
    "position_lr_max_steps",
    "feature_lr",
    "opacity_lr",
    "scaling_lr",
    "rotation_lr",
    "percent_dense",
    "lambda_dssim",
    "densification_interval",
    "opacity_reset_interval",
    "densify_from_iter",
    "densify_until_iter",
    "densify_grad_threshold",
    "random_background",
]
optimization_params_value = [
    35_000,
    0.00016,
    0.0000016,
    0.01,
    30_000,
    0.0025,
    0.05,
    0.005,
    0.001,
    0.01,
    0.2,
    100,
    3000,
    500,
    15_000,
    0.0002,
    False,
]


viewpoint_stack = []
for idx_batch in range(0, 1):
    model_params_value = [
        0,
        dummy_image_path,
        "",
        "images",
        -1,
        False,
        "cuda",
        256,
        False,
    ]
    dataset_for_gs = group_extract(model_params_list, model_params_value)
    gaussians = GaussianModel(dataset_for_gs.sh_degree)
    scene = Scene(dataset_for_gs, gaussians)
    viewpoint_stack.append(scene.getTrainCameras().copy())
    training_setup_for_gs = group_extract(
        optimization_params_list, optimization_params_value
    )
    pipe = group_extract(pipeline_params_list, pipeline_params_value)

background = torch.tensor([0, 0, 0], dtype=torch.float32).to(device)


config_path_vqvae = "./model/configs/aligned_shape_latents/shapevqvae-256.yaml"
model_config_vqvae = get_config_from_file(config_path_vqvae)
if hasattr(model_config_vqvae, "model"):
    model_config_vqvae = model_config_vqvae.model

gs_autoencoder = instantiate_from_config(model_config_vqvae)


ckpt = 0
if torch.cuda.device_count() > 1:
    gs_autoencoder = nn.DataParallel(gs_autoencoder)
else:
    gs_autoencoder = gs_autoencoder
gs_autoencoder.to(device)

vq_commitment_weight = 0.25
vq_codebook_weight = 1.0
recon_weight = 1.0
kl_weight = 0.0

optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=1e-4, betas=[0.9, 0.999])


gs_dataset = gs_dataset(data_path, resol=128, random_permute=True, train=True)
trainDataLoader = Data.DataLoader(
    dataset=gs_dataset, batch_size=bch_size, shuffle=True, num_workers=12
)

gen_vxs_from_pts = PointToVoxel(
    vsize_xyz=[0.2, 0.2, 0.2],
    coors_range_xyz=[-8, -8, -8, 8, 8, 8],
    num_point_features=14,
    max_num_voxels=10000,
    max_num_points_per_voxel=40,
)

voxel_reso = 40
x_y = np.linspace(-8, 8, voxel_reso)
z_res = np.linspace(-8, 8, voxel_reso)
xv, yv, zv = np.meshgrid(x_y, x_y, z_res, indexing="ij")
voxel_centers = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

output_voxel_reso = 40
output_x_y = np.linspace(-8, 8, output_voxel_reso)
output_z_res = np.linspace(-8, 8, output_voxel_reso)
output_xv, output_yv, output_zv = np.meshgrid(
    output_x_y, output_x_y, output_z_res, indexing="ij"
)
output_volume_centers = torch.tensor(
    np.vstack([output_xv.ravel(), output_yv.ravel(), output_zv.ravel()]).T,
    dtype=torch.float32,
)


volume_dims = 40
resolution = 16.0 / volume_dims
origin_offset = torch.tensor(
    np.array([(volume_dims - 1) / 2, (volume_dims - 1) / 2, (volume_dims - 1) / 2])
    * resolution,
    dtype=torch.float32,
).to(device)


for epoch in tqdm(range(num_epochs)):
    for i_batch, UV_gs_batch in enumerate(trainDataLoader):
        UV_gs_batch = UV_gs_batch[0].type(torch.float32).to(device)
        if epoch % 1 == 0 and random_permute == 1:
            UV_gs_batch = UV_gs_batch[:, torch.randperm(UV_gs_batch.size()[1])]
        if epoch % 5 == 0 and epoch > 1 and random_rotation == 1:
            rand_rot_comp = special_ortho_group.rvs(3)
            rand_rot = torch.tensor(
                np.dot(rand_rot_comp, rand_rot_comp.T), dtype=torch.float32
            ).to(UV_gs_batch.device)
            UV_gs_batch[:, :, 4:7] = UV_gs_batch[:, :, 4:7] @ rand_rot
            for bcbc in range(UV_gs_batch.shape[0]):
                shifted_points = UV_gs_batch[bcbc, :, 4:7] + origin_offset
                voxel_indices = torch.floor(shifted_points / resolution)
                voxel_indices = torch.clip(voxel_indices, 0, volume_dims - 1)
                voxel_centers = (voxel_indices - (volume_dims - 1) / 2) * resolution
                UV_gs_batch[bcbc, :, :3] = torch.tensor(
                    voxel_centers, dtype=torch.float32
                )

        loss = 0.0
        loss_render = 0.0
        optimizer.zero_grad()

        shape_embed, UV_gs_recover, quantized_flat, indices, vq_out = gs_autoencoder(
            UV_gs_batch, UV_gs_batch, UV_gs_batch, UV_gs_batch[:, :, :3]
        )

        if enable_rendering_loss == 1:
            if epoch % k_rendering_loss == 0:
                viewpoint_stack_test = []
                random_idx = random.sample(range(0, len(UV_gs_batch)), 2)
                for iik in range(len(random_idx)):
                    idx_batch = random_idx[iik]
                    dummy_image_path = (
                        "/home/qgao/sensei-fs-link/Dataset/scripts/DL3DV-10K-Benchmark/"
                        + folder_path_each[idx_batch]
                        + "/gaussian_splat/"
                    )
                    model_params_value = [
                        0,
                        dummy_image_path,
                        "",
                        "images",
                        -1,
                        False,
                        "cuda",
                        resol,
                        False,
                    ]
                    dataset_for_gs = group_extract(
                        model_params_list, model_params_value
                    )
                    gaussians = GaussianModel(dataset_for_gs.sh_degree)
                    scene = Scene(dataset_for_gs, gaussians)
                    viewpoint_stack_test.append(scene.getTrainCameras().copy())
                    training_setup_for_gs = group_extract(
                        optimization_params_list, optimization_params_value
                    )
                    pipe = group_extract(pipeline_params_list, pipeline_params_value)

                    viewpoint = viewpoint_stack_test[0]
                    recovered_idx = UV_gs_recover[idx_batch]
                    gaussians._xyz = recovered_idx[:, :3]
                    gaussians._features_dc = recovered_idx[:, 3:6][:, None, :]
                    gaussians._features_rest = torch.zeros(
                        [recovered_idx.shape[0], 0, 3]
                    ).to(recovered_idx.device)
                    gaussians._opacity = recovered_idx[:, 6][:, None]
                    gaussians._scaling = recovered_idx[:, 7:10]
                    gaussians._rotation = recovered_idx[:, 10:14]

                    for n_views in range(2):
                        rand_idx = randint(0, len(viewpoint) - 1)
                        view_idx = viewpoint[rand_idx]
                        render_pkg = render(view_idx, gaussians, pipe, background)
                        image = render_pkg["render"]
                        gt_image = view_idx.original_image
                        loss_render += l1_loss(image, gt_image)

        vq_commitment_loss = vq_out.commitment_loss
        vq_codebook_loss = vq_out.codebook_loss
        vq_total_loss = (
            vq_commitment_weight * vq_commitment_loss
            + vq_codebook_weight * vq_codebook_loss
        )

        if loss_usage == "L1":
            recon_loss = (
                torch.norm(
                    UV_gs_recover.reshape(UV_gs_batch.shape[0], -1, 14)
                    - UV_gs_batch[:, :, 4:],
                    p=2,
                )
                / UV_gs_batch.shape[0]
            )
            loss += recon_weight * recon_loss + vq_total_loss + 10 * loss_render

        elif loss_usage == "chamfer":
            recon_loss = torch.mean(
                chamferDist(
                    UV_gs_batch.reshape([bch_size, -1, 14])[:, :, :3],
                    UV_gs_recover.reshape([bch_size, -1, 14])[:, :, :3],
                )
            ) + 0.01 * torch.mean(
                chamferDist(
                    UV_gs_batch.reshape([bch_size, -1, 14])[:, :, 3:],
                    UV_gs_recover.reshape([bch_size, -1, 14])[:, :, 3:],
                )
            )
            loss += recon_loss + vq_total_loss + 10 * loss_render

        elif loss_usage == "sinkhorn":
            sinkhorn_loss_ = sinkhorn_eff(
                UV_gs_batch.contiguous().reshape([bch_size, -1, 14]),
                UV_gs_recover.contiguous().reshape([bch_size, -1, 14]),
            ).mean()
            loss += sinkhorn_loss_ + vq_total_loss + 10 * loss_render

        if epoch % 100 == 0:
            print(
                f"loss={loss.item()},  vq_commitment={vq_commitment_loss.item()}, vq_codebook={vq_codebook_loss.item()}"
            )

        if epoch % 1000 == 0:
            gs_autoencoder.eval()
            recovered_1 = UV_gs_recover[0].reshape(-1, 14)
            gaussians._xyz = recovered_1[:, :3]
            gaussians._features_dc = recovered_1[:, 3:6][:, None, :]
            gaussians._features_rest = torch.zeros([recovered_1.shape[0], 0, 3]).to(
                recovered_idx.device
            )
            gaussians._opacity = recovered_1[:, 6][:, None]
            gaussians._scaling = recovered_1[:, 7:10]
            gaussians._rotation = recovered_1[:, 10:14]
            gaussians.save_ply(save_path + "recovered_vq.ply")

            transform = T.ToPILImage()
            vis_num = 3
            viewpoint = viewpoint_stack[0]

            if epoch >= 10000 and epoch % 10000 == 0:
                gs_autoencoder.train()
                torch.save(vq_out.indices, f"{save_path}vq_indices_{epoch}.pt")
                torch.save(quantized_flat, f"{save_path}vq_quantized_{epoch}.pt")
                torch.save(shape_embed, f"{save_path}vq_shape_embed_{epoch}.pt")
                subpath = f"{int(epoch)}_vqvae.pth"
                torch.save(
                    gs_autoencoder.module.state_dict(), os.path.join(save_path, subpath)
                )
            gs_autoencoder.train()

        loss.backward()
        optimizer.step()


gs_autoencoder.eval()
recovered_1 = UV_gs_recover[0].reshape(-1, 14)
gaussians._xyz = recovered_1[:, :3]
gaussians._features_dc = recovered_1[:, 3:6][:, None, :]
gaussians._features_rest = torch.zeros([recovered_1.shape[0], 0, 3]).to(device)
gaussians._opacity = recovered_1[:, 6][:, None]
gaussians._scaling = recovered_1[:, 7:10]
gaussians._rotation = recovered_1[:, 10:14]
gaussians.save_ply(save_path + "recovered_vq_final.ply")

transform = T.ToPILImage()
vis_num = 3
viewpoint = viewpoint_stack[0]
for i_vis in range(0, vis_num):
    view_i = viewpoint[i_vis]
    render_pkg = render(view_i, gaussians, pipe, background)
    image = render_pkg["render"]
    gt_image = view_i.original_image
    img_recovered = transform(image)
    gt_image = transform(gt_image)
    gt_image.save(f"{save_path}gt_vq_{i_vis}.png")
    img_recovered.save(f"{save_path}reco_vq_{i_vis}.png")

torch.save(vq_out.indices, f"{save_path}vq_indices_final.pt")
torch.save(quantized_flat, f"{save_path}vq_quantized_final.pt")
torch.save(shape_embed, f"{save_path}vq_shape_embed_final.pt")
subpath = f"{int(num_epochs)}_vqvae_final.pth"
torch.save(gs_autoencoder.module.state_dict(), os.path.join(save_path, subpath))
