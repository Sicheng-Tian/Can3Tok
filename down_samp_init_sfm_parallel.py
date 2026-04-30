"""
Parallel version of down_samp_init_sfm.py.
Uses ProcessPoolExecutor to run 3DGS training for multiple scenes concurrently.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from scene.colmap_loader import read_points3D_binary

# ---------- 配置 ----------
data_path = "D:/Code/MyWork/DL3DV-10K/"  # 修改为你的数据集路径
reso_Gaussian = 128  # 128*128 = 16384 个初始点
num_workers = min(8, max(1, os.cpu_count() - 1))  # 并行进程数，默认 8
# --------------------------


def storePly(path, xyz, rgb):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    PlyData([vertex_element]).write(path)


def process_single_scene(args_tuple):
    """
    处理单个场景：读取点云 → 下采样 → 保存 PLY → 运行 3DGS 训练。
    每个进程独立运行，互不干扰。
    """
    scene_name, output_root, reso_Gaussian, train_py_dir = args_tuple

    sparse_dir = os.path.join(data_path, scene_name, "gaussian_splat", "sparse", "0")
    ply_path = os.path.join(sparse_dir, "points3D.ply")

    # ---- 1. 读取/转换点云 ----
    if os.path.isfile(ply_path):
        plydata = PlyData.read(ply_path)
    else:
        bin_path = os.path.join(sparse_dir, "points3D.bin")
        if not os.path.isfile(bin_path):
            return scene_name, False, "bin 文件不存在"

        xyz, rgb, _ = read_points3D_binary(bin_path)
        storePly(ply_path, xyz, rgb)
        plydata = PlyData.read(ply_path)

    # ---- 2. 读取点云属性 ----
    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )

    normals = np.stack(
        (
            np.asarray(plydata.elements[0]["nx"]),
            np.asarray(plydata.elements[0]["ny"]),
            np.asarray(plydata.elements[0]["nz"]),
        ),
        axis=1,
    )

    color_rgb = np.stack(
        (
            np.asarray(plydata.elements[0]["red"]),
            np.asarray(plydata.elements[0]["green"]),
            np.asarray(plydata.elements[0]["blue"]),
        ),
        axis=1,
    )

    num_target = reso_Gaussian**2
    if len(xyz) < num_target:
        return scene_name, False, f"点数不足 ({len(xyz)} < {num_target})"

    # ---- 3. 随机下采样 ----
    idx = np.random.randint(0, len(xyz), size=num_target)
    xyz_ds = xyz[idx]
    rgb_ds = color_rgb[idx]
    normals_ds = normals[idx]

    # ---- 4. 保存下采样后的 PLY ----
    output_ply_path = os.path.join(sparse_dir, f"points3D_{reso_Gaussian}.ply")
    l = ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"]
    dtype_full = [(attr, "f4") for attr in l]
    elmts = np.empty(xyz_ds.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz_ds, normals_ds, rgb_ds), axis=1)
    elmts[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elmts, "vertex")
    PlyData([el]).write(output_ply_path)

    # ---- 5. 运行 3DGS 训练 ----
    output_model_dir = os.path.join(output_root, scene_name)
    os.makedirs(output_model_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "train.py",
        "-s",
        os.path.join(data_path, scene_name, "gaussian_splat"),
        "--model_path",
        output_model_dir,
    ]

    import subprocess

    result = subprocess.run(
        cmd,
        cwd=train_py_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        err_msg = result.stderr.decode("utf-8", errors="replace")[-500:]
        return scene_name, False, f"train.py 失败: {err_msg}"
    return scene_name, True, "完成"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default=data_path, help="DL3DV-10K 数据集根目录"
    )
    parser.add_argument(
        "--output_root", type=str, default="./output_gs_init", help="3DGS 输出根目录"
    )
    parser.add_argument(
        "--reso_Gaussian",
        type=int,
        default=reso_Gaussian,
        help="初始下采样分辨率，默认 128 (128*128=16384 个点)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=num_workers,
        help=f"并行进程数，默认 {num_workers}",
    )
    parser.add_argument(
        "--skip_done", action="store_true", default=True, help="跳过已有输出的场景"
    )
    args = parser.parse_args()

    data_path_global = args.data_path.rstrip("/")
    output_root = args.output_root
    reso_G = args.reso_Gaussian
    workers = args.workers

    # 获取所有场景文件夹
    folder_list = os.listdir(data_path_global)
    exclude = {"benchmark-meta.csv", ".cache", ".huggingface"}
    scene_names = [
        f
        for f in folder_list
        if f not in exclude and os.path.isdir(os.path.join(data_path_global, f))
    ]

    # 排除已完成的场景
    if args.skip_done:
        remaining = []
        for name in scene_names:
            model_dir = os.path.join(output_root, name)
            if os.path.isdir(model_dir):
                ckpt = os.path.join(
                    model_dir, "point_cloud", "iteration_30000", "gs_filtered.ply"
                )
                if os.path.isfile(ckpt):
                    continue
            remaining.append(name)
        skipped = len(scene_names) - len(remaining)
        if skipped > 0:
            print(f"跳过 {skipped} 个已完成场景，剩余 {len(remaining)} 个待处理")
        scene_names = remaining

    print(f"共 {len(scene_names)} 个场景，使用 {workers} 个并行进程")
    print(f"数据路径: {data_path_global}")
    print(f"输出路径: {output_root}")

    train_py_dir = os.path.dirname(os.path.abspath(__file__))

    task_args = [(name, output_root, reso_G, train_py_dir) for name in scene_names]

    success_count = 0
    fail_count = 0
    fail_info = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_single_scene, arg): arg[0] for arg in task_args
        }

        with tqdm(total=len(futures), desc="3DGS 训练进度") as pbar:
            for future in as_completed(futures):
                scene_name = futures[future]
                try:
                    name, ok, msg = future.result()
                    if ok:
                        success_count += 1
                        pbar.set_postfix_str(
                            f"成功: {success_count} | 失败: {fail_count}"
                        )
                    else:
                        fail_count += 1
                        fail_info.append((name, msg))
                        pbar.set_postfix_str(
                            f"成功: {success_count} | 失败: {fail_count}"
                        )
                        print(f"\n[失败] {name}: {msg}")
                except Exception as e:
                    fail_count += 1
                    fail_info.append((scene_name, str(e)))
                    pbar.set_postfix_str(f"成功: {success_count} | 失败: {fail_count}")
                    print(f"\n[异常] {scene_name}: {e}")
                pbar.update(1)

    print("\n========== 完成 ==========")
    print(f"成功: {success_count} | 失败: {fail_count}")
    if fail_info:
        print("\n失败详情:")
        for name, msg in fail_info:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
