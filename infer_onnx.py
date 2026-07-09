# -*- coding: utf-8 -*-
"""
infer_onnx.py — 使用 ONNX Runtime 运行 Any2Full 导出的 ONNX 模型。

用法:
  conda run -n any2full python infer_onnx.py \
    --rgb /path/to/rgb.png \
    --depth /path/to/depth.npy \
    --onnx onnx/Any2Full_vits.onnx \
    --out_dir ./outputs_onnx

说明:
  - 输入 rgb 会被 normalize (ImageNet)
  - 输入 depth 支持 .npy (float, HxW 或 1xHxW) 或 .png (除以 depth_scale)
  - 输入会被 resize 到 ONNX 模型要求的内部工作分辨率 (bicubic / nearest)
  - 输出会被 unresize 回原始尺寸
  - ONNX 模型导出时 resize_to_multiple 已被 no-op，所有 resize 由本脚本负责
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("请先安装 onnxruntime: pip install onnxruntime")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Any2Full ONNX inference")
    parser.add_argument("--rgb", type=str, default=None, help="RGB 图像路径")
    parser.add_argument("--depth", type=str, default=None, help="深度图路径 (.png 或 .npy)")

    parser.add_argument("--rgb_dir", type=str, default=None, help="RGB 目录 (批量模式)")
    parser.add_argument("--depth_dir", type=str, default=None, help="深度目录 (批量模式)")

    parser.add_argument("--onnx", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument("--out_dir", type=str, default="./outputs_onnx")
    parser.add_argument("--depth_scale", type=float, default=100.0, help="PNG 深度的缩放因子")
    parser.add_argument("--providers", type=str, default="CPUExecutionProvider",
                        help="ONNX Runtime providers，逗号分隔")
    return parser.parse_args()


def _load_rgb(path: str):
    rgb_img = Image.open(path).convert("RGB")
    t_rgb = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return t_rgb(rgb_img).unsqueeze(0).numpy()


def _load_depth(path: str, depth_scale: float):
    if path.lower().endswith(".npy"):
        arr = np.load(path).astype(np.float32)
    else:
        arr = np.array(Image.open(path)).astype(np.float32) / depth_scale

    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        arr = arr[0] if arr.shape[0] in (1, 3) else arr[:, :, 0]

    return arr[np.newaxis, np.newaxis, ...]  # (1,1,H,W)


def _resize_to_target(img: np.ndarray, target_h: int, target_w: int):
    """将 (B,C,H,W) 图像 resize 到目标尺寸。
    RGB 用 bicubic，depth 用 nearest（保留稀疏值的精确性）。"""
    _, _, h, w = img.shape
    if h == target_h and w == target_w:
        return img
    is_single_channel = img.shape[1] == 1
    mode = "nearest" if is_single_channel else "bicubic"
    tensor = torch.from_numpy(img)
    resized = F.interpolate(tensor, size=(target_h, target_w), mode=mode,
                           align_corners=False if mode != "nearest" else None)
    return resized.numpy()


def _init_scailing(pred: np.ndarray, sparse_depth: np.ndarray,
                   max_depth: float = 1e3, min_depth: float = 1e-6):
    """
    稀疏深度最小二乘对齐 (复现 Any2Full.init_scailing + disparity_to_depth)。

    pred:         (H,W) 模型原始输出 (ONNX disparity_pre, disparity 空间)
    sparse_depth: (H,W) 输入稀疏深度 (与 pred 同分辨率)
    返回:         (H,W) 对齐后的深度
    """
    sparse_disp = np.where(sparse_depth > 0, 1.0 / (sparse_depth + 1e-8), 0.0)

    idx_valid = np.nonzero(sparse_disp.ravel() > 0.00001)[0]
    if len(idx_valid) == 0:
        return np.where(pred > 0, 1.0 / (np.clip(pred, min=0) + 1e-8), 0.0)

    B = sparse_disp.ravel()[idx_valid]
    A = pred.ravel()[idx_valid]
    A = A + np.array([random.random() for _ in range(len(A))]) * 1e-10
    A_aug = np.column_stack([A, np.ones(len(A))])

    X = np.linalg.pinv(A_aug) @ B

    aligned = pred * X[0] + X[1]
    aligned = np.clip(aligned, 1.0 / max_depth, None)

    depth = np.where(aligned > 0, 1.0 / (aligned + 1e-8), 0.0)
    depth = np.clip(depth, min_depth, max_depth)
    return depth


def _save_outputs(pred: np.ndarray, out_base: Path, depth_range=None):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_base) + ".npy", pred)

    vmin, vmax = depth_range if depth_range else (float(pred.min()), float(pred.max()))
    if vmax > vmin:
        norm = (pred - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(pred)

    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    img = (cmap((norm * 255).astype(np.uint8))[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(img).save(str(out_base) + ".png")


def _collect_pairs(rgb_dir: str, depth_dir: str):
    import glob
    rgb_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        rgb_files.extend(glob.glob(os.path.join(rgb_dir, ext)))
    depth_files = []
    for ext in ("*.png", "*.npy"):
        depth_files.extend(glob.glob(os.path.join(depth_dir, ext)))
    return list(zip(sorted(rgb_files), sorted(depth_files)))


def main():
    args = parse_args()

    providers = [p.strip() for p in args.providers.split(",")]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]

    # 读取 ONNX 模型要求的输入尺寸
    onnx_h = sess.get_inputs()[0].shape[2]
    onnx_w = sess.get_inputs()[0].shape[3]
    print(f"ONNX model loaded: {args.onnx}")
    print(f"  inputs:  {input_names}")
    print(f"  outputs: {output_names}")
    print(f"  expected input size: {onnx_h}x{onnx_w}")

    if args.rgb and args.depth:
        pairs = [(args.rgb, args.depth)]
    elif args.rgb_dir and args.depth_dir:
        pairs = _collect_pairs(args.rgb_dir, args.depth_dir)
    else:
        raise ValueError("请指定 --rgb/--depth (单张) 或 --rgb_dir/--depth_dir (批量)")

    out_dir = Path(args.out_dir)

    for idx, (rgb_path, depth_path) in enumerate(pairs, 1):
        print(f"[{idx}/{len(pairs)}] {rgb_path} | {depth_path}")

        rgb = _load_rgb(rgb_path).astype(np.float32)
        depth = _load_depth(depth_path, args.depth_scale).astype(np.float32)  # (1,1,H,W)

        # 确保 depth 和 rgb 原始尺寸一致
        if rgb.shape[-2:] != depth.shape[-2:]:
            depth = _resize_to_target(depth, rgb.shape[-2], rgb.shape[-1])

        # Resize 到 ONNX 模型要求的内部工作分辨率
        rgb_resized = _resize_to_target(rgb, onnx_h, onnx_w)
        depth_resized = _resize_to_target(depth, onnx_h, onnx_w)

        feeds = {
            input_names[0]: rgb_resized,
            input_names[1]: depth_resized,
        }
        results = sess.run(output_names, feeds)
        depth_pred = results[0][0, 0]  # pred: (onnx_h, onnx_w) — 最终深度

        out_base = out_dir / Path(rgb_path).stem
        _save_outputs(depth_pred, out_base)

    print(f"\nDone. Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
