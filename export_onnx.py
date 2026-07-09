# -*- coding: utf-8 -*-
"""
export_onnx.py — 将 Any2Full .pth.tar 权重导出为 ONNX 格式。

用法:
  conda run -n any2full python export_onnx.py \
    --checkpoint pth/Any2Full_vits.pth.tar \
    --encoder vits \
    --out_path onnx/Any2Full_vits.onnx \
    --height 480 --width 640

说明:
  - 模型内部的 resize_to_multiple / unresize 被 no-op，由推理脚本负责 resize
  - 导出时会根据 --height/--width 自动计算模型内部工作分辨率 (resize_lower_size=518)
  - 输出 pred (B,1,H,W) + disparity_pre (B,1,H,W)，H/W 为内部工作分辨率
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

# 补丁: torch.where(condition, x, y) 在 ONNX 中生成 Where 算子，condition 为 BOOL8。
# NPU int16 量化编译器不支持 BOOL8 + FLOAT16 的组合。
# 将 bool condition 替换为 float 算术: cond_f * input + (1 - cond_f) * other，消除 BOOL8 类型。
# 注意: 使用 cond_f * (input - other) + other 形式避免 ONNX 优化器反转为 Where。
_orig_where = torch.where
def _patched_where(condition, input=None, other=None):
    if input is not None and other is not None and condition.dtype == torch.bool:
        cond_f = condition.float()
        return cond_f * (input - other) + other
    return _orig_where(condition, input, other)
torch.where = _patched_where

_orig_avg_pool2d = F.avg_pool2d
_orig_pad = F.pad

def _patched_adaptive_avg_pool2d(input, output_size):
    # ONNX 不支持 output_size 非输入因子时的 adaptive_avg_pool2d，
    # 用 "pad 到可被整除 → avg_pool2d" 替代
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    _, _, Hin, Win = input.shape
    Hout, Wout = output_size

    # pad 到最近的 output_size 倍数
    Hpad = ((Hin + Hout - 1) // Hout) * Hout
    Wpad = ((Win + Wout - 1) // Wout) * Wout
    if Hpad != Hin or Wpad != Win:
        input = _orig_pad(input, (0, Wpad - Win, 0, Hpad - Hin), mode='constant', value=0.0)
    kH, kW = Hpad // Hout, Wpad // Wout
    return _orig_avg_pool2d(input, kernel_size=(kH, kW), stride=(kH, kW), ceil_mode=False, padding=0, count_include_pad=False)

F.adaptive_avg_pool2d = _patched_adaptive_avg_pool2d

# 在导入模型之前关闭 xFormers，使 MemEffAttention 回退到标准 attention，
# 否则 memory_efficient_attention 在 CPU 下不支持 ONNX 追踪
import model.ours.depth_anything_v2.dinov2_layers.attention as _attn
_attn.XFORMERS_AVAILABLE = False

# 补丁: PyTorch 2.1.0 导出 scaled_dot_product_attention 到 ONNX 有 bug，
# 替换为显式 matmul attention
_orig_attention_forward = _attn.Attention.forward
def _patched_attention_forward(self, x):
    B, N, C = x.shape
    previous_dtype = x.dtype
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    attn = q.float() @ k.float().transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn @ v.float()).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x.to(previous_dtype)
_attn.Attention.forward = _patched_attention_forward
# MemEffAttention 继承自 Attention，回退时也会调用父类 forward，一并替换
_attn.MemEffAttention.forward = _patched_attention_forward

# PromptAttention 继承自 Attention 但有自己的 forward，也调用 scaled_dot_product_attention
# 同样 patch 为显式 matmul attention
import model.ours.depth_anything_v2.dinov2_layers.prompt_attention as _prompt_attn
def _patched_prompt_attention_forward(self, x_ori, prompt=None):
    B, N, C = x_ori.shape
    previous_dtype = x_ori.dtype
    if prompt is None:
        qkv = self.qkv(x_ori).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q.float() @ k.float().transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v.float()).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.to(previous_dtype)
    # prompt is not None 分支
    prompt_value, prompt_mask = prompt[0], prompt[1]
    prompt_qk = self.prompt_depth_qk(x_ori.detach()).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    prompt_q, prompt_k = prompt_qk[0], prompt_qk[1]
    prompt_v = self.prompt_depth_bias(self.prompt_depth_norm(prompt_value)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    prompt_mask = prompt_mask.view(B, 1, 1, N).expand(B, self.num_heads, N, N)
    attn_mask = (prompt_mask == 0).float() * (-1e4)
    attn = (prompt_q.float() @ prompt_k.float().transpose(-2, -1)) * self.scale + attn_mask
    attn = attn.softmax(dim=-1)
    attn = F.dropout(attn, p=self.attn_drop_p, training=self.training)
    prompt_output = (attn @ prompt_v.float()).transpose(1, 2).reshape(B, N, C)
    prompt_output = self.prompt_depth_proj(prompt_output)
    new_prompt = F.dropout(prompt_output, p=self.proj_drop.p, training=self.training)
    return [new_prompt, torch.ones_like(prompt[1])]
_prompt_attn.PromptAttention.forward = _patched_prompt_attention_forward

from model.ours.any2full import Any2Full

# 补丁: sparse_depth_embed.py 中 efficient_nearest_fill 使用了 while not valid_mask.all()
# 这个 while 循环在 ONNX 导出时，如果 dummy depth 全部为正（无空洞），循环不会被 tracing。
# 推理时遇到真实稀疏深度中的空洞，ONNX 模型缺少必要的 dilation 操作，造成噪声。
# 解决方案: 将 while 循环替换为固定次数的 for 循环，保证 ONNX 导出包含完整的 dilation 逻辑。
import model.ours.sparse_depth_embed as _sde

_orig_efficient_nearest_fill = _sde.DepthPatchEmbed.efficient_nearest_fill

def _patched_efficient_nearest_fill(self, depth_heatmap, mask_interp, epsilon=1e-5):
    now_dtype = depth_heatmap.dtype
    valid_mask = (mask_interp >= epsilon)
    kernel = torch.ones(3, 3).to(device=depth_heatmap.device, dtype=now_dtype)

    # max_iter = grid_size，保证即使只有单个有效点也能扩张到整个 patch
    max_iter = mask_interp.shape[-1]
    for _ in range(max_iter):
        expanded_mask = F.conv2d(valid_mask.to(now_dtype),
                                 kernel.unsqueeze(0).unsqueeze(0),
                                 padding=1) > 0
        new_valid = expanded_mask & (~valid_mask)
        denom = F.conv2d(valid_mask.to(now_dtype),
                         kernel.unsqueeze(0).unsqueeze(0), padding=1)
        new_values = F.conv2d(depth_heatmap * valid_mask.to(now_dtype),
                              kernel.unsqueeze(0).unsqueeze(0),
                              padding=1) / (denom + 1e-8)
        depth_heatmap = torch.where(new_valid, new_values, depth_heatmap)
        valid_mask = expanded_mask
    return depth_heatmap

_sde.DepthPatchEmbed.efficient_nearest_fill = _patched_efficient_nearest_fill

# 补丁: F.interpolate(scale_factor=...) 在 ONNX 导出时生成 Shape→Range→Resize，
# Vivante NPU 对 Range 算子支持不好，导致 VerifyGraph 报 invalid connections (cycles)。
# 将 scale_factor 替换为显式 size=，避免生成 Range 算子。
import model.ours.depth_anything_v2.util.blocks as _blocks
import model.ours.depth_anything_v2.dinov2 as _dinov2_mod

# ---- FeatureFusionBlock.forward: scale_factor=2 → size=(H*2, W*2) ----
_orig_ffb_forward = _blocks.FeatureFusionBlock.forward
def _patched_ffb_forward(self, *xs, size=None):
    output = xs[0]
    if len(xs) == 2:
        res = self.resConfUnit1(xs[1])
        output = self.skip_add.add(output, res)
    output = self.resConfUnit2(output)
    if (size is None) and (self.size is None):
        # 原: scale_factor=2 → 改为显式 size=
        h, w = output.shape[2], output.shape[3]
        modifier = {"size": (int(h * 2), int(w * 2))}
    elif size is None:
        modifier = {"size": self.size}
    else:
        modifier = {"size": (int(size[0]), int(size[1]))}
    output = F.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
    output = self.out_conv(output)
    return output
_blocks.FeatureFusionBlock.forward = _patched_ffb_forward

# ---- SparseDepthEmbed.interpolate_pos_encoding: scale_factor=(sx,sy) → size=(int(w0),int(h0)) ----
_orig_sde_interp = _sde.SparseDepthEmbed.interpolate_pos_encoding
def _patched_sde_interp(self, x, w, h):
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return self.pos_embed
    pos_embed = self.pos_embed.float()
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // self.patch_size
    h0 = h // self.patch_size
    w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
    # 原: scale_factor=(sx, sy) → 改为 size=(int(w0), int(h0))
    patch_pos_embed = F.interpolate(
        patch_pos_embed.reshape(1, int(_math.sqrt(N)), int(_math.sqrt(N)), dim).permute(0, 3, 1, 2),
        size=(int(w0), int(h0)),
        mode="bicubic",
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
_sde.SparseDepthEmbed.interpolate_pos_encoding = _patched_sde_interp

# ---- DINOv2.interpolate_pos_encoding: 同上 ----
import math as _math
_orig_dinov2_interp = _dinov2_mod.DinoVisionTransformer.interpolate_pos_encoding
def _patched_dinov2_interp(self, x, w, h):
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return self.pos_embed
    pos_embed = self.pos_embed.float()
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // self.patch_size
    h0 = h // self.patch_size
    w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
    sqrt_N = _math.sqrt(N)
    patch_pos_embed = F.interpolate(
        patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
        size=(int(w0), int(h0)),
        mode="bicubic",
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
_dinov2_mod.DinoVisionTransformer.interpolate_pos_encoding = _patched_dinov2_interp


def compute_internal_size(h: int, w: int, multiple_of: int = 14, resize_lower_size: int = 518):
    """复现 Any2Full.resize_to_multiple 的目标尺寸计算。"""
    scale = max(resize_lower_size / h, resize_lower_size / w)
    new_h = int(((h * scale + multiple_of - 1) // multiple_of) * multiple_of)
    new_w = int(((w * scale + multiple_of - 1) // multiple_of) * multiple_of)
    return new_h, new_w


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Any2Full ONNX export")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth.tar checkpoint")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--out_path", type=str, default="onnx/Any2Full_vits.onnx")
    parser.add_argument("--height", type=int, default=480, help="Export resolution height (must be multiple of 14)")
    parser.add_argument("--width", type=int, default=640, help="Export resolution width (must be multiple of 14)")
    parser.add_argument("--source_height", type=int, default=None, help="Source height for auto-compute (overrides --height/--width)")
    parser.add_argument("--source_width", type=int, default=None, help="Source width for auto-compute (overrides --height/--width)")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="Run onnx-simplifier after export (may fold dynamic shapes)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 计算导出分辨率
    if args.source_height is not None and args.source_width is not None:
        internal_h, internal_w = compute_internal_size(args.source_height, args.source_width)
        print(f"Source size: {args.source_height}x{args.source_width}")
    else:
        internal_h, internal_w = args.height, args.width
        # 验证是 14 的倍数
        assert internal_h % 14 == 0, f"Height {internal_h} must be multiple of 14"
        assert internal_w % 14 == 0, f"Width {internal_w} must be multiple of 14"
    print(f"Export resolution: {internal_h}x{internal_w}")

    # 加载模型 (需要 mock args，forward 中会访问 self.args.init_scailing 等属性)
    # init_scailing=False: 避免导出时遇到 nonzero/pinverse 等 ONNX 不支持的动态操作
    class _Args:
        init_scailing = False
        max_depth = 1e3
        min_depth = 1e-6
        stage = 1
    model = Any2Full(encoder=args.encoder, args=_Args())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    cleaned = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
    model.load_state_dict(cleaned, strict=True)
    model.eval()

    # No-op 掉模型内部的 resize_to_multiple 和 unresize，
    # 让推理脚本负责正确的 resize，避免 ONNX 中 F.interpolate 动态尺寸兼容问题
    def _noop_resize_to_multiple(self, x, multiple_of=14, mode='bilinear', resize_lower_size=518):
        return x, (0, 0)

    def _noop_unresize(self, x, size_diff):
        return x

    Any2Full.resize_to_multiple = _noop_resize_to_multiple
    Any2Full.unresize = _noop_unresize

    # 补丁: get_depth_bias_scale 使用了 prompt_depth[b][mask[b]] 布尔索引，
    # ONNX 导出时会产生 NonZero 算子，pegasus 不支持。
    # 替换为纯 masked arithmetic 的向量化实现。
    def _patched_get_depth_bias_scale(self, prompt_depth):
        B, C, H, W = prompt_depth.shape
        mask_f = (prompt_depth != 0).float()

        count = mask_f.reshape(B, -1).sum(dim=1)  # (B,)
        sum_vals = (prompt_depth * mask_f).reshape(B, -1).sum(dim=1)  # (B,)

        means = sum_vals / (count + 1e-8)

        diff = (prompt_depth - means.view(B, 1, 1, 1)) * mask_f
        var = diff.pow(2).reshape(B, -1).sum(dim=1) / (count + 1e-8)
        stds = torch.sqrt(var + 1e-8)

        # 处理边界: count=0 → mean=0,std=1; count≤1 或 std≈0 → std=1
        means = torch.where(count > 0, means, torch.zeros_like(means))
        stds = torch.where((count > 1) & (stds > 1e-8), stds, torch.ones_like(stds))

        return means, stds

    Any2Full.get_depth_bias_scale = _patched_get_depth_bias_scale

    # 构造 dummy input (内部工作分辨率，已为 14 的倍数)
    # 注意: dummy_depth 必须包含一些零值（空洞），确保 efficient_nearest_fill
    # 的 for 循环被正确 tracing，否则 ONNX 模型缺少 dilation 操作会在真实推理时产生噪声
    dummy_rgb = torch.randn(1, 3, internal_h, internal_w)
    dummy_depth = torch.rand(1, 1, internal_h, internal_w)  # 均匀分布 [0, 1)
    # 在 dummy_depth 中制造一些空洞区域 (设为 0)
    dummy_depth[:, :, :internal_h // 3, :internal_w // 3] = 0.0  # 左上角 1/9 区域为空洞

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        args=(dummy_rgb, dummy_depth),
        f=str(out_path),
        input_names=["rgb", "depth"],
        output_names=["pred"],
        opset_version=args.opset,
        do_constant_folding=True,
    )

    _clean_onnx(str(out_path), output_names=["pred"])

    if args.simplify:
        # onnx-simplifier: 折叠 Shape/Gather/Slice 等路径为常量。
        # 注意: 模型有动态序列长度，simplify 可能错误地将它们折叠为常量，
        # 导致其他分辨率导入时 shape 不匹配。仅在确定输入固定分辨率时使用。
        try:
            import onnx
            import onnxsim
            orig = onnx.load(str(out_path))
            orig_n = len(orig.graph.node)
            simp_model, check = onnxsim.simplify(str(out_path))
            if check:
                onnx.save(simp_model, str(out_path))
                simp_n = len(simp_model.graph.node)
                print(f"  Simplified: {simp_n} nodes ({orig_n - simp_n} removed)")
            else:
                print("  Simplify check failed, keeping original")
        except Exception as e:
            print(f"  Simplify skipped: {e}")

    print(f"ONNX model saved to {out_path}")


def _clean_onnx(model_path: str, output_names: list):
    """清理 ONNX 图: 移除多余输出 + Where false 分支为 Constant → 直接替换为 Constant。

    ONNX 导出时 PyTorch 会为 expand_as() 等操作生成 Where(Equal(const, Mul(COS(1),-1)),
    COS(1), const)。条件恒 False (Equal 对比 [1,-1,-1] 与 Mul(1, -1)=-1 得到 [F,T,T])，
    输出始终等于 false 分支的 constant。NPU 工具链不兼容 BOOL8 类型，直接替换为 Constant。
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(model_path)
    graph = m.graph

    # ---- 1. 移除不在 output_names 中的多余输出 ----
    valid = set(output_names)
    kept = [o for o in graph.output if o.name in valid]
    while len(graph.output) > 0:
        graph.output.pop()
    graph.output.extend(kept)

    # ---- 2. 替换 Where(false_branch=Constant) → Constant ----
    node_map = {n.output[0]: n for n in graph.node}
    new_nodes = []
    removed_count = 0

    for node in graph.node:
        if node.op_type != "Where":
            new_nodes.append(node)
            continue

        # 检查 false 分支 (input[2]) 是否来自 Constant
        false_node = node_map.get(node.input[2])
        if false_node is None or false_node.op_type != "Constant":
            new_nodes.append(node)
            continue

        # 计算 Where 实际输出: const 中 -1 → 1 (PyTorch expand 语义, -1 = keep dim)
        old_tensor_proto = false_node.attribute[0].t
        old_arr = numpy_helper.to_array(old_tensor_proto)
        fixed_arr = old_arr.copy()
        fixed_arr[fixed_arr < 0] = 1
        new_tensor = helper.make_tensor(
            name=node.output[0] + "_const",
            data_type=old_tensor_proto.data_type,
            dims=old_arr.shape,
            vals=fixed_arr.flatten().tolist(),
        )
        new_const = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[node.output[0]],
            value=new_tensor,
        )
        new_nodes.append(new_const)
        removed_count += 1

    if removed_count:
        while len(graph.node) > 0:
            graph.node.pop()
        graph.node.extend(new_nodes)
        print(f"  Replaced {removed_count} Where nodes with Constants")

        # ---- 3. 死代码消除 (基于新 graph) ----
        live = set(o.name for o in graph.output)
        for init in graph.initializer:
            live.add(init.name)
        out_to_node = {}
        for node in graph.node:
            for o in node.output:
                out_to_node[o] = node
        changed = True
        live_ids = set()
        while changed:
            changed = False
            for node in graph.node:
                if id(node) in live_ids:
                    continue
                if any(o in live for o in node.output):
                    live_ids.add(id(node))
                    for inp in node.input:
                        if inp not in live:
                            live.add(inp)
                            changed = True
        keep = [n for n in graph.node if id(n) in live_ids]
        removed = len(graph.node) - len(keep)
        while len(graph.node) > 0:
            graph.node.pop()
        graph.node.extend(keep)
        if removed:
            print(f"  Removed {removed} dead nodes (remaining: {len(keep)})")

    onnx.save(m, model_path)


if __name__ == "__main__":
    main()
