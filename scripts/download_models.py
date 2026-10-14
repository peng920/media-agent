# /// script
# requires-python = ">=3.10"
# dependencies = ["modelscope>=1.18", "safetensors", "torch==2.7.1"]
# ///
"""按模板清单从 ModelScope 下载模型文件到 ComfyUI models 目录。

用法：
    uv run scripts/download_models.py --list          # 查看清单与缺什么
    uv run scripts/download_models.py                 # 下载全部缺失项
    uv run scripts/download_models.py --only qwen-t2i # 只下某个组
    uv run scripts/download_models.py --skip ltx-2b   # 跳过某个组

约定：
- 已存在且大小与远端一致的文件自动跳过（可断点续跑）。
- 下载经 modelscope snapshot_download 完成（国内源，无需 HF）。
- LTX-2.3（ltx-2.3-i2v 模板，22B dev 档）按决定不下，其 5 个文件不在清单里；
  若日后要启用，参考 docs/HANDOFF.md §5 与 backend README 补齐。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

# (组, 仓库, 仓库内路径, 目标相对 models 目录的路径 [, 抽取前缀])
# 抽取前缀：从单文件 checkpoint 里抽出该前缀的键另存（用于 LTX 旧版 VAE——
# Lightricks 仓库的 vae/diffusion_pytorch_model.safetensors 已被替换为 LTX 2.x
# 新格式，与 0.9.x DiT 不兼容；旧 VAE 只能从 0.9.5 checkpoint 抽取）
MANIFEST: list[tuple] = [
    # ---- qwen-t2i：Qwen-Image 2.1 Turbo（turbo 裸跑，不用 LoRA）----
    ("qwen-t2i", "Comfy-Org/Qwen-Image-2.1",
     "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
     "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"),
    ("qwen-t2i", "Comfy-Org/Qwen-Image-2.1",
     "text_encoders/qwen3vl_8b_w4a8.safetensors",
     "text_encoders/qwen3vl_8b_w4a8.safetensors"),
    # T4(Turing) 上 w4a8 若不可加载，模板 resolver 会自动改用 int8_convrot，
    # 备一份在场可免去现场重下
    ("qwen-t2i", "Comfy-Org/Qwen-Image-2.1",
     "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
     "text_encoders/qwen3vl_8b_int8_convrot.safetensors"),
    ("qwen-t2i", "Comfy-Org/Qwen-Image-2.1",
     "vae/qwen_image_2.1_vae_bf16.safetensors",
     "vae/qwen_image_2.1_vae_bf16.safetensors"),
    # ---- ltx-2b：LTX-Video 2B 蒸馏版（低显存档，标书"6GB 跑基础视频模型"）----
    ("ltx-2b", "Lightricks/LTX-Video",
     "ltxv-2b-0.9.6-distilled-04-25.safetensors",
     "diffusion_models/ltxv-2b-0.9.6-distilled-04-25.safetensors"),
    ("ltx-2b", "Lightricks/LTX-Video",
     "ltx-video-2b-v0.9.5.safetensors",
     "vae/ltxv_vae.safetensors", "vae."),
    ("ltx-2b", "city96/t5-v1_1-xxl-encoder-gguf",
     "t5-v1_1-xxl-encoder-Q4_K_M.gguf",
     "text_encoders/t5-v1_1-xxl-encoder-Q4_K_M.gguf"),
    # ---- wan-t2v：Wan 2.1 1.3B 文生视频 480p ----
    ("wan-t2v", "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
     "split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
     "diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors"),
    ("wan-t2v", "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
     "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
     "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    ("wan-t2v", "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
     "split_files/vae/wan_2.1_vae.safetensors",
     "vae/wan_2.1_vae.safetensors"),
    # ---- hy15：HunyuanVideo 1.5 480p i2v step 蒸馏（T4 可用档）----
    # 判断依据：720p/cfg_distilled 需 50 步+CFG 双前向，T4 按小时计；
    # step 蒸馏 CFG=1 单前向 + 8~12 步是 8.3B 家族唯一现实档。SR/720p 档不取。
    ("hy15", "Comfy-Org/HunyuanVideo_1.5_repackaged",
     "split_files/diffusion_models/hunyuanvideo1.5_480p_i2v_step_distilled_fp8_scaled.safetensors",
     "diffusion_models/hunyuanvideo1.5_480p_i2v_step_distilled_fp8_scaled.safetensors"),
    ("hy15", "Comfy-Org/HunyuanVideo_1.5_repackaged",
     "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
     "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    ("hy15", "Comfy-Org/HunyuanVideo_1.5_repackaged",
     "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors",
     "text_encoders/byt5_small_glyphxl_fp16.safetensors"),
    ("hy15", "Comfy-Org/HunyuanVideo_1.5_repackaged",
     "split_files/vae/hunyuanvideo15_vae_fp16.safetensors",
     "vae/hunyuanvideo15_vae_fp16.safetensors"),
    ("hy15", "Comfy-Org/HunyuanVideo_1.5_repackaged",
     "split_files/clip_vision/sigclip_vision_patch14_384.safetensors",
     "clip_vision/sigclip_vision_patch14_384.safetensors"),
    # ---- zimage：Z-Image Turbo 文生图（bf16 档，T4/Turing 可用）----
    ("zimage", "Comfy-Org/z_image_turbo",
     "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
     "diffusion_models/z_image_turbo_bf16.safetensors"),
    ("zimage", "Comfy-Org/z_image_turbo",
     "split_files/vae/ae.safetensors",
     "vae/ae.safetensors"),
    # Qwen3-4B TE：z-image 与 flux2-klein 共用
    ("zimage", "Comfy-Org/z_image_turbo",
     "split_files/text_encoders/qwen_3_4b.safetensors",
     "text_encoders/qwen_3_4b.safetensors"),
    # ---- klein：FLUX.2 klein 4B 文生图（蒸馏 4 步）----
    ("klein", "Comfy-Org/flux2-klein-4B",
     "split_files/diffusion_models/flux-2-klein-4b.safetensors",
     "diffusion_models/flux-2-klein-4b.safetensors"),
    ("klein", "Comfy-Org/flux2-klein-4B",
     "split_files/vae/flux2-vae.safetensors",
     "vae/flux2-vae.safetensors"),
    ("klein", "Comfy-Org/flux2-klein-4B",
     "split_files/text_encoders/qwen_3_4b.safetensors",
     "text_encoders/qwen_3_4b.safetensors"),
]

GROUP_LABELS = {
    "qwen-t2i": "Qwen-Image 2.1 Turbo 文生图（含 int8 编码器备份）",
    "ltx-2b": "LTX-Video 2B 蒸馏版 图生视频（低显存档）",
    "wan-t2v": "Wan 2.1 1.3B 文生视频 480p",
    "hy15": "HunyuanVideo 1.5 480p 图生视频（step 蒸馏，T4 可用）",
    "zimage": "Z-Image Turbo 文生图（T4 可用，替代 Qwen 2.1）",
    "klein": "FLUX.2 klein 4B 文生图（蒸馏 4 步）",
}


def remote_size(repo: str, path: str) -> int | None:
    """查 ModelScope 文件接口拿准确大小（仅用于跳过/校验判断）。"""
    root = path.rsplit("/", 1)[0] if "/" in path else ""
    url = (f"https://modelscope.cn/api/v1/models/{repo}/repo/files"
           f"?Revision=master&Root={urllib.request.quote(root)}")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            import json
            files = json.load(r).get("Data", {}).get("Files", []) or []
        for f in files:
            if f.get("Path") == path and f.get("Type") != "TREE":
                return int(f.get("Size", 0))
    except Exception as e:
        print(f"  [warn] 查询远端大小失败 {repo}/{path}: {e}")
    return None


def extract_prefix_tensors(src: Path, target: Path, prefix: str) -> bool:
    """从单文件 checkpoint 抽取 prefix 开头的键，去掉前缀后另存。"""
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    with safe_open(str(src), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(prefix):
                tensors[k[len(prefix):]] = f.get_tensor(k)
    if not tensors:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(target))
    return True


def download_one(idx: int, total: int, repo: str, path: str, target: Path,
                 tmp_root: Path, prefix: str | None = None) -> bool:
    print(f"[{idx}/{total}] {target.name}  <-  {repo}/{path}" +
          (f"  (抽取 '{prefix}*' 键)" if prefix else ""))
    tmp_dir = tmp_root / f"{idx:02d}_{path.split('/')[-1][:24]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ok = False
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        snapshot_download(model_id=repo, local_dir=str(tmp_dir),
                          allow_patterns=[path])
        src = tmp_dir / path
        if not src.is_file():
            print(f"  [fail] 下载后未找到文件: {src}")
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        if prefix:
            if not extract_prefix_tensors(src, target, prefix):
                print(f"  [fail] checkpoint 中没有 '{prefix}*' 键")
                return False
        else:
            shutil.move(str(src), target)
            size = remote_size(repo, path)
            if size and target.stat().st_size != size:
                print(f"  [warn] 大小不符：本地 {target.stat().st_size} vs 远端 {size}，请重跑校验")
                return False
        print(f"  [ok] {target}  ({target.stat().st_size / 1e9:.2f}GB)")
        ok = True
        return True
    except Exception as e:
        print(f"  [fail] {e}（临时目录保留以续传: {tmp_dir}）")
        return False
    finally:
        if ok:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", default="/data/ComfyUI/models",
                    help="ComfyUI models 目录（默认 /data/ComfyUI/models）")
    ap.add_argument("--list", action="store_true", help="只列出清单与缺失状态")
    ap.add_argument("--only", help="只下载指定组（逗号分隔，如 qwen-t2i,fasthy）")
    ap.add_argument("--skip", help="跳过指定组（逗号分隔）")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    entries = MANIFEST
    if args.only:
        keep = {g.strip() for g in args.only.split(",")}
        entries = [e for e in entries if e[0] in keep]
    if args.skip:
        drop = {g.strip() for g in args.skip.split(",")}
        entries = [e for e in entries if e[0] not in drop]

    todo: list[tuple] = []
    print(f"目标目录: {models_dir}\n")
    last_group = None
    for entry in entries:
        group, repo, path, target_rel = entry[0], entry[1], entry[2], entry[3]
        prefix = entry[4] if len(entry) > 4 else None
        if group != last_group:
            print(f"— {group}: {GROUP_LABELS.get(group, '')}")
            last_group = group
        target = models_dir / target_rel
        if target.is_file() and target.stat().st_size > 0:
            # 抽取条目无法对照远端大小（源是 checkpoint），存在即视为就绪；
            # 如需重新抽取请先删除目标文件
            if prefix or (remote_size(repo, path) == target.stat().st_size):
                print(f"    [存在] {target_rel}")
                continue
            print(f"    [大小异常] {target_rel}")
        else:
            print(f"    [缺失] {target_rel}")
        todo.append((group, repo, path, target, prefix))

    if args.list:
        return 0
    if not todo:
        print("\n全部就绪，无需下载。")
        return 0

    total_bytes = 0
    for _, repo, path, _t, _p in todo:
        s = remote_size(repo, path)
        total_bytes += s or 0
    print(f"\n待下载 {len(todo)} 个文件，约 {total_bytes / 1e9:.1f}GB\n")

    tmp_root = models_dir / ".ms_tmp"
    failed = []
    for i, (group, repo, path, target, prefix) in enumerate(todo, 1):
        if not download_one(i, len(todo), repo, path, target, tmp_root, prefix):
            failed.append(f"{repo}/{path} -> {target}")
    # 仅当整个 .ms_tmp 为空时才移除根目录（避免误删并发运行的临时文件）
    try:
        tmp_root.rmdir()
    except OSError:
        pass

    if failed:
        print("\n以下文件失败，可重跑本脚本续传：")
        for f in failed:
            print(f"  - {f}")
        return 1
    print("\n全部下载完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
