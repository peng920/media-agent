# /// script
# requires-python = ">=3.10"
# dependencies = ["modelscope>=1.18"]
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

# (组, 仓库, 仓库内路径, 目标相对 models 目录的路径)
MANIFEST: list[tuple[str, str, str, str]] = [
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
    # VAE 原名 diffusion_pytorch_model.safetensors，落盘改名为模板期望的 ltxv_vae
    ("ltx-2b", "Lightricks/LTX-Video",
     "vae/diffusion_pytorch_model.safetensors",
     "vae/ltxv_vae.safetensors"),
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
    # ---- fasthy：FastHunyuan 720p 6步蒸馏 ----
    ("fasthy", "city96/FastHunyuan-gguf",
     "fast-hunyuan-video-t2v-720p-Q3_K_S.gguf",
     "diffusion_models/fast-hunyuan-video-t2v-720p-Q3_K_S.gguf"),
    ("fasthy", "Comfy-Org/HunyuanVideo_repackaged",
     "split_files/text_encoders/llava_llama3_fp8_scaled.safetensors",
     "text_encoders/llava_llama3_fp8_scaled.safetensors"),
    ("fasthy", "Comfy-Org/HunyuanVideo_repackaged",
     "split_files/vae/hunyuan_video_vae_bf16.safetensors",
     "vae/hunyuan_video_vae_bf16.safetensors"),
]

GROUP_LABELS = {
    "qwen-t2i": "Qwen-Image 2.1 Turbo 文生图（含 int8 编码器备份）",
    "ltx-2b": "LTX-Video 2B 蒸馏版 图生视频（低显存档）",
    "wan-t2v": "Wan 2.1 1.3B 文生视频 480p",
    "fasthy": "FastHunyuan 文生视频 720p 6步",
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


def download_one(idx: int, total: int, repo: str, path: str, target: Path,
                 tmp_root: Path) -> bool:
    print(f"[{idx}/{total}] {target.name}  <-  {repo}/{path}")
    tmp_dir = tmp_root / f"{idx:02d}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        snapshot_download(model_id=repo, local_dir=str(tmp_dir),
                          allow_patterns=[path])
        src = tmp_dir / path
        if not src.is_file():
            print(f"  [fail] 下载后未找到文件: {src}")
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), target)
        size = remote_size(repo, path)
        if size and target.stat().st_size != size:
            print(f"  [warn] 大小不符：本地 {target.stat().st_size} vs 远端 {size}，请重跑校验")
            return False
        print(f"  [ok] {target}  ({target.stat().st_size / 1e9:.2f}GB)")
        return True
    except Exception as e:
        print(f"  [fail] {e}")
        return False
    finally:
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

    todo: list[tuple[str, str, str, Path]] = []
    print(f"目标目录: {models_dir}\n")
    last_group = None
    for group, repo, path, target_rel in entries:
        if group != last_group:
            print(f"— {group}: {GROUP_LABELS.get(group, '')}")
            last_group = group
        target = models_dir / target_rel
        if target.is_file() and target.stat().st_size > 0:
            size = remote_size(repo, path)
            mark = "存在" if (size is None or size == target.stat().st_size) \
                else f"大小异常(本地 {target.stat().st_size / 1e9:.2f}GB)"
            print(f"    [{mark}] {target_rel}")
            if mark == "存在":
                continue
        else:
            print(f"    [缺失] {target_rel}")
        todo.append((group, repo, path, target))

    if args.list:
        return 0
    if not todo:
        print("\n全部就绪，无需下载。")
        return 0

    total_bytes = 0
    for _, repo, path, _t in todo:
        s = remote_size(repo, path)
        total_bytes += s or 0
    print(f"\n待下载 {len(todo)} 个文件，约 {total_bytes / 1e9:.1f}GB\n")

    tmp_root = models_dir / ".ms_tmp"
    failed = []
    for i, (group, repo, path, target) in enumerate(todo, 1):
        if not download_one(i, len(todo), repo, path, target, tmp_root):
            failed.append(f"{repo}/{path} -> {target}")
    shutil.rmtree(tmp_root, ignore_errors=True)

    if failed:
        print("\n以下文件失败，可重跑本脚本续传：")
        for f in failed:
            print(f"  - {f}")
        return 1
    print("\n全部下载完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
