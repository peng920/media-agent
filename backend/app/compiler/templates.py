import random
from collections.abc import Callable

from pydantic import BaseModel, Field, field_validator


class QwenT2IParams(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1024, ge=256, le=2048, multiple_of=32)
    height: int = Field(default=1024, ge=256, le=2048, multiple_of=32)
    steps: int = Field(default=6, ge=1, le=50)
    seed: int | None = None


# 2026-10-12 真机校准结论：qwen_image_2.1_turbo 需 4096 维文本特征（qwen3vl_8b），
# 盘上 qwen3vl_4b（2560 维）会报 normalized_shape 不匹配。
# 8B 编码器接受任一官方量化变体，按显存占用从省到奢排序。
UNET = "qwen_image_2.1_turbo_int8_convrot.safetensors"
TE_CANDIDATES = [
    "qwen3vl_8b_w4a8.safetensors",
    "qwen3vl_8b_int8_convrot.safetensors",
    "qwen3vl_8b_bf16.safetensors",
]
VAE = "qwen_image_2.1_vae_bf16.safetensors"


def build_qwen_t2i(p: QwenT2IParams, overrides: dict | None = None) -> dict:
    seed = p.seed if p.seed is not None else random.randint(0, 2**48)
    te = (overrides or {}).get("clip_name", TE_CANDIDATES[0])
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "6": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": p.width, "height": p.height, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["5", 0],
                         "latent_image": ["6", 0], "seed": seed, "steps": p.steps,
                         "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                         "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "zimg"}},
    }


class Template:
    """required_models/optional_models：[(node, field, 文件名或候选列表)， ...]
    条目列表形式允许同一节点同一字段出现多次（如双 VAE）。"""

    def __init__(self, id, name, capability, params_model, builder, required_models,
                 resolver=None, optional_models=None):
        self.id, self.name, self.capability = id, name, capability
        self.params_model, self.builder = params_model, builder
        self.required_models = required_models or []
        self.optional_models = optional_models or []
        self._resolver = resolver  # async (engine) -> {"节点入参名": 文件名}

    def build(self, params: BaseModel, overrides: dict | None = None) -> dict:
        return self.builder(params, overrides)

    async def resolve_overrides(self, engine) -> dict:
        return await self._resolver(engine) if self._resolver else {}


async def _resolve_te(engine) -> dict:
    opts = await engine.model_options("CLIPLoader", "clip_name")
    for cand in TE_CANDIDATES:
        if cand in opts:
            return {"clip_name": cand}
    raise RuntimeError(
        "缺少 Qwen-Image 2.1 文本编码器（qwen3vl_8b_w4a8 / int8_convrot / bf16 任一），"
        "请放入 engine 的 text_encoders 目录")


QWEN_T2I = Template(
    id="qwen-image-2.1-turbo",
    name="Qwen-Image 2.1 Turbo 文生图",
    capability="text-to-image",
    params_model=QwenT2IParams,
    builder=build_qwen_t2i,
    required_models=[
        ("UNETLoader", "unet_name", UNET),
        ("CLIPLoader", "clip_name", TE_CANDIDATES),
        ("VAELoader", "vae_name", VAE),
    ],
    resolver=_resolve_te,
)


async def _check_group(engine, entries: list) -> list[str]:
    missing = []
    for node, field, expected in entries:
        cands = expected if isinstance(expected, list) else [expected]
        try:
            opts = await engine.model_options(node, field)
        except Exception as e:
            missing.append(f"{'/'.join(cands)}（无法查询引擎: {e}）")
            continue
        if not any(c in opts for c in cands):
            missing.append(" 或 ".join(cands))
    return missing


async def validate_models(engine, tpl=None) -> dict:
    """返回 {"required": [...], "optional": [...]}；required 为空才允许任务提交。
    tpl=None 时汇总所有模板，否则只查该模板。"""
    targets = [tpl] if tpl is not None else list(TEMPLATES.values())
    required: list[str] = []
    optional: list[str] = []
    for t in targets:
        required += await _check_group(engine, t.required_models)
        optional += await _check_group(engine, t.optional_models)
    return {"required": required, "optional": optional}


# ============ LTX-2.3 图生视频（音视频联合，unsloth GGUF 配方） ============

class LtxI2VParams(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    image: str = Field(min_length=1)  # 引擎 input 目录里的文件名（经 /api/uploads 上传）
    width: int = Field(default=768, ge=256, le=1920, multiple_of=32)
    height: int = Field(default=512, ge=256, le=1920, multiple_of=32)
    length: int = Field(default=25, ge=9, le=121)        # 帧数，8n+1
    steps: int = Field(default=20, ge=1, le=60)
    cfg: float = Field(default=3.0, ge=1.0, le=10.0)     # dev 模型非蒸馏，需 CFG
    fps: float = Field(default=24.0, ge=8.0, le=60.0)
    seed: int | None = None

    @field_validator("length")
    @classmethod
    def _length_8n1(cls, v: int) -> int:
        if (v - 1) % 8 != 0:
            raise ValueError("帧数须为 8n+1（如 25/49/97）")
        return v

LTX_GGUF = "ltx-2.3-22b-dev-Q3_K_S.gguf"
EMB_CONN = "ltx-2.3-22b-dev_embeddings_connectors.safetensors"
LTX_VIDEO_VAE = "ltx-2.3-22b-dev_video_vae.safetensors"
LTX_AUDIO_VAE = "ltx-2.3-22b-dev_audio_vae.safetensors"
GEMMA_CANDIDATES = [
    "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf",
    "gemma-3-12b-it-qat-UD-Q3_K_XL.gguf",
    "gemma-3-12b-it-qat-Q4_K_M.gguf",
    "gemma-3-12b-it-qat-Q5_K_M.gguf",
    "gemma-3-12b-it-qat-Q8_0.gguf",
]
LTX_NEGATIVE = ("blurry, low quality, still frame, frames, watermark, overlay, "
                "titles, has blurbox, has subtitles")


def build_ltx_i2v(p: LtxI2VParams, overrides: dict | None = None) -> dict:
    ov = overrides or {}
    gemma = ov.get("clip_name1", GEMMA_CANDIDATES[0])
    seed = p.seed if p.seed is not None else random.randint(0, 2**48)
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": LTX_GGUF}},
        "2": {"class_type": "DualCLIPLoaderGGUF",
              "inputs": {"clip_name1": gemma, "clip_name2": EMB_CONN, "type": "ltxv"}},
        "3": {"class_type": "VAELoaderKJ",
              "inputs": {"vae_name": LTX_VIDEO_VAE, "device": "main_device", "weight_dtype": "bf16"}},
        "4": {"class_type": "VAELoaderKJ",
              "inputs": {"vae_name": LTX_AUDIO_VAE, "device": "main_device", "weight_dtype": "bf16"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": p.image}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": LTX_NEGATIVE}},
        "8": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["6", 0], "negative": ["7", 0], "frame_rate": p.fps}},
        "9": {"class_type": "LTXVImgToVideo",
              "inputs": {"positive": ["8", 0], "negative": ["8", 1], "vae": ["3", 0],
                         "image": ["5", 0], "width": p.width, "height": p.height,
                         "length": p.length, "batch_size": 1, "strength": 1.0}},
        "10": {"class_type": "EmptyLTXVLatentVideo",
               "inputs": {"width": p.width, "height": p.height, "length": p.length, "batch_size": 1}},
        "11": {"class_type": "LTXVEmptyLatentAudio",
               "inputs": {"frames_number": p.length, "frame_rate": p.fps,
                          "batch_size": 1, "audio_vae": ["4", 0]}},
        "12": {"class_type": "LTXVConcatAVLatent",
               "inputs": {"video_latent": ["10", 0], "audio_latent": ["11", 0]}},
        "13": {"class_type": "LTXVScheduler",
               "inputs": {"steps": p.steps, "max_shift": 2.05, "base_shift": 0.95,
                          "stretch": True, "terminal": 0.1}},
        "14": {"class_type": "CFGGuider",
               "inputs": {"model": ["1", 0], "positive": ["9", 0], "negative": ["9", 1], "cfg": p.cfg}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "16": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "17": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["15", 0], "guider": ["14", 0], "sampler": ["16", 0],
                          "sigmas": ["13", 0], "latent_image": ["12", 0]}},
        "18": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["17", 0]}},
        "19": {"class_type": "VAEDecodeTiled",
               "inputs": {"samples": ["18", 0], "vae": ["3", 0], "tile_size": 512,
                          "overlap": 64, "temporal_size": 4096, "temporal_overlap": 8}},
        "20": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["18", 1], "audio_vae": ["4", 0]}},
        "21": {"class_type": "CreateVideo",
               "inputs": {"images": ["19", 0], "fps": p.fps, "audio": ["20", 0]}},
        "22": {"class_type": "SaveVideo",
               "inputs": {"video": ["21", 0], "filename_prefix": "zimg", "format": "mp4",
                          "codec": "auto"}},
    }


async def _resolve_gemma(engine) -> dict:
    opts = await engine.model_options("DualCLIPLoaderGGUF", "clip_name1")
    for cand in GEMMA_CANDIDATES:
        if cand in opts:
            return {"clip_name1": cand}
    raise RuntimeError("缺少 LTX 文本编码器 gemma-3-12b-it-qat（UD-Q4_K_XL 推荐），"
                       "请放入 engine 的 text_encoders 目录")


LTX_I2V = Template(
    id="ltx-2.3-i2v",
    name="LTX-2.3 图生视频（含音频）",
    capability="image-to-video",
    params_model=LtxI2VParams,
    builder=build_ltx_i2v,
    required_models=[
        ("UnetLoaderGGUF", "unet_name", LTX_GGUF),
        ("DualCLIPLoaderGGUF", "clip_name1", GEMMA_CANDIDATES),
        ("DualCLIPLoaderGGUF", "clip_name2", EMB_CONN),
        ("VAELoaderKJ", "vae_name", LTX_VIDEO_VAE),
        ("VAELoaderKJ", "vae_name", LTX_AUDIO_VAE),
    ],
    resolver=_resolve_gemma,
)

TEMPLATES: dict[str, Template] = {QWEN_T2I.id: QWEN_T2I, LTX_I2V.id: LTX_I2V}


# ============ LTX-Video 2B 0.9.6 distilled 图生视频（低显存档，无音频） ============

class Ltx2bI2VParams(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    image: str = Field(min_length=1)
    width: int = Field(default=768, ge=256, le=1920, multiple_of=32)
    height: int = Field(default=512, ge=256, le=1920, multiple_of=32)
    length: int = Field(default=97, ge=9, le=121)
    steps: int = Field(default=8, ge=1, le=60)         # 蒸馏版 8 步
    cfg: float = Field(default=1.0, ge=1.0, le=10.0)   # 蒸馏版 cfg=1
    fps: float = Field(default=24.0, ge=8.0, le=60.0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: int | None = None

    @field_validator("length")
    @classmethod
    def _length_8n1(cls, v: int) -> int:
        if (v - 1) % 8 != 0:
            raise ValueError("帧数须为 8n+1（如 25/49/97）")
        return v

LTXV_2B = "ltxv-2b-0.9.6-distilled-04-25.safetensors"
LTXV_VAE_CANDS = ["ltxv_vae.safetensors",
                  "diffusion_pytorch_model.safetensors",
                  "ltxv_vae-04-25.safetensors"]
T5_GGUF_CANDS = [
    "t5-v1_1-xxl-encoder-Q4_K_M.gguf",
    "t5-v1_1-xxl-encoder-Q5_K_M.gguf",
    "t5-v1_1-xxl-encoder-Q6_K.gguf",
    "t5-v1_1-xxl-encoder-Q8_0.gguf",
    "t5xxl_fp8_safetensors.gguf",
]
LTX2B_NEGATIVE = ("worst quality, inconsistent motion, blurry, jittery, "
                  "distorted, watermark, text")


def build_ltx_2b_i2v(p: Ltx2bI2VParams, overrides: dict | None = None) -> dict:
    ov = overrides or {}
    te = ov.get("clip_name", T5_GGUF_CANDS[0])
    vae = ov.get("vae_name", LTXV_VAE_CANDS[0])
    te_node = "CLIPLoaderGGUF" if te.endswith(".gguf") else "CLIPLoader"
    seed = p.seed if p.seed is not None else random.randint(0, 2**48)
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": LTXV_2B, "weight_dtype": "default"}},
        "2": {"class_type": te_node, "inputs": {"clip_name": te, "type": "ltxv"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "LoadImage", "inputs": {"image": p.image}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": LTX2B_NEGATIVE}},
        "7": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": p.fps}},
        "8": {"class_type": "LTXVImgToVideo",
              "inputs": {"positive": ["7", 0], "negative": ["7", 1], "vae": ["3", 0],
                         "image": ["4", 0], "width": p.width, "height": p.height,
                         "length": p.length, "batch_size": 1, "strength": p.strength}},
        "9": {"class_type": "EmptyLTXVLatentVideo",
              "inputs": {"width": p.width, "height": p.height, "length": p.length,
                         "batch_size": 1}},
        "10": {"class_type": "LTXVScheduler",
               "inputs": {"steps": p.steps, "max_shift": 2.05, "base_shift": 0.95,
                          "stretch": True, "terminal": 0.1}},
        "11": {"class_type": "CFGGuider",
               "inputs": {"model": ["1", 0], "positive": ["8", 0], "negative": ["8", 1],
                          "cfg": p.cfg}},
        "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "14": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["12", 0], "guider": ["11", 0], "sampler": ["13", 0],
                          "sigmas": ["10", 0], "latent_image": ["9", 0]}},
        "15": {"class_type": "VAEDecodeTiled",
               "inputs": {"samples": ["14", 0], "vae": ["3", 0], "tile_size": 512,
                          "overlap": 64, "temporal_size": 4096, "temporal_overlap": 8}},
        "16": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "fps": p.fps}},
        "17": {"class_type": "SaveVideo",
               "inputs": {"video": ["16", 0], "filename_prefix": "zimg", "format": "mp4",
                          "codec": "auto"}},
    }


T5_SF_CANDS = ["t5xxl_fp8_safetensors.safetensors", "t5xxl_fp16.safetensors",
               "t5xxl_fp16.safetensors.3"]


async def _resolve_2b_te_vae(engine) -> dict:
    ov: dict = {}
    gguf_opts = await engine.model_options("CLIPLoaderGGUF", "clip_name")
    te = next((c for c in T5_GGUF_CANDS if c in gguf_opts), None)
    if te is None:
        sf_opts = await engine.model_options("CLIPLoader", "clip_name")
        te = next((c for c in T5_SF_CANDS if c in sf_opts), None)
    if te is None:
        raise RuntimeError("缺少 T5-XXL 文本编码器（t5-v1_1-xxl-encoder-Q4_K_M.gguf 推荐），"
                           "请放入 engine 的 text_encoders 目录")
    ov["clip_name"] = te
    vae_opts = await engine.model_options("VAELoader", "vae_name")
    vae = next((c for c in LTXV_VAE_CANDS if c in vae_opts), None)
    if vae is None:
        raise RuntimeError("缺少 LTX 2B VAE（ltxv_vae.safetensors），请放入 vae 目录")
    ov["vae_name"] = vae
    return ov


LTX_2B_I2V = Template(
    id="ltx-2b-i2v-distilled",
    name="LTX-Video 2B 蒸馏版 图生视频（低显存）",
    capability="image-to-video",
    params_model=Ltx2bI2VParams,
    builder=build_ltx_2b_i2v,
    required_models=[
        ("UNETLoader", "unet_name", LTXV_2B),
        ("CLIPLoaderGGUF", "clip_name", T5_GGUF_CANDS),
        ("VAELoader", "vae_name", LTXV_VAE_CANDS),
    ],
    resolver=_resolve_2b_te_vae,
)

TEMPLATES[LTX_2B_I2V.id] = LTX_2B_I2V


# ============ Wan 2.1 1.3B 文生视频（480p 轻量档） ============

class WanT2VParams(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    width: int = Field(default=848, ge=256, le=1920, multiple_of=16)
    height: int = Field(default=480, ge=256, le=1920, multiple_of=16)
    length: int = Field(default=81, ge=9, le=161)
    steps: int = Field(default=25, ge=1, le=60)
    cfg: float = Field(default=5.0, ge=1.0, le=15.0)
    fps: float = Field(default=16.0, ge=8.0, le=60.0)
    seed: int | None = None

    @field_validator("length")
    @classmethod
    def _length_4n1(cls, v: int) -> int:
        if (v - 1) % 4 != 0:
            raise ValueError("帧数须为 4n+1（如 25/49/81）")
        return v

WAN_UNET_CANDS = ["wan2.1_t2v_1.3B_fp16.safetensors",
                  "wan2.1_t2v_1.3B_bf16.safetensors"]
WAN_TE_CANDS = ["umt5_xxl_fp8_e4m3fn_scaled.safetensors", "umt5_xxl_fp16.safetensors"]
WAN_VAE_CANDS = ["wan_2.1_vae.safetensors"]
WAN_NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
                "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
                "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def build_wan_t2v(p: WanT2VParams, overrides: dict | None = None) -> dict:
    ov = overrides or {}
    unet = ov.get("unet_name", WAN_UNET_CANDS[0])
    te = ov.get("clip_name", WAN_TE_CANDS[0])
    vae = ov.get("vae_name", WAN_VAE_CANDS[0])
    seed = p.seed if p.seed is not None else random.randint(0, 2**48)
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "wan"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": WAN_NEGATIVE}},
        "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
        "7": {"class_type": "EmptyHunyuanLatentVideo",
              "inputs": {"width": p.width, "height": p.height, "length": p.length,
                         "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["6", 0], "positive": ["4", 0], "negative": ["5", 0],
                         "latent_image": ["7", 0], "seed": seed, "steps": p.steps,
                         "cfg": p.cfg, "sampler_name": "euler", "scheduler": "simple",
                         "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "CreateVideo", "inputs": {"images": ["9", 0], "fps": p.fps}},
        "11": {"class_type": "SaveVideo",
               "inputs": {"video": ["10", 0], "filename_prefix": "zimg", "format": "mp4",
                          "codec": "auto"}},
    }


async def _resolve_wan(engine) -> dict:
    async def pick(node, field, cands):
        opts = await engine.model_options(node, field)
        for c in cands:
            if c in opts:
                return c
        return None
    ov = {}
    ov["unet_name"] = await pick("UNETLoader", "unet_name", WAN_UNET_CANDS)
    ov["clip_name"] = await pick("CLIPLoader", "clip_name", WAN_TE_CANDS)
    ov["vae_name"] = await pick("VAELoader", "vae_name", WAN_VAE_CANDS)
    miss = [k for k, v in ov.items() if v is None]
    if miss:
        raise RuntimeError(f"Wan 2.1 缺少模型文件: {miss}，请按 README 下载放入对应目录")
    return ov


WAN_T2V = Template(
    id="wan-2.1-1.3b-t2v",
    name="Wan 2.1 1.3B 文生视频（480p 轻量）",
    capability="text-to-video",
    params_model=WanT2VParams,
    builder=build_wan_t2v,
    required_models=[
        ("UNETLoader", "unet_name", WAN_UNET_CANDS),
        ("CLIPLoader", "clip_name", WAN_TE_CANDS),
        ("VAELoader", "vae_name", WAN_VAE_CANDS),
    ],
    resolver=_resolve_wan,
)
TEMPLATES[WAN_T2V.id] = WAN_T2V


# ============ FastHunyuan 文生视频（720p 6步蒸馏档） ============

class FastHYT2VParams(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    width: int = Field(default=1280, ge=256, le=1920, multiple_of=16)
    height: int = Field(default=720, ge=256, le=1920, multiple_of=16)
    length: int = Field(default=125, ge=9, le=201)
    steps: int = Field(default=6, ge=1, le=60)        # 官方蒸馏：6 步
    cfg: float = Field(default=7.0, ge=1.0, le=15.0)  # 官方建议 >6
    fps: float = Field(default=24.0, ge=8.0, le=60.0)
    seed: int | None = None

    @field_validator("length")
    @classmethod
    def _length_4n1(cls, v: int) -> int:
        if (v - 1) % 4 != 0:
            raise ValueError("帧数须为 4n+1（如 49/121/125）")
        return v

FASTHY_GGUF_CANDS = ["fast-hunyuan-video-t2v-720p-Q3_K_S.gguf",
                     "fast-hunyuan-video-t2v-720p-Q3_K_M.gguf",
                     "fast-hunyuan-video-t2v-720p-Q4_K_S.gguf"]
FASTHY_TE_CANDS = ["llava_llama3_fp8_scaled.safetensors", "llava_llama3_fp16.safetensors"]
FASTHY_CLIP_L_CANDS = ["clip_l.safetensors", "clip_l_fp16.safetensors"]
FASTHY_VAE_CANDS = ["hunyuan_video_vae_bf16.safetensors", "hunyuan_video_vae_fp32.safetensors"]
FASTHY_NEGATIVE = ("low quality, worst quality, blurry, watermark, text, deformed, "
                   "distorted, flickering")


def build_fasthy_t2v(p: FastHYT2VParams, overrides: dict | None = None) -> dict:
    ov = overrides or {}
    unet = ov.get("unet_name", FASTHY_GGUF_CANDS[0])
    te = ov.get("clip_name2", FASTHY_TE_CANDS[0])
    clip_l = ov.get("clip_name1", FASTHY_CLIP_L_CANDS[0])
    vae = ov.get("vae_name", FASTHY_VAE_CANDS[0])
    seed = p.seed if p.seed is not None else random.randint(0, 2**48)
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}},
        # HunyuanVideo 系文本编码器 = clip_l + llava 双编码器（官方模板接线）
        "2": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": clip_l, "clip_name2": te,
                         "type": "hunyuan_video"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": FASTHY_NEGATIVE}},
        "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 17.0}},
        "7": {"class_type": "EmptyHunyuanLatentVideo",
              "inputs": {"width": p.width, "height": p.height, "length": p.length,
                         "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["6", 0], "positive": ["4", 0], "negative": ["5", 0],
                         "latent_image": ["7", 0], "seed": seed, "steps": p.steps,
                         "cfg": p.cfg, "sampler_name": "euler", "scheduler": "simple",
                         "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "CreateVideo", "inputs": {"images": ["9", 0], "fps": p.fps}},
        "11": {"class_type": "SaveVideo",
               "inputs": {"video": ["10", 0], "filename_prefix": "zimg", "format": "mp4",
                          "codec": "auto"}},
    }


async def _resolve_fasthy(engine) -> dict:
    async def pick(node, field, cands):
        opts = await engine.model_options(node, field)
        for c in cands:
            if c in opts:
                return c
        return None
    ov = {}
    ov["unet_name"] = await pick("UnetLoaderGGUF", "unet_name", FASTHY_GGUF_CANDS)
    ov["clip_name2"] = await pick("DualCLIPLoader", "clip_name2", FASTHY_TE_CANDS)
    ov["clip_name1"] = await pick("DualCLIPLoader", "clip_name1", FASTHY_CLIP_L_CANDS)
    ov["vae_name"] = await pick("VAELoader", "vae_name", FASTHY_VAE_CANDS)
    miss = [k for k, v in ov.items() if v is None]
    if miss:
        raise RuntimeError(f"FastHunyuan 缺少模型文件: {miss}，请按 README 下载放入对应目录")
    return ov


FASTHY_T2V = Template(
    id="fast-hunyuan-t2v-720p",
    name="FastHunyuan 文生视频（720p 6步蒸馏）",
    capability="text-to-video",
    params_model=FastHYT2VParams,
    builder=build_fasthy_t2v,
    required_models=[
        ("UnetLoaderGGUF", "unet_name", FASTHY_GGUF_CANDS),
        ("DualCLIPLoader", "clip_name2", FASTHY_TE_CANDS),
        ("DualCLIPLoader", "clip_name1", FASTHY_CLIP_L_CANDS),
        ("VAELoader", "vae_name", FASTHY_VAE_CANDS),
    ],
    resolver=_resolve_fasthy,
)
TEMPLATES[FASTHY_T2V.id] = FASTHY_T2V


# ============ Z-Image Turbo 文生图（T4/Turing 可用的 bf16 档，替代 Qwen-Image 2.1） ============

class ZImageTurboParams(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1024, ge=256, le=2048, multiple_of=32)
    height: int = Field(default=1024, ge=256, le=2048, multiple_of=32)
    steps: int = Field(default=8, ge=1, le=50)
    seed: int | None = None


ZIMAGE_UNET = "z_image_turbo_bf16.safetensors"
# TE 与 FLUX.2-klein 共用同一份 Qwen3-4B
QWEN3_4B_CANDS = ["qwen_3_4b.safetensors"]
ZIMAGE_VAE = "ae.safetensors"


def build_zimage_turbo(p: ZImageTurboParams, overrides: dict | None = None) -> dict:
    ov = overrides or {}
    te = ov.get("clip_name", QWEN3_4B_CANDS[0])
    seed = p.seed if p.seed is not None else random.randint(0, 2**48)
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": ZIMAGE_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "lumina2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": ZIMAGE_VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": p.width, "height": p.height, "batch_size": 1}},
        "8": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.0}},
        "9": {"class_type": "KSampler",
              "inputs": {"model": ["8", 0], "positive": ["5", 0], "negative": ["6", 0],
                         "latent_image": ["7", 0], "seed": seed, "steps": p.steps,
                         "cfg": 1.0, "sampler_name": "res_multistep", "scheduler": "simple",
                         "denoise": 1.0}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "zimg"}},
    }


ZIMAGE_TURBO = Template(
    id="z-image-turbo",
    name="Z-Image Turbo 文生图（6B 轻量）",
    capability="text-to-image",
    params_model=ZImageTurboParams,
    builder=build_zimage_turbo,
    required_models=[
        ("UNETLoader", "unet_name", ZIMAGE_UNET),
        ("CLIPLoader", "clip_name", QWEN3_4B_CANDS),
        ("VAELoader", "vae_name", ZIMAGE_VAE),
    ],
)
TEMPLATES[ZIMAGE_TURBO.id] = ZIMAGE_TURBO


# ============ FLUX.2 klein 4B 文生视频（蒸馏档 4 步） ============

class Flux2KleinParams(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1024, ge=256, le=2048, multiple_of=32)
    height: int = Field(default=1024, ge=256, le=2048, multiple_of=32)
    steps: int = Field(default=4, ge=1, le=50)   # 蒸馏档 4 步
    seed: int | None = None


KLEIN_UNET_CANDS = ["flux-2-klein-4b.safetensors", "flux-2-klein-4b-fp8.safetensors"]
KLEIN_VAE = "flux2-vae.safetensors"


def build_flux2_klein(p: Flux2KleinParams, overrides: dict | None = None) -> dict:
    ov = overrides or {}
    unet = ov.get("unet_name", KLEIN_UNET_CANDS[0])
    te = ov.get("clip_name", QWEN3_4B_CANDS[0])
    seed = p.seed if p.seed is not None else random.randint(0, 2**48)
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "flux2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": KLEIN_VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "EmptyFlux2LatentImage",
              "inputs": {"width": p.width, "height": p.height, "batch_size": 1}},
        "8": {"class_type": "Flux2Scheduler",
              "inputs": {"steps": p.steps, "width": p.width, "height": p.height}},
        "10": {"class_type": "CFGGuider",
               "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0],
                          "cfg": 1.0}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "13": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["11", 0], "guider": ["10", 0], "sampler": ["12", 0],
                          "sigmas": ["8", 0], "latent_image": ["7", 0]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "zimg"}},
    }


FLUX2_KLEIN = Template(
    id="flux2-klein-4b",
    name="FLUX.2 klein 4B 文生图（蒸馏 4 步）",
    capability="text-to-image",
    params_model=Flux2KleinParams,
    builder=build_flux2_klein,
    required_models=[
        ("UNETLoader", "unet_name", KLEIN_UNET_CANDS),
        ("CLIPLoader", "clip_name", QWEN3_4B_CANDS),
        ("VAELoader", "vae_name", KLEIN_VAE),
    ],
)
TEMPLATES[FLUX2_KLEIN.id] = FLUX2_KLEIN
