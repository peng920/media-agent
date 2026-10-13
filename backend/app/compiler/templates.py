import random
from collections.abc import Callable

from pydantic import BaseModel, Field


class QwenT2IParams(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1024, ge=256, le=2048, multiple_of=32)
    height: int = Field(default=1024, ge=256, le=2048, multiple_of=32)
    steps: int = Field(default=6, ge=1, le=50)
    seed: int | None = None
    use_lora: bool = True


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
LORA = "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors"


def build_qwen_t2i(p: QwenT2IParams, overrides: dict | None = None) -> dict:
    seed = p.seed if p.seed is not None else random.randint(0, 2**48)
    te = (overrides or {}).get("clip_name", TE_CANDIDATES[0])
    model_src = ["1", 0]
    g: dict = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}},
        "6": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": p.width, "height": p.height, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": None, "positive": ["5", 0], "negative": ["5", 0],
                         "latent_image": ["6", 0], "seed": seed, "steps": p.steps,
                         "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                         "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "zimg"}},
    }
    if p.use_lora:
        g["4"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["1", 0], "lora_name": LORA, "strength_model": 1.0}}
        model_src = ["4", 0]
    g["7"]["inputs"]["model"] = model_src
    return g


class Template:
    def __init__(self, id, name, capability, params_model, builder, required_models,
                 resolver=None, optional_models=None):
        self.id, self.name, self.capability = id, name, capability
        self.params_model, self.builder, self.required_models = params_model, builder, required_models
        self.optional_models = optional_models or {}
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
    required_models={
        "UNETLoader": {"unet_name": UNET},
        "CLIPLoader": {"clip_name": TE_CANDIDATES},
        "VAELoader": {"vae_name": VAE},
    },
    resolver=_resolve_te,
    optional_models={
        # LoRA 缺失时关掉 use_lora 仍可生成，故列为可选
        "LoraLoaderModelOnly": {"lora_name": LORA},
    },
)

TEMPLATES: dict[str, Template] = {QWEN_T2I.id: QWEN_T2I}


async def _check_group(engine, groups: dict) -> list[str]:
    missing = []
    for node, fields in groups.items():
        for field, expected in fields.items():
            cands = expected if isinstance(expected, list) else [expected]
            try:
                opts = await engine.model_options(node, field)
            except Exception as e:
                missing.append(f"{'/'.join(cands)}（无法查询引擎: {e}）")
                continue
            if not any(c in opts for c in cands):
                missing.append(" 或 ".join(cands))
    return missing


async def validate_models(engine) -> dict:
    """返回 {"required": [...], "optional": [...]}；required 为空才允许任务提交。"""
    required: list[str] = []
    optional: list[str] = []
    for tpl in TEMPLATES.values():
        required += await _check_group(engine, tpl.required_models)
        optional += await _check_group(engine, tpl.optional_models)
    return {"required": required, "optional": optional}
