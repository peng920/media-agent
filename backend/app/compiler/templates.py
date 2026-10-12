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
UNET = "qwen_image_2.1_turbo_int8_convrot.safetensors"
TE = "qwen3vl_8b_int8_convrot.safetensors"
VAE = "qwen_image_2.1_vae_bf16.safetensors"
LORA = "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors"


def build_qwen_t2i(p: QwenT2IParams) -> dict:
    seed = p.seed if p.seed is not None else random.randint(0, 2**48)
    model_src = ["1", 0]
    g: dict = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE, "type": "qwen_image"}},
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
    def __init__(self, id, name, capability, params_model, builder, required_models):
        self.id, self.name, self.capability = id, name, capability
        self.params_model, self.builder, self.required_models = params_model, builder, required_models

    def build(self, params: BaseModel) -> dict:
        return self.builder(params)


QWEN_T2I = Template(
    id="qwen-image-2.1-turbo",
    name="Qwen-Image 2.1 Turbo 文生图",
    capability="text-to-image",
    params_model=QwenT2IParams,
    builder=build_qwen_t2i,
    required_models={
        "UNETLoader": {"unet_name": UNET},
        "CLIPLoader": {"clip_name": TE},
        "VAELoader": {"vae_name": VAE},
        "LoraLoaderModelOnly": {"lora_name": LORA},
    },
)

TEMPLATES: dict[str, Template] = {QWEN_T2I.id: QWEN_T2I}


async def validate_models(engine) -> list[str]:
    """对照引擎各加载节点的下拉清单，返回缺失的模型文件名列表；空列表=齐备。"""
    missing = []
    for tpl in TEMPLATES.values():
        for node, fields in tpl.required_models.items():
            for field, expected in fields.items():
                try:
                    opts = await engine.model_options(node, field)
                except Exception as e:
                    missing.append(f"{expected}（无法查询引擎: {e}）")
                    continue
                if expected not in opts:
                    missing.append(expected)
    return missing
