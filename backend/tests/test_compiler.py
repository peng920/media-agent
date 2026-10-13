import pytest

from app.compiler.templates import (TEMPLATES, QwenT2IParams, build_qwen_t2i,
                                    validate_models)


def test_build_with_lora():
    g = build_qwen_t2i(QwenT2IParams(prompt="a cat"))
    assert g["7"]["inputs"]["model"] == ["4", 0]
    assert g["4"]["class_type"] == "LoraLoaderModelOnly"
    assert g["9"]["class_type"] == "SaveImage"
    assert g["7"]["inputs"]["cfg"] == 1.0
    assert g["2"]["inputs"]["clip_name"] == "qwen3vl_8b_w4a8.safetensors"  # 默认取候选首选


def test_build_te_override():
    g = build_qwen_t2i(QwenT2IParams(prompt="a cat"),
                       {"clip_name": "qwen3vl_8b_bf16.safetensors"})
    assert g["2"]["inputs"]["clip_name"] == "qwen3vl_8b_bf16.safetensors"


def test_build_without_lora():
    g = build_qwen_t2i(QwenT2IParams(prompt="a cat", use_lora=False))
    assert "4" not in g
    assert g["7"]["inputs"]["model"] == ["1", 0]


def test_seed_random_when_none():
    g1 = build_qwen_t2i(QwenT2IParams(prompt="a cat", seed=None))
    g2 = build_qwen_t2i(QwenT2IParams(prompt="a cat", seed=None))
    assert g1["7"]["inputs"]["seed"] != g2["7"]["inputs"]["seed"]


def test_param_validation():
    with pytest.raises(Exception):
        QwenT2IParams(prompt="x", width=100)  # 非 32 倍数
    with pytest.raises(Exception):
        QwenT2IParams(prompt="")  # 空 prompt


class FakeEngine:
    def __init__(self, opts):
        self.opts = opts

    async def model_options(self, node, field):
        return self.opts[(node, field)]


async def test_validate_models_reports_missing():
    ok = FakeEngine({("UNETLoader", "unet_name"):
                     ["qwen_image_2.1_turbo_int8_convrot.safetensors"]})
    report = await validate_models(ok)
    assert report["required"]
    assert any("qwen3vl_8b" in m for m in report["required"])


async def test_validate_models_lora_is_optional():
    from app.compiler.templates import QWEN_T2I
    opts = {(n, f): ([v] if isinstance(v, list) else [v])
            for n, fs in QWEN_T2I.required_models.items() for f, v in fs.items()}
    opts[("CLIPLoader", "clip_name")] = ["qwen3vl_8b_w4a8.safetensors"]
    report = await validate_models(FakeEngine(opts))  # LoRA 不在场
    assert report["required"] == []
    assert len(report["optional"]) == 1 and "viggle" in report["optional"][0]


async def test_validate_models_accepts_any_te_candidate():
    from app.compiler.templates import QWEN_T2I
    opts = {(n, f): ([v] if isinstance(v, list) else [v])
            for n, fs in QWEN_T2I.required_models.items() for f, v in fs.items()}
    for te in ["qwen3vl_8b_w4a8.safetensors", "qwen3vl_8b_int8_convrot.safetensors"]:
        opts[("CLIPLoader", "clip_name")] = [te]
        assert (await validate_models(FakeEngine(opts)))["required"] == []


async def test_resolve_te_preference_order():
    from app.compiler.templates import TEMPLATES
    both = FakeEngine({("CLIPLoader", "clip_name"):
                       ["qwen3vl_8b_bf16.safetensors", "qwen3vl_8b_int8_convrot.safetensors"]})
    ov = await TEMPLATES["qwen-image-2.1-turbo"].resolve_overrides(both)
    assert ov == {"clip_name": "qwen3vl_8b_int8_convrot.safetensors"}  # w4a8 缺席则取 int8

    none = FakeEngine({("CLIPLoader", "clip_name"): ["qwen3vl_4b_fp8_scaled.safetensors"]})
    with pytest.raises(RuntimeError, match="文本编码器"):
        await TEMPLATES["qwen-image-2.1-turbo"].resolve_overrides(none)
