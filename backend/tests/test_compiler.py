import pytest

from app.compiler.templates import (TEMPLATES, QwenT2IParams, build_qwen_t2i,
                                    validate_models)


def test_build_with_lora():
    g = build_qwen_t2i(QwenT2IParams(prompt="a cat"))
    assert g["7"]["inputs"]["model"] == ["4", 0]
    assert g["4"]["class_type"] == "LoraLoaderModelOnly"
    assert g["9"]["class_type"] == "SaveImage"
    assert g["7"]["inputs"]["cfg"] == 1.0


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
    miss = await validate_models(ok)
    assert miss
    assert any("qwen3vl_8b" in m for m in miss)


async def test_validate_models_all_present():
    from app.compiler.templates import QWEN_T2I
    opts = {(n, f): [v] for n, fs in QWEN_T2I.required_models.items()
            for f, v in fs.items()}
    assert await validate_models(FakeEngine(opts)) == []
