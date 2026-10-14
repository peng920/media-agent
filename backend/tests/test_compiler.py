import pytest

from app.compiler.templates import (TEMPLATES, QwenT2IParams, build_qwen_t2i,
                                    LtxI2VParams, build_ltx_i2v,
                                    Ltx2bI2VParams, build_ltx_2b_i2v,
                                    WanT2VParams, build_wan_t2v,
                                    FastHYT2VParams, build_fasthy_t2v,
                                    ZImageTurboParams, build_zimage_turbo,
                                    Flux2KleinParams, build_flux2_klein,
                                    validate_models)


def test_build_qwen_t2i():
    g = build_qwen_t2i(QwenT2IParams(prompt="a cat"))
    assert g["7"]["inputs"]["model"] == ["1", 0]          # turbo 裸跑，无 LoRA
    assert g["9"]["class_type"] == "SaveImage"
    assert g["7"]["inputs"]["cfg"] == 1.0
    assert g["2"]["inputs"]["clip_name"] == "qwen3vl_8b_w4a8.safetensors"  # 默认取候选首选


def test_build_te_override():
    g = build_qwen_t2i(QwenT2IParams(prompt="a cat"),
                       {"clip_name": "qwen3vl_8b_bf16.safetensors"})
    assert g["2"]["inputs"]["clip_name"] == "qwen3vl_8b_bf16.safetensors"


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


def _full_opts():
    """全部模板的必需条目在场的选项表。"""
    opts: dict = {}
    for tpl in TEMPLATES.values():
        for n, f, v in tpl.required_models:
            opts.setdefault((n, f), [])
            opts[(n, f)] += v if isinstance(v, list) else [v]
    return opts


async def test_validate_models_all_present():
    opts = _full_opts()
    report = await validate_models(FakeEngine(opts))
    assert report["required"] == []
    assert report["optional"] == []                       # 已无可选条目（LoRA 已移除）


async def test_validate_models_accepts_any_te_candidate():
    for te in ["qwen3vl_8b_w4a8.safetensors", "qwen3vl_8b_int8_convrot.safetensors"]:
        opts = _full_opts()
        base = opts[("CLIPLoader", "clip_name")]
        opts[("CLIPLoader", "clip_name")] = [te] + [x for x in base if x != te]
        assert (await validate_models(FakeEngine(opts)))["required"] == []


def test_build_ltx_i2v():
    p = LtxI2VParams(prompt="a cat walking", image="zimg_1_abc.png")
    g = build_ltx_i2v(p)
    assert g["2"]["inputs"]["type"] == "ltxv"
    assert g["2"]["inputs"]["clip_name2"] == "ltx-2.3-22b-dev_embeddings_connectors.safetensors"
    assert g["2"]["inputs"]["clip_name1"] == "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf"
    assert g["5"]["inputs"]["image"] == "zimg_1_abc.png"
    assert g["14"]["inputs"]["positive"] == ["9", 0]   # i2v 条件进 guider
    assert g["17"]["inputs"]["latent_image"] == ["12", 0]
    assert g["22"]["class_type"] == "SaveVideo"
    assert g["21"]["inputs"]["audio"] == ["20", 0]     # 音轨并入视频


def test_build_ltx_gemma_override():
    g = build_ltx_i2v(LtxI2VParams(prompt="x", image="i.png"),
                      {"clip_name1": "gemma-3-12b-it-qat-Q8_0.gguf"})
    assert g["2"]["inputs"]["clip_name1"] == "gemma-3-12b-it-qat-Q8_0.gguf"


def test_build_ltx_2b():
    p = Ltx2bI2VParams(prompt="a bird flying", image="zimg_1_x.png")
    g = build_ltx_2b_i2v(p)
    assert g["2"]["class_type"] == "CLIPLoaderGGUF"      # gguf TE 走 GGUF 加载器
    assert g["2"]["inputs"]["type"] == "ltxv"
    assert g["11"]["inputs"]["cfg"] == 1.0                # 蒸馏档默认
    assert g["17"]["class_type"] == "SaveVideo"
    assert "18" not in g                                  # 无音频路径


def test_build_ltx_2b_safetensors_te_switch():
    g = build_ltx_2b_i2v(Ltx2bI2VParams(prompt="x", image="i.png"),
                         {"clip_name": "t5xxl_fp8_safetensors.safetensors",
                          "vae_name": "diffusion_pytorch_model.safetensors"})
    assert g["2"]["class_type"] == "CLIPLoader"           # safetensors 切回普通加载器
    assert g["3"]["inputs"]["vae_name"] == "diffusion_pytorch_model.safetensors"


def test_ltx_2b_param_validation():
    with pytest.raises(Exception):
        Ltx2bI2VParams(prompt="x", image="i.png", length=24)  # 非 8n+1
    with pytest.raises(Exception):
        Ltx2bI2VParams(prompt="x", image="i.png", strength=1.5)


async def test_resolve_2b_te_fallback_to_safetensors():
    from app.compiler.templates import TEMPLATES
    engine = FakeEngine({("CLIPLoaderGGUF", "clip_name"): [],
                         ("CLIPLoader", "clip_name"): ["t5xxl_fp8_safetensors.safetensors"],
                         ("VAELoader", "vae_name"): ["ltxv_vae.safetensors"]})
    ov = await TEMPLATES["ltx-2b-i2v-distilled"].resolve_overrides(engine)
    assert ov["clip_name"] == "t5xxl_fp8_safetensors.safetensors"


def test_build_wan_t2v():
    g = build_wan_t2v(WanT2VParams(prompt="一只猫在草地上奔跑"))
    assert g["2"]["inputs"]["type"] == "wan"
    assert g["6"]["inputs"]["shift"] == 8.0           # 1.3B t2v 官方 shift
    assert g["7"]["class_type"] == "EmptyHunyuanLatentVideo"
    assert g["8"]["inputs"]["cfg"] == 5.0
    assert g["11"]["class_type"] == "SaveVideo"


def test_build_fasthy_t2v():
    g = build_fasthy_t2v(FastHYT2VParams(prompt="ocean waves at sunset"))
    assert g["1"]["class_type"] == "UnetLoaderGGUF"
    assert g["2"]["class_type"] == "DualCLIPLoader"   # clip_l + llava 双编码器
    assert g["2"]["inputs"]["type"] == "hunyuan_video"
    assert g["2"]["inputs"]["clip_name1"] == "clip_l.safetensors"
    assert g["2"]["inputs"]["clip_name2"] == "llava_llama3_fp8_scaled.safetensors"
    assert g["6"]["inputs"]["shift"] == 17.0          # FastHunyuan 官方 shift
    assert g["8"]["inputs"]["steps"] == 6 and g["8"]["inputs"]["cfg"] == 7.0


def test_build_zimage_turbo():
    g = build_zimage_turbo(ZImageTurboParams(prompt="a cat"))
    assert g["2"]["inputs"]["type"] == "lumina2"      # Qwen3-4B TE 走 lumina2 挂载
    assert g["8"]["inputs"]["shift"] == 3.0           # 官方 AuraFlow shift
    assert g["9"]["inputs"]["cfg"] == 1.0 and g["9"]["inputs"]["steps"] == 8
    assert g["9"]["inputs"]["sampler_name"] == "res_multistep"
    assert g["11"]["class_type"] == "SaveImage"


def test_build_flux2_klein():
    g = build_flux2_klein(Flux2KleinParams(prompt="a cat"))
    assert g["2"]["inputs"]["type"] == "flux2"
    assert g["8"]["inputs"]["steps"] == 4             # Flux2Scheduler 蒸馏 4 步
    assert g["10"]["inputs"]["cfg"] == 1.0
    assert g["13"]["class_type"] == "SamplerCustomAdvanced"
    assert g["15"]["class_type"] == "SaveImage"


def test_t2v_length_validators():
    with pytest.raises(Exception):
        WanT2VParams(prompt="x", length=24)   # 非 4n+1
    with pytest.raises(Exception):
        FastHYT2VParams(prompt="x", length=124)  # 非 4n+1
    assert WanT2VParams(prompt="x", length=81).length == 81


def test_ltx_param_validation():
    with pytest.raises(Exception):
        LtxI2VParams(prompt="x", image="i.png", width=100)   # 非32倍
    with pytest.raises(Exception):
        LtxI2VParams(prompt="x", image="i.png", length=24)   # 非 8n+1
    with pytest.raises(Exception):
        LtxI2VParams(prompt="x")                             # 缺图片


async def test_validate_models_reports_gemma_missing():
    opts = {("UnetLoaderGGUF", "unet_name"): ["ltx-2.3-22b-dev-Q3_K_S.gguf"],
            ("DualCLIPLoaderGGUF", "clip_name2"):
                ["ltx-2.3-22b-dev_embeddings_connectors.safetensors"],
            ("VAELoaderKJ", "vae_name"): ["ltx-2.3-22b-dev_video_vae.safetensors"]}
    report = await validate_models(FakeEngine(opts))
    assert any("gemma" in m for m in report["required"])


async def test_validate_models_scoped_per_template():
    """缺 gemma 不应拖累 qwen 模板的可用性。"""
    from app.compiler.templates import TEMPLATES
    opts = _full_opts()
    del opts[("DualCLIPLoaderGGUF", "clip_name1")]
    qwen_r = await validate_models(FakeEngine(opts), TEMPLATES["qwen-image-2.1-turbo"])
    ltx_r = await validate_models(FakeEngine(opts), TEMPLATES["ltx-2.3-i2v"])
    assert qwen_r["required"] == []
    assert any("gemma" in m for m in ltx_r["required"])


async def test_resolve_te_preference_order():
    from app.compiler.templates import TEMPLATES
    both = FakeEngine({("CLIPLoader", "clip_name"):
                       ["qwen3vl_8b_bf16.safetensors", "qwen3vl_8b_int8_convrot.safetensors"]})
    ov = await TEMPLATES["qwen-image-2.1-turbo"].resolve_overrides(both)
    assert ov == {"clip_name": "qwen3vl_8b_int8_convrot.safetensors"}  # w4a8 缺席则取 int8

    none = FakeEngine({("CLIPLoader", "clip_name"): ["qwen3vl_4b_fp8_scaled.safetensors"]})
    with pytest.raises(RuntimeError, match="文本编码器"):
        await TEMPLATES["qwen-image-2.1-turbo"].resolve_overrides(none)
