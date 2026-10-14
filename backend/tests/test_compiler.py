import pytest

from app.compiler.templates import (TEMPLATES,
                                    Ltx2bI2VParams, build_ltx_2b_i2v,
                                    WanT2VParams, build_wan_t2v,
                                    Hy15I2VParams, build_hy15_i2v,
                                    ZImageTurboParams, build_zimage_turbo,
                                    Flux2KleinParams, build_flux2_klein,
                                    validate_models)


class FakeEngine:
    def __init__(self, opts):
        self.opts = opts

    async def model_options(self, node, field):
        return self.opts[(node, field)]


def _full_opts():
    """全部模板的必需条目在场的选项表。"""
    opts: dict = {}
    for tpl in TEMPLATES.values():
        for n, f, v in tpl.required_models:
            opts.setdefault((n, f), [])
            opts[(n, f)] += v if isinstance(v, list) else [v]
    return opts


async def test_validate_models_all_present():
    report = await validate_models(FakeEngine(_full_opts()))
    assert report["required"] == []
    assert report["optional"] == []


async def test_validate_models_reports_missing_when_empty():
    report = await validate_models(FakeEngine({}))
    assert report["required"]           # 引擎不可查/无文件时必须拦截提交


async def test_validate_models_scoped_per_template():
    """某模板缺文件不应拖累其他模板的可用性。"""
    opts = _full_opts()
    del opts[("UNETLoader", "unet_name")]
    wan_r = await validate_models(FakeEngine(opts), TEMPLATES["wan-2.1-1.3b-t2v"])
    zi_r = await validate_models(FakeEngine(opts), TEMPLATES["z-image-turbo"])
    assert wan_r["required"]
    assert zi_r["required"]             # z-image 的 DiT 同为 UNETLoader，同缺
    opts2 = _full_opts()
    del opts2[("DualCLIPLoader", "clip_name1")]
    hy_r = await validate_models(FakeEngine(opts2), TEMPLATES["hy15-480p-i2v-step-distilled"])
    kl_r = await validate_models(FakeEngine(opts2), TEMPLATES["flux2-klein-4b"])
    assert any("qwen_2.5_vl" in m for m in hy_r["required"])
    assert kl_r["required"] == []       # klein 用单 CLIPLoader，不受影响


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


def test_t2v_length_validators():
    with pytest.raises(Exception):
        WanT2VParams(prompt="x", length=24)   # 非 4n+1
    assert WanT2VParams(prompt="x", length=81).length == 81


def test_build_hy15_i2v():
    p = Hy15I2VParams(prompt="a bird flying", image="zimg_1_x.png")
    g = build_hy15_i2v(p)
    assert g["2"]["inputs"]["type"] == "hunyuan_video_15"   # qwen2.5vl + byt5 双编码器
    assert g["9"]["inputs"]["start_image"] == ["5", 0]      # 起始图直连
    assert g["9"]["inputs"]["clip_vision_output"] == ["6", 0]
    assert g["12"]["inputs"]["cfg"] == 1.0                  # step 蒸馏：CFG 1
    assert g["15"]["class_type"] == "SamplerCustomAdvanced"
    assert g["18"]["class_type"] == "SaveVideo"


def test_hy15_param_validation():
    with pytest.raises(Exception):
        Hy15I2VParams(prompt="x", image="i.png", length=48)   # 非 4n+1
    with pytest.raises(Exception):
        Hy15I2VParams(prompt="x")                             # 缺图片


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


def test_image_templates_param_validation():
    with pytest.raises(Exception):
        ZImageTurboParams(prompt="x", width=100)   # 非 32 倍数
    with pytest.raises(Exception):
        Flux2KleinParams(prompt="")                 # 空 prompt
