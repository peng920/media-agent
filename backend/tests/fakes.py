"""测试用假引擎：即时完成、可注入失败/取消行为。"""

FAKE_HISTORY_DONE = {
    "status": {"completed": True, "status_str": "success"},
    "outputs": {"9": {"images": [
        {"filename": "a.png", "subfolder": "", "type": "output"}]}},
}

FAKE_HISTORY_VIDEO = {
    "status": {"completed": True, "status_str": "success"},
    "outputs": {"22": {"videos": [
        {"filename": "a.mp4", "subfolder": "", "type": "output"}]}},
}


class FakeEngine:
    base_url = "http://fake-engine"

    def __init__(self):
        self.submitted = []
        self.interrupted = False

    async def submit(self, graph, client_id):
        self.submitted.append(graph)
        return "pid-1"

    async def history(self, prompt_id):
        return FAKE_HISTORY_DONE

    async def interrupt(self):
        self.interrupted = True

    async def health(self):
        return {"system": {"comfyui_version": "0.39.0-test"}}

    async def model_options(self, node, field):
        # 与 app.compiler.templates 的模型清单对齐；LoRA 故意缺席 → 可选缺失路径
        return {
            ("CLIPLoader", "clip_name"): ["qwen3vl_8b_w4a8.safetensors",
                                          "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                                          "llava_llama3_fp8_scaled.safetensors"],
            ("CLIPLoaderGGUF", "clip_name"): ["t5-v1_1-xxl-encoder-Q4_K_M.gguf"],
            ("UNETLoader", "unet_name"): ["qwen_image_2.1_turbo_int8_convrot.safetensors",
                                          "ltxv-2b-0.9.6-distilled-04-25.safetensors",
                                          "wan2.1_t2v_1.3B_fp16.safetensors"],
            ("VAELoader", "vae_name"): ["qwen_image_2.1_vae_bf16.safetensors",
                                        "ltxv_vae.safetensors",
                                        "wan_2.1_vae.safetensors",
                                        "hunyuan_video_vae_bf16.safetensors"],
            ("UnetLoaderGGUF", "unet_name"): ["fast-hunyuan-video-t2v-720p-Q3_K_S.gguf"],
            ("DualCLIPLoaderGGUF", "clip_name1"): ["gemma-3-12b-it-qat-UD-Q4_K_XL.gguf"],
            ("DualCLIPLoaderGGUF", "clip_name2"):
                ["ltx-2.3-22b-dev_embeddings_connectors.safetensors"],
            ("VAELoaderKJ", "vae_name"): ["ltx-2.3-22b-dev_video_vae.safetensors",
                                          "ltx-2.3-22b-dev_audio_vae.safetensors"],
        }.get((node, field), [])


class FailingEngine(FakeEngine):
    async def submit(self, graph, client_id):
        raise RuntimeError("engine down")
