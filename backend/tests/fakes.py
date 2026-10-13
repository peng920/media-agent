"""测试用假引擎：即时完成、可注入失败/取消行为。"""

FAKE_HISTORY_DONE = {
    "status": {"completed": True, "status_str": "success"},
    "outputs": {"9": {"images": [
        {"filename": "a.png", "subfolder": "", "type": "output"}]}},
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
            ("CLIPLoader", "clip_name"): ["qwen3vl_8b_w4a8.safetensors"],
            ("UNETLoader", "unet_name"): ["qwen_image_2.1_turbo_int8_convrot.safetensors"],
            ("VAELoader", "vae_name"): ["qwen_image_2.1_vae_bf16.safetensors"],
        }.get((node, field), [])


class FailingEngine(FakeEngine):
    async def submit(self, graph, client_id):
        raise RuntimeError("engine down")
