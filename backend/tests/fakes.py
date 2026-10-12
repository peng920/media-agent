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
        return []


class FailingEngine(FakeEngine):
    async def submit(self, graph, client_id):
        raise RuntimeError("engine down")
