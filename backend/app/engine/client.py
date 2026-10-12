import httpx


class EngineError(Exception):
    pass


class EngineClient:
    """ComfyUI HTTP API 封装（attach 模式，引擎为外部进程）。"""

    def __init__(self, base_url: str = "http://127.0.0.1:8188"):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=60)

    async def _get(self, path: str):
        r = await self._http.get(path)
        r.raise_for_status()
        if "json" in r.headers.get("content-type", ""):
            return r.json()
        return r.content

    async def health(self) -> dict:
        return await self._get("/system_stats")

    async def submit(self, graph: dict, client_id: str) -> str:
        r = await self._http.post("/prompt", json={"client_id": client_id, "prompt": graph})
        if r.status_code != 200:
            raise EngineError(f"引擎拒单({r.status_code}): {r.text[:300]}")
        data = r.json()
        if "error" in data:
            raise EngineError(
                f"图校验失败: {data['error']} nodes={list(data.get('node_errors', {}).keys())}"
            )
        return data["prompt_id"]

    async def history(self, prompt_id: str) -> dict:
        return (await self._get(f"/history/{prompt_id}")).get(prompt_id, {})

    async def interrupt(self) -> None:
        await self._http.post("/interrupt")

    async def download(self, filename: str, subfolder: str = "") -> bytes:
        r = await self._http.get(
            "/view", params={"filename": filename, "subfolder": subfolder, "type": "output"}
        )
        r.raise_for_status()
        return r.content

    async def model_options(self, node: str, field: str) -> list[str]:
        info = (await self._get(f"/object_info/{node}"))[node]
        return info["input"]["required"][field][0]
