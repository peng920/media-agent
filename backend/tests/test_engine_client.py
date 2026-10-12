import httpx
import pytest

from app.engine.client import EngineClient, EngineError


def _client(handler) -> EngineClient:
    ec = EngineClient("http://test")
    ec._http = httpx.AsyncClient(base_url="http://test",
                                 transport=httpx.MockTransport(handler))
    return ec


async def test_submit_ok():
    async def h(request):
        return httpx.Response(200, json={"prompt_id": "pid-1"})
    ec = _client(h)
    assert await ec.submit({"1": {}}, "c1") == "pid-1"


async def test_submit_validation_error():
    async def h(request):
        return httpx.Response(400, json={"error": "bad", "node_errors": {"7": "x"}})
    ec = _client(h)
    with pytest.raises(EngineError):
        await ec.submit({}, "c1")


async def test_history_empty_for_unknown():
    async def h(request):
        return httpx.Response(200, json={})
    ec = _client(h)
    assert await ec.history("nope") == {}


async def test_model_options():
    async def h(request):
        return httpx.Response(200, json={"VAELoader": {"input": {"required": {
            "vae_name": [["a.safetensors", "b.safetensors"]]}}}})
    ec = _client(h)
    assert await ec.model_options("VAELoader", "vae_name") == ["a.safetensors", "b.safetensors"]
