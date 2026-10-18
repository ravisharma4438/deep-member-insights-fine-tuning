import asyncio

import pytest

openai = pytest.importorskip("openai")

import httpx  # noqa: E402  (installed with openai)

from dmi.eval import predict  # noqa: E402


def _status_error(code: int) -> "openai.APIStatusError":
    response = httpx.Response(code, request=httpx.Request("GET", "http://endpoint/v1/models"))
    return openai.APIStatusError(f"Error code: {code}", response=response, body=None)


class FakeClient:
    """client.models.list() fails with the queued errors, then succeeds."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0
        self.models = self

    async def list(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return []


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setattr(predict, "BACKOFF_SCALE", 0.0)


def test_waits_through_cold_start_503s():
    client = FakeClient([_status_error(503), _status_error(503)])
    asyncio.run(predict.wait_until_ready(client, timeout_s=60))
    assert client.calls == 3


def test_bad_key_fails_fast():
    client = FakeClient([_status_error(401)])
    with pytest.raises(SystemExit, match="API key"):
        asyncio.run(predict.wait_until_ready(client, timeout_s=60))
    assert client.calls == 1


def test_endpoint_without_models_route_is_not_waited_on():
    client = FakeClient([_status_error(404)])
    asyncio.run(predict.wait_until_ready(client, timeout_s=60))
    assert client.calls == 1


def test_gives_up_after_timeout():
    client = FakeClient([_status_error(503)] * 100)
    with pytest.raises(SystemExit, match="not ready"):
        asyncio.run(predict.wait_until_ready(client, timeout_s=0))
