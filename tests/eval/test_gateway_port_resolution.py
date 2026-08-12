"""Port resolution for an eval gateway sitting behind a tunnel.

Concurrent evals against *remote* sandboxes each need their own tunnel on
their own port. Daemon state and the setup config are both process-wide
singletons, so without an explicit override every run reads the same port and
the second one dies on bind. ``RLLM_GATEWAY_PORT`` is that override.
"""

from __future__ import annotations

import pytest

import rllm.eval.runner as runner_mod
from rllm.eval.runner import ENV_GATEWAY_PORT, resolve_gateway_port

URL = "https://gw.example.test"


@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch):
    monkeypatch.delenv(ENV_GATEWAY_PORT, raising=False)


def _daemon(monkeypatch, *, url: str, port: int) -> None:
    import rllm.gateway.tunnel as tunnel_mod

    monkeypatch.setattr(tunnel_mod, "live_tunnel", lambda: {"backend": "ngrok", "url": url, "pid": 1, "upstream": f"http://127.0.0.1:{port}"})


def _config_port(monkeypatch, port: int) -> None:
    import rllm.eval.config as config_mod

    monkeypatch.setattr(config_mod, "load_tunnel_config", lambda: {"port": port})


def test_env_override_wins_over_daemon_state(monkeypatch):
    """The whole point: two runs sharing one daemon still get distinct ports."""
    _daemon(monkeypatch, url=URL, port=9091)
    monkeypatch.setenv(ENV_GATEWAY_PORT, "9092")

    assert resolve_gateway_port(URL) == 9092


def test_env_override_wins_over_config(monkeypatch):
    _daemon(monkeypatch, url="https://other.example.test", port=9091)
    _config_port(monkeypatch, 9091)
    monkeypatch.setenv(ENV_GATEWAY_PORT, "9093")

    assert resolve_gateway_port(URL) == 9093


def test_daemon_upstream_used_when_url_matches(monkeypatch):
    _daemon(monkeypatch, url=URL, port=9091)
    _config_port(monkeypatch, 4321)

    assert resolve_gateway_port(URL) == 9091


def test_daemon_ignored_when_url_differs(monkeypatch):
    """A daemon forwarding somewhere else says nothing about this tunnel."""
    _daemon(monkeypatch, url="https://stale.example.test", port=9091)
    _config_port(monkeypatch, 4321)

    assert resolve_gateway_port(URL) == 4321


def test_falls_back_to_config_and_warns(monkeypatch, caplog):
    import rllm.gateway.tunnel as tunnel_mod

    monkeypatch.setattr(tunnel_mod, "live_tunnel", lambda: None)
    _config_port(monkeypatch, 4321)

    with caplog.at_level("WARNING", logger=runner_mod.__name__):
        assert resolve_gateway_port(URL) == 4321

    assert ENV_GATEWAY_PORT in caplog.text


def test_malformed_override_fails_loudly(monkeypatch):
    """A typo in an ops knob must not silently fall through to a shared port."""
    _daemon(monkeypatch, url=URL, port=9091)
    monkeypatch.setenv(ENV_GATEWAY_PORT, "not-a-port")

    with pytest.raises(ValueError):
        resolve_gateway_port(URL)
