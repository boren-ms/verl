import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[2] / ".github/skills/transfer-to-orange/scripts/refresh_source_sas.py"
spec = importlib.util.spec_from_file_location("refresh_source_sas", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_az_login_uses_browser_flow_without_device_code(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.az_login("microsoft.com", {"AZURE_CONFIG_DIR": "/tmp/test-azure-config"})

    assert calls == [
        (
            [
                "az",
                "login",
                "--tenant",
                "microsoft.com",
                "--allow-no-subscriptions",
                "--output",
                "none",
            ],
            {
                "env": {"AZURE_CONFIG_DIR": "/tmp/test-azure-config", "AZURE_CORE_LOGIN_EXPERIENCE_V2": "off"},
                "check": False,
            },
        )
    ]
    assert "--use-device-code" not in calls[0][0]


def test_az_login_failure_forbids_device_fallback(monkeypatch):
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )

    with pytest.raises(RuntimeError, match="Device-code fallback is disabled"):
        module.az_login(module.GREEN_TENANT, {})
