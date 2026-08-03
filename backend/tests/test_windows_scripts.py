from __future__ import annotations

import codecs
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POWERSHELL_SCRIPTS = sorted(PROJECT_ROOT.glob("*.ps1"))
MACOS_SCRIPTS = sorted(PROJECT_ROOT.glob("*.sh"))


def test_powershell_scripts_are_windows_powershell_compatible() -> None:
    assert POWERSHELL_SCRIPTS, "project root should contain PowerShell scripts"

    for script in POWERSHELL_SCRIPTS:
        payload = script.read_bytes()
        assert payload.startswith(codecs.BOM_UTF8), (
            f"{script.name} must start with an UTF-8 BOM for Windows PowerShell 5.1"
        )
        decoded = payload.decode("utf-8-sig")
        assert "\ufffd" not in decoded, f"{script.name} contains replacement characters"
        assert b"\n" not in payload.replace(b"\r\n", b""), (
            f"{script.name} must use CRLF line endings"
        )


def test_macos_scripts_are_utf8_and_executable() -> None:
    assert MACOS_SCRIPTS, "project root should contain macOS shell scripts"

    for script in MACOS_SCRIPTS:
        payload = script.read_bytes()
        assert payload.startswith(b"#!/usr/bin/env bash\n")
        payload.decode("utf-8")
        assert script.stat().st_mode & 0o111, f"{script.name} must be executable"


def test_windows_pytorch_installer_has_scoped_network_fallback() -> None:
    installer = (PROJECT_ROOT / "install-local-model.ps1").read_text(
        encoding="utf-8-sig"
    )
    assert '$TorchVersion = "2.7.1"' in installer
    assert '"--timeout", "180"' in installer
    assert '"--retries", "10"' in installer
    assert '--trusted-host $PytorchHost' in installer
    assert '$PytorchHost = "download.pytorch.org"' in installer
