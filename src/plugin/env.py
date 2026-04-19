# vim: set expandtab shiftwidth=4 softtabstop=4:
"""Conda env discovery + CUDA / cupy / cucim probe.

Pure data layer — no Qt widgets. QSettings used only for persisting the
selected python path across sessions.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from Qt.QtCore import QSettings

SETTINGS_ORG = "ChimeraX"
SETTINGS_APP = "ArtiaX-Plugin"
KEY_PYTHON = "plugin/env_python"
KEY_LAST_VOXEL = "plugin/last_voxel_size"
KEY_DEBUG = "plugin/debug_mode"
KEY_AUTO_LOAD = "plugin/auto_load"
KEY_LAST_TOOL = "plugin/last_tool"
KEY_LAST_INPUT = "plugin/last_input"
KEY_LAST_OUT_DIR = "plugin/last_output_dir"
KEY_LAST_OUT_PREFIX = "plugin/last_out_prefix"
DEFAULT_VOXEL_SIZE = 10.0

PROBE_TIMEOUT_S = 15.0

# Probe script; runs in the target env. Prints a single JSON line.
_PROBE_SRC = (
    "import json\n"
    "r = {'cuda': False, 'device': None, 'mem_gb': None, 'cupy': None, 'cucim': None, "
    "'mrcfile': False, 'starfile': False, 'error': None}\n"
    "try:\n"
    "    import cupy\n"
    "    r['cupy'] = cupy.__version__\n"
    "    r['cuda'] = bool(cupy.cuda.is_available())\n"
    "    if r['cuda']:\n"
    "        try:\n"
    "            props = cupy.cuda.runtime.getDeviceProperties(0)\n"
    "            name = props.get('name', '')\n"
    "            r['device'] = name.decode() if isinstance(name, bytes) else str(name)\n"
    "            mem = props.get('totalGlobalMem', 0)\n"
    "            if mem:\n"
    "                r['mem_gb'] = mem / (1024 ** 3)\n"
    "        except Exception:\n"
    "            pass\n"
    "    import cucim\n"
    "    r['cucim'] = getattr(cucim, '__version__', 'installed')\n"
    "    import mrcfile; r['mrcfile'] = True\n"
    "    import starfile; r['starfile'] = True\n"
    "except Exception as exc:\n"
    "    r['error'] = type(exc).__name__ + ': ' + str(exc)\n"
    "print(json.dumps(r))\n"
)


@dataclass
class EnvStatus:
    ok: bool
    cuda: bool
    device_name: Optional[str]
    mem_gb: Optional[float]
    cupy_version: Optional[str]
    cucim_version: Optional[str]
    mrcfile_ok: bool
    starfile_ok: bool
    ais2star_ok: bool
    error: Optional[str] = None


def list_conda_envs() -> List[Tuple[str, str]]:
    """Return [(env_name, python_path), …] across common conda installations.

    Silent about missing conda / missing directories. Deduplicates by path.
    """
    found: List[Tuple[str, str]] = []

    conda = shutil.which("conda")
    if conda is not None:
        try:
            proc = subprocess.run(
                [conda, "env", "list", "--json"],
                capture_output=True, text=True, timeout=5.0,
            )
            if proc.returncode == 0:
                data = json.loads(proc.stdout)
                for env_path_str in data.get("envs", []):
                    env_path = Path(env_path_str)
                    py = env_path / "bin" / "python"
                    if py.is_file():
                        found.append((env_path.name, str(py)))
        except Exception:
            pass

    if not found:
        candidate_roots = [
            Path.home() / "miniconda3" / "envs",
            Path.home() / "anaconda3" / "envs",
            Path.home() / "miniforge3" / "envs",
            Path.home() / "mambaforge" / "envs",
            Path("/opt/miniconda3/envs"),
            Path("/opt/anaconda3/envs"),
        ]
        for root in candidate_roots:
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                py = child / "bin" / "python"
                if py.is_file():
                    found.append((child.name, str(py)))

    seen: set = set()
    uniq: List[Tuple[str, str]] = []
    for name, path in found:
        if path in seen:
            continue
        seen.add(path)
        uniq.append((name, path))
    return uniq


def probe_env(python_path: str, ais2star_dir: str) -> EnvStatus:
    """Run the probe under `python_path`; verify AIS2star run_step.py exists at `ais2star_dir`."""
    ais2star_ok = Path(ais2star_dir, "run_step.py").is_file()

    def _fail(msg: str) -> EnvStatus:
        return EnvStatus(
            ok=False, cuda=False, device_name=None, mem_gb=None, cupy_version=None,
            cucim_version=None, mrcfile_ok=False, starfile_ok=False,
            ais2star_ok=ais2star_ok, error=msg,
        )

    if not python_path or not Path(python_path).is_file():
        return _fail(f"python not found: {python_path!r}")

    try:
        proc = subprocess.run(
            [python_path, "-c", _PROBE_SRC],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return _fail(f"probe timed out after {PROBE_TIMEOUT_S:.0f}s")
    except Exception as exc:
        return _fail(f"probe launch failed: {exc}")

    if proc.returncode != 0:
        return _fail((proc.stderr or proc.stdout).strip()[:400] or f"probe exit {proc.returncode}")

    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return _fail("probe returned non-JSON output")

    cuda = bool(payload.get("cuda"))
    cupy_v = payload.get("cupy")
    cucim_v = payload.get("cucim")
    mrcfile_ok = bool(payload.get("mrcfile"))
    starfile_ok = bool(payload.get("starfile"))
    ok = cuda and bool(cupy_v) and bool(cucim_v) and mrcfile_ok and starfile_ok and ais2star_ok
    mem_gb_raw = payload.get("mem_gb")
    mem_gb = float(mem_gb_raw) if isinstance(mem_gb_raw, (int, float)) else None
    return EnvStatus(
        ok=ok,
        cuda=cuda,
        device_name=payload.get("device"),
        mem_gb=mem_gb,
        cupy_version=cupy_v,
        cucim_version=cucim_v,
        mrcfile_ok=mrcfile_ok,
        starfile_ok=starfile_ok,
        ais2star_ok=ais2star_ok,
        error=payload.get("error"),
    )


def load_saved_python() -> str:
    return QSettings(SETTINGS_ORG, SETTINGS_APP).value(KEY_PYTHON, "", type=str) or ""


def save_python(path: str) -> None:
    QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(KEY_PYTHON, path)


def load_saved_voxel() -> float:
    raw = QSettings(SETTINGS_ORG, SETTINGS_APP).value(KEY_LAST_VOXEL, DEFAULT_VOXEL_SIZE)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return DEFAULT_VOXEL_SIZE


def save_voxel(value: float) -> None:
    QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(KEY_LAST_VOXEL, float(value))


def _settings() -> QSettings:
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def load_bool(key: str, default: bool) -> bool:
    raw = _settings().value(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.lower() in ("true", "1", "yes")
    try:
        return bool(raw)
    except Exception:
        return default


def save_bool(key: str, value: bool) -> None:
    _settings().setValue(key, bool(value))


def load_string(key: str, default: str = "") -> str:
    val = _settings().value(key, default, type=str)
    return val or ""


def save_string(key: str, value: str) -> None:
    _settings().setValue(key, str(value or ""))


def load_tool_params(tool: str) -> dict:
    raw = _settings().value(f"plugin/{tool}/params", "", type=str) or ""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_tool_params(tool: str, params: dict) -> None:
    _settings().setValue(f"plugin/{tool}/params", json.dumps(params))
