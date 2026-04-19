# vim: set expandtab shiftwidth=4 softtabstop=4:
"""QProcess-driven runner for src/plugin/ais2star/run_step.py.

Design: QProcess + QEventLoop. Looks synchronous to callers (run() blocks
until the subprocess exits), but the Qt event loop keeps turning so the
log streams live and the Stop button can fire kill().
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from Qt.QtCore import QEventLoop, QProcess, QProcessEnvironment

from .schema import TOOL_SCHEMAS

_RUN_STEP_PATH = Path(__file__).parent / "ais2star" / "run_step.py"

EXIT_ABORTED = -1
EXIT_FAILED_TO_START = -2
EXIT_CRASHED = -3


def resolve_out_prefix(spec: dict) -> Path:
    """(input_dir or output_dir) / (out_prefix or input_stem). Always absolute."""
    input_p = Path(spec["input"]).expanduser()
    outdir = (spec.get("output_dir") or "").strip()
    prefix = (spec.get("out_prefix") or "").strip() or input_p.stem
    base = Path(outdir).expanduser() if outdir else input_p.parent
    return (base / prefix).resolve()


def build_tool_argv(spec: dict, out_prefix: Path, params_json_path: str) -> List[str]:
    argv = [
        str(_RUN_STEP_PATH),
        "--tool", spec["tool"],
        "--steps", f"1-{int(spec['max_step'])}",
        "--input", str(Path(spec["input"]).expanduser()),
        "--params-json", params_json_path,
        "--out-prefix", str(out_prefix),
    ]
    if spec.get("debug"):
        argv.append("--debug")
    return argv


def format_equivalent_cli(spec: dict, out_prefix: Path) -> str:
    """Expand run_step.py invocation with params inline (no params-json)
    so the result is copy-pasteable into a terminal.
    """
    parts: List[str] = [
        shlex.quote(spec["env_python"]),
        "-u",
        shlex.quote(str(_RUN_STEP_PATH)),
        "--tool", spec["tool"],
        "--steps", f"1-{int(spec['max_step'])}",
        "--input", shlex.quote(str(Path(spec["input"]).expanduser())),
        "--out-prefix", shlex.quote(str(out_prefix)),
    ]
    if spec.get("debug"):
        parts.append("--debug")
    for name, value in (spec.get("params") or {}).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                parts.append(flag)
        else:
            parts.extend([flag, shlex.quote(str(value))])
    return " ".join(parts)


class Runner:
    """One subprocess at a time; kill() ends it immediately."""

    def __init__(self, append_log: Callable[[str], None]):
        self._append = append_log
        self._process: Optional[QProcess] = None
        self._params_json_path: Optional[str] = None
        self.aborted: bool = False

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.NotRunning

    def run(self, spec: dict) -> int:
        """Start run_step.py for `spec`, block until it finishes. Returns exit code."""
        if self.running:
            raise RuntimeError("Runner is already busy")

        self.aborted = False
        out_prefix = resolve_out_prefix(spec)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)

        fd, self._params_json_path = tempfile.mkstemp(prefix="ais2star_params_", suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(spec.get("params", {}), fh)

        argv = build_tool_argv(spec, out_prefix, self._params_json_path)

        self._process = QProcess()
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._drain_output)

        # Force live stdout streaming: -u disables Python's block buffering,
        # PYTHONUNBUFFERED covers subprocesses / libraries that ignore -u.
        unbuffered_argv = ["-u"] + argv
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self._process.setProcessEnvironment(env)

        self._append(f"$ {shlex.quote(spec['env_python'])} -u " + " ".join(shlex.quote(a) for a in argv) + "\n")

        loop = QEventLoop()
        self._process.finished.connect(loop.quit)
        self._process.start(spec["env_python"], unbuffered_argv)
        if not self._process.waitForStarted(10_000):
            self._append("ERROR: failed to start subprocess (check Python path)\n")
            self._cleanup()
            return EXIT_FAILED_TO_START

        loop.exec()

        self._drain_output()  # flush any stragglers
        if self.aborted:
            code = EXIT_ABORTED
        elif self._process.exitStatus() == QProcess.CrashExit:
            # Signal death (e.g. OOM kill, segfault). QProcess.exitCode() is
            # undefined for a CrashExit, so surface a distinct sentinel so the
            # caller can report it accurately instead of a garbled exit code.
            code = EXIT_CRASHED
        else:
            code = int(self._process.exitCode())
        self._cleanup()
        return code

    def kill(self) -> None:
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            self.aborted = True
            self._append("(user requested stop)\n")
            self._process.kill()

    def _drain_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode(errors="replace")
        if data:
            self._append(data)

    def _cleanup(self) -> None:
        if self._params_json_path and os.path.exists(self._params_json_path):
            try:
                os.remove(self._params_json_path)
            except OSError:
                pass
        self._params_json_path = None
        self._process = None


# ─────────────────────────────────────────────────────────────────────────────
# Auto-open
# ─────────────────────────────────────────────────────────────────────────────


def auto_open_outputs(session, spec: dict, out_prefix: Path) -> int:
    """Open only the MRC outputs of the final (= max_step) step, applying preview_hint."""
    from chimerax.core.commands import run as cmd_run
    from chimerax.map import open_map
    from chimerax.artiax.volume import Tomogram

    steps = TOOL_SCHEMAS[spec["tool"]]
    max_step = int(spec.get("max_step") or 0)
    last = next((s for s in steps if s.number == max_step), None)
    if last is None:
        return 0

    n_opened = 0
    for out in last.outputs:
        path = out_prefix.parent / (out_prefix.name + out.pattern)
        if not path.is_file():
            continue
        try:
            models = open_map(session, str(path))[0]
            if not models:
                continue
            vol = models[0]
            tomo = Tomogram.from_volume(session, vol)
            session.ArtiaX.add_tomogram(tomo)
            cmd = _style_command(vol, tomo.id_string, out.hint)
            cmd_run(session, cmd, log=False)
            n_opened += 1
        except Exception as exc:
            session.logger.warning(f"[ais2star] failed to open {path}: {exc}")
    return n_opened


def _style_command(volume, id_string: str, hint: str) -> str:
    """Build a `volume #id style surface …` command from a preview_hint."""
    cmd = f"volume #{id_string} style surface step 1"
    if hint == "prob01":
        # Only add level 0.99 when data is actually 0..1 normalized
        try:
            mx = float(volume.matrix_value_statistics().maximum)
        except Exception:
            mx = 2.0  # fall back to "keep" behaviour
        if mx <= 1.01:
            cmd += " level 0.99"
    elif hint in ("binary", "label"):
        cmd += " level 0.5"
    # "keep" → leave level alone
    return cmd
