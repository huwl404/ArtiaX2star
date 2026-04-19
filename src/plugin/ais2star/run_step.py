#!/usr/bin/env python3
"""Step-aware driver for fiber2star / mem2star plugin runs.

CLI::

    run_step.py --tool {fiber2star,mem2star} --steps 1-K \
                --input <mrc> [--params-json <json>] \
                [--out-prefix <prefix>] [--debug]

Loads parameter overrides from the JSON file (keys match CLI flags with
underscores, e.g. ``"sigma_z": 1.5``), translates them to the underlying
tool's CLI, injects ``--max-step K`` and ``--tomo <input>``, then invokes
the tool's ``main()`` in-process.

Runs under the filament conda env's Python. Does not import chimerax.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parse_steps(spec: str) -> int:
    """Parse '1-K' or 'K' into the max step integer."""
    spec = spec.strip()
    if "-" in spec:
        _, k = spec.split("-", 1)
        return int(k)
    return int(spec)


def _overrides_to_argv(overrides: dict) -> list:
    """Turn a {param: value} dict into ['--param', 'value', ...]. Booleans become flags."""
    out: list = []
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                out.append(flag)
        else:
            out.extend([flag, str(value)])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run fiber2star / mem2star up to a given pipeline step.")
    p.add_argument("--tool", choices=["fiber2star", "mem2star"], required=True)
    p.add_argument("--steps", default="1-99", help="step range '1-K'; runs steps 1..K inclusive")
    p.add_argument("--input", required=True, help="input MRC path")
    p.add_argument("--params-json", default=None, help="JSON file of tool-specific parameter overrides")
    p.add_argument("--out-prefix", default=None, help="output prefix (default: input stem in same dir)")
    p.add_argument("--debug", action="store_true", help="save intermediate debug MRCs")
    args = p.parse_args()

    max_step = _parse_steps(args.steps)

    overrides: dict = {}
    if args.params_json:
        with open(args.params_json, "r") as fh:
            overrides = json.load(fh)

    # Build the target tool's argv. sys.argv[0] is ignored by argparse.
    tool_argv = [args.tool, "--tomo", args.input, "--max-step", str(max_step)]
    if args.debug:
        tool_argv.append("--debug")
    if args.out_prefix:
        tool_argv.extend(["--out-prefix", args.out_prefix])
    tool_argv.extend(_overrides_to_argv(overrides))

    # Import sibling modules by adding this directory to sys.path (allows the script
    # to be executed with an absolute path, e.g. from ArtiaX's QProcess).
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    if args.tool == "fiber2star":
        from fiber2star import main as tool_main
    else:
        from mem2star import main as tool_main

    sys.argv = tool_argv
    tool_main()


if __name__ == "__main__":
    main()
