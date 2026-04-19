# vim: set expandtab shiftwidth=4 softtabstop=4:

import struct
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from Qt.QtCore import Qt, QTimer
from Qt.QtGui import QFont
from Qt.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import env as env_mod
from . import runner as runner_mod
from ..ArtiaX import SEL_PARTLIST_CHANGED
from .ais2star import package_dir as ais2star_dir
from .cc_table import ConnectedComponentsPanel
from .schema import StepSpec, TOOL_SCHEMAS

STAR_FORMAT = "RELION STAR file"

CUSTOM_LABEL = "(custom)"
DEFAULT_TOOL = "mem2star"

# Uniform layout spacing across every QGroupBox in the Plugin tab. Keeping the
# values centralized makes sure any tweak to the look reads as one decision
# rather than drift across seven groups.
GROUP_MARGINS = (8, 6, 8, 6)  # left, top, right, bottom
GROUP_SPACING = 6             # between rows inside a group, and between groups


def _tighten(layout) -> None:
    """Apply the shared margin + spacing values to a group's inner layout."""
    layout.setContentsMargins(*GROUP_MARGINS)
    layout.setSpacing(GROUP_SPACING)


class PluginWidget(QScrollArea):
    """Right-side 'Plugin' tab for the AIS2star fiber2star / mem2star workflow.

    Phase 2: Environment group.
    Phase 3: Tool picker, Inputs, Pipeline (schema-driven).
    Later phases: Run controls, ParticleList display, Connected Components.
    """

    def __init__(self, session, parent=None, font=None):
        super().__init__(parent)
        self.session = session
        self._envs: List[Tuple[str, str]] = []
        self.env_ok: bool = False
        self._staged_star_path: Optional[Path] = None

        if font is not None:
            self.setFont(font)

        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        inner = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(*GROUP_MARGINS)
        layout.setSpacing(GROUP_SPACING)

        layout.addWidget(self._build_env_group())
        layout.addWidget(self._build_tool_group())
        layout.addWidget(self._build_inputs_group())
        layout.addWidget(self._build_pipeline_group())
        layout.addWidget(self._build_run_group())
        layout.addWidget(self._build_particle_list_group())
        self._cc_panel = ConnectedComponentsPanel(self.session, self._append_log)
        layout.addWidget(self._cc_panel)

        self._runner = runner_mod.Runner(self._append_log)

        # Track the active ParticleList so the "current:" label stays in sync.
        self.session.ArtiaX.triggers.add_handler(
            SEL_PARTLIST_CHANGED, self._on_selected_partlist_changed
        )

        # Restore UI state from the previous session.
        self._restore_session_state()

        inner.setLayout(layout)
        self.setWidget(inner)

        # Defer env discovery to after the event loop returns to idle; running
        # `conda env list --json` synchronously in __init__ lets Marker Placement
        # win the dock-tab focus ahead of the trailing `show ArtiaX Options` in
        # OptionsWindow.__init__.
        QTimer.singleShot(0, self._populate_envs)

    # ------------------------------------------------------------------
    # Environment group
    # ------------------------------------------------------------------

    def _build_env_group(self) -> QGroupBox:
        group = QGroupBox("Environment")
        grid = QGridLayout()
        _tighten(grid)
        grid.setColumnStretch(1, 1)

        grid.addWidget(QLabel("Conda env:"), 0, 0)
        self._env_combo = QComboBox()
        self._env_combo.currentIndexChanged.connect(self._on_env_combo_change)
        self._env_combo.setMinimumContentsLength(8)
        self._env_combo.setMinimumWidth(80)
        grid.addWidget(self._env_combo, 0, 1)
        self._detect_btn = QPushButton("Detect")
        self._detect_btn.setToolTip("Rescan conda envs")
        self._detect_btn.clicked.connect(self._populate_envs)
        grid.addWidget(self._detect_btn, 0, 2)

        grid.addWidget(QLabel("Python:"), 1, 0)
        self._py_edit = QLineEdit()
        self._py_edit.setPlaceholderText("/path/to/env/bin/python")
        self._py_edit.textEdited.connect(self._on_py_edited)
        _make_path_field(self._py_edit)
        grid.addWidget(self._py_edit, 1, 1)
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.clicked.connect(self._browse_python)
        grid.addWidget(self._browse_btn, 1, 2)

        grid.addWidget(QLabel("Status:"), 2, 0, Qt.AlignTop)
        self._status_label = QLabel("(not tested)")
        self._status_label.setWordWrap(True)
        self._status_label.setTextFormat(Qt.RichText)
        grid.addWidget(self._status_label, 2, 1)

        self._test_btn = QPushButton("Test env")
        self._test_btn.setToolTip("Probe the selected python for cupy / cucim / CUDA / AIS2star")
        self._test_btn.clicked.connect(self._test_env)
        grid.addWidget(self._test_btn, 2, 2)

        group.setLayout(grid)
        return group

    def _populate_envs(self) -> None:
        self._envs = env_mod.list_conda_envs()
        saved = env_mod.load_saved_python()

        self._env_combo.blockSignals(True)
        self._env_combo.clear()
        for name, path in self._envs:
            self._env_combo.addItem(name, path)
        self._env_combo.addItem(CUSTOM_LABEL, "")

        selected_path = saved
        if not selected_path and self._envs:
            preferred = next((p for n, p in self._envs if n == "filament"), self._envs[0][1])
            selected_path = preferred

        _set_path_text(self._py_edit, selected_path)
        self._sync_combo_to_path(selected_path)
        self._env_combo.blockSignals(False)

    def _sync_combo_to_path(self, path: str) -> None:
        for i, (_, p) in enumerate(self._envs):
            if p == path:
                self._env_combo.setCurrentIndex(i)
                return
        self._env_combo.setCurrentIndex(self._env_combo.count() - 1)

    def _on_env_combo_change(self, idx: int) -> None:
        path = self._env_combo.itemData(idx) or ""
        if path:
            _set_path_text(self._py_edit, path)
            self._mark_untested()

    def _on_py_edited(self, _text: str) -> None:
        self._env_combo.blockSignals(True)
        self._sync_combo_to_path(self._py_edit.text().strip())
        self._env_combo.blockSignals(False)
        self._mark_untested()

    def _browse_python(self) -> None:
        start = self._py_edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Select Python interpreter", start)
        if path:
            _set_path_text(self._py_edit, path)
            self._on_py_edited(path)

    def _mark_untested(self) -> None:
        self.env_ok = False
        self._status_label.setText("(not tested)")

    def _test_env(self) -> None:
        py = self._py_edit.text().strip()
        self._status_label.setText("Testing…")
        self._test_btn.setEnabled(False)
        try:
            status = env_mod.probe_env(py, ais2star_dir())
        finally:
            self._test_btn.setEnabled(True)

        self.env_ok = status.ok
        cuda_extra = (
            f" ({status.mem_gb:.0f} GB)"
            if status.cuda and status.mem_gb is not None
            else ""
        )
        checks = [
            ("CUDA", status.cuda, cuda_extra),
            ("cupy", bool(status.cupy_version), ""),
            ("cucim", bool(status.cucim_version), ""),
            ("mrcfile", status.mrcfile_ok, ""),
            ("starfile", status.starfile_ok, ""),
            ("AIS2star", status.ais2star_ok, ""),
        ]
        parts = [f"{n} {'✓' if ok else '✗'}{extra}" for n, ok, extra in checks]
        line = " · ".join(parts)
        if status.ok:
            self._status_label.setText(f"<span style='color:#2a8a2a'>{line}</span>")
            env_mod.save_python(py)
        else:
            tail = f" — {status.error}" if status.error else ""
            self._status_label.setText(f"<span style='color:#c02828'>{line}</span>{tail}")

    # ------------------------------------------------------------------
    # Tool group (radio picker)
    # ------------------------------------------------------------------

    def _build_tool_group(self) -> QGroupBox:
        group = QGroupBox("Tool")
        layout = QHBoxLayout()
        _tighten(layout)

        self._tool_buttons = QButtonGroup(self)
        self._fiber_radio = QRadioButton("fiber2star")
        self._mem_radio = QRadioButton("mem2star")
        self._tool_buttons.addButton(self._fiber_radio, 0)
        self._tool_buttons.addButton(self._mem_radio, 1)
        if DEFAULT_TOOL == "mem2star":
            self._mem_radio.setChecked(True)
        else:
            self._fiber_radio.setChecked(True)

        layout.addWidget(self._fiber_radio)
        layout.addWidget(self._mem_radio)
        layout.addStretch(1)
        group.setLayout(layout)
        return group

    @property
    def selected_tool(self) -> str:
        return "fiber2star" if self._fiber_radio.isChecked() else "mem2star"

    # ------------------------------------------------------------------
    # Inputs group
    # ------------------------------------------------------------------

    def _build_inputs_group(self) -> QGroupBox:
        group = QGroupBox("Inputs")
        grid = QGridLayout()
        _tighten(grid)
        grid.setColumnStretch(1, 1)

        # Row 0: Mode + Debug mode
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._mode_group = QButtonGroup(self)
        self._mode_single_radio = QRadioButton("Single")
        self._mode_batch_radio = QRadioButton("Batch")
        self._mode_single_radio.setChecked(True)
        self._mode_group.addButton(self._mode_single_radio, 0)
        self._mode_group.addButton(self._mode_batch_radio, 1)
        mode_row.addWidget(self._mode_single_radio)
        mode_row.addWidget(self._mode_batch_radio)
        mode_row.addStretch(1)
        self._debug_cb = QCheckBox("Debug mode")
        self._debug_cb.setToolTip("Save intermediate MRCs for each pipeline step")
        self._debug_cb.setChecked(True)
        mode_row.addWidget(self._debug_cb)
        mode_wrap = QWidget()
        mode_wrap.setLayout(mode_row)
        grid.addWidget(mode_wrap, 0, 0, 1, 3)

        # Row 1: Input area — container holding both pages; hidden page is
        # excluded from QVBoxLayout's size hint, so the groupbox shrinks when
        # Single is active (QStackedWidget's sizeHint is max of all children,
        # so we avoid it here).
        input_container = QWidget()
        input_box = QVBoxLayout()
        input_box.setContentsMargins(0, 0, 0, 0)
        self._single_page = self._build_single_input()
        self._batch_page = self._build_batch_input()
        self._batch_page.setVisible(False)
        input_box.addWidget(self._single_page)
        input_box.addWidget(self._batch_page)
        input_container.setLayout(input_box)
        grid.addWidget(input_container, 1, 0, 1, 3)
        self._mode_single_radio.toggled.connect(self._on_mode_changed)
        self._mode_batch_radio.toggled.connect(self._on_mode_changed)

        # Row 2: Output dir
        grid.addWidget(QLabel("Output dir:"), 2, 0)
        self._outdir_edit = QLineEdit()
        self._outdir_edit.setPlaceholderText("(default: alongside input)")
        _make_path_field(self._outdir_edit)
        grid.addWidget(self._outdir_edit, 2, 1)
        browse_out = QPushButton("Browse…")
        browse_out.clicked.connect(self._browse_outdir)
        grid.addWidget(browse_out, 2, 2)

        # Row 3: Out prefix (Debug moved up to Mode row)
        grid.addWidget(QLabel("Out prefix:"), 3, 0)
        self._prefix_edit = QLineEdit()
        self._prefix_edit.setPlaceholderText("(default: input stem)")
        _make_path_field(self._prefix_edit)
        grid.addWidget(self._prefix_edit, 3, 1, 1, 2)

        group.setLayout(grid)
        return group

    def _build_single_input(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Input MRC:"))
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("/path/to/tomo.mrc")
        _make_path_field(self._input_edit)
        row.addWidget(self._input_edit, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_input)
        row.addWidget(browse)
        w.setLayout(row)
        return w

    def _build_batch_input(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout()
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(QLabel("Input files:"))
        self._batch_list = QListWidget()
        self._batch_list.setFixedHeight(90)
        self._batch_list.setSelectionMode(QListWidget.ExtendedSelection)
        v.addWidget(self._batch_list)
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add files…")
        add_btn.clicked.connect(self._batch_add)
        btn_row.addWidget(add_btn)
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(self._batch_remove)
        btn_row.addWidget(rm_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._batch_clear)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch(1)
        v.addLayout(btn_row)
        w.setLayout(v)
        return w

    def _on_mode_changed(self, *_args) -> None:
        is_batch = self._mode_batch_radio.isChecked()
        self._single_page.setVisible(not is_batch)
        self._batch_page.setVisible(is_batch)
        # In batch mode, debug is forced off (avoid disk blowup across many files).
        if is_batch:
            self._debug_cb.setChecked(False)
            self._debug_cb.setEnabled(False)
        else:
            self._debug_cb.setEnabled(True)

    def _browse_input(self) -> None:
        start = self._input_edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Select input volume", start,
            "Volume (*.mrc *.rec *.map *.mrcs)",
        )
        if path:
            _set_path_text(self._input_edit, path)

    def _browse_outdir(self) -> None:
        start = self._outdir_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select output directory", start)
        if path:
            _set_path_text(self._outdir_edit, path)

    def _batch_add(self) -> None:
        start = str(Path.home())
        if self._batch_list.count() > 0:
            last = self._batch_list.item(self._batch_list.count() - 1).text()
            parent_dir = str(Path(last).expanduser().parent)
            if parent_dir:
                start = parent_dir
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add input volumes", start,
            "Volume (*.mrc *.rec *.map *.mrcs)",
        )
        existing = {self._batch_list.item(i).text() for i in range(self._batch_list.count())}
        for p in paths:
            if p not in existing:
                self._batch_list.addItem(p)

    def _batch_remove(self) -> None:
        for item in self._batch_list.selectedItems():
            self._batch_list.takeItem(self._batch_list.row(item))

    def _batch_clear(self) -> None:
        self._batch_list.clear()

    def _batch_files(self) -> List[str]:
        return [self._batch_list.item(i).text() for i in range(self._batch_list.count())]

    # ------------------------------------------------------------------
    # Pipeline group (schema-driven)
    # ------------------------------------------------------------------

    def _build_pipeline_group(self) -> QGroupBox:
        group = QGroupBox("Pipeline")
        layout = QVBoxLayout()
        _tighten(layout)

        # Hold both pipelines as siblings in a plain QVBoxLayout and toggle
        # visibility on tool change. QStackedWidget would reserve the union
        # size of both pages; QBoxLayout excludes hidden widgets from its
        # sizeHint, so the groupbox shrinks to the visible pipeline.
        self._pipeline_pages: Dict[str, _PipelinePage] = {}
        for tool_name, steps in TOOL_SCHEMAS.items():
            page = _PipelinePage(steps)
            page.setVisible(tool_name == DEFAULT_TOOL)
            self._pipeline_pages[tool_name] = page
            layout.addWidget(page)

        self._fiber_radio.toggled.connect(self._on_tool_changed)
        self._mem_radio.toggled.connect(self._on_tool_changed)

        # Debug checkbox drives per-step output-filename hints.
        self._debug_cb.toggled.connect(self._on_debug_toggled)
        self._on_debug_toggled(self._debug_cb.isChecked())

        group.setLayout(layout)
        return group

    def _on_tool_changed(self, *_args) -> None:
        active = self.selected_tool
        for tool_name, page in self._pipeline_pages.items():
            page.setVisible(tool_name == active)

    def _on_debug_toggled(self, checked: bool) -> None:
        for page in self._pipeline_pages.values():
            page.set_debug_visible(bool(checked))

    # ------------------------------------------------------------------
    # Run-spec accessor (consumed by Phase 4 runner)
    # ------------------------------------------------------------------

    def collect_run_spec(self) -> dict:
        tool = self.selected_tool
        page = self._pipeline_pages[tool]
        max_step, checked_steps, params = page.collect()
        mode = "batch" if self._mode_batch_radio.isChecked() else "single"
        return {
            "tool": tool,
            "env_python": self._py_edit.text().strip(),
            "env_ok": self.env_ok,
            "mode": mode,
            "input": self._input_edit.text().strip(),
            "batch_inputs": self._batch_files() if mode == "batch" else [],
            "output_dir": self._outdir_edit.text().strip(),
            "out_prefix": self._prefix_edit.text().strip(),
            "debug": bool(self._debug_cb.isChecked()),
            "max_step": max_step,
            "checked_steps": sorted(checked_steps),
            "params": params,
        }

    # ------------------------------------------------------------------
    # Run group
    # ------------------------------------------------------------------

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Run")
        layout = QVBoxLayout()
        _tighten(layout)

        btn_row = QHBoxLayout()
        self._run_last_btn = QPushButton("Run to last checked")
        self._run_last_btn.clicked.connect(self._on_run_last_clicked)
        self._run_all_btn = QPushButton("Run all")
        self._run_all_btn.clicked.connect(self._on_run_all_clicked)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        # Distribute buttons evenly with equal gaps on both ends and between.
        btn_row.addStretch(1)
        btn_row.addWidget(self._run_last_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._run_all_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._run_status = QLabel("Idle")
        layout.addWidget(self._run_status)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(2000)
        self._log_view.setFixedHeight(180)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.TypeWriter)
        mono.setPointSize(max(6, self.font().pointSize() - 1))
        self._log_view.setFont(mono)
        layout.addWidget(self._log_view)

        group.setLayout(layout)
        return group

    def _append_log(self, text: str) -> None:
        self._log_view.moveCursor(self._log_view.textCursor().MoveOperation.End)
        self._log_view.insertPlainText(text)
        self._log_view.moveCursor(self._log_view.textCursor().MoveOperation.End)

    def _on_run_last_clicked(self) -> None:
        self._run_spec(run_all=False)

    def _on_run_all_clicked(self) -> None:
        self._run_spec(run_all=True)

    def _on_stop_clicked(self) -> None:
        self._runner.kill()

    def _run_spec(self, *, run_all: bool) -> None:
        if self._runner.running:
            return

        if run_all:
            self._pipeline_pages[self.selected_tool].check_all()

        spec = self.collect_run_spec()

        err, files = self._validate_spec(spec)
        if err:
            self._append_log(f"ERROR: {err}\n")
            return

        tab = _find_tab_ancestor(self)
        saved_tab = tab.currentIndex() if tab is not None else -1

        is_batch = spec["mode"] == "batch"
        failures: List[Tuple[str, str]] = []
        n_ok = 0

        self._set_running(True)
        try:
            for idx, fpath in enumerate(files):
                single_spec = dict(spec)
                single_spec["input"] = str(fpath)
                if is_batch:
                    self._append_log(f"\n=== [{idx + 1}/{len(files)}] {fpath} ===\n")

                code = self._runner.run(single_spec)

                if code == 0:
                    n_ok += 1
                    out_prefix = runner_mod.resolve_out_prefix(single_spec)
                    if not is_batch:
                        n_vols = runner_mod.auto_open_outputs(self.session, single_spec, out_prefix)
                        self._append_log(f"[done] exit 0 · opened {n_vols} debug volume(s)\n")
                        # Phase 6: auto-load produced STAR as ParticleList.
                        if self._auto_load_cb.isChecked() and single_spec["max_step"] == len(TOOL_SCHEMAS[single_spec["tool"]]):
                            self._auto_load_produced_star(single_spec, out_prefix)
                    else:
                        self._append_log("[done] exit 0\n")
                    # Phase 8: persist params + mirror expanded CLI to ChimeraX Log.
                    self._save_session_state(single_spec)
                    cli = runner_mod.format_equivalent_cli(single_spec, out_prefix)
                    self.session.logger.info(
                        f"[ArtiaX-Plugin] {single_spec['tool']} {'batch ' if is_batch else ''}"
                        f"run OK — equivalent command:\n{cli}"
                    )
                elif code == runner_mod.EXIT_ABORTED:
                    self._append_log("[stopped by user]\n")
                    break
                elif code == runner_mod.EXIT_FAILED_TO_START:
                    self._append_log("[failed to launch] click Test env to check the environment.\n")
                    failures.append((str(fpath), "failed to launch"))
                elif code == runner_mod.EXIT_CRASHED:
                    self._append_log(
                        "[crashed] subprocess died from a signal (OOM kill / segfault). "
                        "Check GPU memory and the log above for the last step that ran.\n"
                    )
                    failures.append((str(fpath), "crashed"))
                else:
                    self._append_log(f"[failed] exit {code} · click Test env to check the environment.\n")
                    failures.append((str(fpath), f"exit {code}"))
        finally:
            self._set_running(False)

        if is_batch:
            self._append_log(f"\n[batch done] {n_ok} ok, {len(failures)} failed\n")
            for path, reason in failures:
                self._append_log(f"  FAIL: {path} ({reason})\n")

        if tab is not None and saved_tab >= 0:
            tab.setCurrentIndex(saved_tab)

    def _validate_spec(self, spec: dict) -> Tuple[str, List[Path]]:
        """Return (error_message, resolved_file_list). Empty error means OK."""
        if not spec["env_python"]:
            return "Python path is empty (pick a conda env).", []
        if int(spec["max_step"]) <= 0:
            return "no step is checked.", []

        if spec["mode"] == "batch":
            raw = spec.get("batch_inputs") or []
            if not raw:
                return "batch mode has no input files.", []
            files = [Path(p).expanduser() for p in raw]
        else:
            if not spec["input"]:
                return "input MRC is empty.", []
            files = [Path(spec["input"]).expanduser()]

        for f in files:
            if not f.is_file():
                return f"input not found: {f}", []
        return "", files

    def _set_running(self, busy: bool) -> None:
        self._run_last_btn.setEnabled(not busy)
        self._run_all_btn.setEnabled(not busy)
        self._stop_btn.setEnabled(busy)
        self._run_status.setText("Running…" if busy else "Idle")

    # ------------------------------------------------------------------
    # Session persistence (Phase 8)
    # ------------------------------------------------------------------

    def _restore_session_state(self) -> None:
        """Load last-used tool / inputs / per-tool params from QSettings."""
        last_tool = env_mod.load_string(env_mod.KEY_LAST_TOOL, DEFAULT_TOOL)
        if last_tool == "fiber2star":
            self._fiber_radio.setChecked(True)
        elif last_tool == "mem2star":
            self._mem_radio.setChecked(True)
        self._debug_cb.setChecked(env_mod.load_bool(env_mod.KEY_DEBUG, True))
        self._auto_load_cb.setChecked(env_mod.load_bool(env_mod.KEY_AUTO_LOAD, True))
        inp = env_mod.load_string(env_mod.KEY_LAST_INPUT, "")
        if inp:
            _set_path_text(self._input_edit, inp)
        outd = env_mod.load_string(env_mod.KEY_LAST_OUT_DIR, "")
        if outd:
            _set_path_text(self._outdir_edit, outd)
        outp = env_mod.load_string(env_mod.KEY_LAST_OUT_PREFIX, "")
        if outp:
            _set_path_text(self._prefix_edit, outp)
        for tool_name, page in self._pipeline_pages.items():
            saved = env_mod.load_tool_params(tool_name)
            if saved:
                page.apply_params(saved)

    def _save_session_state(self, spec: dict) -> None:
        env_mod.save_bool(env_mod.KEY_DEBUG, bool(spec.get("debug")))
        env_mod.save_bool(env_mod.KEY_AUTO_LOAD, bool(self._auto_load_cb.isChecked()))
        env_mod.save_string(env_mod.KEY_LAST_TOOL, spec["tool"])
        env_mod.save_string(env_mod.KEY_LAST_INPUT, spec.get("input", ""))
        env_mod.save_string(env_mod.KEY_LAST_OUT_DIR, spec.get("output_dir", ""))
        env_mod.save_string(env_mod.KEY_LAST_OUT_PREFIX, spec.get("out_prefix", ""))
        env_mod.save_tool_params(spec["tool"], spec.get("params") or {})

    # ------------------------------------------------------------------
    # Particle List group
    # ------------------------------------------------------------------

    def _build_particle_list_group(self) -> QGroupBox:
        group = QGroupBox("Particle List")
        box = QVBoxLayout()
        _tighten(box)

        self._auto_load_cb = QCheckBox("Auto-load produced STAR as ParticleList")
        self._auto_load_cb.setChecked(True)
        box.addWidget(self._auto_load_cb)

        load_row = QHBoxLayout()
        self._load_star_btn = QPushButton("Load STAR…")
        self._load_star_btn.clicked.connect(self._on_load_star_clicked)
        load_row.addWidget(self._load_star_btn)
        self._current_star_label = QLabel("current: (none)")
        self._current_star_label.setWordWrap(True)
        load_row.addWidget(self._current_star_label, 1)
        box.addLayout(load_row)

        # Single row: Voxel + Marker + Axes + one Apply button.
        # Labels shed their colons for uniform styling with the rest of the UI;
        # spinboxes share tooltips with their labels so hover works anywhere.
        voxel_tip = (
            "Voxel size in Å; written to Pixelsize Factors → Origin on the "
            "active ParticleList.\n↑ : larger scale for particle coordinates.\n"
            "↓ : smaller scale."
        )
        marker_tip = "Marker sphere radius in Å (→ pl.radius). Auto-set to 2 × voxel on load."
        axes_tip = "Particle-axes length in Å (→ pl.axes_size). Auto-set to 4 × voxel on load."

        apply_row = QHBoxLayout()
        apply_row.addWidget(QLabel("Voxel (Å)"))
        self._voxel_spin = QDoubleSpinBox()
        self._voxel_spin.setRange(0.01, 10000.0)
        self._voxel_spin.setDecimals(3)
        self._voxel_spin.setSingleStep(0.1)
        self._voxel_spin.setValue(env_mod.load_saved_voxel())
        self._voxel_spin.setMinimumWidth(30)
        self._voxel_spin.setToolTip(voxel_tip)
        apply_row.addWidget(self._voxel_spin, 1)
        apply_row.addSpacing(6)
        apply_row.addWidget(QLabel("Marker (Å)"))
        self._marker_spin = QDoubleSpinBox()
        self._marker_spin.setRange(0.01, 10000.0)
        self._marker_spin.setDecimals(2)
        self._marker_spin.setSingleStep(1.0)
        self._marker_spin.setValue(self._voxel_spin.value() * 2.0)
        self._marker_spin.setMinimumWidth(30)
        self._marker_spin.setToolTip(marker_tip)
        apply_row.addWidget(self._marker_spin, 1)
        apply_row.addSpacing(6)
        apply_row.addWidget(QLabel("Axes (Å)"))
        self._axes_spin = QDoubleSpinBox()
        self._axes_spin.setRange(0.01, 10000.0)
        self._axes_spin.setDecimals(2)
        self._axes_spin.setSingleStep(1.0)
        self._axes_spin.setValue(self._voxel_spin.value() * 4.0)
        self._axes_spin.setMinimumWidth(30)
        self._axes_spin.setToolTip(axes_tip)
        apply_row.addWidget(self._axes_spin, 1)
        apply_row.addSpacing(6)
        self._apply_all_btn = QPushButton("Apply")
        self._apply_all_btn.clicked.connect(self._on_apply_all_clicked)
        apply_row.addWidget(self._apply_all_btn)
        box.addLayout(apply_row)

        group.setLayout(box)
        return group

    def _active_partlist(self):
        artia = self.session.ArtiaX
        pid = artia.selected_partlist
        if pid is None:
            return None
        return artia.partlists.get(pid)

    def _on_selected_partlist_changed(self, _name, _data) -> None:
        if self._staged_star_path is not None:
            return  # keep the "(staged)" label until Apply commits the load
        pl = self._active_partlist()
        if pl is None:
            self._current_star_label.setText("current: (none)")
        else:
            self._current_star_label.setText(f"current: {pl.name}")

    def _on_load_star_clicked(self) -> None:
        start = str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Load STAR file", start, "STAR (*.star)"
        )
        if not path:
            return
        self._staged_star_path = Path(path)
        self._current_star_label.setText(
            f"current: (staged) {self._staged_star_path.name} — set Voxel size and click Apply"
        )

    def _on_apply_all_clicked(self) -> None:
        voxel = float(self._voxel_spin.value())
        if voxel <= 0:
            self._append_log("ERROR: Voxel size must be > 0.\n")
            return
        env_mod.save_voxel(voxel)

        if self._staged_star_path is not None:
            self._commit_staged_star(voxel)
            return

        pl = self._active_partlist()
        if pl is None:
            self._append_log("(no active ParticleList to apply to)\n")
            return
        pl.origin_pixelsize = voxel
        pl.radius = float(self._marker_spin.value())
        pl.axes_size = float(self._axes_spin.value())

    def _commit_staged_star(self, voxel: float) -> None:
        # Preserve the Plugin tab around open_partlist, which otherwise jumps
        # options_window to the Visualization tab via OPTIONS_PARTLIST_CHANGED.
        tab = _find_tab_ancestor(self)
        saved_tab = tab.currentIndex() if tab is not None else -1

        path = self._staged_star_path
        self._staged_star_path = None
        pl = self._load_star_into_session(path)
        if pl is None:
            self._current_star_label.setText("current: (none)")
            self._append_log(f"ERROR: failed to load {path}\n")
            return
        self._apply_display(pl, voxel)
        self._current_star_label.setText(f"current: {pl.name}")

        if tab is not None and saved_tab >= 0:
            tab.setCurrentIndex(saved_tab)

    def _load_star_into_session(self, path: Path):
        try:
            self.session.ArtiaX.open_partlist(str(path), STAR_FORMAT)
        except Exception as exc:
            self._append_log(f"ERROR: open_partlist failed: {exc}\n")
            return None
        return self._active_partlist()

    def _apply_display(self, partlist, voxel: float) -> None:
        """Apply voxel → origin_pixelsize, then reset marker/axes to voxel × 2/4 and apply."""
        partlist.origin_pixelsize = voxel
        marker = voxel * 2.0
        axes = voxel * 4.0
        self._marker_spin.setValue(marker)
        self._axes_spin.setValue(axes)
        partlist.radius = marker
        partlist.axes_size = axes
        env_mod.save_voxel(voxel)

    def _auto_load_produced_star(self, spec: dict, out_prefix: Path) -> None:
        """Called from _run_spec on a successful single-mode run when auto-load is on."""
        star_path = out_prefix.parent / (out_prefix.name + "_particles.star")
        if not star_path.is_file():
            return
        # Preferred voxel source is the final step's MRC — it lives in the same
        # (already-binned) coordinate space as the STAR particles, so no extra
        # bin multiplication is needed. Fall back to input MRC header × bin
        # only if no final-step MRC is available.
        voxel = _read_final_step_voxel(spec, out_prefix)
        if voxel <= 0:
            params = spec.get("params") or {}
            v = float(params.get("voxel_size", -1.0) or -1.0)
            if v <= 0:
                v = _read_mrc_voxel(Path(spec["input"]).expanduser())
            voxel = v * max(1, int(params.get("bin", 1) or 1))

        if voxel <= 0:
            self._append_log(
                "[auto-load] could not determine voxel size; STAR not loaded.\n"
            )
            return

        self._voxel_spin.setValue(voxel)
        pl = self._load_star_into_session(star_path)
        if pl is None:
            return
        self._apply_display(pl, voxel)
        self._current_star_label.setText(f"current: {pl.name}")
        self._append_log(
            f"[auto-load] {star_path.name} → #{pl.id_string} · voxel {voxel:.3f} Å\n"
        )


class _PipelinePage(QWidget):
    """Renders one tool's pipeline: a stack of checkable step groups with param rows."""

    def __init__(self, steps: List[StepSpec], parent=None):
        super().__init__(parent)
        self._steps: List[StepSpec] = steps
        # per-step: (step_number, QGroupBox, {param_name: editor}, debug_label_or_None)
        self._step_entries: List[Tuple[int, QGroupBox, Dict[str, QWidget], QLabel]] = []
        self._syncing: bool = False

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(0, 0, 0, 0)

        for step in steps:
            layout.addWidget(self._build_step_group(step))
        layout.addStretch(1)
        self.setLayout(layout)

        # Wire radio-like check behavior after all groups exist.
        for step_num, group, _editors, _dlabel in self._step_entries:
            group.toggled.connect(lambda checked, n=step_num: self._on_step_toggled(n, checked))

    def _build_step_group(self, step: StepSpec) -> QGroupBox:
        group = QGroupBox(f"{step.number} · {step.name}")
        group.setCheckable(True)
        group.setChecked(True)

        grid = QGridLayout()
        # 5-column layout: label / editor / spacer / label / editor.
        # cols 1 & 4 stretch equally → both spinboxes get the same width.
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(4, 1)
        grid.setColumnMinimumWidth(2, 18)
        grid.setHorizontalSpacing(6)
        row = 0

        # Debug-output hint (visible only when Debug mode is on).
        if step.outputs:
            patterns = " · ".join(o.pattern for o in step.outputs)
            debug_label = QLabel(f"<i>debug: {patterns}</i>")
        else:
            debug_label = QLabel("<i>debug: (none)</i>")
        debug_label.setWordWrap(True)
        debug_label.setVisible(False)  # PluginWidget sets visibility from Debug checkbox
        grid.addWidget(debug_label, row, 0, 1, 5)
        row += 1

        editors: Dict[str, QWidget] = {}
        if not step.params:
            # Plain-italic HTML instead of setStyleSheet — a stylesheet would
            # reset the inherited ArtiaX font and render this line too large.
            note = QLabel("<i>(no parameters for this step)</i>")
            grid.addWidget(note, row, 0, 1, 5)
        else:
            for idx, pspec in enumerate(step.params):
                label_text = f"{pspec.label} ({pspec.suffix})" if pspec.suffix else pspec.label
                label = QLabel(label_text)
                label.setToolTip(pspec.tooltip)
                # Word-wrap on the long param names (e.g. "curvature-consistency-weight",
                # "mask-normal-smooth-sigma (voxels)") lets the Plugin tab's minimum
                # width drop below the ArtiaX Options dock's floor; at normal widths
                # each label still renders on a single line.
                label.setWordWrap(True)
                if pspec.kind == "int":
                    editor: QWidget = QSpinBox()
                    editor.setRange(-1_000_000, 1_000_000)
                    editor.setValue(int(pspec.default))
                    if pspec.step > 0:
                        editor.setSingleStep(max(1, int(pspec.step)))
                else:
                    dsb = QDoubleSpinBox()
                    # 3 decimals covers every default in the schema (smallest is
                    # 0.05) while shrinking the spinbox's minimumSizeHint by ~30 px
                    # versus 5 decimals.
                    dsb.setDecimals(3)
                    dsb.setRange(-1e9, 1e9)
                    dsb.setValue(float(pspec.default))
                    step_size = pspec.step if pspec.step > 0 else _single_step_for(float(pspec.default))
                    dsb.setSingleStep(step_size)
                    editor = dsb
                editor.setToolTip(pspec.tooltip)
                # Skip col 2 (spacer): first pair goes in cols 0/1, second in 3/4.
                grid_row = row + (idx // 2)
                col_base = (idx % 2) * 3
                grid.addWidget(label, grid_row, col_base)
                grid.addWidget(editor, grid_row, col_base + 1)
                editors[pspec.name] = editor

        group.setLayout(grid)
        self._step_entries.append((step.number, group, editors, debug_label))
        return group

    def _on_step_toggled(self, step_num: int, checked: bool) -> None:
        """Radio-like: clicking step K → steps 1..K checked, K+1..N unchecked."""
        if self._syncing:
            return
        self._syncing = True
        try:
            if checked:
                for num, group, _e, _d in self._step_entries:
                    group.setChecked(num <= step_num)
            else:
                for num, group, _e, _d in self._step_entries:
                    if num >= step_num:
                        group.setChecked(False)
        finally:
            self._syncing = False

    def check_all(self) -> None:
        """Mark every step's checkbox ON (used by Run all)."""
        self._syncing = True
        try:
            for _num, group, _e, _d in self._step_entries:
                group.setChecked(True)
        finally:
            self._syncing = False

    def apply_params(self, params: dict) -> None:
        """Push saved values into the spinbox editors (per-tool restore)."""
        for _num, _group, editors, _d in self._step_entries:
            for name, editor in editors.items():
                if name not in params:
                    continue
                try:
                    if isinstance(editor, QSpinBox):
                        editor.setValue(int(params[name]))
                    elif isinstance(editor, QDoubleSpinBox):
                        editor.setValue(float(params[name]))
                except (TypeError, ValueError):
                    pass

    def set_debug_visible(self, visible: bool) -> None:
        for _num, _g, _e, dlabel in self._step_entries:
            if dlabel is not None:
                dlabel.setVisible(visible)

    def collect(self) -> Tuple[int, Set[int], Dict[str, object]]:
        """Return (max_step, checked_step_numbers, flat_params_dict).

        - max_step: highest checked step number (0 if none checked)
        - checked_steps: set of step numbers whose group is checked (drives which
          debug MRCs auto-open)
        - params: flat dict of all parameter values across all steps; the runner
          forwards them to the tool as CLI flags. Values from unchecked steps
          are still included (the tool uses its max_step guard to skip them).
        """
        max_step = 0
        checked: Set[int] = set()
        params: Dict[str, object] = {}
        for step_num, group, editors, _dlabel in self._step_entries:
            if group.isChecked():
                checked.add(step_num)
                if step_num > max_step:
                    max_step = step_num
            for name, editor in editors.items():
                if isinstance(editor, QSpinBox):
                    params[name] = int(editor.value())
                elif isinstance(editor, QDoubleSpinBox):
                    params[name] = float(editor.value())
        return max_step, checked, params


def _find_tab_ancestor(widget: QWidget):
    """Walk up parents until a QTabWidget is found; None if not hosted in one."""
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QTabWidget):
            return parent
        parent = parent.parentWidget()
    return None


def _make_path_field(edit: QLineEdit) -> None:
    """Prepare a QLineEdit to hold long filesystem paths inside a narrow dock.

    - Narrow minimum width so the Plugin tab can shrink without forcing a
      horizontal scrollbar.
    - Tail-display: when the stored text overflows the visible width we want
      the end of the path visible (filename, not drive prefix), so we leave the
      cursor at the end after any programmatic `setText`.
    """
    edit.setMinimumWidth(60)


def _set_path_text(edit: QLineEdit, text: str) -> None:
    """setText + cursor-to-end so the path tail is visible when it overflows."""
    edit.setText(text)
    edit.setCursorPosition(len(text))


def _read_mrc_voxel(path: Path) -> float:
    """Return X voxel size (Å) from an MRC file header, or 0.0 on failure.

    Reads only the 56 header bytes we need: NX (int32 @ 0), cella.x (float32 @ 40).
    voxel_size = cella.x / NX. Avoids pulling in mrcfile just for this lookup.
    """
    try:
        with open(path, "rb") as f:
            hdr = f.read(56)
        if len(hdr) < 52:
            return 0.0
        nx = struct.unpack("<i", hdr[0:4])[0]
        cella_x = struct.unpack("<f", hdr[40:44])[0]
        if nx <= 0 or not cella_x or cella_x <= 0:
            return 0.0
        return float(cella_x) / float(nx)
    except Exception:
        return 0.0


def _read_final_step_voxel(spec: dict, out_prefix: Path) -> float:
    """Voxel size (Å) of the pipeline's final-step MRC, or 0.0 if none exist.

    The final step's MRC is already binned and aligned with STAR coords, so
    its header voxel is exactly what the ParticleList needs — no extra bin
    multiplication, no risk of header/param disagreement on the input side.
    """
    steps = TOOL_SCHEMAS[spec["tool"]]
    if not steps:
        return 0.0
    final = steps[-1]
    for out in final.outputs:
        path = out_prefix.parent / (out_prefix.name + out.pattern)
        if path.is_file():
            v = _read_mrc_voxel(path)
            if v > 0:
                return v
    return 0.0


def _single_step_for(default: float) -> float:
    """Pick a spinbox single-step that matches the parameter's magnitude."""
    mag = abs(default)
    if mag >= 100:
        return 10.0
    if mag >= 10:
        return 1.0
    if mag >= 1:
        return 0.1
    if mag >= 0.1:
        return 0.01
    return 0.001
