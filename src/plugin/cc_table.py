# vim: set expandtab shiftwidth=4 softtabstop=4:
"""Connected-components panel: horizontal CC table + edit/delete/save actions.

Binds to the current ArtiaX active ParticleList (one at a time). Column-select
in the table mirrors into `partlist.selected_particles` (a boolean numpy mask)
so the 3D view highlights the same particles, matching right-click select.

Delete routes through `partlist.delete_data(ids)` — the high-level API that
cleans up `_map`, `_data`, markers, and the collection model atomically.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from Qt.QtCore import QAbstractTableModel, QModelIndex, Qt
from Qt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
)

from . import star_io
from ..ArtiaX import SEL_PARTLIST_CHANGED


class _CCModel(QAbstractTableModel):
    """Horizontal layout: 2 rows (CC ID / # particles), one column per CC."""

    V_HEADERS = ["CC ID", "# particles"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: List[tuple] = []  # (cc_id, count), sorted by count desc

    def set_rows(self, rows) -> None:
        self.beginResetModel()
        self._rows = sorted(rows, key=lambda r: (-r[1], r[0]))
        self.endResetModel()

    def clear(self) -> None:
        self.set_rows([])

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else 2

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Vertical:
            return self.V_HEADERS[section]
        return None  # horizontal header hidden in view

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        cc_id, count = self._rows[index.column()]
        if index.row() == 0:
            return "ALL" if cc_id == 0 else str(cc_id)
        return str(count)

    def cc_id_at_column(self, col: int) -> int:
        return self._rows[col][0]


class ConnectedComponentsPanel(QGroupBox):
    def __init__(self, session, append_log: Callable[[str], None], parent=None):
        super().__init__("Connected Components (active ParticleList)", parent)
        self.session = session
        self._append = append_log
        self._groups: Dict[int, List] = {}
        self._build_ui()
        self.session.ArtiaX.triggers.add_handler(
            SEL_PARTLIST_CHANGED, self._on_partlist_changed
        )
        self._refresh()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        self._model = _CCModel(self)
        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QAbstractItemView.SelectColumns)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.horizontalHeader().setVisible(False)
        # Fixed row heights keyed to font metrics so _autofit_table_height is
        # accurate before the first paint (ResizeToContents returns 0 before
        # the widget is rendered, which leaves a blank band under the headers
        # until the user loads a STAR).
        vh = self._table.verticalHeader()
        vh.setSectionResizeMode(QHeaderView.Fixed)
        vh.setDefaultSectionSize(self._table.fontMetrics().height() + 6)
        self._table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        layout.addWidget(self._table)

        self._summary = QLabel("Selection: 0 CCs · 0 particles")
        layout.addWidget(self._summary)
        self._table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        # Edit row. Tooltips live on both labels and spinboxes so hovering
        # anywhere along the pair surfaces the help.
        shift_z_tip = "Shift (voxels) along +Z applied to every selected particle."
        shift_n_tip = (
            "Shift (voxels) along each particle's own outward normal "
            "(RELION ZYZ: n = (−cos ψ · sin τ, sin ψ · sin τ, cos τ))."
        )
        flip_tip = "Flip each particle's normal: tilt → 180−tilt, psi → psi+180."

        edit_row = QHBoxLayout()
        shift_z_label = QLabel("shift-z (voxel)")
        shift_z_label.setToolTip(shift_z_tip)
        edit_row.addWidget(shift_z_label)
        self._shift_z_spin = self._make_spin(shift_z_tip)
        edit_row.addWidget(self._shift_z_spin, 1)
        edit_row.addSpacing(6)
        shift_n_label = QLabel("shift-along-normal (voxel)")
        shift_n_label.setToolTip(shift_n_tip)
        edit_row.addWidget(shift_n_label)
        self._shift_n_spin = self._make_spin(shift_n_tip)
        edit_row.addWidget(self._shift_n_spin, 1)
        edit_row.addSpacing(6)
        self._flip_cb = QCheckBox("flip-normal")
        self._flip_cb.setToolTip(flip_tip)
        edit_row.addWidget(self._flip_cb)
        layout.addLayout(edit_row)

        # Single action row: 4 buttons evenly distributed.
        act_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply edits")
        self._apply_btn.clicked.connect(self._on_apply_edits)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._on_delete_selected)
        self._save_btn = QPushButton("Save (overwrite)")
        self._save_btn.setToolTip("Overwrite the original STAR file (no .bak)")
        self._save_btn.clicked.connect(self._on_save)
        self._save_as_btn = QPushButton("Save As…")
        self._save_as_btn.clicked.connect(self._on_save_as)
        act_row.addStretch(1)
        act_row.addWidget(self._apply_btn)
        act_row.addStretch(1)
        act_row.addWidget(self._delete_btn)
        act_row.addStretch(1)
        act_row.addWidget(self._save_btn)
        act_row.addStretch(1)
        act_row.addWidget(self._save_as_btn)
        act_row.addStretch(1)
        layout.addLayout(act_row)

        self.setLayout(layout)

    @staticmethod
    def _make_spin(tooltip: str = "") -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(-1e6, 1e6)
        sb.setDecimals(3)
        sb.setSingleStep(1.0)
        sb.setValue(0.0)
        sb.setMinimumWidth(30)
        if tooltip:
            sb.setToolTip(tooltip)
        return sb

    # ------------------------------------------------------------- state

    def _active_partlist(self):
        artia = self.session.ArtiaX
        pid = artia.selected_partlist
        return None if pid is None else artia.partlists.get(pid)

    def _on_partlist_changed(self, _name, _data) -> None:
        self._refresh()

    def _refresh(self) -> None:
        pl = self._active_partlist()
        if pl is None:
            self._groups = {}
            self._model.clear()
            self._autofit_table_height()
            self._on_selection_changed()
            return
        self._groups = star_io.group_particles_by_cc(pl)
        self._model.set_rows([(cc, len(pids)) for cc, pids in self._groups.items()])
        self._autofit_table_height()
        self._on_selection_changed()

    def _autofit_table_height(self) -> None:
        """Pin the table height to exactly 2 content rows (+ frame, + hscrollbar
        when columns exist). Row size is taken from `defaultSectionSize`, which
        is valid before the widget is painted — `verticalHeader().length()`
        returns 0 pre-render, which is why an empty band used to show up below
        the headers until the first STAR was loaded."""
        rh = self._table.verticalHeader().defaultSectionSize()
        row_h = rh * self._model.rowCount()
        frame = 2 * self._table.frameWidth()
        hbar = (
            self._table.horizontalScrollBar().sizeHint().height()
            if self._model.columnCount() > 0
            else 0
        )
        self._table.setFixedHeight(row_h + frame + hbar)

    def _selected_cc_ids(self) -> List[int]:
        cols = self._table.selectionModel().selectedColumns()
        return [self._model.cc_id_at_column(idx.column()) for idx in cols]

    def _selected_particle_ids(self) -> List:
        out: List = []
        for cc in self._selected_cc_ids():
            out.extend(self._groups.get(cc, []))
        return out

    def _on_selection_changed(self, *_args) -> None:
        pids = self._selected_particle_ids()
        n_ccs = len(self._selected_cc_ids())
        self._summary.setText(f"Selection: {n_ccs} CCs · {len(pids)} particles")
        # Mirror the table selection onto the ParticleList → highlights in the
        # 3D view, same effect as right-click-select in ChimeraX.
        pl = self._active_partlist()
        if pl is None or pl.size == 0:
            return
        all_ids = pl.particle_ids  # numpy array of uuid strings, ordered
        # Dict lookup stays O(1) per particle, so building the mask is O(N+M)
        # regardless of partlist size — faster than `pid in set` iteration for
        # 50k+ particles, and faster than np.isin over object-dtype arrays.
        pos = {pid: i for i, pid in enumerate(all_ids)}
        mask = np.zeros(len(all_ids), dtype=bool)
        for pid in pids:
            idx = pos.get(pid)
            if idx is not None:
                mask[idx] = True
        try:
            pl.selected_particles = mask
        except Exception as exc:
            self._append(f"[warn] failed to sync selection: {exc}\n")

    # ----------------------------------------------------------- actions

    def _on_apply_edits(self) -> None:
        pl = self._active_partlist()
        if pl is None:
            self._append("(no active ParticleList)\n")
            return
        pids = self._selected_particle_ids()
        if not pids:
            self._append("(no CC selected)\n")
            return

        shift_z = float(self._shift_z_spin.value())
        shift_n = float(self._shift_n_spin.value())
        flip = self._flip_cb.isChecked()
        data = pl.data
        # origin_coord is stored in physical Å (pos_x × pixelsize_ori), so UI
        # shifts expressed in voxels must be converted before they're added.
        # Without this multiplication, 200 "voxels" was silently treated as
        # 200 Å and only moved the particle by ~20 voxels at typical pixelsizes.
        voxel_A = float(pl.origin_pixelsize)
        dz_A = shift_z * voxel_A
        dn_A = shift_n * voxel_A

        for pid in pids:
            p = data[pid]
            if shift_z != 0.0 or shift_n != 0.0:
                x, y, z = p.origin_coord
                if shift_z != 0.0:
                    z += dz_A
                if shift_n != 0.0:
                    nx, ny, nz = _normal_zyz_xyz(
                        float(p["ang_1"]), float(p["ang_2"]), float(p["ang_3"])
                    )
                    x += nx * dn_A
                    y += ny * dn_A
                    z += nz * dn_A
                p.origin_coord = (x, y, z)
            if flip:
                # Invert the particle's own Z axis: tilt' = 180−tilt, psi' = psi+180.
                p["ang_2"] = 180.0 - float(p["ang_2"])
                p["ang_3"] = float(p["ang_3"]) + 180.0

        pl.update_places()
        from ..particle.ParticleList import PARTLIST_CHANGED
        pl.triggers.activate_trigger(PARTLIST_CHANGED, pl)
        self._append(f"[applied] {len(pids)} particles modified\n")

    def _on_delete_selected(self) -> None:
        pl = self._active_partlist()
        if pl is None:
            self._append("(no active ParticleList)\n")
            return
        pids = self._selected_particle_ids()
        if not pids:
            self._append("(no CC selected)\n")
            return
        # Use the high-level delete that cleans up _map + markers + collection
        # model atomically; pl.data.delete_particles(…) alone leaves stale
        # markers in the 3D view.
        pl.delete_data(list(pids))
        self._append(f"[deleted] {len(pids)} particles\n")
        self._refresh()

    def _on_save(self) -> None:
        pl = self._active_partlist()
        if pl is None:
            self._append("(no active ParticleList)\n")
            return
        path = getattr(pl.data, "file_name", None)
        if not path:
            self._append("ERROR: no STAR path on current ParticleList; use Save As….\n")
            return
        self._save_to(pl, str(path))

    def _on_save_as(self) -> None:
        pl = self._active_partlist()
        if pl is None:
            self._append("(no active ParticleList)\n")
            return
        current = getattr(pl.data, "file_name", None) or str(Path.home())
        path, _ = QFileDialog.getSaveFileName(
            self, "Save STAR as", str(current), "STAR (*.star)"
        )
        if not path:
            return
        if not path.endswith(".star"):
            path += ".star"
        self._save_to(pl, path)

    def _save_to(self, pl, path: str) -> None:
        try:
            from chimerax.artiax.io import save_particle_list
            save_particle_list(
                self.session, path, pl, format_name="RELION STAR file"
            )
        except Exception as exc:
            self._append(f"ERROR: save failed: {exc}\n")
            return
        self._append(f"[saved] {path}\n")


def _normal_zyz_xyz(rot_deg: float, tilt_deg: float, psi_deg: float) -> tuple:
    """Particle's Z axis in lab frame, per RELION ZYZ convention
    (A·(0,0,1) = (−cos ψ · sin τ,  sin ψ · sin τ,  cos τ))."""
    t = math.radians(tilt_deg)
    psi = math.radians(psi_deg)
    return (
        -math.cos(psi) * math.sin(t),
        math.sin(psi) * math.sin(t),
        math.cos(t),
    )
