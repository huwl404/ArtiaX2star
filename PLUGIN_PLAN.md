# ArtiaX × AIS2star Plugin

Plugin tab under **ArtiaX Options**. Drives AIS2star's fiber2star / mem2star
pipelines step-by-step against a filament conda env (through `QProcess`),
auto-opens each step's debug MRC, and hands the final STAR to ArtiaX's
ParticleList stack for visualisation + per-connected-component editing.

```
Plugin tab  →  env picker · tool radio · inputs · pipeline editor
            →  [Run to last checked | Run all | Stop]  +  log tail
            →  Particle List controls (voxel / marker / axes)
            →  Connected Components table · edit · save

filament conda env (subprocess)
     │ run_step.py --tool X --steps 1-K --input M --params-json J [--debug]
     └── src/plugin/ais2star/{fiber2star,mem2star}.py
                → _*.mrc debug files · *_particles.star
```

ChimeraX's Python never imports cupy/cucim. All heavy work runs in the env
the user selects.

---

## Files

| Path | Role |
|---|---|
| `src/options_window.py` | Register Plugin tab on the Options QTabWidget |
| `src/plugin/plugin_widget.py` | Plugin tab UI + run loop |
| `src/plugin/env.py` | Conda env discovery · CUDA / cupy / cucim probe · QSettings |
| `src/plugin/schema.py` | Steps + parameters (kind, default, step, suffix, three-segment tooltip, debug outputs) |
| `src/plugin/runner.py` | `QProcess + QEventLoop` driver · `format_equivalent_cli` · auto-open |
| `src/plugin/star_io.py` | Group active ParticleList's particles by `aisCcId` |
| `src/plugin/cc_table.py` | Connected Components panel (table · edit · delete · save) |
| `src/plugin/ais2star/{fiber2star,mem2star}.py` | Copies of AIS2star; add `--bin` (fiber), `--max-step`, `aisCcId` column |
| `src/plugin/ais2star/run_step.py` | Thin CLI driver |

AIS2star upstream repo is never modified; `src/plugin/ais2star/` is the owned
copy the runner spawns.

---

## Pipeline steps

### fiber2star (8)
1. **Load & Bin** — `voxel-size`, `bin`
2. **Smooth** — `sigma-z`, `sigma-xy` → `_1_smoothed.mrc`
3. **K-mode Direction** — `high-threshold`, `mode2-ratio-thr` → `_2_high_mask.mrc`, `_2_dir_mode{0,1}.mrc`
4. **Binary Mask + Opening** — `threshold`, `opening-radius` → `_3_opened.mrc`
5. **Elongation Filter** — `min-aspect-ratio`, `direction-filter-angle` → `_4_elongated.mrc`, `_4_rejected.mrc`
6. **Skeletonize** — `bridge-radius` → `_5_skel_raw`, `_6_skel_bridged`, `_7_branches_mode`, `_8_segments_mode{0,1}`
7. **Greedy B-spline** — `fiber-diameter`, `erase-radius(-scale)`, `min-fiber-length`, `spline-smoothness`, `signal-weight`, `bending-weight`, `curvature-consistency-weight` → `_9_fibers_mode{0,1}.mrc`
8. **Sample + Write STAR** — `spacing`, `shift-z` → `_particles.star`; auto-open surfaces step 7's fiber MRCs

`min-candidate-score` is not exposed (generalises poorly).

### mem2star (6)
1. **Load & Bin** — `voxel-size`, `bin`
2. **Smooth** — `sigma-z`, `sigma-xy`, `border-width` → `_01_smoothed.mrc`
3. **Ridge Midband** — `mask-hessian-sigma`, `mask-normal-smooth-sigma`, `prob-min`, `sheet-saliency-threshold`, `min-component-size` → `_02_roi`, `_03a/b/c_*`, `_04a/b/c/d_*`
4. **Junction Cut** — `cut-junction-sigma`, `cut-junction-threshold`, `midband-close-iters` → `_05_junction_field`, `_06a/b_cut_*`, `_07a–e_midband_*`
5. **Estimate Outward Normals** — no params → `_07f_normals_oriented_bs.mrc`
6. **FPS Sample + Write STAR** — `spacing`, `shift-z`, `shift-along-normal` → `_08_sample_points.mrc`, `_particles.star`

Each parameter's tooltip is three-segment: function sentence · `↑ :` increase
effect · `↓ :` decrease effect.

---

## Behavior

### Run
- **Run to last checked** — run steps 1..K where K is the highest-checked step group. Debug on + single mode → MRCs of the last checked step auto-open.
- **Run all** — checks every step, runs 1..N, auto-opens last step.
- **Stop** — `QProcess.kill()`; in batch mode aborts the whole queue.
- Test env is not required before Run; a failed run prints a hint pointing to Test env.

### Pipeline checkboxes (radio-like)
Clicking step K → steps 1..K checked, K+1..N unchecked. Unchecking K → K..N unchecked.

### Single vs Batch
- Single: one MRC input. Auto-open runs after success if **Auto-load produced STAR** is on.
- Batch: file list; Debug is forced off (disk blow-up). No auto-open; end-of-run summary lists failed files only.

### Auto-open display style
Schema `preview_hint` governs the `level` suffix:
- `prob01` → `level 0.99` if `volume.matrix_value_statistics().maximum ≤ 1.01`; else no override
- `binary` / `label` → `level 0.5`
- `keep` → no level override

### Particle List
- **Auto-load** (single + Auto-load checked + max_step == final): load `_particles.star`; Voxel ← input MRC header × bin (or `--voxel-size` × bin); Marker = 2×vx; Axes = 4×vx. All applied once at open.
- **Load STAR…** — stage a path; user sets Voxel and clicks **Apply** to commit the load. Voxel defaults to last value (QSettings).
- **Apply** — one button writes voxel → `origin_pixelsize`, marker → `radius`, axes → `axes_size` on the active ParticleList.
- Active ParticleList tracked via `SEL_PARTLIST_CHANGED`.
- Loading a STAR must not jump the Options window to the Visualization tab — plugin saves and restores `QTabWidget.currentIndex()` around the call.

### Connected Components
- Horizontal table: row 0 = `CC ID`, row 1 = `# particles`, one column per CC, sorted by count desc. Horizontal scrollbar on overflow.
- Particles missing the `aisCcId` column fall back to CC 0, which renders as a single **ALL** column — so a vanilla STAR still gets a usable CC panel.
- Column selection mirrors to `partlist.selected_particles = boolean_mask` — equivalent of ChimeraX right-click select, highlights in the 3D view.
- **Apply edits** — shift-z (voxel), shift-along-normal (voxel, RELION ZYZ convention: `n = (−cos ψ · sin τ, sin ψ · sin τ, cos τ)`), flip-normal (tilt → 180−tilt, psi → psi+180). Writes via `particle.origin_coord` and `particle["ang_2/3"]`; redraw via `partlist.update_places() + PARTLIST_CHANGED`.
- **Delete** — `partlist.delete_data(ids)`; cleans up `_map / _data / markers / collection_model` atomically.
- **Save (overwrite)** — `save_particle_list(session, pl.data.file_name, pl, "RELION STAR file")`. No `.bak`, no confirmation. `aisCcId` survives the round-trip via `RELIONParticleData.as_dictionary` → all `_data_keys`.
- **Save As…** — same call, new path; `.star` appended if missing.

### Persistence (QSettings, `plugin/` namespace)
- `env_python` · `last_voxel_size` · `debug_mode` · `auto_load`
- `last_tool` · `last_input` · `last_output_dir` · `last_out_prefix`
- `plugin/{tool}/params` — JSON blob of last successful-run parameter values per tool

Restored in `_restore_session_state()` at widget init; saved in
`_save_session_state(spec)` on every successful run.

### Command mirror
Every successful run writes a fully expanded one-line CLI (params inlined,
paths shlex-quoted) to ChimeraX Log via `runner.format_equivalent_cli`.

---

## Non-goals

- No `artiax plugin run …` ChimeraX command layer. UI is the only driver.
- No heavy deps installed into ChimeraX's Python. All cupy/cucim lives in the filament env.
- No `.bak` backups, no confirmation dialog on Save.
- No intermediate-result disk cache (full run is ~20–30 s, fast enough).
- No anisotropic voxel size; one value covers X = Y = Z.
- No multi-ParticleList view in the CC panel — one active list at a time.

---

## Dev loop

```bash
# First time:
chimerax --cmd "devel install /home/user/code/ArtiaX2star ; exit"

# Every code change:
chimerax   # restart — devel install softlinks src/, no wheel rebuild
```

Smoke a run_step subprocess directly from the filament env:
```bash
$FILAMENT_PY src/plugin/ais2star/run_step.py \
  --tool mem2star --steps 1-3 --input test.mrc --debug --out-prefix /tmp/x
```

---

## Qt traps (if extending)

- **Don't use `QStackedWidget`** for size-adaptive panes — `QStackedLayout.sizeHint` is the union of all children, and `QSizePolicy.Ignored` on hidden pages doesn't help. Use `QVBoxLayout + setVisible(True/False)`; `QBoxLayout::sizeHint` skips hidden items (`QLayoutItem::isEmpty` returns true for hidden widgets).
- **Subprocess log live-stream**: pass `-u` AND set `PYTHONUNBUFFERED=1` — Python's block-buffering on pipes hides output until exit otherwise.
- **`_populate_envs` subprocess must defer** (`QTimer.singleShot(0, …)`). Running it inside `__init__` lets Marker Placement steal the dock tab.
- **`setStyleSheet('color: …')` on a QLabel resets font inheritance** → the label renders at default size, not the ArtiaX small font. Use HTML `<i>…</i>` instead.
- **`add_particlelist` jumps to Visualization tab** via `OPTIONS_PARTLIST_CHANGED`. Save/restore `QTabWidget.currentIndex()` around any open_partlist / auto-load call.
- **Widget min-widths**: narrow docks clip the Plugin tab if any row forces a large `minimumSize`. Use short labels and `spinbox.setMinimumWidth(30)`.

---

## Files & anchors

| Target | Location |
|---|---|
| Tab registration | `src/options_window.py` — `_build_full_ui` adds `self.tabs.addTab(self.plugin_area, 'Plugin')`; `_build_plugin_widget` passes `self.font` |
| Triggers consumed | `SEL_PARTLIST_CHANGED` (`src/ArtiaX.py:74–94`) |
| STAR loader | `session.ArtiaX.open_partlist(path, "RELION STAR file")` |
| STAR saver | `chimerax.artiax.io.save_particle_list(session, path, pl, format_name=…)` |
| ParticleList setters | `pl.origin_pixelsize`, `pl.radius`, `pl.axes_size`, `pl.selected_particles = bool_mask` |
| Particle editors | `p.origin_coord = (x,y,z)`; `p["ang_1/2/3"]`; `p["aisCcId"]` |
| Particle redraw | `pl.update_places()` + `pl.triggers.activate_trigger(PARTLIST_CHANGED, pl)` |
| Particle delete | `pl.delete_data(list(ids))` |
