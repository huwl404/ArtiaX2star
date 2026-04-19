# vim: set expandtab shiftwidth=4 softtabstop=4:
"""Pipeline step schemas for fiber2star and mem2star.

Drives the Plugin-tab UI:
  - One QGroupBox per StepSpec, checkable (checked == "run through this step").
  - Param rows inside, built from ParamSpec (kind, default, three-segment tooltip).
  - After a run, the runner auto-opens each checked step's OutputSpec MRCs and
    applies a display style based on `hint` ("prob01" | "binary" | "label" | "keep").

Tooltip convention (enforced by convention, not code):
    Line 1: function — what this param controls in this step's behavior.
    Line 2: "↑ : <half-sentence on the effect / cost of increasing>"
    Line 3: "↓ : <half-sentence on the effect / cost of decreasing>"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass(frozen=True)
class ParamSpec:
    name: str       # argparse dest (underscored), e.g. "sigma_z"
    label: str      # display label, e.g. "sigma-z"
    kind: str       # "float" | "int"
    default: Any
    tooltip: str    # three-segment string; see module docstring
    step: float = 0.0  # spinbox single-step; 0 means pick from default magnitude
    suffix: str = ""   # unit appended to label, e.g. "Å", "voxels", "°"


@dataclass(frozen=True)
class OutputSpec:
    pattern: str    # suffix appended to out_prefix, e.g. "_01_smoothed.mrc"
    hint: str       # "prob01" | "binary" | "label" | "keep"


@dataclass(frozen=True)
class StepSpec:
    number: int
    name: str
    params: List[ParamSpec] = field(default_factory=list)
    outputs: List[OutputSpec] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Shared parameter tooltips
# ─────────────────────────────────────────────────────────────────────────────

_VOXEL_SIZE_TIP = (
    "Voxel size (Å) used throughout the pipeline.\n"
    "↑ : a positive value overrides whatever the MRC header stores.\n"
    "↓ : 0 or negative reads the voxel size from the MRC header."
)

_BIN_TIP = (
    "Integer downsampling factor applied right after load; subsequent steps and "
    "STAR coordinates live in binned space.\n"
    "↑ : fewer voxels, much faster; fine detail lost.\n"
    "↓ : full resolution; slower, more GPU memory."
)


# ─────────────────────────────────────────────────────────────────────────────
# fiber2star — 8 steps
# ─────────────────────────────────────────────────────────────────────────────

FIBER2STAR_STEPS: List[StepSpec] = [
    StepSpec(
        number=1, name="Load & Bin",
        params=[
            ParamSpec("voxel_size", "voxel-size", "float", -1.0, _VOXEL_SIZE_TIP, suffix="Å"),
            ParamSpec("bin", "bin", "int", 1, _BIN_TIP),
        ],
        outputs=[],
    ),
    StepSpec(
        number=2, name="Smooth",
        params=[
            ParamSpec(
                "sigma_z", "sigma-z", "float", 1.5,
                "Gaussian σ along Z to bridge discontinuities between slices.\n"
                "↑ : smoother, bridges broken fibers across slices; thin fibers blur out.\n"
                "↓ : preserves fine fibers; Z-streak artifacts leak through.",
                suffix="voxels",
            ),
            ParamSpec(
                "sigma_xy", "sigma-xy", "float", 0.0,
                "Gaussian σ in the XY plane, usually 0.\n"
                "↑ : denoises in-plane; thin filaments thicken and may fuse.\n"
                "↓ : no in-plane smoothing (default).",
                step=0.1, suffix="voxels",
            ),
        ],
        outputs=[OutputSpec("_1_smoothed.mrc", "keep")],
    ),
    StepSpec(
        number=3, name="K-mode Direction",
        params=[
            ParamSpec(
                "high_threshold", "high-threshold", "float", 254.0,
                "Threshold on the input volume selecting high-confidence voxels for direction PCA.\n"
                "↑ : only the most confident voxels vote; may miss weak fiber systems.\n"
                "↓ : more voxels vote; noisier direction estimate."
            ),
            ParamSpec(
                "mode2_ratio_thr", "mode2-ratio-thr", "float", 0.25,
                "Eigenvalue ratio λ2/λ1 that triggers activating a second mode (K=2).\n"
                "↑ : harder to split; tends to produce one mode only.\n"
                "↓ : more sensitive; splits into two orthogonal modes earlier."
            ),
        ],
        outputs=[
            OutputSpec("_2_high_mask.mrc", "binary"),
            OutputSpec("_2_dir_mode0.mrc", "keep"),
            OutputSpec("_2_dir_mode1.mrc", "keep"),
        ],
    ),
    StepSpec(
        number=4, name="Binary Mask + Opening",
        params=[
            ParamSpec(
                "threshold", "threshold", "float", 252.0,
                "Threshold on the input volume for the main binary mask.\n"
                "↑ : keeps only high-confidence voxels; holes along thin fibers.\n"
                "↓ : catches weaker signal; more spurious mask regions."
            ),
            ParamSpec(
                "opening_radius", "opening-radius", "int", 3,
                "Ball opening radius (voxels) to remove thin bridges between CCs.\n"
                "↑ : removes thicker bridges; may also erode real fibers.\n"
                "↓ : keeps thin features; bridges remain.",
                suffix="voxels",
            ),
        ],
        outputs=[OutputSpec("_3_opened.mrc", "binary")],
    ),
    StepSpec(
        number=5, name="Elongation Filter",
        params=[
            ParamSpec(
                "min_aspect_ratio", "min-aspect-ratio", "float", 2.0,
                "Minimum CC length / cross-span aspect ratio to count as elongated.\n"
                "↑ : keeps only very elongated CCs; rejects short candidates.\n"
                "↓ : keeps shorter CCs; admits more noise."
            ),
            ParamSpec(
                "direction_filter_angle", "direction-filter-angle", "float", 20.0,
                "Max angle (deg) from mode direction for CCs and skeleton branches to be kept.\n"
                "↑ : wider accepted cone; more off-axis noise.\n"
                "↓ : stricter; only CCs closely aligned with mode survive.",
                suffix="°",
            ),
        ],
        outputs=[
            OutputSpec("_4_elongated.mrc", "binary"),
            OutputSpec("_4_rejected.mrc", "label"),
        ],
    ),
    StepSpec(
        number=6, name="Skeletonize",
        params=[
            ParamSpec(
                "bridge_radius", "bridge-radius", "int", 3,
                "Dilation radius (voxels) used to bridge skeleton micro-gaps before branching.\n"
                "↑ : bridges larger gaps; may fuse parallel skeletons.\n"
                "↓ : preserves separation; more broken skeletons.",
                suffix="voxels",
            ),
        ],
        outputs=[
            OutputSpec("_5_skel_raw.mrc", "binary"),
            OutputSpec("_6_skel_bridged.mrc", "binary"),
            OutputSpec("_7_branches_mode.mrc", "label"),
            OutputSpec("_8_segments_mode0.mrc", "label"),
            OutputSpec("_8_segments_mode1.mrc", "label"),
        ],
    ),
    StepSpec(
        number=7, name="Greedy B-spline",
        params=[
            ParamSpec(
                "fiber_diameter", "fiber-diameter", "float", 500.0,
                "Expected fiber diameter (Å); sets the local erase / absorption radius.\n"
                "↑ : spans small gaps; risks fusing close-packed fibers.\n"
                "↓ : separates adjacent fibers cleanly; may cut one fiber at thin spots.",
                suffix="Å",
            ),
            ParamSpec(
                "erase_radius", "erase-radius", "float", -1.0,
                "Absolute erase radius in Å. ≤ 0 means auto from fiber_diameter × erase_radius_scale.\n"
                "↑ : more aggressive erasing; fewer fibers extracted.\n"
                "↓ : less erasing; more overlap; more (possibly redundant) fibers.",
                suffix="Å",
            ),
            ParamSpec(
                "erase_radius_scale", "erase-radius-scale", "float", 1.1,
                "Erase radius = fiber_radius × this scale when erase_radius ≤ 0.\n"
                "↑ : auto erase grows; fewer fibers.\n"
                "↓ : auto erase shrinks; more parallel fibers kept."
            ),
            ParamSpec(
                "min_fiber_length", "min-fiber-length", "float", 1000.0,
                "Minimum arc length (Å) required to accept a fiber candidate.\n"
                "↑ : rejects short fragments; may lose real short filaments near borders.\n"
                "↓ : keeps short filaments; more noise contamination.",
                suffix="Å",
            ),
            ParamSpec(
                "spline_smoothness", "spline-smoothness", "float", 1.5,
                "B-spline smoothing strength.\n"
                "↑ : smoother curves; may cut corners at real bends.\n"
                "↓ : follows skeleton closely; wobbles on noisy segments."
            ),
            ParamSpec(
                "signal_weight", "signal-weight", "float", 15.0,
                "Weight of the signal-confidence term in the greedy candidate score.\n"
                "↑ : favors candidates through high-confidence voxels.\n"
                "↓ : favors long / straight candidates regardless of signal strength."
            ),
            ParamSpec(
                "bending_weight", "bending-weight", "float", 0.05,
                "Weight of the B-spline bending-energy penalty.\n"
                "↑ : penalizes curvy fibers; prefers straight ones.\n"
                "↓ : allows tighter bends; may follow skeletal noise."
            ),
            ParamSpec(
                "curvature_consistency_weight", "curvature-consistency-weight", "float", 0.2,
                "Weight of the curvature-consistency penalty along a candidate.\n"
                "↑ : penalizes abrupt curvature changes; smoother sweeps.\n"
                "↓ : admits kinks along a single fiber."
            ),
        ],
        outputs=[
            OutputSpec("_9_fibers_mode0.mrc", "binary"),
            OutputSpec("_9_fibers_mode1.mrc", "binary"),
        ],
    ),
    StepSpec(
        number=8, name="Sample + Write STAR",
        params=[
            ParamSpec(
                "spacing", "spacing", "float", 40.0,
                "Particle sampling spacing along each fiber (Å).\n"
                "↑ : fewer particles per fiber; sparser along the fiber.\n"
                "↓ : more particles; denser along fibers.",
                suffix="Å",
            ),
            ParamSpec(
                "shift_z", "shift-z", "float", 0.0,
                "Global Z shift added to every STAR coordinate (voxel units).\n"
                "↑ : particles shifted toward +Z.\n"
                "↓ : particles shifted toward -Z.",
                step=1.0, suffix="voxels",
            ),
        ],
        # Full-pipeline preview: show step 7's fiber MRCs alongside the STAR.
        outputs=[
            OutputSpec("_9_fibers_mode0.mrc", "binary"),
            OutputSpec("_9_fibers_mode1.mrc", "binary"),
        ],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# mem2star — 6 steps
# ─────────────────────────────────────────────────────────────────────────────

MEM2STAR_STEPS: List[StepSpec] = [
    StepSpec(
        number=1, name="Load & Bin",
        params=[
            ParamSpec("voxel_size", "voxel-size", "float", -1.0, _VOXEL_SIZE_TIP, suffix="Å"),
            ParamSpec("bin", "bin", "int", 2, _BIN_TIP),
        ],
        outputs=[],
    ),
    StepSpec(
        number=2, name="Smooth",
        params=[
            ParamSpec(
                "sigma_z", "sigma-z", "float", 1.5,
                "Gaussian σ along Z; suppresses per-slice noise before ridge detection.\n"
                "↑ : cleaner midband across bad slices; thin membranes blur out.\n"
                "↓ : preserves fine detail; layer-streak artifacts leak through.",
                suffix="voxels",
            ),
            ParamSpec(
                "sigma_xy", "sigma-xy", "float", 0.0,
                "Gaussian σ in the XY plane, usually 0.\n"
                "↑ : denoises in-plane; membrane edges soften.\n"
                "↓ : no in-plane smoothing (default).",
                step=0.1, suffix="voxels",
            ),
            ParamSpec(
                "border_width", "border-width", "int", 4,
                "Border-stripping width (voxels); zeros a margin around the volume.\n"
                "↑ : more aggressive masking; loses membranes near edges.\n"
                "↓ : keeps edges; risk of boundary artifacts in ridge detection.",
                suffix="voxels",
            ),
        ],
        outputs=[OutputSpec("_01_smoothed.mrc", "keep")],
    ),
    StepSpec(
        number=3, name="Ridge Midband",
        params=[
            ParamSpec(
                "mask_hessian_sigma", "mask-hessian-sigma", "float", 3.0,
                "Hessian σ (voxels) for ridge normal estimation.\n"
                "↑ : smoother normals, more tolerant of noise; spatial resolution of normals drops.\n"
                "↓ : sharper normals; noisier direction on thin ridges.",
                suffix="voxels",
            ),
            ParamSpec(
                "mask_normal_smooth_sigma", "mask-normal-smooth-sigma", "float", 1.5,
                "Smoothing σ (voxels) applied to the n⊗n normal tensor.\n"
                "↑ : more coherent normal field; less sensitive to voxel noise.\n"
                "↓ : follows local normals faithfully; noisier field.",
                suffix="voxels",
            ),
            ParamSpec(
                "prob_min", "prob-min", "float", 32.0,
                "Minimum probability for Hessian to run (loose ROI).\n"
                "↑ : only high-prob regions get normals; misses weak membranes.\n"
                "↓ : covers more signal; more compute, possibly more noise."
            ),
            ParamSpec(
                "sheet_saliency_threshold", "sheet-saliency-threshold", "float", 0.5,
                "Minimum sheet-saliency S for a voxel to count as membrane.\n"
                "↑ : only the crispest membranes pass; more misses.\n"
                "↓ : admits weaker sheets; more spurious fragments."
            ),
            ParamSpec(
                "min_component_size", "min-component-size", "int", 1000,
                "Drop connected components smaller than this (voxels); also used as junction-cut criterion.\n"
                "↑ : discards smaller fragments; may lose small real membranes.\n"
                "↓ : keeps small CCs; more noise.",
                suffix="voxels",
            ),
        ],
        outputs=[
            OutputSpec("_02_roi_thresholded.mrc", "binary"),
            OutputSpec("_03a_saliency.mrc", "prob01"),
            OutputSpec("_03b_saliency_passed.mrc", "binary"),
            OutputSpec("_03c_ridge_passed.mrc", "binary"),
            OutputSpec("_04a_midband_intersected.mrc", "binary"),
            OutputSpec("_04b_midband_dropsmall.mrc", "binary"),
            OutputSpec("_04c_midband_labels.mrc", "label"),
            OutputSpec("_04d_normals_ballstick.mrc", "keep"),
        ],
    ),
    StepSpec(
        number=4, name="Junction Cut",
        params=[
            ParamSpec(
                "cut_junction_sigma", "cut-junction-sigma", "float", 3.0,
                "n⊗n smoothing σ (voxels) used for the junction field; ≈ bridge width.\n"
                "↑ : detects wider junctions; more bridges candidates.\n"
                "↓ : only narrow bridges trigger the detector.",
                suffix="voxels",
            ),
            ParamSpec(
                "cut_junction_threshold", "cut-junction-threshold", "float", 0.07,
                "Threshold on the junction field used to carve bridges (0–1).\n"
                "↑ : cuts fewer bridges; keeps more merged CCs.\n"
                "↓ : cuts more aggressively; may over-split a single membrane."
            ),
            ParamSpec(
                "midband_close_iters", "midband-close-iters", "int", 10,
                "Binary-closing iterations after junction cut to heal small holes.\n"
                "↑ : closes larger holes; may re-bridge separated pieces.\n"
                "↓ : preserves cuts exactly; small holes remain."
            ),
        ],
        outputs=[
            OutputSpec("_05_junction_field.mrc", "prob01"),
            OutputSpec("_06a_cut_thresholded.mrc", "binary"),
            OutputSpec("_06b_cut_separating_only.mrc", "binary"),
            OutputSpec("_07a_midband_cut.mrc", "binary"),
            OutputSpec("_07b_midband_holefilled.mrc", "binary"),
            OutputSpec("_07c_midband_dropsmall.mrc", "binary"),
            OutputSpec("_07d_midband_labels.mrc", "label"),
            OutputSpec("_07e_midband_smoothed.mrc", "binary"),
        ],
    ),
    StepSpec(
        number=5, name="Estimate Outward Normals",
        params=[],  # uses fields computed in step 3/4
        outputs=[OutputSpec("_07f_normals_oriented_bs.mrc", "keep")],
    ),
    StepSpec(
        number=6, name="FPS Sample + Write STAR",
        params=[
            ParamSpec(
                "spacing", "spacing", "float", 200.0,
                "Target particle spacing (Å) on the membrane surface.\n"
                "↑ : fewer particles; sparser sampling.\n"
                "↓ : more particles; denser sampling.",
                suffix="Å",
            ),
            ParamSpec(
                "shift_z", "shift-z", "float", 0.0,
                "Global Z shift (voxel units) added to every STAR coordinate.\n"
                "↑ : particles shifted toward +Z.\n"
                "↓ : particles shifted toward -Z.",
                step=1.0, suffix="voxels",
            ),
            ParamSpec(
                "shift_along_normal", "shift-along-normal", "float", 0.0,
                "Per-particle shift (voxel units) along its own surface normal.\n"
                "↑ : particles pushed outward from the membrane.\n"
                "↓ : particles pushed inward (opposite of local normal).",
                step=1.0, suffix="voxels",
            ),
        ],
        outputs=[OutputSpec("_08_sample_points.mrc", "binary")],
    ),
]


TOOL_SCHEMAS = {
    "fiber2star": FIBER2STAR_STEPS,
    "mem2star": MEM2STAR_STEPS,
}
