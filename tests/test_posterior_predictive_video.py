"""The posterior-predictive video check never overwrites, never renders beyond its limits unasked,
and names what its synthetic is made from.

The files of the check are listed in
``Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.md``. The engine, the
viewer notebook and the companions below are tested here.

The engine (``posterior_predictive_video_runner``):

1. a render refuses, before any work, to write over an existing clip, figure or trajectory; the stem
   of a declared-configuration render carries neither the nuisance tag nor the configuration, so
   another attempt or a variant needs its own run label, which gives it a distinct stem;
2. ``simulate_and_render`` refuses an existing trajectory instead of deleting it, and the engine
   deletes no file;
3. the comparison figure names the synthetic source in the clip file's words (a declared-configuration
   render shows no "MAP", and a MAP selection appears only when a MAP was read) and states the receptors
   the render actually started with.

The receptor-total match
(``Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Count_Match.py``).
``iterate_count`` receives its renders from a callable, so every path runs on synthetic detection
responses, with no simulation:

4. a linear detection response converges to the value the arithmetic predicts (the cell-0 case);
5. no experimental detections, no visible host, no synthetic detection and a non-finite update each
   end in their own undefined status, with no count;
6. a saturating detection response ends as "unresolved under detector saturation" (inside the prior
   box, and beyond it under the user's decision) instead of chasing an unattainable density;
7. an exhausted iteration budget and a count above --max-count carry no count, and no render, the
   first included, exceeds the ceiling;
8. nothing is rendered outside the prior box without the user's decision: not the first count, not
   an update, not the converged value. Under that decision a count outside it is accepted, flagged,
   not clipped. The render guard, which also covers the check renders, refuses before rendering;
9. an existing result is refused before any computation; an attempt label gives a distinct folder.

The player render (``Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Player.py``):

10. the headless copy of the viewer notebook differs from it only in the clip path (and the settings
    given) and in dropping the scrubber's widget call, whose helpers stay for the player; the player
    file is named after the clip; a truncated player is detected from the notebook's report.

The display windows (engine, viewer notebook, player script):

11. the four windows (percentile, the default at (0, 99.99); experimental; synthetic; full) are one window shared by
    every panel over all frames; each clip's pixels below and above it are counted; the engine, the
    notebook and the player list the same modes, and the notebook's windows equal the engine's;
12. the comparison figure is drawn from the clip alone: the labels rebuilt from a clip are the render's
    (declared or MAP reaction-diffusion, fixed or MAP imaging, pinned or drawn nuisance, a labeling arm),
    the labeling line is the plan's own description with its numbers compacted, the provenance lines wrap
    within the panel with compact imaging and reaction-diffusion values, and the render draws its figure from the
    fields it saves;
13. the figure redraw names the figure as the engine does, replaces an existing one only when asked,
    and leaves no partial file.

The companions of the bright tail: a labeling arm keeps the labeled subunits; the tail structure tells
spots from single pixels; the spot count reuses the count match's detection rule; each comparison and
count writes to a folder keyed by its source renders' run label.

Runnable with ``python -m pytest`` or directly:
``MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python tests/test_posterior_predictive_video.py``.
"""
import ast
import contextlib
import importlib.util
import io
import json
import re
import tempfile
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
from srm_and_sbi_monomer_dimer_alp import parameterization as bio
from srm_and_sbi_monomer_dimer_alp import posterior_predictive_video_runner as ppv
from srm_and_sbi_monomer_dimer_alp.labeling import LABELING_SET_COLUMNS, resolve_labeling
from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow, detector_workflow

REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / "srm_and_sbi_monomer_dimer_alp/posterior_predictive_video_runner.py"
SCRIPT = REPO / "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Count_Match.py"
TOML = REPO / "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Declared_RDS_FAB.toml"
_spec = importlib.util.spec_from_file_location("count_match", SCRIPT)
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)
PLAYER = REPO / "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Player.py"
_spec = importlib.util.spec_from_file_location("player", PLAYER)
pl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pl)
ARM = REPO / "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Labeling_Arm.py"
_spec = importlib.util.spec_from_file_location("labeling_arm", ARM)
arm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arm)
TAIL = REPO / "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Brightness_Tail_Structure.py"
_spec = importlib.util.spec_from_file_location("tail_structure", TAIL)
tail = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tail)
SPOTS = REPO / "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Spot_Count.py"
_spec = importlib.util.spec_from_file_location("spot_count", SPOTS)
spots = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spots)
REDRAW = REPO / "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Redraw_Figure.py"
_spec = importlib.util.spec_from_file_location("redraw_figure", REDRAW)
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)


# ==========================================================================================
# The engine
# ==========================================================================================

def _call_name(func):
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def _run_calls():
    """The calls of ``run_posterior_predictive_video``, in source order."""
    tree = ast.parse(ENGINE.read_text())
    run = next(n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "run_posterior_predictive_video")
    return sorted((n for n in ast.walk(run) if isinstance(n, ast.Call)),
                  key=lambda n: (n.lineno, n.col_offset))


def _declared_stem(run_label=None):
    return ppv._build_stem("SRM_AND_SBI_MONOMER_DIMER_ALP", "2S_50FPS", "MET-FAB", 0, None, "cell-sgm",
                           "20S", run_label=run_label, declared_rds=True, map_block="rds")


def test_a_render_never_overwrites_and_a_variant_needs_its_own_label():
    # REF and a variant of the same recording resolve to one stem: only the label separates them.
    assert len({_declared_stem(), _declared_stem("REF"), _declared_stem("VAR_PB")}) == 3
    with tempfile.TemporaryDirectory() as tmp:
        paths = ppv.render_output_paths(tmp, _declared_stem("REF"))
        assert sorted(paths) == ["clip", "figure", "trajectory"]
        ppv.refuse_existing_outputs(paths)                  # nothing there: passes
        for existing in paths.values():
            existing.write_bytes(b"earlier render")
            try:
                ppv.refuse_existing_outputs(paths)
            except SystemExit as exc:
                assert "refusing to overwrite" in str(exc) and "--run-label" in str(exc)
            else:
                raise AssertionError(f"an existing {existing.name} must be refused")
            existing.unlink()
    # The run refuses before any work: before the MAP, the recording, the simulation and any write.
    names = [_call_name(c.func) for c in _run_calls()]
    refusal = names.index("refuse_existing_outputs")
    for work in ("_load_map_theta", "read_recording", "simulate_and_render", "savez_compressed",
                 "draw_comparison_figure"):
        assert refusal < names.index(work), work


def test_an_existing_trajectory_is_refused_and_the_engine_deletes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        trajectory = Path(tmp) / "Earlier_Trajectory.h5"
        trajectory.write_bytes(b"earlier trajectory")
        try:
            ppv.simulate_and_render("FAB", None, None, None, 10, trajectory)
        except FileExistsError:
            pass
        else:
            raise AssertionError("an existing trajectory must be refused")
        assert trajectory.read_bytes() == b"earlier trajectory"
    calls = {_call_name(n.func) for n in ast.walk(ast.parse(ENGINE.read_text())) if isinstance(n, ast.Call)}
    assert not calls & {"unlink", "remove", "rmtree", "rmdir"}, calls & {"unlink", "remove", "rmtree", "rmdir"}


def test_the_figure_names_the_synthetic_source():
    assert ppv.synthetic_source_label("rds", declared_biology=True) == "SYNTH (declared reaction-diffusion)"
    assert ppv.synthetic_source_label("rds") == "SYNTH (MAP reaction-diffusion)"
    assert ppv.synthetic_source_label("imaging", fixed_imaging=True) == "SYNTH (fixed imaging)"
    assert ppv.synthetic_source_label("imaging") == "SYNTH (MAP imaging)"
    # The run saves its fields and draws the figure from those same fields (comparison_figure_inputs
    # hands the clip's label and labeling row to the figure, where both are required arguments).
    calls = _run_calls()
    save = next(c for c in calls if _call_name(c.func) == "savez_compressed")
    figure_call = next(c for c in calls if _call_name(c.func) == "draw_comparison_figure")
    assert any(k.arg is None and isinstance(k.value, ast.Name) and k.value.id == "fields" for k in save.keywords)
    assert isinstance(figure_call.args[0], ast.Name) and figure_call.args[0].id == "fields"
    assert (save.lineno, save.col_offset) < (figure_call.lineno, figure_call.col_offset)
    # ... under the display settings the user gave (not the default by omission).
    norm, pair = figure_call.args[3], figure_call.args[4]
    assert isinstance(norm, ast.Attribute) and norm.attr == "display_norm"
    assert (isinstance(pair, ast.Call) and _call_name(pair.func) == "tuple"
            and isinstance(pair.args[0], ast.Attribute) and pair.args[0].attr == "display_percentiles")
    # n_subunits, n_dyes, n_labeled_subunits, monomers_0, monomers_visible_0, dimers_0, dimers_visible_0,
    # dimers_two_labeled_0, occupancy_monomer, occupancy_dimer (labeling.LABELING_SET_COLUMNS)
    row = np.array([1000, 261, 136, 860, 124, 70, 10, 2, 0.155, 0.155])
    # A declared-configuration biology render: the figure names its source and shows no MAP.
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    experimental = rng.integers(150, 400, size=(4, 16, 16)).astype(np.uint16)
    synth = rng.integers(150, 400, size=(4, 16, 16)).astype(np.uint16)
    rds = np.array([bio.prior_center(e) for e in bio.PARAMETERIZATION])
    with tempfile.TemporaryDirectory() as tmp, matplotlib.rc_context({"svg.fonttype": "none"}):
        path = Path(tmp) / "Comparison.svg"                 # text kept as text, so it can be read back
        ppv._save_comparison_png(
            path, experimental, synth, "FAB", 0, None, "full", rds, ppv._fixed_imaging_theta(),
            synth_label=ppv.synthetic_source_label("rds", declared_biology=True), labeling_row=row,
            imaging_label="FIXED imaging (calibrated Nuisance_DLI + MET SCOPE)",
            rds_label="DECLARED reaction-diffusion (FAB_DECLARED_RDS, absolute)",
            motion_desc="from the declared reaction-diffusion configuration (not a draw)",
            labeling_desc="FAB_POISSON, occupancy 0.155")
        text = re.sub(r'"data:[^"]*"', '""', path.read_text())   # drop the embedded image data
    assert "SYNTH (declared reaction-diffusion) FAB c0" in text
    assert "Posterior-predictive video check: experimental vs SYNTH (declared reaction-diffusion)" in text
    assert not re.search(r"\bMAP\b", text), re.findall(r".{0,40}\bMAP\b.{0,40}", text)
    # The receptors the render started with, stated explicitly.
    assert "1000 subunits = 860 monomers + 70 dimers (930 receptors)" in text
    assert "labeled: 136 subunits carrying 261 dyes" in text
    # Long provenance lines wrap inside the panel (no line exceeds the width); both halves are stated.
    assert "visible: 124 of 860 monomers + 10 of 70 dimers (2 with both" in text
    assert "subunits labeled) = 134 spots" in text
    assert "display: norm full [min, max] = [150, 399] ADU" in text
    assert "outside it: experimental 0 below, 0 above; synthetic 0 below," in text
    # The brightness range of each row is explicit in a colorbar.
    assert "display window (full [min, max]): 150 to 399 ADU" in text
    assert "projection window [min, max]:" in text
    # The provenance block itself: every line within the width, the imaging and reaction-diffusion values
    # compact, the SCOPE footnote only when a value is flagged.
    import matplotlib.axes
    def provenance_block(imaging, rds_vector):
        blocks, real_text = [], matplotlib.axes.Axes.text
        def recording_text(self, x, y, s, *a, **k):
            if k.get("family") == "monospace":
                blocks.append(s)
            return real_text(self, x, y, s, *a, **k)
        matplotlib.axes.Axes.text = recording_text
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ppv._save_comparison_png(Path(tmp) / "C.png", experimental, synth, "FAB", 0, None, "full",
                                         rds_vector, imaging, synth_label="SYNTH (declared reaction-diffusion)",
                                         labeling_row=row, labeling_desc=ppv.labeling_description(
                                             resolve_labeling("FAB", "binomial:1.6412:4").record()))
        finally:
            matplotlib.axes.Axes.text = real_text
        assert len(blocks) == 1
        return blocks[0]
    imaging = ppv._fixed_imaging_theta({"sigma_r": 0.23714, "prob_photo_bleach": 0.056234})
    rds_vector = rds.copy()
    rds_vector[[e["KEY"] for e in bio.PARAMETERIZATION].index("ratio_dimer_monomer_initial")] = 0.0316228
    block = provenance_block(imaging, rds_vector)
    assert max(len(line) for line in block.split("\n")) <= ppv.PROVENANCE_WIDTH
    assert "sigma_r=0.237" in block and "p_bleach=0.056" in block and "0.2371" not in block
    assert "r_B/A=0.032" in block and "0.0316" not in block
    flat = " ".join(line.strip() for line in block.split("\n"))      # the text with its wraps undone
    assert "Binomial(sites=4, mean=1.641)" in flat
    assert "(* outside prior / SCOPE box)" not in block
    assert "(* outside prior / SCOPE box)" in provenance_block(ppv._fixed_imaging_theta({"mu_pc": 5000.0}), rds)
    # A sensitivity render names its --set-imaging overrides in the imaging header, so it never
    # reads as the untouched setup; without overrides the header is the bare role.
    base = "FIXED imaging (calibrated Nuisance_DLI + MET SCOPE)"
    assert ppv.imaging_role_label(base) == base and ppv.imaging_role_label(base, {}) == base
    assert (ppv.imaging_role_label(base, {"mu_pc": 250.731571, "sigma_pc": 0.6744})
            == base + " WITH OVERRIDES (mu_pc=250.7, sigma_pc=0.674)")
    # Every number the provenance prints is compact: at most three decimal places, and a value below 0.01
    # keeps three significant digits.
    for value, shown in ((0.15508300855884316, "0.155"), (0.125, "0.125"), (0.08108, "0.081"), (1.585, "1.585"),
                         (250.731571, "250.7"), (139.0, "139"), (0.003162, "0.00316"), (41.84, "41.84"),
                         (0.0, "0"), (-0.0, "0"), (3162.0, "3162"), (0.6744, "0.674"), (12345.6, "12350"),
                         (-0.15508, "-0.155"), (1e-7, "1e-07")):
        assert ppv.compact_number(value) == shown, (value, ppv.compact_number(value))
    # The percentile window takes the user's pair and names it; an impossible pair is refused.
    assert ppv.check_display_percentiles((0, 99.99)) == ppv.DISPLAY_PERCENTILES == (0.0, 99.99)
    for bad in ((99.9, 1), (-1, 50), (0, 100.5), (50, 50), (1,), "ab"):
        try:
            ppv.check_display_percentiles(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} must be refused")
    with tempfile.TemporaryDirectory() as tmp, matplotlib.rc_context({"svg.fonttype": "none"}):
        path = Path(tmp) / "Comparison.svg"
        ppv._save_comparison_png(
            path, experimental, synth, "FAB", 0, None, "percentile", rds, ppv._fixed_imaging_theta(),
            synth_label=ppv.synthetic_source_label("rds", declared_biology=True), labeling_row=row,
            display_percentiles=(1, 99.9))
        text = re.sub(r'"data:[^"]*"', '""', path.read_text())
    assert "norm percentile [p1, p99.9]" in text


# ==========================================================================================
# The receptor-total match
# ==========================================================================================

V = 0.12391                    # visible hosts per subunit at the MET-FAB declared visibility, r = 0.081
BOX = (316.2, 3162.3)
KW = dict(initial=600, tolerance=0.03, max_iterations=6, max_count=10000, min_elasticity=0.3,
          prior_box=BOX)
DECIDED = dict(KW, allow_outside_prior=True)             # the user's decision to render beyond the box


def _linear(eta=0.67, calls=None):
    """Detection proportional to the visible hosts, which equal their expectation; ``calls``
    collects the counts rendered."""
    def render(n_r, it):
        if calls is not None:
            calls.append(n_r)
        return n_r * V, eta * n_r * V
    return render


def test_linear_response_converges_to_the_predicted_count():
    out = cm.iterate_count(44.34, V, _linear(), **KW)
    assert out["status"] == "matched", out
    assert abs(out["matched_count_total"] - 44.34 / (0.67 * V)) <= 1
    assert out["matched_count_total"] == 534


def test_undefined_paths_carry_no_count():
    assert cm.iterate_count(0.0, V, _linear(), **KW)["status"] == "undefined_no_experimental_detections"
    assert cm.iterate_count(float("nan"), V, _linear(), **KW)["status"] == "undefined_no_experimental_detections"
    assert cm.iterate_count(44.0, 0.0, _linear(), **KW)["status"] == "undefined_no_visibility"
    out = cm.iterate_count(44.0, V, lambda n, it: (0, 0.0), **KW)
    assert out["status"] == "undefined_no_visible_hosts" and out["matched_count_total"] is None
    out = cm.iterate_count(44.0, V, lambda n, it: (70, 0.0), **KW)
    assert out["status"] == "undefined_no_synthetic_detections" and out["matched_count_total"] is None
    out = cm.iterate_count(44.0, V, lambda n, it: (70, float("inf")), **KW)
    assert out["status"] == "undefined_no_synthetic_detections"


def test_saturation_is_reported_not_chased():
    """Detection that caps below the recording's count cannot reach it: unresolved, no count."""
    def capped(cap):
        return lambda n_r, it: (n_r * V, min(0.67 * n_r * V, cap))
    out = cm.iterate_count(200.0, V, capped(150.0), **dict(KW, initial=1000))
    assert out["status"] == "unresolved_saturation", out
    assert out["matched_count_total"] is None and "saturation" in out["reason"]
    assert max(r["count_total"] for r in out["iterations"]) <= BOX[1]
    out = cm.iterate_count(400.0, V, capped(300.0), **dict(DECIDED, initial=3000))
    assert out["status"] == "unresolved_saturation", out
    assert out["matched_count_total"] is None
    assert max(r["count_total"] for r in out["iterations"]) <= 10000


def test_budget_and_count_ceiling_carry_no_count():
    # One iteration from 600 proposes 534 (11 % away): the budget ends before agreement.
    out = cm.iterate_count(44.34, V, _linear(), **dict(KW, max_iterations=1))
    assert out["status"] == "not_converged" and out["matched_count_total"] is None
    assert out["last_count_total"] == 600
    # A target needing about 20000 receptors under a 10000 ceiling, rendering beyond the box decided.
    calls = []
    out = cm.iterate_count(1660.0, V, _linear(calls=calls), **dict(DECIDED, initial=3000, max_iterations=10))
    assert out["status"] == "unresolved_max_count", out
    assert out["matched_count_total"] is None and max(calls) <= 10000
    # The ceiling holds from the first render: an initial count above it renders nothing.
    for kw in (KW, DECIDED):
        calls = []
        out = cm.iterate_count(44.34, V, _linear(calls=calls), **dict(kw, initial=20000))
        assert out["status"] == "unresolved_max_count" and calls == [], (out, calls)


def test_nothing_is_rendered_outside_the_prior_without_the_users_decision():
    target = 0.67 * V * 4000.0                  # the arithmetic's answer is N_R = 4000, outside the box
    calls = []
    out = cm.iterate_count(target, V, _linear(calls=calls), **dict(KW, initial=3000))
    assert out["status"] == "user_decision_outside_prior", out
    assert out["matched_count_total"] is None and calls == [3000]
    # The first count is tested before its render too.
    calls = []
    out = cm.iterate_count(target, V, _linear(calls=calls), **dict(KW, initial=5000))
    assert out["status"] == "user_decision_outside_prior" and calls == [], (out, calls)
    # A converged value just outside the box (rendered 3150, proposed 3200) is not adopted either.
    calls = []
    out = cm.iterate_count(0.67 * V * 3200.0, V, _linear(calls=calls), **dict(KW, initial=3150))
    assert out["status"] == "user_decision_outside_prior" and calls == [3150], (out, calls)
    # Under the user's decision the count is rendered and accepted, flagged, not clipped.
    calls = []
    out = cm.iterate_count(target, V, _linear(calls=calls), **dict(DECIDED, initial=3000))
    assert out["status"] == "matched_outside_prior" and out["matched_count_total"] == 4000, out
    assert calls == [3000, 4000]
    # The render guard (the iteration's and the check renders') refuses before rendering.
    for n, decided, allowed in ((20000, False, False), (20000, True, False), (4000, False, False),
                                (4000, True, True), (1000, False, True)):
        calls = []
        render = cm.guarded_render(_linear(calls=calls), max_count=10000, prior_box=BOX,
                                   allow_outside_prior=decided)
        try:
            render(n, 100)
        except RuntimeError as exc:
            assert not allowed and "refusing to render" in str(exc) and calls == [], (n, decided)
        else:
            assert allowed and calls == [n], (n, decided)


def test_an_initial_count_beyond_the_limits_stops_the_run_before_any_work():
    common = ["--total-time-seconds", "2", "--kind", "MET-FAB", "--cell", "0", "--nuisance-tag", "REF",
              "--declared-rds", str(TOML), "--attempt-label", "TEST_REFUSED_BEFORE_ANY_WORK"]
    for extra, expected in ((["--initial-count", "20000"], "exceeds --max-count"),
                            (["--initial-count", "5000"], "outside the prior box")):
        try:
            cm.main(common + extra)
        except SystemExit as exc:
            assert expected in str(exc) and "--initial-count" in str(exc), str(exc)
        else:
            raise AssertionError(f"{extra} must stop the run")


def test_existing_results_are_refused_before_any_work():
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "X_Count_Match_MET-FAB_Cell_0"
        paths = {0: (Path(f"{stem}.json"), Path(f"{stem}.md"))}
        cm.preflight_outputs(paths)                          # nothing there: passes
        Path(f"{stem}.json").write_text("{}")
        try:
            cm.preflight_outputs(paths)
        except SystemExit as exc:
            assert "refusing to overwrite" in str(exc)
        else:
            raise AssertionError("an existing result must be refused")
    assert cm.attempt_folder_name("FAB_DECLARED_RDS", "REF") == "FAB_DECLARED_RDS_REF"
    assert cm.attempt_folder_name("FAB_DECLARED_RDS", None, "check 2") == "FAB_DECLARED_RDS_CANONICAL_check_2"


# ==========================================================================================
# The player render
# ==========================================================================================

def test_the_headless_copy_drops_only_the_scrubber_widget():
    import nbformat
    original = nbformat.read(pl.NOTEBOOK, as_version=4)
    copy = pl.prepare_notebook(nbformat.read(pl.NOTEBOOK, as_version=4), "/data/X_Synthetic_Video.npz",
                               norm_mode="full", play_every=2, norm_percentiles=(1, 99.9))
    assert len(copy.cells) == len(original.cells)
    code_o = [c for c in original.cells if c.cell_type == "code"]
    code_c = [c for c in copy.cells if c.cell_type == "code"]
    # Configuration: the clip path and the settings given, nothing else.
    cfg = pl._source(code_c[1])
    assert 'CLIP_PATH = "/data/X_Synthetic_Video.npz"' in cfg and 'NORM_MODE = "full"' in cfg
    assert pl.CLIP_PLACEHOLDER not in cfg
    original_cfg = pl._source(code_o[1])
    assert 'NORM_MODE = "percentile"' in original_cfg            # the notebook's default: fixed over all frames
    assert "NORM_PERCENTILES = (0.0, 99.99)" in original_cfg     # the editable pair's default
    assert "NORM_PERCENTILES = (1, 99.9)" in cfg
    assert "SECOND_CLIP_PATH = None" in cfg                      # the optional third panel stays off in a player
    assert cfg.count("SECOND_CLIP_PATH = None") == 1
    assert (original_cfg.replace(pl.CLIP_PLACEHOLDER, 'CLIP_PATH = "/data/X_Synthetic_Video.npz"')
            .replace('NORM_MODE = "percentile"', 'NORM_MODE = "full"')
            .replace("NORM_PERCENTILES = (0.0, 99.99)", "NORM_PERCENTILES = (1, 99.9)") == cfg)
    for bad in ((99.9, 1), (-1, 50), (0, 100.5), (50, 50)):
        try:
            pl.prepare_notebook(nbformat.read(pl.NOTEBOOK, as_version=4), "/data/X_Synthetic_Video.npz",
                                norm_percentiles=bad)
        except SystemExit as exc:
            assert "0 <= lower < upper <= 100" in str(exc), bad
        else:
            raise AssertionError(f"{bad} must be refused")
    # Scrubber: the helpers the player uses stay, the widget call goes.
    scrub = pl._source(code_c[2])
    assert "def _roi(" in scrub and "H, W = " in scrub and "interact(" not in scrub
    assert pl._source(code_o[2]).startswith(scrub.split("\n# Headless render")[0])
    # Player: the stride given, and the report the script reads.
    player = pl._source(code_c[3])
    assert "PLAY_EVERY = 2" in player and 'mpl.rcParams["animation.embed_limit"]' in player
    assert "frames embedded" in player
    # Unchanged cells stay byte-identical.
    assert pl._source(code_c[0]) == pl._source(code_o[0])
    for a, b in zip(original.cells, copy.cells):
        if a.cell_type == "markdown":
            assert pl._source(a) == pl._source(b)
    # The player file is named after the clip; the truncation report is read from the outputs.
    assert pl.player_path("/data/X_Synthetic_Video.npz").name == "X_Player.html"
    assert pl.player_path("/data/X_Synthetic_Video.npz", "/out") == Path("/out/X_Player.html")
    try:
        pl.player_path("/data/X_Comparison.png")
    except SystemExit:
        pass
    else:
        raise AssertionError("only a clip names a player")
    # A zoomed player: the zoom (and a center) set on their own lines, and its own file name.
    zoomed = pl.prepare_notebook(nbformat.read(pl.NOTEBOOK, as_version=4), "/data/X_Synthetic_Video.npz",
                                 play_zoom=4, play_center=(100, 140))
    zp = pl._source([c for c in zoomed.cells if c.cell_type == "code"][3])
    assert zp.count("PLAY_ZOOM = 4.0") == 1 and zp.count("PLAY_CX, PLAY_CY = 100, 140") == 1
    assert pl.player_path("/data/X_Synthetic_Video.npz", play_zoom=4) == Path("/data/X_Player_Zoom4.html")
    assert pl.player_path("/data/X_Synthetic_Video.npz", play_zoom=2.0).name == "X_Player_Zoom2.html"
    assert pl.player_path("/data/X_Synthetic_Video.npz", play_zoom=1.5).name == "X_Player_Zoom1p5.html"
    assert pl.player_path("/data/X_Synthetic_Video.npz", play_zoom=1).name == "X_Player.html"
    for bad in (0.5, 0, -2):
        try:
            pl.prepare_notebook(nbformat.read(pl.NOTEBOOK, as_version=4), "/data/X_Synthetic_Video.npz", play_zoom=bad)
        except SystemExit as exc:
            assert "at least 1" in str(exc)
        else:
            raise AssertionError(f"zoom {bad} must be refused")
    copy.cells[-1]["outputs"] = [nbformat.v4.new_output(
        "stream", name="stdout", text="player: 271.9 MB, 778 of 1000 frames embedded  <-- TRUNCATED: raise\n")]
    assert pl.player_report(copy) == (271.9, 778, 1000, "<-- TRUNCATED: raise")


# ==========================================================================================
# The labeling arm and the tail structure
# ==========================================================================================

def test_a_labeling_arm_keeps_the_labeled_subunits_and_gives_each_at_least_one_dye():
    from srm_and_sbi_monomer_dimer_alp.labeling import LABELING_LAWS
    rng = np.random.default_rng(0)
    source = rng.poisson(1.64, size=1000)
    law = LABELING_LAWS["FAB_BINOMIAL"]
    counts = arm.arm_dye_counts(source, law, np.random.default_rng(1))
    assert np.array_equal(counts > 0, source > 0)                  # the same subunits are labeled
    assert counts[source > 0].min() >= 1 and counts.max() <= 4      # at least one dye, at most the law's sites
    assert counts.dtype == np.int64
    # The positive-conditioned draw follows the law above zero: no zeros, the right support.
    k = arm.positive_conditioned_counts(law, 5000, np.random.default_rng(2))
    assert k.min() == 1 and k.max() <= 4 and abs(k.mean() - 1.64 / (1 - law.probability_zero)) < 0.05
    assert arm.histogram(np.array([0, 1, 1, 3])) == {1: 2, 3: 1}


def test_the_tail_structure_tells_spots_from_single_pixels():
    stack = np.full((20, 32, 32), 1000, dtype=np.uint16)
    stack[3, 10, 10] = 20000                                        # one single-pixel, single-frame event
    single = tail.structure(stack, 10_000)
    assert single["npix"] == 1 and single["blobs"] == 1 and single["blobs_3plus"] == 0
    assert single["frac_neighbor"] == 0.0 and single["sites"] == 1 and single["frames_per_site_max"] == 1
    stack[5:9, 20:23, 20:23] = 15000                                # a 3x3 spot revisited over four frames
    spot = tail.structure(stack, 10_000)
    assert spot["npix"] == 1 + 36 and spot["blobs_3plus"] == 4 and spot["frames_per_site_max"] == 4
    assert spot["frac_neighbor"] > 0.9
    assert tail.structure(np.zeros((2, 4, 4), dtype=np.uint16), 10_000)["npix"] == 0
    w = tail.window_stats(stack, 10_000, exp_max=16000)              # only the single-pixel event exceeds 16000
    assert w["quantiles"]["max"] == 20000 and w["above_experimental_max"] == 1
    assert tail.window_stats(stack, 10_000, exp_max=12000)["above_experimental_max"] == 37
    assert "above_experimental_max" not in tail.window_stats(stack, 10_000, None)


def test_the_spot_count_reuses_the_count_match_rule_and_counts_fixed_windows():
    # One detection rule: the spot count loads the count match's ``detected_per_frame`` from its file.
    assert spots.COUNT_MATCH == SCRIPT and spots.count_match_module().detected_per_frame is not None
    assert spots.count_match_module().N_SIGMA == cm.N_SIGMA and spots.count_match_module().HALF_PX == cm.HALF_PX
    # The windows are the first and the last frames of the clip, never overlapping its edges.
    assert spots.windows(1000, 100, 100) == {"opening 100 frames": (0, 100), "closing 100 frames": (900, 1000)}
    assert spots.windows(10, 10, 10) == {"opening 10 frames": (0, 10), "closing 10 frames": (0, 10)}
    for bad in ((10, 11, 5), (10, 5, 11), (10, 0, 5)):
        try:
            spots.windows(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} must be refused")
    s = spots.summarize(np.array([3, 5, 4, 8]))
    assert s["mean"] == 5.0 and s["median"] == 4.5 and s["min"] == 3 and s["max"] == 8 and s["per_frame"] == [3, 5, 4, 8]
    assert spots.fmt_ratio(80.19, 44.34) == "80.2 (1.81)" and spots.fmt_ratio(5.0, 0.0) == "5.0"
    # A synthetic stack with three well-separated bright spots is counted as three spots per frame.
    stack = np.full((4, 96, 96), 1200, dtype=np.uint16)
    rng = np.random.default_rng(0)
    stack = (stack + rng.normal(0, 40, size=stack.shape)).astype(np.uint16)
    yy, xx = np.mgrid[-6:7, -6:7]
    spot = (12000 * np.exp(-(xx ** 2 + yy ** 2) / (2 * 1.5 ** 2))).astype(np.uint16)
    for cx, cy in ((20, 20), (60, 30), (40, 70)):
        stack[:, cy - 6:cy + 7, cx - 6:cx + 7] += spot
    counts = spots.detected_per_frame(stack)
    assert counts.shape == (4,) and np.array_equal(counts, [3, 3, 3, 3]), counts



# ==========================================================================================
# The display windows and the figure drawn from the clip
# ==========================================================================================

def _fixture_fields(experimental, synth, *, workflow="biology", map_block="rds", rds_source="declared",
                    map_theta=None, map_source="declared", labeling_record=None, imaging_identity=None,
                    synth_label="SYNTH (declared reaction-diffusion)", rds_outside=()):
    """A clip's fields as the engine saves them (the keys of its ``np.savez_compressed`` call)."""
    plan = resolve_labeling("FAB")
    record = ({"scenario": {"name": "FAB_DECLARED_RDS"}, "path": "Declared_RDS_FAB.toml", "sha256": "0" * 64}
              if rds_source == "declared" else {"basis": "fixture"})
    return dict(
        experimental=np.asarray(experimental, dtype=np.uint16), synth=np.asarray(synth, dtype=np.uint16),
        imaging_physical=ppv._fixed_imaging_theta(), imaging_keys=np.array(det.DETECTOR_IMAGING_KEYS),
        map_theta=np.array([]) if map_theta is None else np.asarray(map_theta, dtype=float),
        map_keys=np.array(["k"]), map_block=map_block, workflow=workflow,
        rds_provenance=np.array([bio.prior_center(e) for e in bio.PARAMETERIZATION]),
        rds_keys=np.array(ppv._NUISANCE_KEYS), rds_source=rds_source, rds_record_json=json.dumps(record),
        rds_outside_prior=np.array(list(rds_outside), dtype=str), condition="FAB", labeling_law=plan.law_name,
        dye_counts=np.zeros(10, dtype=np.int64), n_subunits=1000, n_dyes=261,
        labeling_columns=np.array(LABELING_SET_COLUMNS),
        labeling_row=np.array([1000, 261, 136, 860, 124, 70, 10, 2, 0.155, 0.155]),
        occupancy_source=plan.occupancy_source, occupancy_monomer=0.155, occupancy_dimer=0.155,
        labeling_record_json=json.dumps(labeling_record if labeling_record is not None else plan.record()),
        imaging_record_json=json.dumps(imaging_identity if imaging_identity is not None else {"artifact": "REF"}),
        synth_label=synth_label, package_version="test", kind="MET-FAB", cell=0, chunk=-1,
        map_source=map_source, seed=7, frame_time_seconds=0.02, n_frames=int(np.asarray(experimental).shape[0]),
        experimental_tif="/data/Experiment_FAB_Cell_0_20S_RAW.tif", imaging_desc="fixture imaging",
        nuisance_tag="REF")


def _partly_overlapping_clips():
    """An experimental and a synthetic clip whose four windows all differ, with planted extremes."""
    exp = np.full((3, 8, 8), 1000, dtype=np.uint16)
    exp[0, 0, 0], exp[1, 1, 1] = 300, 40000
    syn = np.full((3, 8, 8), 1100, dtype=np.uint16)
    syn[0, 2, 2], syn[2, 3, 3], syn[2, 4, 4] = 200, 30000, 25000
    return exp, syn


def test_the_display_windows_and_their_default():
    assert ppv.DISPLAY_NORMS == ("percentile", "experimental", "synthetic", "full")
    assert ppv.DISPLAY_NORM_DEFAULT == "percentile" and ppv.DISPLAY_PERCENTILES == (0.0, 99.99)
    parser = ppv.build_parser("test")
    assert parser.get_default("display_norm") == "percentile"
    assert tuple(next(a for a in parser._actions if a.dest == "display_norm").choices) == ppv.DISPLAY_NORMS
    exp, syn = _partly_overlapping_clips()
    w = ppv.display_window(exp, syn, "experimental")
    assert w["clim"] == (300.0, 40000.0) and w["desc"] == "experimental [min, max]"
    assert w["outside"] == {"experimental": (0, 0), "synthetic": (1, 0)}
    assert (ppv.display_window_line(w)
            == "display: norm experimental [min, max] = [300, 40000] ADU\n"
               "outside it: experimental 0 below, 0 above; synthetic 1 below, 0 above")
    w = ppv.display_window(exp, syn, "synthetic")
    assert w["clim"] == (200.0, 30000.0) and w["desc"] == "synthetic [min, max]"
    assert w["outside"] == {"experimental": (0, 1), "synthetic": (0, 0)}
    w = ppv.display_window(exp, syn, "full")
    assert w["clim"] == (200.0, 40000.0) and w["desc"] == "full [min, max]"
    assert w["outside"] == {"experimental": (0, 0), "synthetic": (0, 0)}
    w = ppv.display_window(exp, syn, "percentile", (1, 99))
    lo = min(np.percentile(exp, 1), np.percentile(syn, 1)); hi = max(np.percentile(exp, 99), np.percentile(syn, 99))
    assert w["clim"] == (float(lo), float(hi)) and w["desc"] == "percentile [p1, p99]"
    assert w["outside"]["synthetic"] == (int((syn < lo).sum()), int((syn > hi).sum()))
    for bad in ("autoscale", "min-max-experiment", "Full", None):
        try:
            ppv.display_window(exp, syn, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} must be refused")
    try:
        ppv.display_window(exp, syn, "percentile", (99, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("an impossible percentile pair must be refused")
    # The run checks the mode and the pair before any work, the dry run included.
    names = [_call_name(c.func) for c in _run_calls()]
    for check in ("check_display_norm", "check_display_percentiles"):
        assert names.index(check) < names.index("refuse_existing_outputs"), check
    run = next(n for n in ast.parse(ENGINE.read_text()).body
               if isinstance(n, ast.FunctionDef) and n.name == "run_posterior_predictive_video")
    dry = next(n for n in ast.walk(run) if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute)
               and n.test.attr == "dry_run")
    for c in _run_calls():
        if _call_name(c.func) in ("check_display_norm", "check_display_percentiles"):
            assert c.lineno < dry.lineno, _call_name(c.func)


def test_the_figure_is_drawn_from_the_clip_alone():
    # The labeling line is the plan's own description with compact numbers, for every condition, an override
    # and an ad-hoc law.
    for plan in (resolve_labeling("FAB"), resolve_labeling("INLB"), resolve_labeling("INLB", None, "A=0.5,B=1.0"),
                 resolve_labeling("FAB", "FAB_BINOMIAL", "0.1422"), resolve_labeling("FAB", "poisson:1.64123")):
        line = ppv.labeling_description(json.loads(json.dumps(plan.record())))
        assert line.startswith(f"{plan.condition} {plan.law_name} = {ppv._law_text(plan.record()['law'])}, occupancy ")
        assert f"({plan.occupancy_source}); visible per subunit " in line
        # At most three decimal places; the law's name is an identifier, printed as stored.
        assert not re.search(r"\d\.\d{4,}", line.replace(plan.law_name, "")), line
    assert "Poisson(mean=1.641)" in ppv.labeling_description(resolve_labeling("FAB", "poisson:1.64123").record())
    assert (ppv.labeling_description(resolve_labeling("FAB").record())
            == "FAB FAB_POISSON = Poisson(mean=1.64), occupancy 0.155 (derived); visible per subunit A 0.125, B 0.125")
    assert "occupancy A=0.5, B=1 (override)" in ppv.labeling_description(
        json.loads(json.dumps(resolve_labeling("INLB", None, "A=0.5,B=1.0").record())))
    exp, syn = _partly_overlapping_clips()
    bio_cfg, det_cfg = biology_workflow(), detector_workflow()
    # A declared biology render.
    x = ppv.comparison_figure_inputs(_fixture_fields(exp, syn), bio_cfg)
    assert x["imaging_label"] == "FIXED imaging (calibrated Nuisance_DLI tag REF + MET SCOPE)"
    assert x["rds_label"] == "DECLARED reaction-diffusion (FAB_DECLARED_RDS, absolute)"
    assert x["motion_desc"] == "from the declared reaction-diffusion configuration (not a draw)"
    assert x["sel_desc"] is None and not x["fixed_imaging"] and not x["fixed_nuisance"]
    assert len(x["rds_table"]) == len(bio.PARAMETERIZATION) and x["synth_label"] == "SYNTH (declared reaction-diffusion)"
    assert x["labeling_desc"] == ppv.labeling_description(resolve_labeling("FAB").record())
    assert x["cell"] == 0 and x["kind"] == "MET-FAB"
    assert np.array_equal(x["experimental"], exp) and np.array_equal(x["synth"], syn)
    # A labeling arm: the source's labeled subunits kept, the source's trajectory reused.
    arm_record = dict(resolve_labeling("FAB").record(), law_name="FAB_BINOMIAL",
                      law={"family": "binomial", "mean": 1.64, "shape": 4, "description": "Binomial(sites=4, mean=1.64)"},
                      arm={"label": "BINOMIAL4"})
    x = ppv.comparison_figure_inputs(_fixture_fields(exp, syn, labeling_record=arm_record), bio_cfg)
    assert x["motion_desc"] == "the source render's trajectory (reused, not re-simulated)"
    assert x["labeling_desc"].startswith("FAB FAB_BINOMIAL = Binomial(sites=4, mean=1.64), occupancy ")
    assert "(derived; the source's labeled subunits kept); visible per subunit A 0.125, B 0.125" in x["labeling_desc"]
    # A MAP biology render names its selection.
    x = ppv.comparison_figure_inputs(_fixture_fields(exp, syn, rds_source="map", map_theta=[1.0], map_source="cell-sgm",
                                                     synth_label="SYNTH (MAP reaction-diffusion)"), bio_cfg)
    assert x["sel_desc"] == "cell-sgm" and x["rds_label"] == "INFERRED reaction-diffusion (MAP theta, absolute)"
    assert x["motion_desc"] == "from the MAP reaction-diffusion parameters (not a draw)"
    # A detector render at fixed imaging with a pinned nuisance, and one with MAP imaging and a drawn nuisance.
    x = ppv.comparison_figure_inputs(_fixture_fields(exp, syn, workflow="detector", map_block="imaging",
                                                     rds_source="pinned", map_source="cell-sgm",
                                                     imaging_identity={"set_imaging_overrides": {"mu_pc": 250.7}},
                                                     synth_label="SYNTH (fixed imaging)"), det_cfg)
    assert x["imaging_label"] == "FIXED imaging (MET values) WITH OVERRIDES (mu_pc=250.7)"
    assert x["fixed_imaging"] and x["fixed_nuisance"] and x["sel_desc"] is None and x["rds_table"] is None
    assert x["rds_label"] == "NUISANCE reaction-diffusion (marginalized)" and x["motion_desc"] is None
    x = ppv.comparison_figure_inputs(_fixture_fields(exp, syn, workflow="detector", map_block="imaging",
                                                     rds_source="drawn", map_theta=[1.0] * 6, map_source="cell-sgm",
                                                     synth_label="SYNTH (MAP imaging)"), det_cfg)
    assert x["imaging_label"] == "INFERRED imaging (MAP theta, absolute)" and x["sel_desc"] == "cell-sgm"
    assert not x["fixed_imaging"] and not x["fixed_nuisance"]
    # A detector render at a declared configuration, with MAP and with fixed imaging.
    for theta, fixed in (([1.0] * 6, False), (None, True)):
        x = ppv.comparison_figure_inputs(_fixture_fields(exp, syn, workflow="detector", map_block="imaging",
                                                         rds_source="declared", map_theta=theta, map_source="cell-sgm",
                                                         synth_label="SYNTH (MAP imaging)"), det_cfg)
        assert x["motion_desc"] == "from the declared reaction-diffusion configuration"
        assert x["rds_label"] == "DECLARED reaction-diffusion (FAB_DECLARED_RDS, absolute)"
        assert x["fixed_imaging"] is fixed and x["rds_table"] is None
    # A biology sensitivity render names its imaging overrides; out-of-prior keys are carried to the figure.
    x = ppv.comparison_figure_inputs(_fixture_fields(exp, syn, imaging_identity={"set_imaging_overrides": {"mu_pc": 250.7}},
                                                     rds_outside=("count_total",)), bio_cfg)
    assert x["imaging_label"] == "FIXED imaging (calibrated Nuisance_DLI tag REF + MET SCOPE) WITH OVERRIDES (mu_pc=250.7)"
    assert x["rds_outside"] == ("count_total",)
    # The figure itself, drawn from the fields under the default window, states the window and the
    # pixels outside it; the same clip saved and reopened as an npz gives the same inputs.
    import matplotlib
    matplotlib.use("Agg")
    fields = _fixture_fields(exp, syn)
    with tempfile.TemporaryDirectory() as tmp, matplotlib.rc_context({"svg.fonttype": "none"}):
        path = Path(tmp) / "Comparison.svg"
        ppv.draw_comparison_figure(fields, path, bio_cfg)
        text = re.sub(r'"data:[^"]*"', '""', path.read_text())
        np.savez_compressed(Path(tmp) / "X_Synthetic_Video.npz", **fields)
        reopened = ppv.comparison_figure_inputs(np.load(Path(tmp) / "X_Synthetic_Video.npz"), bio_cfg)
    w = ppv.display_window(exp, syn, "percentile")                 # the default window
    lo, hi = w["clim"]
    assert f"display: norm percentile [p0, p99.99] = [{lo:.0f}, {hi:.0f}] ADU" in text
    assert f"display window (percentile [p0, p99.99]): {lo:.0f} to {hi:.0f} ADU" in text
    assert "DECLARED reaction-diffusion (FAB_DECLARED_RDS, absolute)" in text
    direct = ppv.comparison_figure_inputs(fields, bio_cfg)
    for key, value in direct.items():
        other = reopened[key]
        assert (np.array_equal(value, other) if isinstance(value, np.ndarray) else value == other), key


def _notebook_config(clip, norm_mode=None, second=None):
    """The viewer's configuration cell as the player prepares it, with SECOND_CLIP_PATH set if given."""
    import nbformat
    copy = pl.prepare_notebook(nbformat.read(pl.NOTEBOOK, as_version=4), clip, norm_mode=norm_mode)
    cfg = pl._source([c for c in copy.cells if c.cell_type == "code"][1])
    if second is not None:
        assert cfg.count("SECOND_CLIP_PATH = None") == 1
        cfg = cfg.replace("SECOND_CLIP_PATH = None", f'SECOND_CLIP_PATH = "{second}"')
    return cfg


def _run_config(cfg):
    namespace, out = {"np": np}, io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(cfg, "<configuration cell>", "exec"), namespace)
    return namespace, out.getvalue()


def test_the_notebook_windows_equal_the_engine_windows():
    exp, syn = _partly_overlapping_clips()
    syn2 = syn.copy(); syn2[1, 5, 5] = 50000; syn2[1, 6, 6] = 150        # a second render, brighter and darker
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "A_Synthetic_Video.npz"
        second = Path(tmp) / "B_Synthetic_Video.npz"
        other = Path(tmp) / "C_Synthetic_Video.npz"
        np.savez_compressed(clip, **_fixture_fields(exp, syn))
        np.savez_compressed(second, **_fixture_fields(exp, syn2))
        np.savez_compressed(other, **_fixture_fields(exp + 1, syn2))
        # The notebook's default is the percentile window at (0, 99.99), and every mode matches the engine's.
        ns, printed = _run_config(_notebook_config(clip))
        assert ns["NORM_MODE"] == "percentile" and ns["clim"]() == ppv.display_window(exp, syn, "percentile")["clim"]
        for mode in ppv.DISPLAY_NORMS:
            ns, printed = _run_config(_notebook_config(clip, mode))
            w = ppv.display_window(exp, syn, mode)
            assert ns["clim"]() == w["clim"], mode
            assert f"NORM_MODE={mode}  window [{w['clim'][0]:.0f}, {w['clim'][1]:.0f}] ADU" in printed, printed
            (eb, ea), (sb, sa) = w["outside"]["experimental"], w["outside"]["synthetic"]
            assert f"outside it: experimental {eb} below, {ea} above; synthetic {sb} below, {sa} above" in printed
        # Three panels: the synthetic window spans both synthetic clips, and it is one window for all.
        ns, printed = _run_config(_notebook_config(clip, "synthetic", second))
        assert ns["clim"]() == (150.0, 50000.0) and len(ns["PANELS"]) == 3
        assert "second synthetic 0 below, 0 above" in printed and "experimental 0 below, 0 above" in printed
        ns, printed = _run_config(_notebook_config(clip, "experimental", second))
        # syn2 keeps syn's planted 200 ADU pixel and adds 150 and 50000: two below the window, one above.
        assert ns["clim"]() == (300.0, 40000.0) and "second synthetic 2 below, 1 above" in printed
        # A clip of another recording is refused, and so is an unknown mode, in the configuration cell.
        for cfg in (_notebook_config(clip, None, other),
                    _notebook_config(clip).replace('NORM_MODE = "percentile"', 'NORM_MODE = "min-max-experiment"')):
            try:
                _run_config(cfg)
            except ValueError:
                pass
            else:
                raise AssertionError("the configuration cell must refuse this")


def test_the_engine_notebook_and_player_list_the_same_modes():
    import nbformat
    assert pl.NORM_MODES == ppv.DISPLAY_NORMS
    cfg = pl._source([c for c in nbformat.read(pl.NOTEBOOK, as_version=4).cells if c.cell_type == "code"][1])
    assert ast.literal_eval(re.search(r"^NORM_MODES = (\(.*?\))$", cfg, flags=re.M).group(1)) == ppv.DISPLAY_NORMS
    assert re.search(r'^NORM_MODE = "([a-z]+)"$', cfg, flags=re.M).group(1) == ppv.DISPLAY_NORM_DEFAULT
    for mode in ppv.DISPLAY_NORMS:                       # every mode is accepted and set on exactly one line
        copy = pl.prepare_notebook(nbformat.read(pl.NOTEBOOK, as_version=4), "/data/X_Synthetic_Video.npz",
                                   norm_mode=mode)
        text = pl._source([c for c in copy.cells if c.cell_type == "code"][1])
        assert len(re.findall(r'^NORM_MODE = "', text, flags=re.M)) == 1 and f'NORM_MODE = "{mode}"' in text
        try:
            pl.main(["--dry-run", "--norm-mode", mode, "/nonexistent/X_Synthetic_Video.npz"])
        except SystemExit as exc:
            assert "clip not found" in str(exc), exc       # the option parser accepted the mode
    try:
        pl.prepare_notebook(nbformat.read(pl.NOTEBOOK, as_version=4), "/data/X_Synthetic_Video.npz",
                            norm_mode="autoscale")
    except SystemExit as exc:
        assert "--norm-mode must be one of" in str(exc)
    else:
        raise AssertionError("an unknown mode must be refused")


def test_the_figure_redraw_replaces_only_when_asked():
    assert rd.figure_path("/data/X_Synthetic_Video.npz") == Path("/data/X_Comparison.png")
    assert rd.figure_path("/data/X_Synthetic_Video.npz", "/out") == Path("/out/X_Comparison.png")
    for bad in ("/data/X_Comparison.png", "/data/X.npz"):
        try:
            rd.figure_path(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"{bad} names no clip")
    exp, syn = _partly_overlapping_clips()
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "X_Synthetic_Video.npz"
        try:
            rd.plan_redraws([clip])
        except SystemExit as exc:
            assert "clip not found" in str(exc)
        else:
            raise AssertionError("a missing clip must be refused")
        np.savez_compressed(clip, **_fixture_fields(exp, syn))
        figure = Path(tmp) / "X_Comparison.png"
        assert rd.plan_redraws([clip]) == [(clip, figure)]
        figure.write_bytes(b"earlier figure")
        try:
            rd.plan_redraws([clip])
        except SystemExit as exc:
            assert "without --replace" in str(exc)
        else:
            raise AssertionError("an existing figure is replaced only under --replace")
        assert figure.read_bytes() == b"earlier figure"
        assert rd.plan_redraws([clip], replace=True) == [(clip, figure)]
        try:                                              # without --replace, an existing figure stays
            rd.redraw(clip, figure, "experimental", (0, 99.99))
        except SystemExit as exc:
            assert "without --replace" in str(exc)
        else:
            raise AssertionError("a figure is replaced only when asked")
        assert figure.read_bytes() == b"earlier figure"
        # A failed draw leaves the old figure and no partial file.
        import matplotlib.figure
        real_savefig = matplotlib.figure.Figure.savefig
        def failing_savefig(self, fname, *a, **k):
            Path(fname).write_bytes(b"half a figure")
            raise OSError(28, "No space left on device")
        matplotlib.figure.Figure.savefig = failing_savefig
        try:
            rd.redraw(clip, figure, "experimental", (0, 99.99), replace=True)
        except OSError:
            pass
        else:
            raise AssertionError("the failing draw must raise")
        finally:
            matplotlib.figure.Figure.savefig = real_savefig
        assert figure.read_bytes() == b"earlier figure"
        assert sorted(p.name for p in Path(tmp).iterdir()) == ["X_Comparison.png", "X_Synthetic_Video.npz"]
        rd.redraw(clip, figure, "experimental", (0, 99.99), replace=True)
        assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert sorted(p.name for p in Path(tmp).iterdir()) == ["X_Comparison.png", "X_Synthetic_Video.npz"]
        # Two clips that would draw the same figure are refused.
        other = Path(tmp) / "sub"
        other.mkdir()
        np.savez_compressed(other / "X_Synthetic_Video.npz", **_fixture_fields(exp, syn))
        try:
            rd.plan_redraws([clip, other / "X_Synthetic_Video.npz"], out_dir=Path(tmp) / "out")
        except SystemExit as exc:
            assert "the same figure" in str(exc)
        else:
            raise AssertionError("duplicate figure targets must be refused")


def test_each_comparison_and_count_has_its_own_folder():
    for module in (tail, spots):                          # labels are written as the engine writes them
        assert module.label_token("REF-DENSITY") == "REF_DENSITY" == module.label_token("REF_DENSITY")
        assert module.label_token("BINOMIAL4") == "BINOMIAL4"
        assert (ppv._build_stem("A", "2S_50FPS", "MET-FAB", 0, None, "cell-sgm", "20S", run_label="REF-DENSITY",
                                declared_rds=True, map_block="rds").endswith("_" + module.label_token("REF-DENSITY")))
    for module, analysis in ((tail, "Tail_Structure"), (spots, "Spot_Count")):
        a = module.output_dir(Path("/posit"), "ALIAS_FAB", "2S_50FPS", "REF")
        b = module.output_dir(Path("/posit"), "ALIAS_FAB", "2S_50FPS", "REF_DENSITY")
        assert a == Path(f"/posit/ALIAS_FAB_2S_50FPS_Posterior_Predictive_Video_{analysis}_REF")
        assert b.name == f"ALIAS_FAB_2S_50FPS_Posterior_Predictive_Video_{analysis}_REF_DENSITY" and a != b
        try:
            module.output_dir(Path("/posit"), "ALIAS_FAB", "2S_50FPS", "--")
        except SystemExit:
            pass
        else:
            raise AssertionError("a label without a character must be refused")


def test_a_labeling_arm_records_itself_and_refuses_before_any_work():
    exp, syn = _partly_overlapping_clips()
    source = _fixture_fields(exp, syn)
    source["dye_counts"] = np.array([0, 1, 3, 0, 2], dtype=np.int64)
    counts = np.array([0, 2, 1, 0, 1], dtype=np.int64)
    from srm_and_sbi_monomer_dimer_alp.labeling import LABELING_LAWS
    fields = arm.arm_fields(source, "SRC_Synthetic_Video.npz", counts, source["labeling_row"], syn.copy(),
                            "FAB_BINOMIAL", LABELING_LAWS["FAB_BINOMIAL"], "BINOMIAL4")
    record = json.loads(fields["labeling_record_json"])
    assert record["law_name"] == "FAB_BINOMIAL" and record["arm"]["label"] == "BINOMIAL4"
    assert record["arm"]["source_dye_counts"] == {"1": 1, "2": 1, "3": 1}      # JSON keys are strings
    assert fields["synth_label"] == "SYNTH (declared reaction-diffusion; labeling arm FAB_BINOMIAL)"
    assert fields["n_dyes"] == 4 and np.array_equal(fields["dye_counts"], counts)
    x = ppv.comparison_figure_inputs(fields, biology_workflow())
    assert x["motion_desc"] == "the source render's trajectory (reused, not re-simulated)"
    assert "(derived; the source's labeled subunits kept)" in x["labeling_desc"]
    assert x["labeling_desc"].startswith("FAB FAB_BINOMIAL = Binomial(sites=4, mean=1.64)")
    # The arm refuses an existing output before rendering, and names its own option.
    tree = ast.parse(ARM.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = sorted((n for n in ast.walk(main) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset))
    names = [_call_name(c.func) for c in calls]
    assert names.index("refuse_existing_outputs") < names.index("render_arm")
    refusal = calls[names.index("refuse_existing_outputs")]
    assert any(k.arg == "hint" and "--arm-label" in k.value.value for k in refusal.keywords)
    with tempfile.TemporaryDirectory() as tmp:
        existing = Path(tmp) / "X_Synthetic_Video.npz"
        existing.write_bytes(b"earlier arm")
        try:
            ppv.refuse_existing_outputs({"clip": existing}, hint="Another labeling arm of this render needs its own --arm-label.")
        except SystemExit as exc:
            assert str(exc).endswith("needs its own --arm-label.") and "--run-label" not in str(exc)
        else:
            raise AssertionError("an existing arm output must be refused")


def test_the_spot_count_report_states_each_recordings_receptor_total():
    def results(totals):
        cells = {}
        for cell, n_r in totals.items():
            window = {"experimental": {"mean": 100.0}, "R (FAB_POISSON)": {"mean": 90.0}}
            cells[str(cell)] = {"count_total_declared": n_r, "visible_spots_0": {"R (FAB_POISSON)": 150},
                                "eta_opening": {"R (FAB_POISSON)": 0.6},
                                "windows": {"opening 100 frames": window, "closing 100 frames": window}}
        return {"package_version": "test", "kind": "MET-FAB", "source_label": "R", "arm_labels": [],
                "columns": ["R (FAB_POISSON)"], "imaging_vector": {"mu_r": 1.585}, "cells": cells}
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "Spot_Count.md"
        spots.write_report(results({0: 550.0, 16: 3162.0}), report)
        text = report.read_text()
        assert "The receptor total is declared per recording (column N_R)" in text
        assert "| 0 | 550 | 150 | 0.60 |" in text and "| 16 | 3162 | 150 | 0.60 |" in text
        assert "Every render declares" not in text and "same receptor total" not in text
        spots.write_report(results({0: 1000.0, 5: 1000.0}), report)
        assert "Every render declares the receptor total N_R = 1000." in report.read_text()


def test_every_video_renderer_defaults_to_the_percentile_window():
    import inspect
    import nbformat
    from srm_and_sbi_monomer_dimer_alp import visualization_dli
    # The engine, the figure redraw and the player script (through the notebook).
    assert ppv.DISPLAY_NORM_DEFAULT == "percentile" and ppv.DISPLAY_PERCENTILES == (0.0, 99.99)
    assert pl.NORM_MODES[0] == "percentile" and ppv.build_parser("t").get_default("display_norm") == "percentile"
    exp, syn = _partly_overlapping_clips()
    calls = []
    real_redraw = rd.redraw
    rd.redraw = lambda clip, figure, display_norm, pair, replace=False: calls.append((display_norm, tuple(pair)))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "X_Synthetic_Video.npz"
            np.savez_compressed(clip, **_fixture_fields(exp, syn))
            rd.main([str(clip), "--out-dir", str(Path(tmp) / "out")])
    finally:
        rd.redraw = real_redraw
    assert calls == [("percentile", (0.0, 99.99))], calls
    def config_cell(path):
        return "".join(c.source for c in nbformat.read(path, as_version=4).cells if c.cell_type == "code")
    ppv_nb = config_cell(pl.NOTEBOOK)
    assert 'NORM_MODE = "percentile"' in ppv_nb and "NORM_PERCENTILES = (0.0, 99.99)" in ppv_nb
    # The training-set viewer: one fixed window per simulation, for its scrubber and its player.
    scrubber = config_cell(REPO / "notebooks/Video_Scrubber.ipynb")
    assert "DISPLAY_PERCENTILES = (0.0, 99.99)" in scrubber and scrubber.count("vmin=vmin, vmax=vmax") == 2
    # The brightness-audit viewer and its side-by-side videos.
    audit = config_cell(REPO / "notebooks/SRM_AND_SBI_MONOMER_DIMER_ALP_Brightness_Stationarity_Audit_Video.ipynb")
    assert 'NORM_MODE = "percentile"' in audit and "PCTL = 99.99" in audit
    assert "autoscale" not in (REPO / "notebooks/SRM_AND_SBI_MONOMER_DIMER_ALP_Brightness_Stationarity_Audit_Video.ipynb").read_text()
    assert "np.percentile(s, 99.99)" in (REPO / "Script_Bank/Analysis/assemble_side_by_side.py").read_text()
    # The package's interactive player.
    assert inspect.signature(visualization_dli.animate_video).parameters["display_percentiles"].default == (0.0, 99.99)


def test_the_viewer_cells_show_the_window_and_the_labeling():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nbformat
    exp, syn = _partly_overlapping_clips()
    inlb = json.loads(json.dumps(resolve_labeling("INLB", None, "A=0.5,B=1.0").record()))
    arm_record = dict(resolve_labeling("FAB").record(), arm={"label": "BINOMIAL4"})
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "A_Synthetic_Video.npz"
        np.savez_compressed(clip, **_fixture_fields(exp, syn, labeling_record=inlb))
        arm_clip = Path(tmp) / "B_Synthetic_Video.npz"
        np.savez_compressed(arm_clip, **_fixture_fields(exp, syn, labeling_record=arm_record))
        copy = pl.prepare_notebook(nbformat.read(pl.NOTEBOOK, as_version=4), clip)
        cells = [pl._source(c) for c in copy.cells if c.cell_type == "code"]
        ns, printed = _run_config(cells[1])
        # The labeling line: a per-species occupancy (it raised before), compact numbers; an arm is marked.
        assert "occupancy A=0.5, B=1 (override)" in printed, printed
        _, arm_printed = _run_config(_notebook_config(arm_clip))
        assert "(derived; the source's labeled subunits kept)" in arm_printed
        # The notebook's number format is the engine's.
        for v in (0.15508, 0.0316, 0.003162, 250.73, 12345.6, 0.0, -0.0, 1e-7, 41.84, 0.6744, 139.0, 1.585):
            assert ns["_num"](v) == ppv.compact_number(v), v
        # The scrubber and the player draw every panel under the one window and name it in their titles.
        lo, hi = ppv.display_window(exp, syn, "percentile")["clim"]
        ns["plt"] = plt                                          # the notebook's first cell imports it
        exec(compile(cells[2], "<scrubber cell>", "exec"), ns)
        plt.close("all")
        ns["show"](1, 4, 4, 1.0)
        fig = plt.gcf()
        assert all(im.get_clim() == (lo, hi) for a in fig.axes for im in a.get_images())
        assert f"display window [{lo:.0f}, {hi:.0f}] ADU (percentile)" in fig._suptitle.get_text()
        plt.close("all")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(cells[3], "<player cell>", "exec"), ns)
        assert "3 of 3 frames embedded" in out.getvalue()
        assert all(im.get_clim() == (lo, hi) for im in ns["panels"])
        assert f"window [{lo:.0f}, {hi:.0f}] ADU (percentile)" in ns["fig"]._suptitle.get_text()
        plt.close("all")


def test_the_training_set_viewer_uses_one_percentile_window_per_simulation():
    import types
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nbformat
    cells = ["".join(c.source) for c in nbformat.read(REPO / "notebooks/Video_Scrubber.ipynb", as_version=4).cells
             if c.cell_type == "code"]
    rng = np.random.default_rng(3)
    videos = rng.integers(200, 1500, size=(2, 6, 16, 16)).astype(np.uint16)
    videos[0, 2, 3, 3] = 40000                                   # one hot pixel must not set the top
    ns = {"np": np, "plt": plt, "videos": videos, "n_sims": 2, "n_frames": 6,
          "interact": lambda *a, **k: None, "IntSlider": lambda **k: None,
          "PARAMETERS": types.SimpleNamespace(simulation=types.SimpleNamespace(
              timing=types.SimpleNamespace(frame_time_seconds=0.02)))}
    helper = cells[1][cells[1].index("# The color window of a simulation"):]
    exec(compile(helper, "<window helper>", "exec"), ns)
    expected = tuple(float(np.percentile(videos[0], q)) for q in (0.0, 99.99))
    assert ns["clip_window"](0) == expected and expected[1] < 40000
    exec(compile(cells[2], "<scrubber cell>", "exec"), ns)
    plt.close("all")
    ns["show"](0, 2)
    assert plt.gcf().axes[0].get_images()[0].get_clim() == expected
    plt.close("all")
    exec(compile(cells[3].replace("HTML(anim.to_jshtml())", "anim.to_jshtml()"), "<player cell>", "exec"), ns)
    assert ns["im"].get_clim() == expected and f"window [{expected[0]:.0f}, {expected[1]:.0f}]" in ns["ax"].get_title()
    plt.close("all")

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
