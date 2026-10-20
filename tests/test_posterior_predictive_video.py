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
5. no experimental detections, no fluorescent host, no synthetic detection and a non-finite update each
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
    # n_subunits, n_dyes, n_labeled_subunits, monomers_0, monomers_fluorescent_0, dimers_0, dimers_fluorescent_0,
    # dimers_two_labeled_0, occupancy_monomer, occupancy_dimer, dimers_one_probe_0, dimers_two_probe_0,
    # two_probe_share (labeling.LABELING_SET_COLUMNS)
    row = np.array([1000, 261, 136, 860, 124, 70, 10, 2, 1.0, 0.5037, 68, 2, 0.00744])
    # A declared-configuration biology render: the figure names its source and shows no MAP.
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    experimental = rng.integers(150, 400, size=(4, 16, 16)).astype(np.uint16)
    synth = rng.integers(150, 400, size=(4, 16, 16)).astype(np.uint16)
    fab_table = bio.parameterization_of("FAB")
    rds = np.array([bio.prior_center(e) for e in fab_table])
    with tempfile.TemporaryDirectory() as tmp, matplotlib.rc_context({"svg.fonttype": "none"}):
        path = Path(tmp) / "Comparison.svg"                 # text kept as text, so it can be read back
        ppv._save_comparison_png(
            path, experimental, synth, "FAB", 0, None, "full", rds, ppv._fixed_imaging_theta(),
            synth_label=ppv.synthetic_source_label("rds", declared_biology=True), labeling_row=row,
            rds_table=fab_table,
            imaging_label="FIXED imaging (calibrated Nuisance_DLI + MET SCOPE)",
            rds_label="DECLARED reaction-diffusion (FAB_DECLARED_RDS, absolute)",
            motion_desc="from the declared reaction-diffusion configuration (not a draw)",
            labeling_desc="FAB_POISSON, retained population, occupancy 0.0148")
        text = re.sub(r'"data:[^"]*"', '""', path.read_text())   # drop the embedded image data
    assert "SYNTH (declared reaction-diffusion) FAB c0" in text
    assert "Posterior-predictive video check: experimental vs SYNTH (declared reaction-diffusion)" in text
    assert not re.search(r"\bMAP\b", text), re.findall(r".{0,40}\bMAP\b.{0,40}", text)
    # The receptors the render started with, stated explicitly.
    assert "1000 subunits = 860 monomers + 70 dimers (930 receptors)" in text
    assert "labeled: 136 subunits carrying 261 dyes" in text
    # Long provenance lines wrap inside the panel (no line exceeds the width); both halves are stated.
    assert "fluorescent: 124 of 860 monomers + 10 of 70 dimers (2 with" in text
    assert "both subunits labeled) = 134 spots" in text
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
                                         labeling_row=row, rds_table=fab_table,
                                         labeling_desc=ppv.labeling_description(
                                             resolve_labeling("FAB", "binomial:1.6412:4").record()))
        finally:
            matplotlib.axes.Axes.text = real_text
        assert len(blocks) == 1
        return blocks[0]
    imaging = ppv._fixed_imaging_theta({"sigma_r": 0.23714, "prob_photo_bleach": 0.056234})
    rds_vector = rds.copy()
    rds_vector[[e["KEY"] for e in fab_table].index("fraction_basal_dimers_reference")] = 0.0316228
    block = provenance_block(imaging, rds_vector)
    assert max(len(line) for line in block.split("\n")) <= ppv.PROVENANCE_WIDTH
    assert "sigma_r=0.237" in block and "p_bleach=0.056" in block and "0.2371" not in block
    # the fraction row's compact value, labeled by the row's LABEL stripped of the TeX markers
    assert "f_B^ref=0.032" in block and "0.0316" not in block and "r_B/A" not in block
    assert "R_active" not in block, "a Fab vector carries no allocation row"
    # the table is required: a vector of another width is refused, never relabeled by position
    try:
        provenance_block(imaging, np.append(rds, 0.5))
    except ValueError as exc:
        assert "must correspond" in str(exc)
    else:
        raise AssertionError("a vector of another width must be refused")
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
            rds_table=fab_table, display_percentiles=(1, 99.9))
        text = re.sub(r'"data:[^"]*"', '""', path.read_text())
    assert "norm percentile [p1, p99.9]" in text


# ==========================================================================================
# The receptor-total match
# ==========================================================================================

V = 0.011894875                # fluorescent hosts per TRUE subunit under MET-FAB at the declared f = 0.075
BOX = (1000.0, 100000.0)
KW = dict(initial=6000, tolerance=0.03, max_iterations=6, max_count=100000, min_elasticity=0.3,
          prior_box=BOX)
DECIDED = dict(KW, allow_outside_prior=True)             # the user's decision to render beyond the box


def _linear(eta=0.67, calls=None):
    """Detection proportional to the fluorescent hosts, which equal their expectation; ``calls``
    collects the counts rendered."""
    def render(n_r, it):
        if calls is not None:
            calls.append(n_r)
        return n_r * V, eta * n_r * V
    return render


def test_linear_response_converges_to_the_predicted_count():
    target = 0.67 * V * 5340.0
    out = cm.iterate_count(target, V, _linear(), **KW)
    assert out["status"] == "matched", out
    assert abs(out["matched_count_total"] - target / (0.67 * V)) <= 1
    assert out["matched_count_total"] == 5340


def test_undefined_paths_carry_no_count():
    assert cm.iterate_count(0.0, V, _linear(), **KW)["status"] == "undefined_no_experimental_detections"
    assert cm.iterate_count(float("nan"), V, _linear(), **KW)["status"] == "undefined_no_experimental_detections"
    assert cm.iterate_count(44.0, 0.0, _linear(), **KW)["status"] == "undefined_no_fluorescence"
    out = cm.iterate_count(44.0, V, lambda n, it: (0, 0.0), **KW)
    assert out["status"] == "undefined_no_fluorescent_hosts" and out["matched_count_total"] is None
    out = cm.iterate_count(44.0, V, lambda n, it: (70, 0.0), **KW)
    assert out["status"] == "undefined_no_synthetic_detections" and out["matched_count_total"] is None
    out = cm.iterate_count(44.0, V, lambda n, it: (70, float("inf")), **KW)
    assert out["status"] == "undefined_no_synthetic_detections"


def test_saturation_is_reported_not_chased():
    """Detection that caps below the recording's count cannot reach it: unresolved, no count."""
    def capped(cap):
        return lambda n_r, it: (n_r * V, min(0.67 * n_r * V, cap))
    out = cm.iterate_count(200.0, V, capped(150.0), **dict(KW, initial=10000))
    assert out["status"] == "unresolved_saturation", out
    assert out["matched_count_total"] is None and "saturation" in out["reason"]
    assert max(r["count_total"] for r in out["iterations"]) <= BOX[1]
    out = cm.iterate_count(400.0, V, capped(300.0), **dict(DECIDED, initial=30000))
    assert out["status"] == "unresolved_saturation", out
    assert out["matched_count_total"] is None
    assert max(r["count_total"] for r in out["iterations"]) <= 100000


def test_budget_and_count_ceiling_carry_no_count():
    # One iteration from 6000 proposes 5340 (11 % away): the budget ends before agreement.
    out = cm.iterate_count(0.67 * V * 5340.0, V, _linear(), **dict(KW, max_iterations=1))
    assert out["status"] == "not_converged" and out["matched_count_total"] is None
    assert out["last_count_total"] == 6000
    # A target needing about 200000 receptors under a 100000 ceiling, rendering beyond the box decided.
    calls = []
    out = cm.iterate_count(0.67 * V * 200000.0, V, _linear(calls=calls), **dict(DECIDED, initial=30000, max_iterations=10))
    assert out["status"] == "unresolved_max_count", out
    assert out["matched_count_total"] is None and max(calls) <= 100000
    # The ceiling holds from the first render: an initial count above it renders nothing.
    for kw in (KW, DECIDED):
        calls = []
        out = cm.iterate_count(0.67 * V * 5340.0, V, _linear(calls=calls), **dict(kw, initial=200000))
        assert out["status"] == "unresolved_max_count" and calls == [], (out, calls)


def test_nothing_is_rendered_outside_the_prior_without_the_users_decision():
    target = 0.67 * V * 600.0                   # the arithmetic's answer is N_total = 600, below the box floor
    calls = []
    out = cm.iterate_count(target, V, _linear(calls=calls), **dict(KW, initial=1000))
    assert out["status"] == "user_decision_outside_prior", out
    assert out["matched_count_total"] is None and calls == [1000]
    # The first count is tested before its render too.
    calls = []
    out = cm.iterate_count(target, V, _linear(calls=calls), **dict(KW, initial=500))
    assert out["status"] == "user_decision_outside_prior" and calls == [], (out, calls)
    # A converged value just outside the box (rendered 1050, proposed 990) is not adopted either.
    calls = []
    out = cm.iterate_count(0.67 * V * 990.0, V, _linear(calls=calls), **dict(KW, initial=1050))
    assert out["status"] == "user_decision_outside_prior" and calls == [1050], (out, calls)
    # Under the user's decision the count is rendered and accepted, flagged, not clipped.
    calls = []
    out = cm.iterate_count(target, V, _linear(calls=calls), **dict(DECIDED, initial=1000))
    assert out["status"] == "matched_outside_prior" and out["matched_count_total"] == 600, out
    assert calls == [1000, 600]
    # The render guard (the iteration's and the check renders') refuses before rendering.
    for n, decided, allowed in ((200000, False, False), (200000, True, False), (500, False, False),
                                (500, True, True), (10000, False, True)):
        calls = []
        render = cm.guarded_render(_linear(calls=calls), max_count=100000, prior_box=BOX,
                                   allow_outside_prior=decided)
        try:
            render(n, 100)
        except RuntimeError as exc:
            assert not allowed and "refusing to render" in str(exc) and calls == [], (n, decided)
        else:
            assert allowed and calls == [n], (n, decided)


def test_an_initial_count_beyond_the_limits_stops_the_run_before_any_work():
    """The entry point runs end to end but isolated from the data bank: its output folder is redirected
    to a temporary directory, and the work function (the recording read, every render, the reports) is
    replaced by a stub that records the call and refuses -- a guard regression trips the stub instead of
    writing anywhere. The last case checks that the isolation is live: a count inside the limits reaches
    the stub, under the temporary folder."""
    from unittest import mock
    from srm_and_sbi_monomer_dimer_alp import experiment_support
    common = ["--total-time-seconds", "2", "--kind", "MET-FAB", "--cell", "0", "--nuisance-tag", "REF",
              "--declared-rds", str(TOML), "--attempt-label", "TEST_REFUSED_BEFORE_ANY_WORK"]
    real_context, reached = cm._context, []

    def forbidden(what):
        def _stub(*args, **kwargs):
            raise AssertionError(f"{what} must not be reached by this test")
        return _stub

    with tempfile.TemporaryDirectory() as tmp:
        def isolated_context(args):
            ctx = real_context(args)
            ctx["folder"] = Path(tmp) / ctx["folder"].name
            ctx["stems"] = {c: ctx["folder"] / s.name for c, s in ctx["stems"].items()}
            return ctx

        def recording_match_one(args, ctx, cell):
            reached.append(Path(ctx["folder"]))
            raise AssertionError("match_one must not be reached by this test")

        with mock.patch.object(cm, "_context", isolated_context), \
             mock.patch.object(cm, "match_one", recording_match_one), \
             mock.patch.object(experiment_support, "read_recording", forbidden("read_recording")), \
             mock.patch.object(ppv, "simulate_and_render", forbidden("simulate_and_render")):
            for extra, expected in ((["--initial-count", "200000"], "exceeds --max-count"),
                                    (["--initial-count", "500"], "outside the prior box")):
                try:
                    cm.main(common + extra)
                except SystemExit as exc:
                    assert expected in str(exc) and "--initial-count" in str(exc), str(exc)
                else:
                    raise AssertionError(f"{extra} must stop the run")
            assert reached == [], reached
            try:
                cm.main(common + ["--initial-count", "10000"])
            except AssertionError as exc:
                assert "match_one must not be reached" in str(exc), str(exc)
            else:
                raise AssertionError("a count inside the limits must reach the (stubbed) work function")
            assert len(reached) == 1 and str(reached[0]).startswith(tmp), reached
        assert not any(Path(tmp).rglob("*")), "nothing is written before the work function"


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
                    synth_label="SYNTH (declared reaction-diffusion)", rds_outside=(), condition="FAB",
                    rds_keys=None, rds_provenance=None):
    """A clip's fields as the engine saves them (the keys of its ``np.savez_compressed`` call); the
    reaction-diffusion vector and its keys are the ``condition``'s unless given."""
    plan = resolve_labeling(condition)
    record = ({"scenario": {"name": f"{condition}_DECLARED_RDS"}, "path": f"Declared_RDS_{condition}.toml",
               "sha256": "0" * 64} if rds_source == "declared" else {"basis": "fixture"})
    table = bio.parameterization_of(condition)
    composition = bio.realize_initial_composition(
        count_total=1e4, fraction_basal_dimers_reference=0.5,
        relative_count_active_dimers_initial=(0.0 if condition == "FAB" else 0.5), condition=condition)
    return dict(
        experimental=np.asarray(experimental, dtype=np.uint16), synth=np.asarray(synth, dtype=np.uint16),
        imaging_physical=ppv._fixed_imaging_theta(), imaging_keys=np.array(det.DETECTOR_IMAGING_KEYS),
        map_theta=np.array([]) if map_theta is None else np.asarray(map_theta, dtype=float),
        map_keys=np.array(["k"]), map_block=map_block, workflow=workflow,
        rds_provenance=(np.array([bio.prior_center(e) for e in table]) if rds_provenance is None
                        else np.asarray(rds_provenance, dtype=float)),
        rds_keys=np.array(ppv._rds_keys(condition) if rds_keys is None else list(rds_keys)),
        rds_source=rds_source, rds_record_json=json.dumps(record),
        rds_declared_json=json.dumps(bio.declared_row_values(condition)),
        initial_composition_json=json.dumps(ppv.initial_composition_record(composition)),
        rds_outside_prior=np.array(list(rds_outside), dtype=str), condition=condition, labeling_law=plan.law_name,
        dye_counts=np.zeros(10, dtype=np.int64), n_subunits=1000, n_dyes=261,
        labeling_columns=np.array(LABELING_SET_COLUMNS),
        labeling_row=np.array([1000, 261, 136, 860, 124, 70, 10, 2, 1.0, 0.5037, 68, 2, 0.00744]),
        occupancy_source=plan.occupancy_source, occupancy_monomer=1.0, occupancy_dimer=0.5037,
        labeling_record_json=json.dumps(labeling_record if labeling_record is not None else plan.record()),
        imaging_record_json=json.dumps(imaging_identity if imaging_identity is not None else {"artifact": "REF"}),
        synth_label=synth_label, package_version="test", kind=f"MET-{condition}", cell=0, chunk=-1,
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
        assert line.startswith(f"{plan.condition} {plan.law_name} = {ppv._law_text(plan.record()['law'])}, ")
        assert f"({plan.occupancy_source}); fluorescent per " in line
        # At most three decimal places; the law's name is an identifier, printed as stored.
        assert not re.search(r"\d\.\d{4,}", line.replace(plan.law_name, "")), line
    assert "Poisson(mean=1.641)" in ppv.labeling_description(resolve_labeling("FAB", "poisson:1.64123").record())
    assert (ppv.labeling_description(resolve_labeling("FAB").record())
            == "FAB FAB_POISSON = Poisson(mean=1.64), retained population: probe classes drawn, two-probe share 0.007, "
               "occupancy 0.015 (derived); fluorescent per retained subunit A 0.806, B 0.406")
    assert "occupancy coins A=0.5, B=1 (override)" in ppv.labeling_description(
        json.loads(json.dumps(resolve_labeling("INLB", None, "A=0.5,B=1.0").record())))
    exp, syn = _partly_overlapping_clips()
    bio_cfg, det_cfg = biology_workflow(), detector_workflow()
    # A declared biology render.
    x = ppv.comparison_figure_inputs(_fixture_fields(exp, syn), bio_cfg)
    assert x["imaging_label"] == "FIXED imaging (calibrated Nuisance_DLI tag REF + MET SCOPE)"
    assert x["rds_label"] == "DECLARED reaction-diffusion (FAB_DECLARED_RDS, absolute)"
    assert x["motion_desc"] == "from the declared reaction-diffusion configuration (not a draw)"
    assert x["sel_desc"] is None and not x["fixed_imaging"] and not x["fixed_nuisance"]
    assert len(x["rds_table"]) == len(bio.parameterization_of("FAB")) == 11
    assert [e["KEY"] for e in x["rds_table"]] == ppv._rds_keys("FAB")
    assert x["synth_label"] == "SYNTH (declared reaction-diffusion)"
    assert x["labeling_desc"] == ppv.labeling_description(resolve_labeling("FAB").record())
    assert x["cell"] == 0 and x["kind"] == "MET-FAB"
    assert np.array_equal(x["experimental"], exp) and np.array_equal(x["synth"], syn)
    # A labeling arm: the source's labeled subunits kept, the source's trajectory reused.
    arm_record = dict(resolve_labeling("FAB").record(), law_name="FAB_BINOMIAL",
                      law={"family": "binomial", "mean": 1.64, "shape": 4, "description": "Binomial(sites=4, mean=1.64)"},
                      arm={"label": "BINOMIAL4"})
    x = ppv.comparison_figure_inputs(_fixture_fields(exp, syn, labeling_record=arm_record), bio_cfg)
    assert x["motion_desc"] == "the source render's trajectory (reused, not re-simulated)"
    assert x["labeling_desc"].startswith("FAB FAB_BINOMIAL = Binomial(sites=4, mean=1.64), retained population: ")
    assert "(derived; the source's labeled subunits kept); fluorescent per retained subunit A 0.806, B 0.406" in x["labeling_desc"]
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
    assert x["fixed_imaging"] and x["fixed_nuisance"] and x["sel_desc"] is None
    assert len(x["rds_table"]) == 11, "the detector's nuisance is labeled by the condition's biology table"
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
        assert x["fixed_imaging"] is fixed and len(x["rds_table"]) == 11
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


def test_an_inlb_fixture_states_twelve_values_by_key_and_a_permuted_fixture_is_refused():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.axes
    exp, syn = _partly_overlapping_clips()
    bio_cfg = biology_workflow()
    inlb_keys = ppv._rds_keys("INLB")
    assert len(inlb_keys) == 12 and inlb_keys[:11] == ppv._rds_keys("FAB")
    assert inlb_keys[-1] == "relative_count_active_dimers_initial"
    fields = _fixture_fields(exp, syn, condition="INLB")
    x = ppv.comparison_figure_inputs(fields, bio_cfg)
    assert [e["KEY"] for e in x["rds_table"]] == inlb_keys and x["nuisance"].shape == (12,)
    assert x["kind"] == "MET-INLB" and x["rds_label"] == "DECLARED reaction-diffusion (INLB_DECLARED_RDS, absolute)"
    # the figure's provenance states all twelve values, each by its key's label
    blocks, real_text = [], matplotlib.axes.Axes.text
    def recording_text(self, px, py, s, *a, **k):
        if k.get("family") == "monospace":
            blocks.append(s)
        return real_text(self, px, py, s, *a, **k)
    matplotlib.axes.Axes.text = recording_text
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ppv.draw_comparison_figure(fields, Path(tmp) / "C.png", bio_cfg)
    finally:
        matplotlib.axes.Axes.text = real_text
    flat = " ".join(line.strip() for line in blocks[0].split("\n"))
    for entry in bio.parameterization_of("INLB"):
        label = entry["LABEL"].translate({ord(c): None for c in "${}\\"})
        assert f"{label}=" in flat, label
    assert "R_active=0.5" in flat and "f_B^ref=0.5" in flat
    # the clip's composition record carries the starting counts of the InlB prior center at N = 1e4
    composition = json.loads(fields["initial_composition_json"])
    assert (composition["n_monomers"], composition["n_dimers_one_probe"], composition["n_dimers_two_probe"]) == (79, 302, 48)
    # a permuted key order, the other condition's keys, and a clip without keys are refused
    for bad in (list(reversed(inlb_keys)), ppv._rds_keys("FAB") + ["rate_dissociation"], ppv._rds_keys("FAB")):
        bad_fields = _fixture_fields(exp, syn, condition="INLB", rds_keys=bad,
                                     rds_provenance=np.ones(len(bad)))
        try:
            ppv.comparison_figure_inputs(bad_fields, bio_cfg)
        except ValueError as exc:
            assert "rds_keys" in str(exc) and "refused" in str(exc), str(exc)
        else:
            raise AssertionError(f"keys {bad} must be refused")
    keyless = dict(fields)
    del keyless["rds_keys"]
    try:
        ppv.comparison_figure_inputs(keyless, bio_cfg)
    except ValueError as exc:
        assert "rds_keys" in str(exc)
    else:
        raise AssertionError("a clip without rds_keys must be refused")
    # a Fab fixture carrying the allocation row as its twelfth key is refused as well
    try:
        ppv.comparison_figure_inputs(_fixture_fields(exp, syn, condition="FAB", rds_keys=inlb_keys,
                                                     rds_provenance=np.ones(12)), bio_cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("a Fab clip with twelve keys must be refused")
    # the vector sources take the condition: a drawn InlB nuisance has twelve values inside the box,
    # a pinned one sits at the prior centers (0.5 for both fraction rows)
    drawn = ppv._draw_nuisance_physical("INLB", np.random.default_rng(0))
    assert drawn.shape == (12,) and ppv.rds_outside_prior(drawn, "INLB") == []
    assert ppv._draw_nuisance_physical("FAB", np.random.default_rng(0)).shape == (11,)
    pinned = ppv._fixed_nuisance_physical("INLB")
    assert pinned[inlb_keys.index("fraction_basal_dimers_reference")] == 0.5 and pinned[-1] == 0.5
    try:
        ppv._fixed_nuisance_physical("FAB", overrides={"relative_count_active_dimers_initial": 0.3})
    except ValueError:
        pass
    else:
        raise AssertionError("the allocation is not a Fab nuisance key")
    try:
        ppv.rds_outside_prior(drawn, "FAB")
    except ValueError:
        pass
    else:
        raise AssertionError("a twelve-value vector is not a Fab vector")
    # The run records, in every clip, the keys of its reaction-diffusion vector, the rows its condition
    # declares as constants and the window-start composition it realized: a declared render of each
    # condition, end to end, with the simulation stubbed (the stub places exactly the composition the
    # engine realized, as the generator does) and the imaging resolution replaced by a fixed vector.
    from unittest import mock
    toml_text = TOML.read_text()
    calls = []

    def stub_simulate(condition, rds_vector, imaging_physical, plan, n_frames, traj_path, seed=None, verbose=False):
        calls.append((condition, np.asarray(rds_vector, dtype=float).copy()))
        c = bio.population_from_theta(np.asarray(rds_vector, dtype=float), condition).item(())
        row = dict(n_subunits=c.n_subunits, n_dyes=c.n_subunits, n_labeled_subunits=c.n_subunits,
                   monomers_0=c.n_monomers, monomers_fluorescent_0=c.n_monomers, dimers_0=c.n_dimers,
                   dimers_fluorescent_0=c.n_dimers, dimers_two_labeled_0=c.n_dimers_two_probe,
                   occupancy_monomer=1.0, occupancy_dimer=1.0, dimers_one_probe_0=c.n_dimers_one_probe,
                   dimers_two_probe_0=c.n_dimers_two_probe, two_probe_share=float(plan.two_probe_share))
        rng = np.random.default_rng(1)
        return {"synth": rng.integers(150, 400, size=(n_frames, 8, 8)).astype(float),
                "dye_counts": np.ones(c.n_subunits, dtype=np.int64),
                "labeling_row": np.array([row[k] for k in LABELING_SET_COLUMNS], dtype=float),
                "n_subunits": c.n_subunits}

    real_spec = ppv._ppv_spec
    for condition, extra in (("FAB", ""), ("INLB", '\n[parameters.relative_count_active_dimers_initial]\n'
                                                     'value = 0.5\nbasis = "prior center (test fixture)"\n'
                                                     'source = "test fixture"\n')):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            declared = tmp_dir / f"Declared_RDS_{condition}.toml"
            declared.write_text(toml_text.replace('condition = "FAB"', f'condition = "{condition}"') + extra)
            recording = tmp_dir / f"Experiment_{condition}_Cell_0_20S_RAW.tif"
            recording.write_bytes(b"not read: the recording readers are stubbed")

            def isolated_spec(cfg, args, _recording=recording, _out=tmp_dir / "out"):
                spec = real_spec(cfg, args)
                spec["experimental_tif"], spec["out_dir"] = _recording, _out
                spec["imaging_physical"] = lambda a, map_theta: (ppv._fixed_imaging_theta(), "fixture imaging",
                                                                 {"source": "fixture"})
                return spec

            args = ppv.build_parser("t").parse_args(["--total-time-seconds", "2", "--kind", f"MET-{condition}",
                                                    "--cell", "0", "--declared-rds", str(declared),
                                                    "--display-norm", "full", "--seed", "3"])
            rng = np.random.default_rng(2)
            with mock.patch.object(ppv, "_ppv_spec", isolated_spec), \
                 mock.patch.object(ppv, "inspect_recording", lambda path: (50, 8, 8)), \
                 mock.patch.object(ppv, "read_recording",
                                   lambda path: rng.integers(150, 400, size=(50, 8, 8)).astype(np.uint16)), \
                 mock.patch.object(ppv, "simulate_and_render", stub_simulate):
                assert ppv.run_posterior_predictive_video(biology_workflow(), args) == 0
            clip_path = next((tmp_dir / "out").glob("*_Synthetic_Video.npz"))
            assert next((tmp_dir / "out").glob("*_Comparison.png")).stat().st_size > 0
            with np.load(clip_path) as clip:
                keys = [str(k) for k in clip["rds_keys"]]
                assert keys == ppv._rds_keys(condition) and len(keys) == (11 if condition == "FAB" else 12)
                assert str(clip["condition"]) == condition and clip["rds_provenance"].shape == (len(keys),)
                assert json.loads(str(clip["rds_declared_json"])) == bio.declared_row_values(condition)
                recorded = json.loads(str(clip["initial_composition_json"]))
                value = dict(zip(keys, clip["rds_provenance"]))
                expect = bio.realize_initial_composition(
                    count_total=value["count_total"], fraction_basal_dimers_reference=value["fraction_basal_dimers_reference"],
                    relative_count_active_dimers_initial=value.get("relative_count_active_dimers_initial", 0.0),
                    condition=condition)
                assert recorded == ppv.initial_composition_record(expect)
                assert (recorded["n_monomers"], recorded["n_dimers"]) == ((127, 20) if condition == "FAB"
                                                                          else (expect.n_monomers, expect.n_dimers))
                if condition == "INLB":
                    assert recorded["n_pairs_allocated"] > 0, "the InlB render places allocated pairs"
            assert calls[-1][0] == condition and calls[-1][1].shape == (len(keys),)


def test_a_fraction_outside_the_box_is_refused_at_load_and_never_reaches_simulate_and_render():
    from unittest import mock
    from srm_and_sbi_monomer_dimer_alp import experiment_support
    text = TOML.read_text()
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "Declared_RDS_FAB_bad.toml"
        bad.write_text(text.replace("value = 0.075", "value = 1.2"))
        try:
            ppv.load_declared_rds(bad, "FAB", 0)
        except ValueError as exc:
            assert "[0, 1]" in str(exc) and "fraction_basal_dimers_reference" in str(exc), str(exc)
        else:
            raise AssertionError("a declared f of 1.2 must be refused at load")
        # the same through a --set-rds override, and the allocation key under Fab
        try:
            ppv.load_declared_rds(TOML, "FAB", 0, overrides={"fraction_basal_dimers_reference": 1.2})
        except ValueError as exc:
            assert "[0, 1]" in str(exc)
        else:
            raise AssertionError("an override of 1.2 must be refused")
        with_alloc = Path(tmp) / "Declared_RDS_FAB_alloc.toml"
        with_alloc.write_text(text + '\n[parameters.relative_count_active_dimers_initial]\nvalue = 0.0\n'
                                     'basis = "x"\nsource = "y"\n')
        try:
            ppv.load_declared_rds(with_alloc, "FAB", 0)
        except ValueError as exc:
            assert "declares" in str(exc) and "relative_count_active_dimers_initial" in str(exc)
        else:
            raise AssertionError("the allocation key under Fab must be refused")
        # the valid file loads with the condition's keys and the composition realizes
        vector, record = ppv.load_declared_rds(TOML, "FAB", 0)
        assert record["keys"] == ppv._rds_keys("FAB") and record["condition"] == "FAB"
        composition = ppv._preflight_initial_composition(vector, "FAB", "declared")
        assert (composition.n_monomers, composition.n_dimers) == (127, 20)
    # Both declared files reproduce, at every declared receptor total, the counts of the ratio form they
    # replaced (r = n_B / n_A = 0.08108, realized by the earlier initializer, transcribed here): the
    # fraction 0.075 changes the coordinate, not the population.
    p = bio.occupancy_of("FAB")

    def ratio_form(n_total, r=0.08108):
        n_monomers_true, n_dimers_true = n_total / (1.0 + 2.0 * r), r * n_total / (1.0 + 2.0 * r)
        return int(round(p * n_monomers_true)), int(round(p * (2.0 - p) * n_dimers_true))

    density = TOML.with_name(TOML.stem + "_Density_Adjusted.toml")
    checked = 0
    for path, cells in ((TOML, (0,)), (density, (0, 4, 5, 16, 26, 32, 43, 45, 49, 53))):
        for cell in cells:
            vector, record = ppv.load_declared_rds(path, "FAB", cell)
            value = dict(zip(record["keys"], vector))
            assert value["fraction_basal_dimers_reference"] == 0.075 and record["outside_prior"] == []
            c = ppv._preflight_initial_composition(vector, "FAB", "declared")
            assert (c.n_monomers, c.n_dimers) == ratio_form(value["count_total"]), (path.name, cell)
            checked += 1
    assert checked == 11
    # the preflight refuses a vector with f = 1.2 for every source name, before any simulation
    fab_table = bio.parameterization_of("FAB")
    vector = np.array([bio.prior_center(e) for e in fab_table])
    vector[ppv._rds_keys("FAB").index("fraction_basal_dimers_reference")] = 1.2
    for source in ("map", "pinned", "drawn"):
        try:
            ppv._preflight_initial_composition(vector, "FAB", source)
        except SystemExit as exc:
            assert source in str(exc) and "cannot be realized" in str(exc) and "clipped" in str(exc)
        else:
            raise AssertionError("f = 1.2 must be refused by the preflight")
    # in the run, the preflight precedes the simulation and follows the vector's resolution
    names = [_call_name(c.func) for c in _run_calls()]
    assert names.index("_preflight_initial_composition") < names.index("simulate_and_render")
    assert names.index("rds_outside_prior") < names.index("_preflight_initial_composition")
    # end to end with the MAP source: a MAP with f = 1.2 stops the run before simulate_and_render
    real_spec = ppv._ppv_spec

    def isolated_spec(cfg, args):
        S = real_spec(cfg, args)
        S["map_npz"] = Path(tmp_dir) / "MAP.npz"
        S["experimental_tif"] = Path(tmp_dir) / "Experiment_FAB_Cell_0_20S_RAW.tif"
        S["out_dir"] = Path(tmp_dir) / "out"
        S["imaging_physical"] = lambda a, map_theta: (ppv._fixed_imaging_theta(), "fixture imaging", {"source": "fixture"})
        return S

    def forbidden(what):
        def _stub(*args, **kwargs):
            raise AssertionError(f"{what} must not be reached")
        return _stub

    with tempfile.TemporaryDirectory() as tmp_dir:
        (Path(tmp_dir) / "MAP.npz").write_bytes(b"not read: _load_map_theta is stubbed")
        (Path(tmp_dir) / "Experiment_FAB_Cell_0_20S_RAW.tif").write_bytes(b"not read: inspect_recording is stubbed")
        args = ppv.build_parser("t").parse_args(["--total-time-seconds", "2", "--kind", "MET-FAB", "--cell", "0"])
        with mock.patch.object(ppv, "_ppv_spec", isolated_spec), \
             mock.patch.object(ppv, "_load_map_theta", lambda *a, **k: vector), \
             mock.patch.object(ppv, "inspect_recording", lambda path: (1000, 256, 256)), \
             mock.patch.object(ppv, "read_recording", lambda path: np.zeros((1000, 4, 4), dtype=np.uint16)), \
             mock.patch.object(experiment_support, "read_recording", forbidden("read_recording")), \
             mock.patch.object(ppv, "simulate_and_render", forbidden("simulate_and_render")):
            try:
                ppv.run_posterior_predictive_video(biology_workflow(), args)
            except SystemExit as exc:
                assert "map" in str(exc) and "cannot be realized" in str(exc), str(exc)
            else:
                raise AssertionError("a MAP with f = 1.2 must stop the run")
        assert not (Path(tmp_dir) / "out").exists(), "the refusal comes before any output folder"


def test_count_match_fluorescence_equals_the_helper_and_the_fluorescence_function():
    from srm_and_sbi_monomer_dimer_alp.labeling import expected_fluorescent_complexes
    plan = resolve_labeling("FAB")
    n_total = 1e7
    for f in (0.0, 0.075, 0.25, 1.0):
        pop = bio.realize_population(count_total=n_total, fraction_basal_dimers_reference=f,
                                     relative_count_active_dimers_initial=0.0, condition="FAB")
        expected = float(expected_fluorescent_complexes(pop, plan)) / n_total
        v = cm.fluorescent_hosts_per_subunit(plan, f)
        assert np.isclose(v, expected, rtol=2e-4), (f, v, expected)
    assert np.isclose(cm.fluorescent_hosts_per_subunit(plan, 0.075), 0.011894875, atol=2e-7)
    assert np.isclose(V, 0.011894875)


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


def test_a_labeling_arm_refuses_a_source_clip_of_another_table_before_any_work():
    """The labeling arm's preflight (``assert_source_keys``): the source clip's stored ``rds_keys``
    must be the condition's, key for key and in order. A clip without them, with the other
    condition's keys or with the retired ratio row stops the run before any output, under
    ``--dry-run`` too; the condition's own keys pass and the dry run writes nothing."""
    import dataclasses
    exp, syn = _partly_overlapping_clips()
    fab_keys = ppv._rds_keys("FAB")
    stale = {"no keys": None, "the other condition's keys": ppv._rds_keys("INLB"),
             "the retired ratio row": [k if k != "fraction_basal_dimers_reference" else "ratio_dimer_monomer_initial"
                                       for k in fab_keys]}
    original = bio.PARAMETERS.machine
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        object.__setattr__(bio.PARAMETERS, "machine",
                           dataclasses.replace(original, data_bank_root=root, scratch_data_bank_root=None))
        try:
            paths = biology_workflow().paths.with_condition("FAB")
            label = bio.RunTiming(total_time_seconds=2.0, frames=bio.PARAMETERS.simulation.timing).label
            out_dir = root / paths.posit_subdir / f"{paths.project_alias}_{label}_Posterior_Predictive_Video"
            out_dir.mkdir(parents=True)
            stem = ppv._build_stem(paths.project_alias, label, "MET-FAB", 0, None, "cell-sgm", "20S",
                                   run_label="REF", declared_rds=True, map_block="rds")
            source = ppv.render_output_paths(out_dir, stem)
            source["trajectory"].write_bytes(b"never read before the preflight")
            arm_clip = ppv.render_output_paths(out_dir, f"{stem}_BINOMIAL4")["clip"]
            argv = ["--total-time-seconds", "2", "--kind", "MET-FAB", "--cell", "0", "--source-label", "REF",
                    "--labeling-law", "FAB_BINOMIAL", "--arm-label", "BINOMIAL4"]
            for name, keys in stale.items():
                fields = _fixture_fields(exp, syn)
                if keys is None:
                    del fields["rds_keys"]
                else:
                    fields["rds_keys"] = np.array(keys)
                np.savez_compressed(source["clip"], **fields)
                for extra in ([], ["--dry-run"]):
                    before = sorted(q.name for q in out_dir.iterdir())
                    try:
                        arm.main(argv + extra)
                    except SystemExit as exc:
                        assert "rds_keys" in str(exc) and "refused" in str(exc), (name, str(exc))
                    else:
                        raise AssertionError(f"a source clip with {name} was accepted")
                    assert sorted(q.name for q in out_dir.iterdir()) == before and not arm_clip.exists()
            np.savez_compressed(source["clip"], **_fixture_fields(exp, syn))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                assert arm.main(argv + ["--dry-run"]) == 0
            assert "[DRY RUN] cell 0" in out.getvalue() and not arm_clip.exists()
        finally:
            object.__setattr__(bio.PARAMETERS, "machine", original)


def test_the_spot_count_report_states_each_recordings_receptor_total():
    def results(totals):
        cells = {}
        for cell, n_r in totals.items():
            window = {"experimental": {"mean": 100.0}, "R (FAB_POISSON)": {"mean": 90.0}}
            cells[str(cell)] = {"count_total_declared": n_r, "fluorescent_complexes_0": {"R (FAB_POISSON)": 150},
                                "eta_opening": {"R (FAB_POISSON)": 0.6},
                                "windows": {"opening 100 frames": window, "closing 100 frames": window}}
        return {"package_version": "test", "kind": "MET-FAB", "source_label": "R", "arm_labels": [],
                "columns": ["R (FAB_POISSON)"], "imaging_vector": {"mu_r": 1.585}, "cells": cells}
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "Spot_Count.md"
        spots.write_report(results({0: 550.0, 16: 3162.0}), report)
        text = report.read_text()
        assert "The receptor total N is declared per recording (column N)" in text
        assert "| 0 | 550 | 150 | 0.60 |" in text and "| 16 | 3162 | 150 | 0.60 |" in text
        assert "Every render declares" not in text and "same receptor total" not in text
        spots.write_report(results({0: 1000.0, 5: 1000.0}), report)
        assert "Every render declares the receptor total N = 1000." in report.read_text()


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
        assert "occupancy coins A=0.5, B=1 (override)" in printed, printed
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
