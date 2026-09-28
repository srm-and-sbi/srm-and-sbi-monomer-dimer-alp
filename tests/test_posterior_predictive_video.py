"""The posterior-predictive video check never overwrites, never renders beyond its limits unasked,
and names what its synthetic is made from.

The files of the check are listed in
``Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.md``. Two of them are
held here.

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

Runnable with ``python -m pytest`` or directly:
``MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python tests/test_posterior_predictive_video.py``.
"""
import ast
import importlib.util
import re
import tempfile
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import parameterization as bio
from srm_and_sbi_monomer_dimer_alp import posterior_predictive_video_runner as ppv

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
                 "_save_comparison_png"):
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
    # The run hands the clip file's label and the render's labeling row to the figure, where both are
    # required arguments.
    figure_call = next(c for c in _run_calls() if _call_name(c.func) == "_save_comparison_png")
    assert {"synth_label", "labeling_row"} <= {k.arg for k in figure_call.keywords}
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
    assert "visible: 124 of 860 monomers + 10 of 70 dimers (2 with both subunits labeled) = 134 spots" in text


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
                               norm_mode="full", play_every=2)
    assert len(copy.cells) == len(original.cells)
    code_o = [c for c in original.cells if c.cell_type == "code"]
    code_c = [c for c in copy.cells if c.cell_type == "code"]
    # Configuration: the clip path and the settings given, nothing else.
    cfg = pl._source(code_c[1])
    assert 'CLIP_PATH = "/data/X_Synthetic_Video.npz"' in cfg and 'NORM_MODE = "full"' in cfg
    assert pl.CLIP_PLACEHOLDER not in cfg
    assert (pl._source(code_o[1]).replace(pl.CLIP_PLACEHOLDER, 'CLIP_PATH = "/data/X_Synthetic_Video.npz"')
            .replace('NORM_MODE = "autoscale"', 'NORM_MODE = "full"') == cfg)
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
    copy.cells[-1]["outputs"] = [nbformat.v4.new_output(
        "stream", name="stdout", text="player: 271.9 MB, 778 of 1000 frames embedded  <-- TRUNCATED: raise\n")]
    assert pl.player_report(copy) == (271.9, 778, 1000, "<-- TRUNCATED: raise")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
