"""The lean trajectory reader of the DLI stage (`extract_subunit_positions`) against the dense one.

The renderer only ever gathers, per dye, the coordinates of the particle hosting its subunit in each
frame. The dense reader builds a (frame, particle id, 3, rank) tensor first, whose id axis holds every
ReaDDy id that ever appears (a fresh one per reaction and mode switch), so long recordings exhaust
memory; the lean reader gathers the per-subunit positions directly. These tests hold the contract on a
stubbed ReaDDy trajectory with id churn (mode switches, association, dissociation):

1. the lean positions equal the gather of the dense tensor through the lineage, value for value, at
   every documented duration, with float64 and float32 stored positions;
2. the two dye-track builders agree for zero, one and several dyes per subunit;
3. the renderer produces identical frames from either source of tracks under a fixed seed, and refuses
   an ambiguous call;
4. the lean reader allocates no frames-by-historical-ids tensor: its peak allocation is a small fraction
   of that tensor's size and its output scales with frames x subunits;
5. the shared DLI runner, switched to the lean path, writes for both workflows, in the ordinary and the
   debug path and over successive simulations, exactly the stored videos and labeling records that the
   dense path yields for the same seeds; the run reads and writes a temporary data bank only.

CPU only; no ReaDDy simulation is run. ``MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python
tests/test_subunit_positions.py``.
"""
import dataclasses
import re
import shutil
import sys
import tempfile
import tracemalloc
from pathlib import Path

import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import readdy                                                                       # noqa: E402
from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det          # noqa: E402
from srm_and_sbi_monomer_dimer_alp import labeling as lab                           # noqa: E402
from srm_and_sbi_monomer_dimer_alp import simulation_dli_runner as dli_runner       # noqa: E402
from srm_and_sbi_monomer_dimer_alp.io import (convert_video_dtype, theta_set_schema,   # noqa: E402
                                              write_theta_set)
from srm_and_sbi_monomer_dimer_alp.parameterization import (PARAMETERIZATION, PARAMETERS,   # noqa: E402
                                                            RunTiming)
from srm_and_sbi_monomer_dimer_alp.simulation_dli_support import (                  # noqa: E402
    build_dye_tracks, build_dye_tracks_from_subunit_positions, render_dli_video)
from srm_and_sbi_monomer_dimer_alp.simulation_rds_support import (                  # noqa: E402
    collapse_species_axis, extract_subunit_lineage, extract_subunit_positions, extract_trajectory_poses)
from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow, detector_workflow   # noqa: E402

RDS = PARAMETERS.simulation.rds
TYPE_NAMES = RDS.particle_type_names                     # ('A_f', 'A_s', 'A_i', 'B_f', 'B_s', 'B_i')
RANK = {name: rank for rank, name in enumerate(TYPE_NAMES)}
STEPS_PER_FRAME = PARAMETERS.simulation.timing.steps_per_frame
HALF_BOX = np.array(PARAMETERS.simulation.stem.box_size, dtype=float) / 2
DOCUMENTED_SECONDS = (1.0, 2.0, 5.0, 10.0, 20.0)
MODES = ("f", "s", "i")


def _frames(seconds):
    return RunTiming(total_time_seconds=seconds, frames=PARAMETERS.simulation.timing).frame_count


class _Record:
    """One ReaDDy reaction record: its kind, educt ids and product ids."""
    def __init__(self, kind, educts, products):
        self.type = kind
        self.educts = np.asarray(educts, dtype=np.int64)
        self.products = np.asarray(products, dtype=np.int64)
        self.reaction_label = f"{kind} (stub)"


class StubTrajectory:
    """The parts of ``readdy.Trajectory`` the readers, the lineage and the labeling use, over a
    synthetic recording with id churn: every frame some particles switch mode (conversion), two
    monomers may associate (fusion) and a dimer may dissociate (fission), each product with a fresh
    id as ReaDDy assigns it; positions random-walk in the box, stored box-centered as ReaDDy stores
    them, in a per-frame order that is not sorted by id; one trailing observable beyond the frames,
    which the readers drop."""

    def __init__(self, n_frames, n_monomers, n_dimers, seed, p_convert=0.08, p_fuse=0.15, p_split=0.15,
                 float32=False):
        rng = np.random.default_rng(seed)
        self.particle_types = dict(RANK)
        next_id = [0]

        def fresh():
            next_id[0] += 1
            return next_id[0] - 1

        def mode():
            return str(rng.choice(MODES))

        particles = [[fresh(), f"A_{mode()}", rng.uniform(-HALF_BOX, HALF_BOX)] for _ in range(n_monomers)]
        particles += [[fresh(), f"B_{mode()}", rng.uniform(-HALF_BOX, HALF_BOX)] for _ in range(n_dimers)]
        self._spans, self._ranks, self._souls, self._poses = [], [], [], []
        self._record_times, self._records = [], []
        self.events = {"conversion": 0, "fusion": 0, "fission": 0}
        for frame in range(n_frames + 1):                       # + the trailing observable
            step = frame * STEPS_PER_FRAME
            if frame > 0:
                records = []
                for p in particles:
                    p[2] = np.clip(p[2] + rng.normal(0.0, 40.0, 3), -HALF_BOX, HALF_BOX)
                for p in particles:
                    if rng.random() < p_convert:
                        old = p[0]
                        p[0], p[1] = fresh(), f"{p[1][0]}_{mode()}"
                        records.append(_Record("conversion", [old], [p[0]]))
                        self.events["conversion"] += 1
                monomers = [p for p in particles if p[1].startswith("A")]
                if len(monomers) >= 2 and rng.random() < p_fuse:
                    i, j = rng.choice(len(monomers), 2, replace=False)
                    a, b = monomers[i], monomers[j]
                    particles.remove(a)
                    particles.remove(b)
                    dimer = [fresh(), f"B_{mode()}", (a[2] + b[2]) / 2]
                    particles.append(dimer)
                    records.append(_Record("fusion", [a[0], b[0]], [dimer[0]]))
                    self.events["fusion"] += 1
                dimers = [p for p in particles if p[1].startswith("B")]
                if dimers and rng.random() < p_split:
                    d = dimers[int(rng.integers(len(dimers)))]
                    particles.remove(d)
                    left = [fresh(), f"A_{mode()}", np.clip(d[2] + np.array([6.0, 0.0, 0.0]), -HALF_BOX, HALF_BOX)]
                    right = [fresh(), f"A_{mode()}", np.clip(d[2] - np.array([6.0, 0.0, 0.0]), -HALF_BOX, HALF_BOX)]
                    particles += [left, right]
                    records.append(_Record("fission", [d[0]], [left[0], right[0]]))
                    self.events["fission"] += 1
                if records:
                    self._record_times.append(step)
                    self._records.append(records)
            order = rng.permutation(len(particles))
            self._spans.append(step)
            self._souls.append(np.array([particles[i][0] for i in order], dtype=np.int64))
            self._ranks.append(np.array([RANK[particles[i][1]] for i in order], dtype=np.int64))
            pos = np.array([particles[i][2] for i in order], dtype=np.float64)
            self._poses.append(pos.astype(np.float32) if float32 else pos)
        self.n_subunits = n_monomers + 2 * n_dimers

    def read_observable_particles(self):
        return np.asarray(self._spans, dtype=np.int64), self._ranks, self._souls, self._poses

    def read_observable_reactions(self):
        return np.asarray(self._record_times, dtype=np.int64), self._records

    def read_observable_reaction_counts(self):
        return None, {"reactions": {}}


def _dense_gather(tray, lineage):
    soul_poses = collapse_species_axis(extract_trajectory_poses(tray))
    frames = np.arange(soul_poses.shape[0])[:, None]
    return soul_poses[frames, lineage.host_index, :]


def _imaging_vector(rng):
    low, high = np.array(det.theta_lower_bound()), np.array(det.theta_upper_bound())
    slow, shigh = np.array(det.scope_lower_bound()), np.array(det.scope_upper_bound())
    return np.concatenate([np.power(10, rng.uniform(low, high)), np.power(10, rng.uniform(slow, shigh))])


# ---- 1. lean == dense gather at every documented duration -----------------------------------

def test_lean_positions_equal_the_dense_gather_at_every_documented_duration():
    for seconds in DOCUMENTED_SECONDS:
        for float32 in (False, True):
            tray = StubTrajectory(_frames(seconds), n_monomers=14, n_dimers=6, seed=int(seconds * 10) + int(float32),
                                  float32=float32)
            assert all(v > 0 for v in tray.events.values()), (seconds, tray.events)
            lineage = extract_subunit_lineage(tray)
            assert lineage.n_frames == _frames(seconds) and lineage.n_subunits == tray.n_subunits
            assert lineage.soul_ids.shape[0] > tray.n_subunits            # ids grew with the churn
            lean = extract_subunit_positions(tray, lineage)
            assert lean.shape == (_frames(seconds), tray.n_subunits, 3) and lean.dtype == np.float64
            assert np.isfinite(lean).all()
            dense = _dense_gather(tray, lineage)
            assert np.array_equal(lean, dense), (seconds, float32)
    # the consistency checks: a lineage with another frame count, or with the same count but ids the
    # trajectory does not hold (a lineage with the same count and reused ids is not detectable)
    a = StubTrajectory(50, 6, 3, seed=1)
    b = StubTrajectory(100, 6, 3, seed=2)
    c = StubTrajectory(50, 6, 3, seed=3)
    for other, needle in ((extract_subunit_lineage(b), "lineage holds 100 frames"),
                          (extract_subunit_lineage(c), "does not hold the host particle")):
        try:
            extract_subunit_positions(a, other)
        except ValueError as e:
            assert needle in str(e)
        else:
            raise AssertionError(f"accepted an inconsistent lineage ({needle})")


# ---- 2. the two dye-track builders agree --------------------------------------------------------

def test_dye_tracks_agree_for_zero_one_and_several_dyes():
    tray = StubTrajectory(_frames(2.0), n_monomers=10, n_dimers=5, seed=21)
    lineage = extract_subunit_lineage(tray)
    lean = extract_subunit_positions(tray, lineage)
    soul_poses = collapse_species_axis(extract_trajectory_poses(tray))
    rng = np.random.default_rng(0)
    for counts in (np.zeros(tray.n_subunits, dtype=np.int64), np.ones(tray.n_subunits, dtype=np.int64),
                   rng.integers(0, 4, tray.n_subunits)):
        pos_dense, sub_dense = build_dye_tracks(soul_poses, lineage.host_index, counts)
        pos_lean, sub_lean = build_dye_tracks_from_subunit_positions(lean, counts)
        assert np.array_equal(sub_dense, sub_lean) and sub_lean.shape == (int(counts.sum()),)
        assert pos_lean.shape == (lineage.n_frames, int(counts.sum()), 2)
        assert np.array_equal(pos_dense, pos_lean)
    for bad in (np.ones(tray.n_subunits + 1, dtype=np.int64), -np.ones(tray.n_subunits, dtype=np.int64)):
        try:
            build_dye_tracks_from_subunit_positions(lean, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid dye counts")


# ---- 3. one rendering calculation, two sources of tracks ---------------------------------------

def test_rendered_frames_are_identical_from_either_source_of_tracks():
    rng = np.random.default_rng(5)
    for seconds, counts_of in ((1.0, lambda n: rng.integers(0, 3, n)), (2.0, lambda n: rng.integers(0, 3, n)),
                               (1.0, lambda n: np.zeros(n, dtype=np.int64))):
        tray = StubTrajectory(_frames(seconds), n_monomers=8, n_dimers=4, seed=int(seconds * 100))
        lineage = extract_subunit_lineage(tray)
        lean = extract_subunit_positions(tray, lineage)
        soul_poses = collapse_species_axis(extract_trajectory_poses(tray))
        counts = counts_of(tray.n_subunits)
        imaging = _imaging_vector(rng)
        dense_frames = render_dli_video(soul_poses, lineage.host_index, counts, imaging, seed=11)
        lean_frames = render_dli_video(None, lineage.host_index, counts, imaging, seed=11, subunit_positions=lean)
        assert dense_frames.shape == lean_frames.shape == (256, 256, _frames(seconds))
        assert np.array_equal(dense_frames, lean_frames), seconds
        without_lineage = render_dli_video(None, None, counts, imaging, seed=11, subunit_positions=lean)
        assert np.array_equal(without_lineage, lean_frames)
    counts = np.ones(tray.n_subunits, dtype=np.int64)
    for kwargs, needle in ((dict(soul_poses=soul_poses, subunit_positions=lean), "exactly one source"),
                           (dict(soul_poses=None, subunit_positions=None), "exactly one source"),
                           (dict(soul_poses=soul_poses, host_index=None), "needs host_index"),
                           (dict(soul_poses=None, subunit_positions=lean[:, :-1]), "does not belong")):
        call = dict(soul_poses=None, host_index=lineage.host_index, dye_counts=counts, imaging_physical=imaging, seed=1)
        call.update(kwargs)
        try:
            render_dli_video(**call)
        except ValueError as e:
            assert needle in str(e), (needle, str(e))
        else:
            raise AssertionError(f"accepted an ambiguous call ({needle})")


# ---- 4. no frames-by-historical-ids tensor ------------------------------------------------------

def test_the_lean_reader_allocates_no_frames_by_ids_tensor():
    n_frames = 100
    tray = StubTrajectory(n_frames, n_monomers=150, n_dimers=75, seed=77, p_convert=0.5, p_fuse=0.3, p_split=0.3)
    lineage = extract_subunit_lineage(tray)
    n_ids = int(lineage.soul_ids.shape[0])
    assert n_ids > 20 * tray.n_subunits, n_ids                            # heavy churn: many historical ids
    dense_bytes = n_frames * n_ids * 3 * len(TYPE_NAMES) * 8               # what the dense reader would allocate
    tracemalloc.start()
    tracemalloc.reset_peak()
    lean = extract_subunit_positions(tray, lineage)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert lean.shape == (n_frames, tray.n_subunits, 3)
    assert lean.nbytes == n_frames * tray.n_subunits * 3 * 8
    assert peak < 0.05 * dense_bytes, (peak, dense_bytes)
    assert np.array_equal(lean, _dense_gather(tray, lineage))


# ---- 5. the shared runner, both workflows, ordinary and debug, successive simulations --------

def _stub_factory(n_frames, n_monomers=6, n_dimers=3):
    """``readdy.Trajectory`` replaced by a stub keyed by the trajectory file name's task and sim."""
    def factory(filename):
        m = re.search(r"_TASK_(\d+)_SIM_(\d+)(?:_[A-Z]+)?\.h5$", str(filename))   # ..._SIM_3_EVAL.h5
        assert m, filename
        task, sim = int(m.group(1)), int(m.group(2))
        return StubTrajectory(n_frames, n_monomers, n_dimers, seed=1000 * task + sim)
    return factory


def _reference_outputs(factory, paths, root, timing_label, split, seed, task, n_sims, imaging_source):
    """What the dense path yields for the runner's recorded draws: one video and one labeling row per
    simulation, computed through the dense reader and `render_dli_video` with soul_poses."""
    condition = "FAB"
    plan = lab.resolve_labeling(condition, None, None)
    scope = zarr.open(str(paths.record_set_path("Nuisance_SCOPE_Theta_Set", task, root, timing_label, True, split)), mode="r")[:]
    imaging = zarr.open(str(imaging_source), mode="r")[:]
    videos, rows = [], []
    for sim in range(n_sims):
        tray = factory(paths.trajectory_path(task, sim, root, timing_label, split))
        lineage = extract_subunit_lineage(tray)
        soul_poses = collapse_species_axis(extract_trajectory_poses(tray))
        dyes, row = lab.label_trajectory(plan, tray, lineage, lab.labeling_rng(seed, task, sim))
        frames = render_dli_video(soul_poses, lineage.host_index, dyes,
                                  np.concatenate([imaging[sim], scope[sim]]), seed=seed)
        videos.append(convert_video_dtype(np.moveaxis(frames, 2, 0), bits_from=16, bits_to=8))
        rows.append(row)
    return np.stack(videos), np.stack(rows)


def _run(cfg, argv):
    args = dli_runner.build_dli_parser().parse_args(argv)
    dli_runner.run_dli(cfg, args)
    return args


def test_the_shared_runner_writes_the_dense_path_result_for_both_workflows():
    original_machine, original_trajectory = PARAMETERS.machine, readdy.Trajectory
    real_posit = original_machine.root_for("EVAL") / PARAMETERS.paths.posit_subdir
    nuisance_files = sorted(f for f in real_posit.glob("SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_REF_Nuisance_DLI*")
                            if f.is_file()) if real_posit.exists() else []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "Data_Bank"
        root.mkdir()
        object.__setattr__(PARAMETERS, "machine",
                           dataclasses.replace(original_machine, data_bank_root=root, scratch_data_bank_root=None))
        try:
            # detector workflow at 1 s: two simulations, the ordinary path
            factory = _stub_factory(_frames(1.0))
            readdy.Trajectory = factory
            cfg = detector_workflow()
            paths = cfg.paths.with_condition("FAB")
            _run(cfg, ["--total-time-seconds", "1", "--tasks", "1", "--task-simulations", "2",
                       "--split", "eval", "--condition", "FAB", "--seed", "3"])
            video = zarr.open(str(paths.video_set_path(0, root, "1S_50FPS", True, "EVAL")), mode="r")[:]
            labeling = zarr.open(str(paths.record_set_path("Labeling_Set", 0, root, "1S_50FPS", True, "EVAL")), mode="r")[:]
            ref_video, ref_rows = _reference_outputs(factory, paths, root, "1S_50FPS", "EVAL", 3, 0, 2,
                                                     paths.theta_set_path(0, root, "1S_50FPS", True, "EVAL"))
            assert video.shape == (2, _frames(1.0), 256, 256) and video.dtype == np.uint8
            assert np.array_equal(video, ref_video) and np.array_equal(labeling, ref_rows)
            assert not np.array_equal(video[0], video[1])                 # two different simulations
            # the debug path, on a fresh task index, must produce the same result and no dense tensor
            _run(cfg, ["--total-time-seconds", "1", "--task-id", "1", "--task-simulations", "2",
                       "--split", "eval", "--condition", "FAB", "--seed", "3", "--debug"])
            video = zarr.open(str(paths.video_set_path(1, root, "1S_50FPS", True, "EVAL")), mode="r")[:]
            labeling = zarr.open(str(paths.record_set_path("Labeling_Set", 1, root, "1S_50FPS", True, "EVAL")), mode="r")[:]
            ref_video, ref_rows = _reference_outputs(factory, paths, root, "1S_50FPS", "EVAL", 3, 1, 2,
                                                     paths.theta_set_path(1, root, "1S_50FPS", True, "EVAL"))
            assert np.array_equal(video, ref_video) and np.array_equal(labeling, ref_rows)
            # biology workflow at 2 s, when this machine holds the reference Nuisance_DLI artifact
            if not nuisance_files:
                print("  (no REF Nuisance_DLI artifact on this machine; the biology end-to-end run is skipped)")
                return
            posit = root / PARAMETERS.paths.posit_subdir
            posit.mkdir(parents=True)
            for f in nuisance_files:
                shutil.copy2(f, posit / f.name)
            factory = _stub_factory(_frames(2.0))
            readdy.Trajectory = factory
            cfg = biology_workflow()
            paths = cfg.paths.with_condition("FAB")
            theta = np.array([[10 ** ((e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]) / 2) if e["LOG_FLAG"]
                               else (e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]) / 2 for e in PARAMETERIZATION]] * 2)
            write_theta_set(paths.theta_set_path(0, root, "2S_50FPS", True, "EVAL"), theta,
                            theta_set_schema(PARAMETERIZATION, condition="FAB", timing_label="2S_50FPS", generator="rds"))
            _run(cfg, ["--total-time-seconds", "2", "--tasks", "1", "--task-simulations", "2",
                       "--split", "eval", "--condition", "FAB", "--seed", "9", "--nuisance-tag", "REF"])
            video = zarr.open(str(paths.video_set_path(0, root, "2S_50FPS", True, "EVAL")), mode="r")[:]
            labeling = zarr.open(str(paths.record_set_path("Labeling_Set", 0, root, "2S_50FPS", True, "EVAL")), mode="r")[:]
            ref_video, ref_rows = _reference_outputs(
                factory, paths, root, "2S_50FPS", "EVAL", 9, 0, 2,
                paths.record_set_path("Nuisance_DLI_Theta_Set", 0, root, "2S_50FPS", True, "EVAL"))
            assert video.shape == (2, _frames(2.0), 256, 256)
            assert np.array_equal(video, ref_video) and np.array_equal(labeling, ref_rows)
        finally:
            readdy.Trajectory = original_trajectory
            object.__setattr__(PARAMETERS, "machine", original_machine)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
