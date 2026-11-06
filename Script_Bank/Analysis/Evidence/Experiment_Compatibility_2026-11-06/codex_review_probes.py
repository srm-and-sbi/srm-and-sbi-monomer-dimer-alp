# Unchanged copy of Codex's review_probes.py (Special_Analyses/EXPERIMENT_COMPATIBILITY_REVIEW_2026-11-06), run from this
# evidence folder after the corrections so its rewritten archives and outputs land here, never in Codex's folder.
"""Independent read-only probes of the uncommitted compatibility implementation.

All rewritten archives, CSVs and synthetic fixtures stay inside this review folder.
"""

import copy
import contextlib
import csv
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WORKSPACE = Path('/home/mars-fias/Documents/Projects/RCL_Agent_Projects')
REPO = WORKSPACE / 'srm-and-sbi/srm-and-sbi-monomer-dimer-alp'
OUT = Path(__file__).resolve().parent
EVIDENCE = REPO / 'Script_Bank/Analysis/Evidence/Experiment_Compatibility_2026-11-06'
sys.path[:0] = [str(REPO), str(REPO / 'tests'), str(REPO / 'Script_Bank/Analysis')]

from srm_and_sbi_monomer_dimer_alp import experiment_aggregation as agg
from srm_and_sbi_monomer_dimer_alp import detector_nuisance_dli as ndli
from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
import SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI as companion
import test_experiment_compatibility as fixtures


class ReportCapture:
    def __init__(self, path):
        self.dump_dir = path
        path.mkdir(parents=True, exist_ok=True)
        self.tables = []

    def table(self, title, headers, rows, note=''):
        self.tables.append({'title': title, 'headers': headers, 'rows': rows, 'note': note})

    def check(self, *args, **kwargs):
        pass

    def prose(self, *args, **kwargs):
        pass

    def save_figure(self, name, fig, caption=''):
        fig.clear()


def rewritten_archive(name, mutate):
    source = next((EVIDENCE / 'review_copies').glob('*_FAB_*/aggregation_candidates.npz'))
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    manifest = json.loads(str(arrays.pop('manifest_json')))
    mutate(arrays, manifest)
    path = OUT / 'probe_archives' / (name + '.npz')
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **arrays, manifest_json=np.asarray(json.dumps(manifest)))
    try:
        archive = agg.load_candidate_archive(path)
        return {'accepted': True}, archive
    except Exception as error:
        return {'accepted': False, 'error': type(error).__name__ + ': ' + str(error)}, None


def spec(archive, components, overrides=None):
    result = {'block': {'posterior_sample_pool_choice': 'empirical_representatives',
                        'candidate_archive': archive['path'], 'candidate_archive_sha256': archive['sha256'],
                        'rationale': 'Independent review probe; not a scientific selection.'},
              'component': components}
    if overrides:
        result['override'] = overrides
    return result


def component(archive, weight, policy):
    return {'source_experiment': archive['manifest']['product_label'], 'condition': 'FAB', 'window_set': 'all',
            'grouping': 'per_cell', 'statistic': 'sgm', 'weight': weight, 'member_weighting': policy}


def main():
    results = {}
    source = next((EVIDENCE / 'review_copies').glob('*_FAB_*/aggregation_candidates.npz'))
    archive = agg.load_candidate_archive(source)

    def wrong_scale(arrays, manifest):
        manifest['compatibility']['protocol']['sgm_scale'][0] *= 2
    results['protocol_scale_conflict'], _ = rewritten_archive('protocol_scale_conflict', wrong_scale)

    def wrong_draw_counts(arrays, manifest):
        key = manifest['compatibility']['blocks'][0]['arrays']['reference_draws']
        arrays[key] = np.ones_like(arrays[key])
    results['wrong_reference_draw_counts'], _ = rewritten_archive('wrong_reference_draw_counts', wrong_draw_counts)

    def split_peer_population(arrays, manifest):
        section = manifest['compatibility']
        block = section['blocks'][0]
        names = block['arrays']
        group_index = arrays[names['group_index']].copy()
        rank = arrays[names['ranks']].copy()
        draws = arrays[names['reference_draws']].copy()
        assert len(group_index) == 60 and block['construction'] == 'per_cell'
        for part, selected in enumerate((np.arange(30), np.arange(30, 60))):
            b = block if part == 0 else copy.deepcopy(block)
            if part:
                b['index'] = len(section['blocks'])
                b['arrays'] = {key: 'split_peer_probe_' + key for key in names}
                section['blocks'].append(b)
            b['n_groups'] = len(selected)
            new_groups = group_index[selected]
            new_rank = rank[np.ix_(selected, selected)]
            arrays[b['arrays']['group_index']] = new_groups
            arrays[b['arrays']['ranks']] = new_rank
            arrays[b['arrays']['reference_draws']] = draws[selected]
            arrays['compatibility_n_references'][new_groups] = 29
            arrays['compatibility_score'][new_groups] = np.nanmedian(new_rank, axis=1)
    result, split_archive = rewritten_archive('split_peer_population', split_peer_population)
    if split_archive is not None:
        comp = agg.archive_component(split_archive, kind='FAB', window_set='all', construction='per_cell')
        p, _, _ = ndli.member_probabilities('posterior_rank_median', 1.0, comp)
        original = agg.archive_component(archive, kind='FAB', window_set='all', construction='per_cell')
        p0, _, _ = ndli.member_probabilities('posterior_rank_median', 1.0, original)
        result['accepted_normalized_selection'] = bool(np.isclose(p.sum(), 1))
        result['max_probability_change'] = float(np.abs(p - p0).max())
        result['references_per_cell_in_archive'] = 29
        result['required_references_for_60_cell_context'] = 59
    results['split_peer_population'] = result

    R = companion._resolve(2.0, 'FAB')
    target = companion._target(type('Args', (), {'condition': 'FAB'})(), R)

    for policy in ('uniform', 'posterior_rank_median'):
        components = [component(archive, 1.0, 'uniform'), component(archive, 0.0, policy)]
        V, P, record = ndli.representatives_from_spec(spec(archive, components), archive, det.DETECTOR_PARAMETERIZATION,
                                                    R['plo'], R['phi'], target=target)
        nu = ndli.NuisanceDLI.from_representatives(R['imaging_keys'], V, P, record, prior_low=R['plo'], prior_high=R['phi'])
        reporter = ReportCapture(OUT / ('zero_component_' + policy))
        result = {'valid_builder_result': True, 'zero_component_record': {
            key: record['components'][1][key] for key in ('weight', 'n_members', 'n_positive_members', 'n_zero_members', 'effective_members')}}
        try:
            companion._report_representatives(reporter, nu, R, archive)
            result['report_completed'] = True
            row = next(table for table in reporter.tables if table['title'] == 'empirical_representatives: selection')['rows'][1]
            result['reported_members_positive_zero'] = row[8]
        except Exception as error:
            result['report_completed'] = False
            result['error'] = type(error).__name__ + ': ' + str(error)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                selection = spec(archive, components)
                companion._build_representatives(SimpleNamespace(condition='FAB', dry_run=True), R,
                                                  selection['block'], selection,
                                                  OUT / ('dry_run_' + policy + '.npz'), OUT / 'not_written.toml')
            result['dry_run_completed'] = True
        except Exception as error:
            result['dry_run_completed'] = False
            result['dry_run_error'] = type(error).__name__ + ': ' + str(error)
        results['zero_component_' + policy] = result

    values = {'mu_r': 1.7, 'sigma_r': .2, 'mu_pc': 110., 'sigma_pc': .5, 'prob_photo_bleach': .05, 'lambda_rate': 5.}
    overrides = {key: {'value': value, 'reason': 'Independent override-collision review probe.'} for key, value in values.items()}
    V, P, record = ndli.representatives_from_spec(spec(archive, [component(archive, 1.0, 'posterior_rank_median')], overrides),
                                                archive, det.DETECTOR_PARAMETERIZATION, R['plo'], R['phi'], target=target)
    nu = ndli.NuisanceDLI.from_representatives(R['imaging_keys'], V, P, record, prior_low=R['plo'], prior_high=R['phi'])
    reporter = ReportCapture(OUT / 'override_collisions')
    companion._report_representatives(reporter, nu, R, archive)
    with (reporter.dump_dir / 'representatives_rows.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    results['override_collision_row_memberships'] = {
        'stored_rows': len(rows), 'unique_final_vectors': len(np.unique(V, axis=0)),
        'rows_missing_memberships': sum(not row['memberships (component, member, cell, chunk, contribution)'] for row in rows),
        'source_members': sum(len(comp['members']) for comp in record['components']),
    }

    rng = np.random.default_rng(555)
    ids = [(0, c, w) for c in (3, 7, 11) for w in range(2)]
    centers = {3: [-.9, .1, .1], 7: [-.5, .5, .5], 11: [-.1, .9, .9]}
    cloud = np.stack([np.array(centers[c]) + rng.normal(scale=.005, size=(30, 3)) for _, c, _ in ids])
    coll = fixtures._collection(cloud, ids, geometry=fixtures.S.window_geometry(n_frames=100, step_frames=100, span_frames=200))
    case = OUT / 'all_zero_scores'; case.mkdir(exist_ok=True)
    zero_archive, _, _, _ = fixtures._archive(case, coll, {'all': None})
    results['all_zero_policy_availability'] = {'description': ndli.component_weighting_availability(zero_archive)[('FAB', 'all', 'per_cell')]}
    try:
        ndli.representatives_from_spec(fixtures._spec(zero_archive, fixtures._comp('per_cell', 1., member_weighting='posterior_rank_median')),
                                      zero_archive, fixtures.TABLE, fixtures.PLO, fixtures.PHI, target=fixtures.TARGET)
        results['all_zero_policy_availability']['build_accepted'] = True
    except Exception as error:
        results['all_zero_policy_availability']['build_accepted'] = False
        results['all_zero_policy_availability']['error'] = str(error)

    (OUT / 'review_probes.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
