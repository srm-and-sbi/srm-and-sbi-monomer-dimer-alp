"""Read-only checks of construction-wide products, exact histograms and legacy rebuilding."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import tomllib

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
REPO = BASE.parents[1] / 'srm-and-sbi-monomer-dimer-alp'
sys.path.insert(0, str(REPO))
from srm_and_sbi_monomer_dimer_alp import experiment_aggregation as agg
from srm_and_sbi_monomer_dimer_alp import detector_nuisance_dli as ndli
from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det

POSIT = Path('/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit')


def check(condition, tag):
    arc_path = POSIT / f'SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_{condition}_2S_50FPS_{tag}_MAP_Experiment_COHORT60RANKSR4/aggregation_candidates.npz'
    arc = agg.load_candidate_archive(arc_path)
    a = arc['arrays']
    old_arc = agg.load_candidate_archive(arc_path.parent.with_name(arc_path.parent.name.replace('RANKSR4', 'RANKSR3')) / arc_path.name)
    lo = np.asarray(a['prior_low'], float)
    hi = np.asarray(a['prior_high'], float)
    out = {'archive_format': arc['format_version'], 'comparison_scope': arc['compatibility']['comparison_scope'], 'constructions': {}}
    with np.load(BASE / 'WEIGHTING_SCOPE_REVIEW_2026-11-10' / f'{condition}_construction_wide_ranks.npz', allow_pickle=False) as z:
        reference = {k: z[k] for k in z.files}
    space = arc['compatibility']['spaces']['spatial_brightness']
    for construction in ('per_cell', 'per_window_position', 'per_cell_window'):
        block = next(b for b in space['blocks'] if b['window_set'] == 'all' and b['construction'] == construction)
        gi = block['group_index']
        expected_gi = reference[f'{construction}_group_index']
        order = np.asarray([list(gi).index(g) for g in expected_gi])
        assert np.array_equal(block['ranks'][np.ix_(order, order)], reference[f'{construction}_ranks'], equal_nan=True)
        scores = np.asarray([space['by_group'][int(g)]['score'] for g in expected_gi])
        assert np.array_equal(scores, reference[f'{construction}_score'])
        out['constructions'][construction] = {'n': len(gi), 'directed_ranks_identical': len(gi)*(len(gi)-1), 'scores_identical': True}
    invariant_keys = ['construction', 'window_set', 'kind', 'cell', 'chunk', 'n_draws', 'n_windows', 'members_json',
                      'sgm_log10', 'median_log10', 'quantiles_log10', 'sgm_source_cell', 'sgm_source_chunk', 'sgm_source_draw']
    assert all(np.array_equal(a[k], old_arc['arrays'][k]) for k in invariant_keys)
    out['unchanged_group_arrays'] = len(invariant_keys)
    review_path = next((HERE.parent / condition).glob('*Nuisance_DLI.npz'))
    nu = ndli.NuisanceDLI.load(review_path)
    old_path = POSIT / f'SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_{condition}_2S_50FPS_RANK4_Nuisance_DLI.npz'
    old = ndli.NuisanceDLI.load(old_path)
    assert np.array_equal(nu.samples, old.samples)
    assert nu.selection['overrides'] == old.selection['overrides']
    expected = {int(g): float(p) for g, p in zip(reference['per_cell_window_group_index'], reference['per_cell_window_probability'])}
    error = max(abs(nu.probabilities[m['stored_row']] - expected[m['group_index']]) for m in nu.selection['components'][0]['members'])
    assert error < 1e-15
    out['artifact'] = {'max_probability_error': error, 'stored_vectors_unchanged': True, 'overrides_unchanged': True,
                       'effective_rows': nu.selection['effective_rows'], 'zero_rows': int(np.count_nonzero(nu.probabilities == 0))}
    # Compare each exported histogram bin directly with the sum of stored probabilities satisfying its boundaries.
    report = review_path.with_name(review_path.stem + '_Analysis')
    with (report / 'nuisance_marginal_histograms.csv').open() as f:
        bins = list(csv.DictReader(f))
    checked = 0
    for key in nu.parameter_keys:
        j = list(nu.parameter_keys).index(key)
        rows = [r for r in bins if r['parameter'] == key]
        if rows[0]['kind'] == 'point_mass':
            assert np.unique(nu.samples[:, j]).size == 1
            assert abs(float(rows[0]['selected_probability_mass']) - 1) < 1e-12
            checked += 1
            continue
        for n, r in enumerate(rows):
            low, high = float(r['low_stored_log10']), float(r['high_stored_log10'])
            mask = (nu.samples[:, j] >= low) & ((nu.samples[:, j] <= high) if n == len(rows)-1 else (nu.samples[:, j] < high))
            actual = float(nu.probabilities[mask].sum())
            assert abs(actual - float(r['selected_probability_mass'])) < 1e-12, (condition, key, n)
            checked += 1
    out['exact_histogram_bins_and_point_masses_checked'] = checked
    # A historical rank-weighted spec now refuses omission of rank_scope; a migrated COPY reproduces its distribution.
    old_spec_path = old_path.with_name(old_path.stem + '_Spec.toml')
    try:
        ndli.load_spec(old_spec_path, list(nu.parameter_keys), lo, hi, table=det.DETECTOR_PARAMETERIZATION)
    except ValueError as exc:
        out['legacy_spec_missing_scope_refusal'] = str(exc)
    else:
        raise AssertionError('legacy spec unexpectedly accepted without an explicit scope')
    migrated = tomllib.loads(old_spec_path.read_text())
    for component in migrated['component']:
        if component.get('member_weighting') == 'posterior_rank_median':
            component['rank_scope'] = 'within_cell'
    vectors, probability, _ = ndli.representatives_from_spec(migrated, old_arc, det.DETECTOR_PARAMETERIZATION, lo, hi, target=old_arc['manifest']['target'])
    assert np.array_equal(vectors, old.samples)
    assert np.allclose(probability, old.probabilities, atol=1e-15, rtol=0)
    out['legacy_migrated_copy_reproduces_artifact'] = True
    out['legacy_artifact_sha256'] = hashlib.sha256(old_path.read_bytes()).hexdigest()
    return out


if __name__ == '__main__':
    results = {}
    for condition, tag in (('FAB', 'C128V42A'), ('INLB', 'C128V42AP3LR32')):
        results[condition] = check(condition, tag)
        print(json.dumps({condition: results[condition]}), flush=True)
        (HERE / 'product_checks.json').write_text(json.dumps(results, indent=2)+'\n')
