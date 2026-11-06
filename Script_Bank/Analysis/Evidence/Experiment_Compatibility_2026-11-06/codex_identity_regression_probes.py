# Unchanged copy of Codex's identity_regression_probes.py (EXPERIMENT_COMPATIBILITY_REVIEW_2026-11-06/REREVIEW_R2), run from
# this evidence folder after the identity-guard correction so its rewritten archives and outputs land here.
"""Check that distinct reference groups also have distinct cell/window identities."""

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path('/home/mars-fias/Documents/Projects/RCL_Agent_Projects')
REPO = ROOT / 'srm-and-sbi/srm-and-sbi-monomer-dimer-alp'
OUT = Path(__file__).resolve().parent
EVIDENCE = REPO / 'Script_Bank/Analysis/Evidence/Experiment_Compatibility_2026-11-06'
sys.path.insert(0, str(REPO))
from srm_and_sbi_monomer_dimer_alp import experiment_aggregation as agg


def main():
    source = next((EVIDENCE / 'review_copies').glob('*_FAB_*/aggregation_candidates.npz'))
    results = {}
    for construction, field in [('per_cell', 'cell'), ('per_cell_window', 'chunk')]:
        with np.load(source, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        manifest = json.loads(str(arrays.pop('manifest_json')))
        block = next(b for b in manifest['compatibility']['blocks'] if b['construction'] == construction and b['window_set'] == 'all')
        indices = arrays[block['arrays']['group_index']]
        before = arrays[field][indices].copy()
        arrays[field][indices[1]] = arrays[field][indices[0]]
        path = OUT / ('duplicate_' + field + '_identity.npz')
        np.savez_compressed(path, **arrays, manifest_json=np.asarray(json.dumps(manifest)))
        record = {'construction': construction, 'changed_field': field, 'candidate_groups': len(indices),
                  'distinct_ids_before': len(np.unique(before)), 'distinct_ids_after': len(np.unique(arrays[field][indices]))}
        try:
            archive = agg.load_candidate_archive(path)
            record['accepted'] = True
            if construction == 'per_cell':
                comp = agg.archive_component(archive, kind='FAB', window_set='all', construction='per_cell')
                record['exported_member_cell_ids'] = [member['cell'] for member in comp['identities'][:4]]
        except Exception as error:
            record['accepted'] = False
            record['error'] = type(error).__name__ + ': ' + str(error)
        results[construction] = record
    (OUT / 'identity_regression_reproduction.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
