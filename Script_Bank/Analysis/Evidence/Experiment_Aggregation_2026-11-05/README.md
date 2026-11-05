# Experiment aggregation and empirical Nuisance DLI (implementation evidence, 2026-11-05)

Evidence of the implementation brief `Special_Analyses/EXPERIMENT_AGGREGATION_AND_NUISANCE_DLI_2026-11-05/CLAUDE_IMPLEMENTATION_PLAN.md`
(definitions in its `DOCUMENTATION_DRAFT.md`). The indexed completion note is `COMPLETION_NOTE.md`.

| item | file(s) |
|---|---|
| CPU regeneration of the two Experiment reports as review copies (two cells per condition, retained collections of 2026-11-03); the `_R1` files are the reruns after the review corrections (label `PILOT2CELLSR1`) | `regenerate_FAB_R1.log`, `regenerate_INLB_R1.log` (+ `.time`: wall and peak memory); first pass `regenerate_FAB.log`, `regenerate_INLB.log` |
| Copies of the two review copies' report, aggregation exports and aggregation figures (the data-bank folders `..._MAP_Experiment_PILOT2CELLSR1/` are the records; the first-pass `_PILOT2CELLS` folders are superseded) | `review_copies/<folder>/` |
| Template emission dry-run on the InlB archive (prints the comparison; writes nothing) | `emit_template_INLB_dryrun.log` |
| ILLUSTRATIVE representatives build on the Fab archive (interface example; example weights are not a decision) | `example_representatives_build.py`, `example_build.log`, `example_build/` (spec template, filled example spec, artifact, report, figures) |
| Codex's review probes rerun unchanged on the corrected state (every negative case refused) | `review_probes_after_corrections.json` (+ `.stderr.log`) |
| First-pass review copies, example build and logs, kept for the record | `_superseded_before_review/` |
| Checksums of this folder | `SHA256SUMS` |

Runs: PC profile `mars_pc`, package 0.1.45 (uncommitted), CPU only; nothing in the data bank other than the review
folders (`_PILOT2CELLS`, then `_PILOT2CELLSR1` after the review corrections of section 9 of the completion note) was written, and the adopted Experiment products, estimators and SEL artifacts keep their checksums.
