# Review of construction-wide weighting and exact empirical marginal plots

Written for the lead and Claude. Date: 2026-11-10. Reviewed working-tree release 0.1.49 against baseline `532c943` (0.1.48). This review did not edit repository files, production specs/artifacts or data-bank reports, connect to remote machines, install packages or submit jobs.

## Verdict

**The numerical implementation passes review. No numerical or archive-scope blocker was found in the checked paths.** It implements the clarified construction-wide target, reproduces the independent exploratory calculation, preserves the historical RANK4 distribution and computes empirical marginals directly from stored probabilities. Biology generation remains on hold pending the lead's adoption and execution decisions.

One presentation refinement and a short documentation clarification are requested below. They do not change the scoring, probabilities, bin edges or selected vectors.

## Verified independently

- **51 targeted tests pass:** comparison-scope 8/8, compatibility 18/18, aggregation 17/17, selection-user 8/8. Logs and `test_results.json` are in this directory. The complete 337-test suite was reported by Claude; this review did not rerun it.
- Both complete-cohort format-4 archives load under the declared construction scope. Every four-coordinate matrix and score matches Codex's exploratory results for all three rank-scored constructions: 3,540 per-cell entries, 90 per-position entries and 359,400 cell–window entries per condition, **726,060 directed entries across both conditions**.
- The selected group identities and checked SGM/median/quantile/source arrays remain unchanged from R3. Both revised artifacts retain the RANK4 vectors and overrides row by row.
- Artifact probabilities reproduce the exploratory probabilities: maximum error zero for FAB and approximately 1.7e-18 for INLB. Effective rows and zero counts reproduce: 427.30/1 and 317.97/64.
- **Every histogram export checked by direct membership summation:** 120 bin masses plus two point masses per condition. The values equal sums of the stored probabilities of the vectors in each bin. Histogram calculation is independent of resampling; sampler draws remain a separate validation.
- Both historical RANK4 artifacts still load with their adoption checksums unchanged. Their original specs now correctly refuse an omitted scope under the new constructor. Adding `rank_scope = "within_cell"` to an in-memory migrated COPY reproduces each stored historical distribution. Neither source spec was modified.
- FAB's new marginal histogram and INLB's probability grid were inspected visually. The latter displays actual global row probabilities and varying cell totals, with the equal-member baseline correctly separated.

Evidence: `check_products.py`, `product_checks.json`, `product_checks.log`, targeted test logs and `test_results.json`.

## R1 — Nonblocking presentation refinement: retain data visibility in the marginal histograms

Location: `srm_and_sbi_monomer_dimer_alp/visualization_aggregation.py`, `figure_weighted_histograms`, approximately lines 559–579.

The exact bin masses are correct. The prior-bound `axvline` annotations currently participate in autoscaling, so a narrow distribution is squeezed into a small portion of the panel. FAB brightness shows the problem clearly: the prior's high bound extends the axis far beyond the vectors, even though the purpose is to compare the selected weights against the equal-member curve.

Requested refinement: for binned panels, set the visible limits from the full support of the plotted stored vectors/baseline, with modest padding. Do not crop to a posterior quantile or remove tails. Prior or reference markers outside the data-support view should be represented by a labeled edge indicator or textual bound and must not silently widen the axis. The exact bin edges and mass export stay unchanged. Keep point-mass annotation text inside its panel; the FAB flicker note currently extends into the right margin. Point-mass panels can have their own sensible display range because there is no binned shape to inspect.

This is report rendering only. It does not require re-scoring the cohorts, resampling posteriors or rebuilding the numerical artifact distribution. A focused plotting/report check is appropriate after the change.

## R2 — Documentation clarification: explain legacy scope migration and the per-cell exception

Locations: the constructor companion's `rank_scope` field description (approximately line 192) and its legacy-readback/migration explanation.

Requiring an explicit `rank_scope` is an acceptable, deliberate specification change. Do not introduce a silent default solely to keep old ranked specs working. Document that old artifacts load unchanged, whereas a copied old RANK4 spec needs this field under its `[[component]]` table:

```toml
member_weighting = "posterior_rank_median"
rank_scope = "within_cell"
```

Preserve the adopted source specs as historical records; show the one-line migration in a new copy or example rather than overwriting their archived content.

Also qualify the companion table's blanket association between `construction` and format 4. The code intentionally and correctly accepts legacy **per-cell** scores as construction-wide because those scores already compare all cells and have the same reference population and values. Keep that behavior. The new full cell–window and per-position scope needs compatible new scores, not merely a relabeled legacy archive. Explain scope as a property of the recorded comparison population, not an absolute format-number restriction.

No additional Experiment CLI switch for `within_cell` is required for this revision. Keeping engine support, archived reports/data and explicit legacy selection is enough; the standard report can compute the intended construction-wide policy.

## Other implementation decisions

- Keeping unweighted legacy sample-pool plotting outside the exact representative path is within the authorized scope. It does not affect the new weighted selection. A separate extension is unnecessary for this release.
- The larger pairwise CSV is a consequence of exporting the complete population matrix. It is acceptable as a linked supporting file. Do not show hundreds of thousands of rows inline.
- The lost successful console logs are disclosed. They need not trigger another full report regeneration merely to recreate logs. Preserve the commands, successful-product manifests and independent read-back/numerical verification. Do not represent refused-run logs as successful-run logs.
- The implementation keeps each condition and acquisition period separate. Zero probabilities, including a whole zero-mass cell, are valid selection outcomes and are not data-quality verdicts.
- Existing RANK4 remains the within-cell distribution. A revised adopted distribution needs a distinct production identity; `RANK4POP` is a reasonable proposed tag if the lead chooses it. No production tag is adopted by this review.

## Operational decisions, separate from the weighting review

### Reuse and transfer

The inventory gives strong evidence that the intended current-model tiers are available: the condition/split/task-name coverage is complete, and eight sampled parameter records match release 0.1.42 and October 21. Full per-folder content counts remain pending. Accordingly, change the inventory's opening claim of "complete" tiers to "complete task-name inventory; content completeness still to be checked." This is a wording correction, not a finding that files are missing.

Reuse was recommended to save repeated computation and preserve the same biological realizations, not as an unconditional requirement. An eight-hour full transfer should not be compared directly with a single 1,000-simulation task's RDS time: regenerating the corpus means 550 such tasks, and wall time depends on the available parallel layout.

**There is no need to copy all 1.28 TiB before a pilot.** From the recorded TRAIN/TEST averages, ten TRAIN tasks per condition would be about 46.6 GiB combined. At the cited 46 MB/s that is approximately 18 minutes of payload transfer, excluding connection setup, verification and variation in task sizes. One task per condition would be about one tenth of that. Measure actual selected folder sizes and route throughput before treating this as a forecast. Check selected pilot folders and their matching Theta records, stage only those, and run the DLI preflight against them; bulk staging can be decided separately.

**Lead's correction, 2026-11-10:** exclude the PC relay from the transfer plan because it is significantly slower on the available connections. Use a direct authenticated JUPITER–Goethe route if transferring the tiers. The approximately 18-minute pilot estimate above is conditional on sustaining the quoted 46 MB/s on the direct route; it is not a measured direct-route runtime and does not apply to a PC relay. A smaller pilot transfer still reduces the volume, but its runtime must be evaluated using the actual direct-route throughput. Do not create credentials or start a large transfer on the basis of this review alone.

The lead can still choose regeneration based on a real wall-time/compute/storage comparison. It is a logistics decision, not a requirement of the revised nuisance weights.

### Goethe package installation

An editable installation of this successor package with dependencies left unchanged is a routine preflight prerequisite, not another scientific selection. Once the approved source reaches Goethe, use the existing environment and repository installation instructions (`pip install -e . --no-deps`, adapting only if the documented build setup requires it). Verify import and version from a neutral working directory so PYTHONPATH/current-directory effects do not hide a missing installation. Do not upgrade dependencies or submit generation as part of that check.

### Commit and subsequent work

After the small report/documentation refinements, this is ready for the lead's code-checkpoint decision. Commit/push/synchronization/verified backup and production adoption remain actions for Claude under the lead's authorization. The uncommitted review checkpoint can be backed up without waiting for a commit.

The subsequent sequence is: select a distinct production identity for the reviewed population-weighted artifact; build and verify it through the normal companion/loader; complete the source-tier staging or regeneration decision and machine preflight; perform the small predictive/generation pilot using the revised selection; then decide the full biology campaign. No generation was authorized or resumed by this review.
