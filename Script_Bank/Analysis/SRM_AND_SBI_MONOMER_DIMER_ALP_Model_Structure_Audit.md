# Model-structure audit

**Script:** `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Model_Structure_Audit.py`
**Output:** `<data_bank_root>/Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_Model_Structure_Audit/` (report `.md` + `audit_summary.json`)
**Scope:** the separated stoichiometry–mobility generator introduced in 0.1.2 (two molecular species × three mobility modes = six particle types, seventeen generated reaction channels, twelve learnable parameters, the shared parameter-conversion rule). A correctness check of the model STRUCTURE, not an inference validation.

## Why it exists

Subunit conservation alone cannot detect a wrong reaction channel: a network that sends two slow monomers into a fast dimer conserves subunits just as well as the intended one. The model specification's implementation increment therefore asks for deterministic checks that run without a simulation, before any approved run, and a few tiny run-level checks after approval. This script is that check, in the same shape as the labeling audit: prespecified criteria, one report, every number in a JSON.

## Deterministic tier (no simulation; always runs)

| check | criterion |
|---|---|
| D1 channels | `reaction_channels` yields exactly 17 unique channels: 6 association fusions (one per unordered pair of monomer modes; product = dimer in the SLOWER parent's mode; one shared microscopic rate), 3 dissociation fissions (dimer → two monomers in the dimer's mode), 8 switching conversions (four shared rates × two species; adjacent modes only). `build_system` registers exactly these names. |
| D2 diffusion | `D[X, m] = D_A · species_factor · mode_factor` for every type; on 20 000 prior draws plus every box corner, immobile < slow ≤ fast within a species and dimer ≤ monomer within a mode (the disjoint-range guarantee checked at import). |
| D3 transforms | `to_flow(to_physical(u)) == u` on prior draws and both box corners; the linear initial dimer fraction passes through untouched at 0 and 1; a blanket `10**u` differs from `to_physical` exactly on that row; every table `VALUE` is its row's prior center. |
| D4 composition | `realize_initial_composition` for totals 1..40 at x_B ∈ {0, 1/3, 1/2, 1}: conservation N_R = n_A + 2 n_B, dimer cap ⌊N_R/2⌋, one leftover monomer at odd totals with x_B = 1, requested vs realized fraction; non-integer totals round, sub-one totals floor to one monomer, x_B outside [0, 1] is rejected. The association reference equals 4π(2D_A)r / ((4/3)πr³) = 6 D_A / r², and λ_on = R_ON · λ_ref on every fusion. |
| D5 stationary law | `stationary_mode_law` sums to one and satisfies detailed balance on every link, over 200 prior draws. |
| D6 occupancy | `rank_to_species` maps every monomer type to A and every dimer type to B; a per-species occupancy yields identical probabilities for A_f, A_s, A_i (and for B_f, B_s, B_i); an occupancy keyed by particle type is rejected; a trajectory carrying the legacy A/B/C types is rejected. |
| D7 fission placement | Fission daughters are placed at `SimulationStem.fission_product_distance_nm` = 2 × the reaction distance (20 nm), outside the fusion radius, and `build_system` passes that field. Declared convention (2026-09-09): at the 2 ms sub-step an eligible pair reacts with probability ≈ 1 for every D_A in range, so daughters placed AT the radius would re-fuse at the next step unless they diffused apart within one step (≈ 3% for fast, ≈ 50% per step for immobile daughters at the prior center), making the effective unbinding rate mode dependent. The one-step rebinding probabilities under the old placement are reported as the documented reason. |

## Run tier (`--run`; tiny simulations; explicit approval required)

| check | criterion |
|---|---|
| R1 conservation | Reactive 1 s run at the prior center: the lineage replays with every subunit covered once per frame (the extractor fails loud otherwise) and its subunit count equals the realized N_R; the per-frame subunit total from the particles observable is constant. |
| R2 stationarity | Isolated chain (association off, monomers only, 600 particles, 10 s): time-averaged mode occupancies over the second half within 0.03 of `stationary_mode_law`. |
| R3 visible fractions | 400 static labelings on the reactive lineage at a declared per-species occupancy (A 0.8, B 0.6), both conditions: monomer `p_occ (1 − P0)` and dimer `1 − (1 − p_occ (1 − P0))²` within 0.02. |
| R4 both conditions | One 20-frame video per condition rendered from the SAME trajectory through the production renderer; both carry finite signal. |
| R5 boundary | In the 1 s run, particles may lie outside the imaged field (open lateral boundary, no confining potential) while the subunit total stays constant; the fraction outside is reported, not judged. |

## Status

- Deterministic tier: **PASS** (6/6) on 2026-09-09 at the introduction of the generator (0.1.2), PC profile `mars_pc`.
- Deterministic tier: **PASS** (7/7) on 2026-09-09 after the fission-placement convention (D7 added), PC profile `mars_pc`.
- Run tier: not executed; awaits the lead's approval (it is compute, however small).

## Reading the report

A deterministic FAIL is a structural defect in the generator or the table and blocks everything downstream. A run-tier FAIL in R1 is a lineage or network defect; in R2 a switching-rate or initial-mode defect; in R3 an occupancy or labeling defect; in R4 a renderer/lineage handoff defect; R5 is descriptive. The JSON carries the channel list, the per-check numbers, and the tolerances.
