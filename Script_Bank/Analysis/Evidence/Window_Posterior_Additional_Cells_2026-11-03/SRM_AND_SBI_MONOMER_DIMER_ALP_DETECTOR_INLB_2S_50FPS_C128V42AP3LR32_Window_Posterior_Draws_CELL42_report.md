# Window-posterior draws (INLB, 2S_50FPS_C128V42AP3LR32)

Fresh joint draws for cells [42], ten windows each, 10000 complete draws per window (flow-draws, pool mode unrestricted as in the source), with the adopted estimator (weights sha256 178e7dd91b41f7e55a816b784349b5cf00c19700fe795d6976a425d0eeac92b8) on cuda:0 (GPU), seed 20261103. Source references joined from `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_MAP_Experiment.npz` (sha256 bc764157da595fe002b9ca2c85b9e61bb0bc0b09464e036057bfc3a1263ff1ef). Reference SEL: `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_SEL_Nuisance_DLI.npz`. Stored coordinates are the estimator's (log10); physical values are ten to the power of the stored values. The fresh summaries are computed from the retained cloud; the source MAP is a reference from the source run, not a result of this cloud. Package 0.1.44; torch 2.7.1+cu118. Collection label `CELL42` (output-only; same estimator and source as the unlabeled collection).

## Checks

| check | result | detail |
|---|---|---|
| checkpoint_matches_source | PASS | loaded weights sha256 178e7dd91b41f7e55a816b784349b5cf00c19700fe795d6976a425d0eeac92b8; source product recorded 178e7dd91b41f7e55a816b784349b5cf00c19700fe795d6976a425d0eeac92b8 |
| cloud_shape | PASS | (10, 10000, 6) float32; expected (10, 10000, 6) float32 |
| cloud_finite | PASS | every draw finite |
| window_identities | PASS | 10 windows; each cell carries chunks 0..9 once; cells [42] |
| parameter_order_and_transform | PASS | ['mu_r', 'sigma_r', 'mu_pc', 'sigma_pc', 'prob_photo_bleach', 'lambda_rate']; parameterization.to_flow (log10 for log rows, linear rows as-is) |
| quantiles_recompute | PASS | quantiles recomputed from the saved cloud agree with the saved ones (max |diff| 0.0e+00 in log10; the saved cloud is float32, the summary was taken on the same float32 draws) |
| sgm_is_a_cloud_row | PASS | the saved SGM equals the cloud row at its index, and recomputing the medoid with the prior-width scale returns the same row |
| physical_endpoints_reproduce | PASS | physical endpoints = 10 ** stored log10 quantiles; widths are computed inside each window from these |
| source_join_by_identity | PASS | source MAP/quantiles/SGM joined by (kind_index, cell, chunk); row indices recorded |
| checkpoint_hash_recorded | PASS | 178e7dd91b41f7e55a816b784349b5cf00c19700fe795d6976a425d0eeac92b8 |
| source_product_unchanged | PASS | source npz sha256 bc764157da595fe002b9ca2c85b9e61bb0bc0b09464e036057bfc3a1263ff1ef |
| reference_artifact_unchanged | PASS | SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_SEL_Nuisance_DLI.npz sha256 2b2d98f99e4fd0c5c3734280cb613582df3bdbaa387e4f953ec71b7592ceac6f |
| code_unchanged_during_run | PASS | implementation hash c86416260f1f8b85 |

## Runtime and storage

- seconds per window: 3.2, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.2 (total sampling 24 s; wall 44 s)
- peak GPU memory 914 MiB
- `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_Window_Posterior_Draws_CELL42.npz`: 2.1 MiB compressed (2.3 MiB of float32 draws)

## Cell 42: fresh summaries per window (physical units) and the source references

**mu_r** (pixels (sqrt(2) times Gaussian sigma); SEL 2.256; prior [1.000, 1.995])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.278 | 2.268 | [2.194, 2.362] | 0.1686 | 2.276 | 2.280 | -0.000 | yes | 100.0 % |
| 1 | 2.246 | 2.242 | [2.163, 2.327] | 0.1636 | 2.244 | 2.245 | +0.000 | yes | 100.0 % |
| 2 | 2.245 | 2.239 | [2.161, 2.325] | 0.1634 | 2.243 | 2.244 | +0.000 | yes | 100.0 % |
| 3 | 2.228 | 2.204 | [2.147, 2.310] | 0.1638 | 2.228 | 2.228 | -0.000 | yes | 100.0 % |
| 4 | 2.222 | 2.233 | [2.139, 2.302] | 0.1622 | 2.222 | 2.222 | +0.000 | yes | 100.0 % |
| 5 | 2.198 | 2.202 | [2.117, 2.277] | 0.1606 | 2.198 | 2.198 | +0.000 | yes | 100.0 % |
| 6 | 2.187 | 2.166 | [2.107, 2.264] | 0.1569 | 2.188 | 2.188 | -0.000 | yes | 100.0 % |
| 7 | 2.193 | 2.200 | [2.111, 2.273] | 0.1619 | 2.195 | 2.194 | -0.000 | yes | 100.0 % |
| 8 | 2.170 | 2.179 | [2.090, 2.247] | 0.1569 | 2.172 | 2.170 | -0.000 | no | 99.9 % |
| 9 | 2.172 | 2.148 | [2.093, 2.251] | 0.1580 | 2.174 | 2.171 | +0.000 | no | 100.0 % |

**sigma_r** (dimensionless ln-spread; SEL 0.4029; prior [0.1000, 0.5623])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.4334 | 0.4300 | [0.3666, 0.5194] | 0.1527 | 0.4321 | 0.4336 | -0.000 | yes | 1.1 % |
| 1 | 0.4496 | 0.4462 | [0.3861, 0.5297] | 0.1436 | 0.4478 | 0.4490 | +0.001 | yes | 1.5 % |
| 2 | 0.4509 | 0.4506 | [0.3889, 0.5299] | 0.1410 | 0.4492 | 0.4505 | +0.000 | yes | 1.3 % |
| 3 | 0.4545 | 0.4571 | [0.3945, 0.5317] | 0.1371 | 0.4524 | 0.4539 | +0.001 | yes | 1.6 % |
| 4 | 0.4529 | 0.4467 | [0.3931, 0.5265] | 0.1333 | 0.4513 | 0.4538 | -0.001 | yes | 1.1 % |
| 5 | 0.4572 | 0.4567 | [0.3982, 0.5317] | 0.1335 | 0.4557 | 0.4568 | +0.000 | yes | 1.4 % |
| 6 | 0.4556 | 0.4515 | [0.3970, 0.5267] | 0.1298 | 0.4528 | 0.4548 | +0.001 | yes | 1.0 % |
| 7 | 0.4523 | 0.4536 | [0.3945, 0.5236] | 0.1291 | 0.4498 | 0.4520 | +0.000 | yes | 0.9 % |
| 8 | 0.4583 | 0.4591 | [0.3997, 0.5267] | 0.1271 | 0.4557 | 0.4582 | +0.000 | yes | 1.0 % |
| 9 | 0.4520 | 0.4395 | [0.3945, 0.5222] | 0.1277 | 0.4500 | 0.4520 | +0.000 | yes | 1.0 % |

**mu_pc** (photons per active dye per frame; SEL 152.90; prior [100.00, 562.34])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 149.85 | 150.44 | [102.29, 195.15] | 92.86 | 151.77 | 149.38 | +0.001 | yes | 4.1 % |
| 1 | 166.90 | 157.32 | [119.19, 208.70] | 89.50 | 169.35 | 166.58 | +0.001 | yes | 1.0 % |
| 2 | 189.09 | 181.76 | [136.89, 232.98] | 96.09 | 192.30 | 189.12 | -0.000 | yes | 0.3 % |
| 3 | 232.65 | 231.41 | [173.65, 280.00] | 106.36 | 236.60 | 233.33 | -0.001 | no | 0.0 % |
| 4 | 240.10 | 245.78 | [183.00, 287.50] | 104.50 | 243.07 | 240.50 | -0.001 | no | 0.0 % |
| 5 | 194.97 | 190.36 | [145.52, 235.57] | 90.04 | 199.82 | 194.89 | +0.000 | yes | 0.1 % |
| 6 | 196.14 | 192.05 | [144.78, 235.53] | 90.75 | 201.16 | 196.25 | -0.000 | yes | 0.2 % |
| 7 | 218.48 | 212.43 | [164.60, 259.37] | 94.76 | 221.40 | 218.38 | +0.000 | no | 0.0 % |
| 8 | 178.44 | 175.97 | [132.06, 217.54] | 85.48 | 185.38 | 179.03 | -0.001 | yes | 0.5 % |
| 9 | 170.44 | 161.89 | [124.43, 208.79] | 84.36 | 178.60 | 171.29 | -0.002 | yes | 0.6 % |

**sigma_pc** (dimensionless ln-spread; SEL 0.8588; prior [0.1778, 1.000])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.9883 | 0.9832 | [0.8106, 1.126] | 0.3150 | 0.9863 | 0.9879 | +0.000 | yes | 44.8 % |
| 1 | 0.9678 | 0.9705 | [0.7992, 1.099] | 0.3002 | 0.9674 | 0.9686 | -0.000 | yes | 36.3 % |
| 2 | 0.9600 | 0.9701 | [0.7764, 1.102] | 0.3252 | 0.9551 | 0.9580 | +0.001 | yes | 34.2 % |
| 3 | 0.9351 | 0.9262 | [0.7197, 1.096] | 0.3760 | 0.9195 | 0.9360 | -0.000 | yes | 28.1 % |
| 4 | 0.9245 | 0.9205 | [0.6891, 1.099] | 0.4094 | 0.9078 | 0.9233 | +0.001 | yes | 26.9 % |
| 5 | 0.9237 | 0.9446 | [0.7045, 1.087] | 0.3828 | 0.9046 | 0.9268 | -0.001 | yes | 25.2 % |
| 6 | 0.9120 | 0.8945 | [0.6736, 1.095] | 0.4213 | 0.8828 | 0.9143 | -0.001 | yes | 24.6 % |
| 7 | 0.9013 | 0.9216 | [0.6286, 1.111] | 0.4821 | 0.8670 | 0.9025 | -0.001 | yes | 24.6 % |
| 8 | 0.9096 | 0.9046 | [0.6817, 1.085] | 0.4035 | 0.8755 | 0.9087 | +0.000 | yes | 22.9 % |
| 9 | 0.9065 | 0.9231 | [0.6662, 1.088] | 0.4219 | 0.8594 | 0.9003 | +0.003 | yes | 22.8 % |

**prob_photo_bleach** (probability per 100 frames; SEL 0.0500; prior [0.0100, 0.3162])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1072 | 0.1043 | [0.0181, 0.3128] | 0.2947 | 0.1321 | 0.1097 | -0.010 | yes | 5.5 % |
| 1 | 0.1072 | 0.1064 | [0.0172, 0.3154] | 0.2983 | 0.1364 | 0.1093 | -0.008 | yes | 5.9 % |
| 2 | 0.1098 | 0.1058 | [0.0171, 0.3229] | 0.3057 | 0.1365 | 0.1070 | +0.011 | yes | 6.4 % |
| 3 | 0.1084 | 0.0998 | [0.0156, 0.3225] | 0.3069 | 0.1375 | 0.1090 | -0.002 | yes | 6.7 % |
| 4 | 0.1081 | 0.1057 | [0.0156, 0.3240] | 0.3084 | 0.1398 | 0.1043 | +0.016 | yes | 6.7 % |
| 5 | 0.1041 | 0.0961 | [0.0153, 0.3108] | 0.2955 | 0.1406 | 0.1028 | +0.005 | yes | 6.0 % |
| 6 | 0.1017 | 0.0947 | [0.0154, 0.3080] | 0.2927 | 0.1425 | 0.0986 | +0.013 | yes | 5.8 % |
| 7 | 0.1006 | 0.0907 | [0.0147, 0.3034] | 0.2887 | 0.1422 | 0.0997 | +0.004 | yes | 5.6 % |
| 8 | 0.0973 | 0.0922 | [0.0148, 0.3075] | 0.2927 | 0.1449 | 0.0986 | -0.006 | yes | 5.8 % |
| 9 | 0.0993 | 0.0918 | [0.0143, 0.3038] | 0.2895 | 0.1460 | 0.0971 | +0.010 | yes | 5.8 % |

**lambda_rate** (per second; SEL 2.000; prior [1.000, 10.00])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.042 | 4.099 | [2.837, 6.904] | 4.067 | 3.997 | 4.046 | -0.000 | no | 0.2 % |
| 1 | 4.451 | 4.483 | [3.170, 7.415] | 4.245 | 4.409 | 4.440 | +0.001 | no | 0.3 % |
| 2 | 4.106 | 4.115 | [2.860, 7.332] | 4.472 | 4.134 | 4.138 | -0.003 | no | 0.4 % |
| 3 | 3.520 | 3.571 | [2.310, 7.125] | 4.816 | 3.658 | 3.515 | +0.001 | no | 0.3 % |
| 4 | 3.165 | 3.212 | [1.993, 7.047] | 5.054 | 3.298 | 3.177 | -0.002 | yes | 0.3 % |
| 5 | 3.524 | 3.576 | [2.282, 7.330] | 5.048 | 3.686 | 3.498 | +0.003 | no | 0.3 % |
| 6 | 3.087 | 3.093 | [1.901, 7.028] | 5.127 | 3.331 | 3.071 | +0.002 | yes | 0.3 % |
| 7 | 2.433 | 2.564 | [1.426, 6.498] | 5.073 | 2.667 | 2.405 | +0.005 | yes | 0.3 % |
| 8 | 3.208 | 3.293 | [2.012, 6.965] | 4.953 | 3.501 | 3.209 | -0.000 | no | 0.4 % |
| 9 | 3.092 | 3.188 | [1.901, 7.100] | 5.199 | 3.494 | 3.138 | -0.006 | yes | 0.3 % |

### Cell 42: joint depth of the reference vectors

| chunk | reference | prior-scaled distance to the fresh SGM | share of draws farther | inside every marginal 90 % |
|---|---|---|---|---|
| 0 | SEL | 0.388 | 0.194 | no |
| 0 | source MAP | 0.070 | 0.991 | yes |
| 0 | source SGM | 0.032 | 1.000 | yes |
| 0 | fresh median vector | 0.013 | 1.000 | yes |
| 1 | SEL | 0.424 | 0.142 | no |
| 1 | source MAP | 0.084 | 0.975 | yes |
| 1 | source SGM | 0.028 | 1.000 | yes |
| 1 | fresh median vector | 0.035 | 1.000 | yes |
| 2 | SEL | 0.406 | 0.178 | no |
| 2 | source MAP | 0.081 | 0.979 | yes |
| 2 | source SGM | 0.034 | 0.999 | yes |
| 2 | fresh median vector | 0.026 | 1.000 | yes |
| 3 | SEL | 0.412 | 0.203 | no |
| 3 | source MAP | 0.096 | 0.969 | yes |
| 3 | source SGM | 0.033 | 1.000 | yes |
| 3 | fresh median vector | 0.030 | 1.000 | yes |
| 4 | SEL | 0.413 | 0.223 | no |
| 4 | source MAP | 0.083 | 0.984 | yes |
| 4 | source SGM | 0.046 | 0.999 | yes |
| 4 | fresh median vector | 0.020 | 1.000 | yes |
| 5 | SEL | 0.354 | 0.324 | no |
| 5 | source MAP | 0.117 | 0.940 | yes |
| 5 | source SGM | 0.040 | 0.999 | yes |
| 5 | fresh median vector | 0.031 | 1.000 | yes |
| 6 | SEL | 0.310 | 0.472 | yes |
| 6 | source MAP | 0.127 | 0.930 | yes |
| 6 | source SGM | 0.039 | 1.000 | yes |
| 6 | fresh median vector | 0.030 | 1.000 | yes |
| 7 | SEL | 0.292 | 0.560 | no |
| 7 | source MAP | 0.138 | 0.921 | yes |
| 7 | source SGM | 0.032 | 1.000 | yes |
| 7 | fresh median vector | 0.043 | 0.999 | yes |
| 8 | SEL | 0.307 | 0.475 | no |
| 8 | source MAP | 0.139 | 0.904 | yes |
| 8 | source SGM | 0.029 | 1.000 | yes |
| 8 | fresh median vector | 0.022 | 1.000 | yes |
| 9 | SEL | 0.287 | 0.551 | no |
| 9 | source MAP | 0.158 | 0.882 | yes |
| 9 | source SGM | 0.041 | 1.000 | yes |
| 9 | fresh median vector | 0.047 | 0.999 | yes |

The share of draws farther from the window's SGM than the reference is a distance rank relative to the SGM in prior-scaled estimator coordinates: near 1 means most draws lie farther from the SGM than the reference does; near 0 means almost every draw is closer to the SGM than the reference is. It is a distance rank, not a posterior-density or joint-credible-region statement.

### Cell 42: within-window correlations (Pearson, estimator coordinates), chunks (0, 4, 9)

chunk 0:

|  | mu_r | sigma_r | mu_pc | sigma_pc | prob_photo_bleach | lambda_rate |
|---|---|---|---|---|---|---|
| mu_r | +1.00 / +1.00 | +0.11 / +0.10 | +0.26 / +0.23 | -0.12 / -0.12 | +0.04 / +0.08 | +0.11 / +0.11 |
| sigma_r | +0.11 / +0.10 | +1.00 / +1.00 | +0.20 / +0.16 | -0.16 / -0.14 | -0.15 / -0.15 | +0.09 / +0.08 |
| mu_pc | +0.26 / +0.23 | +0.20 / +0.16 | +1.00 / +1.00 | -0.59 / -0.60 | -0.38 / -0.39 | +0.44 / +0.45 |
| sigma_pc | -0.12 / -0.12 | -0.16 / -0.14 | -0.59 / -0.60 | +1.00 / +1.00 | +0.31 / +0.30 | -0.93 / -0.92 |
| prob_photo_bleach | +0.04 / +0.08 | -0.15 / -0.15 | -0.38 / -0.39 | +0.31 / +0.30 | +1.00 / +1.00 | -0.23 / -0.23 |
| lambda_rate | +0.11 / +0.11 | +0.09 / +0.08 | +0.44 / +0.45 | -0.93 / -0.92 | -0.23 / -0.23 | +1.00 / +1.00 |
(Pearson / Spearman)

chunk 4:

|  | mu_r | sigma_r | mu_pc | sigma_pc | prob_photo_bleach | lambda_rate |
|---|---|---|---|---|---|---|
| mu_r | +1.00 / +1.00 | +0.10 / +0.09 | +0.20 / +0.15 | -0.19 / -0.19 | +0.03 / +0.06 | +0.19 / +0.20 |
| sigma_r | +0.10 / +0.09 | +1.00 / +1.00 | +0.15 / +0.11 | -0.11 / -0.11 | -0.15 / -0.15 | +0.09 / +0.09 |
| mu_pc | +0.20 / +0.15 | +0.15 / +0.11 | +1.00 / +1.00 | -0.42 / -0.44 | -0.34 / -0.36 | +0.34 / +0.35 |
| sigma_pc | -0.19 / -0.19 | -0.11 / -0.11 | -0.42 / -0.44 | +1.00 / +1.00 | +0.15 / +0.14 | -0.98 / -0.97 |
| prob_photo_bleach | +0.03 / +0.06 | -0.15 / -0.15 | -0.34 / -0.36 | +0.15 / +0.14 | +1.00 / +1.00 | -0.11 / -0.11 |
| lambda_rate | +0.19 / +0.20 | +0.09 / +0.09 | +0.34 / +0.35 | -0.98 / -0.97 | -0.11 / -0.11 | +1.00 / +1.00 |
(Pearson / Spearman)

chunk 9:

|  | mu_r | sigma_r | mu_pc | sigma_pc | prob_photo_bleach | lambda_rate |
|---|---|---|---|---|---|---|
| mu_r | +1.00 / +1.00 | +0.10 / +0.11 | +0.18 / +0.15 | -0.17 / -0.17 | +0.02 / +0.04 | +0.18 / +0.18 |
| sigma_r | +0.10 / +0.11 | +1.00 / +1.00 | +0.17 / +0.14 | -0.12 / -0.11 | -0.14 / -0.14 | +0.10 / +0.10 |
| mu_pc | +0.18 / +0.15 | +0.17 / +0.14 | +1.00 / +1.00 | -0.68 / -0.72 | -0.27 / -0.27 | +0.61 / +0.65 |
| sigma_pc | -0.17 / -0.17 | -0.12 / -0.11 | -0.68 / -0.72 | +1.00 / +1.00 | +0.10 / +0.10 | -0.98 / -0.97 |
| prob_photo_bleach | +0.02 / +0.04 | -0.14 / -0.14 | -0.27 / -0.27 | +0.10 / +0.10 | +1.00 / +1.00 | -0.07 / -0.07 |
| lambda_rate | +0.18 / +0.18 | +0.10 / +0.10 | +0.61 / +0.65 | -0.98 / -0.97 | -0.07 / -0.07 | +1.00 / +1.00 |
(Pearson / Spearman)

Figures: `figures/joint_cell_42_chunk_0.png`, `figures/joint_cell_42_chunk_4.png`, `figures/joint_cell_42_chunk_9.png`, `figures/marginals_cell_42.png`, `figures/correlations_cell_42.png`.

## Files

- `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_Window_Posterior_Draws_CELL42.npz`: posterior_samples_cloud (windows x draws x parameters, float32, log10), kind_index/cell/chunk, fresh_quantiles, fresh_sgm, fresh_sgm_index, draws_outside_prior, source_map_estimate / source_posterior_quantiles / source_posterior_sgm / source_scores / source_row_index (joined references), prior box, sgm_scale, collection_manifest_json.
- `window_summaries.csv`, `correlations.csv`, `joint_depth.csv`, `figures/`, `provenance.json`.

## Reading rules

- Every row of the cloud is one complete joint draw; columns are never resampled or rank-matched independently.
- Outside-prior draws are retained and counted (unrestricted flow draws, as in the source).
- Correlations are reported in estimator coordinates (log10) with Spearman beside Pearson; the physical-coordinate Pearson is in `correlations.csv`.
- Fresh and source summaries come from different Monte Carlo samples; their differences (dex) measure sampling reproducibility, not a change of the posterior.

## Erratum (2026-11-03, after review)

The explanatory sentence under "joint depth of the reference vectors" was reworded from the saved text: the share of draws farther than the reference is a distance rank relative to the window SGM, not a statement about posterior density ("dense core"). Numbers, draws, figures and provenance are unchanged; nothing was resampled.
