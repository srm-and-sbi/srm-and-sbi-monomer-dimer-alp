# Window-posterior draws (INLB, 2S_50FPS_C128V42AP3LR32)

Fresh joint draws for cells [11], ten windows each, 10000 complete draws per window (flow-draws, pool mode unrestricted as in the source), with the adopted estimator (weights sha256 178e7dd91b41f7e55a816b784349b5cf00c19700fe795d6976a425d0eeac92b8) on cuda:0 (GPU), seed 20261103. Source references joined from `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_MAP_Experiment.npz` (sha256 bc764157da595fe002b9ca2c85b9e61bb0bc0b09464e036057bfc3a1263ff1ef). Reference SEL: `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_SEL_Nuisance_DLI.npz`. Stored coordinates are the estimator's (log10); physical values are ten to the power of the stored values. The fresh summaries are computed from the retained cloud; the source MAP is a reference from the source run, not a result of this cloud. Package 0.1.44; torch 2.7.1+cu118.

## Checks

| check | result | detail |
|---|---|---|
| checkpoint_matches_source | PASS | loaded weights sha256 178e7dd91b41f7e55a816b784349b5cf00c19700fe795d6976a425d0eeac92b8; source product recorded 178e7dd91b41f7e55a816b784349b5cf00c19700fe795d6976a425d0eeac92b8 |
| cloud_shape | PASS | (10, 10000, 6) float32; expected (10, 10000, 6) float32 |
| cloud_finite | PASS | every draw finite |
| window_identities | PASS | 10 windows; each cell carries chunks 0..9 once; cells [11] |
| parameter_order_and_transform | PASS | ['mu_r', 'sigma_r', 'mu_pc', 'sigma_pc', 'prob_photo_bleach', 'lambda_rate']; parameterization.to_flow (log10 for log rows, linear rows as-is) |
| quantiles_recompute | PASS | quantiles recomputed from the saved cloud agree with the saved ones (max |diff| 3.4e-13 in log10; the saved cloud is float32, the summary was taken on the same float32 draws) |
| sgm_is_a_cloud_row | PASS | the saved SGM equals the cloud row at its index, and recomputing the medoid with the prior-width scale returns the same row |
| physical_endpoints_reproduce | PASS | physical endpoints = 10 ** stored log10 quantiles; widths are computed inside each window from these |
| source_join_by_identity | PASS | source MAP/quantiles/SGM joined by (kind_index, cell, chunk); row indices recorded |
| checkpoint_hash_recorded | PASS | 178e7dd91b41f7e55a816b784349b5cf00c19700fe795d6976a425d0eeac92b8 |
| source_product_unchanged | PASS | source npz sha256 bc764157da595fe002b9ca2c85b9e61bb0bc0b09464e036057bfc3a1263ff1ef |
| reference_artifact_unchanged | PASS | SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_SEL_Nuisance_DLI.npz sha256 2b2d98f99e4fd0c5c3734280cb613582df3bdbaa387e4f953ec71b7592ceac6f |
| code_unchanged_during_run | PASS | implementation hash c86416260f1f8b85 |

## Runtime and storage

- seconds per window: 3.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3 (total sampling 24 s; wall 45 s)
- peak GPU memory 914 MiB
- `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_Window_Posterior_Draws.npz`: 2.1 MiB compressed (2.3 MiB of float32 draws)

## Cell 11: fresh summaries per window (physical units) and the source references

**mu_r** (pixels (sqrt(2) times Gaussian sigma); SEL 2.256; prior [1.000, 1.995])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.318 | 2.333 | [2.231, 2.404] | 0.1732 | 2.316 | 2.319 | -0.000 | yes | 100.0 % |
| 1 | 2.342 | 2.312 | [2.254, 2.430] | 0.1767 | 2.334 | 2.342 | -0.000 | yes | 100.0 % |
| 2 | 2.345 | 2.359 | [2.258, 2.440] | 0.1817 | 2.336 | 2.344 | +0.000 | no | 100.0 % |
| 3 | 2.329 | 2.315 | [2.243, 2.421] | 0.1773 | 2.321 | 2.330 | -0.000 | yes | 100.0 % |
| 4 | 2.297 | 2.303 | [2.210, 2.392] | 0.1815 | 2.292 | 2.296 | +0.000 | yes | 100.0 % |
| 5 | 2.283 | 2.306 | [2.198, 2.373] | 0.1750 | 2.282 | 2.283 | +0.000 | yes | 100.0 % |
| 6 | 2.251 | 2.222 | [2.171, 2.332] | 0.1613 | 2.249 | 2.250 | +0.000 | yes | 100.0 % |
| 7 | 2.232 | 2.250 | [2.151, 2.312] | 0.1616 | 2.232 | 2.232 | -0.000 | yes | 100.0 % |
| 8 | 2.226 | 2.204 | [2.146, 2.305] | 0.1591 | 2.226 | 2.226 | -0.000 | yes | 100.0 % |
| 9 | 2.187 | 2.177 | [2.110, 2.266] | 0.1564 | 2.190 | 2.188 | -0.000 | yes | 100.0 % |

**sigma_r** (dimensionless ln-spread; SEL 0.4029; prior [0.1000, 0.5623])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.4043 | 0.3943 | [0.3398, 0.4858] | 0.1460 | 0.4029 | 0.4042 | +0.000 | yes | 0.3 % |
| 1 | 0.3664 | 0.3625 | [0.2994, 0.4512] | 0.1517 | 0.3656 | 0.3662 | +0.000 | yes | 0.1 % |
| 2 | 0.3153 | 0.3196 | [0.2480, 0.4002] | 0.1522 | 0.3179 | 0.3160 | -0.001 | no | 0.0 % |
| 3 | 0.3561 | 0.3656 | [0.2875, 0.4433] | 0.1558 | 0.3568 | 0.3557 | +0.000 | yes | 0.1 % |
| 4 | 0.3665 | 0.3613 | [0.2966, 0.4517] | 0.1550 | 0.3708 | 0.3668 | -0.000 | yes | 0.1 % |
| 5 | 0.3877 | 0.3883 | [0.3196, 0.4725] | 0.1530 | 0.3911 | 0.3889 | -0.001 | yes | 0.2 % |
| 6 | 0.4154 | 0.4146 | [0.3513, 0.4930] | 0.1417 | 0.4144 | 0.4152 | +0.000 | yes | 0.3 % |
| 7 | 0.4200 | 0.4287 | [0.3593, 0.4960] | 0.1367 | 0.4196 | 0.4203 | -0.000 | yes | 0.3 % |
| 8 | 0.4180 | 0.4062 | [0.3560, 0.4908] | 0.1349 | 0.4173 | 0.4168 | +0.001 | yes | 0.2 % |
| 9 | 0.4248 | 0.4368 | [0.3682, 0.4955] | 0.1273 | 0.4235 | 0.4255 | -0.001 | yes | 0.3 % |

**mu_pc** (photons per active dye per frame; SEL 152.90; prior [100.00, 562.34])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 280.62 | 284.29 | [205.73, 370.99] | 165.27 | 279.32 | 280.89 | -0.000 | no | 0.0 % |
| 1 | 168.06 | 161.66 | [107.30, 228.01] | 120.71 | 168.82 | 167.36 | +0.002 | yes | 3.4 % |
| 2 | 91.25 | 91.19 | [50.98, 139.86] | 88.88 | 93.24 | 91.56 | -0.001 | no | 62.3 % |
| 3 | 117.10 | 119.24 | [69.61, 168.51] | 98.90 | 119.41 | 117.40 | -0.001 | yes | 27.1 % |
| 4 | 80.95 | 80.97 | [47.60, 124.32] | 76.71 | 84.55 | 80.63 | +0.002 | no | 76.8 % |
| 5 | 99.89 | 99.02 | [60.45, 143.67] | 83.23 | 104.72 | 99.90 | -0.000 | no | 50.2 % |
| 6 | 153.41 | 146.48 | [105.43, 194.83] | 89.40 | 155.85 | 152.93 | +0.001 | yes | 3.2 % |
| 7 | 152.61 | 149.42 | [107.55, 191.13] | 83.58 | 155.59 | 152.89 | -0.001 | yes | 3.0 % |
| 8 | 157.50 | 151.97 | [113.46, 195.27] | 81.81 | 161.29 | 157.85 | -0.001 | yes | 2.1 % |
| 9 | 204.32 | 197.68 | [164.58, 239.33] | 74.75 | 204.07 | 203.96 | +0.001 | no | 0.0 % |

**sigma_pc** (dimensionless ln-spread; SEL 0.8588; prior [0.1778, 1.000])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.7990 | 0.7731 | [0.5369, 1.042] | 0.5046 | 0.7566 | 0.7983 | +0.000 | yes | 9.2 % |
| 1 | 0.8727 | 0.8789 | [0.6209, 1.113] | 0.4916 | 0.8589 | 0.8741 | -0.001 | yes | 20.0 % |
| 2 | 0.9733 | 0.9689 | [0.7398, 1.190] | 0.4505 | 0.9497 | 0.9714 | +0.001 | yes | 42.2 % |
| 3 | 0.9259 | 0.9300 | [0.6817, 1.146] | 0.4641 | 0.9061 | 0.9245 | +0.001 | yes | 29.7 % |
| 4 | 0.9154 | 0.9121 | [0.6922, 1.101] | 0.4087 | 0.8769 | 0.9160 | -0.000 | yes | 24.8 % |
| 5 | 0.8784 | 0.8491 | [0.6365, 1.080] | 0.4435 | 0.8405 | 0.8811 | -0.001 | yes | 17.2 % |
| 6 | 0.8077 | 0.8091 | [0.5534, 1.018] | 0.4649 | 0.7770 | 0.8086 | -0.000 | yes | 7.0 % |
| 7 | 0.7844 | 0.8037 | [0.5335, 0.9937] | 0.4602 | 0.7469 | 0.7839 | +0.000 | yes | 4.5 % |
| 8 | 0.7566 | 0.7620 | [0.5156, 0.9606] | 0.4450 | 0.7142 | 0.7563 | +0.000 | yes | 2.1 % |
| 9 | 0.6559 | 0.6290 | [0.4294, 0.8377] | 0.4083 | 0.6215 | 0.6542 | +0.001 | no | 0.1 % |

**prob_photo_bleach** (probability per 100 frames; SEL 0.0500; prior [0.0100, 0.3162])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1006 | 0.0932 | [0.0185, 0.3180] | 0.2995 | 0.1112 | 0.0991 | +0.006 | yes | 5.6 % |
| 1 | 0.1043 | 0.1100 | [0.0210, 0.3176] | 0.2966 | 0.1095 | 0.1045 | -0.001 | yes | 5.4 % |
| 2 | 0.1104 | 0.1004 | [0.0228, 0.3233] | 0.3004 | 0.1171 | 0.1094 | +0.004 | yes | 5.6 % |
| 3 | 0.1044 | 0.1026 | [0.0201, 0.3050] | 0.2849 | 0.1140 | 0.1060 | -0.007 | yes | 4.6 % |
| 4 | 0.1041 | 0.1083 | [0.0196, 0.2976] | 0.2779 | 0.1184 | 0.1019 | +0.009 | yes | 4.4 % |
| 5 | 0.1017 | 0.1015 | [0.0187, 0.2876] | 0.2689 | 0.1188 | 0.1005 | +0.005 | yes | 4.1 % |
| 6 | 0.0969 | 0.0988 | [0.0172, 0.2832] | 0.2660 | 0.1214 | 0.0957 | +0.006 | yes | 4.0 % |
| 7 | 0.0964 | 0.0819 | [0.0160, 0.2787] | 0.2627 | 0.1249 | 0.0971 | -0.003 | yes | 3.8 % |
| 8 | 0.0936 | 0.0896 | [0.0161, 0.2830] | 0.2669 | 0.1246 | 0.0955 | -0.009 | yes | 3.8 % |
| 9 | 0.0922 | 0.0850 | [0.0147, 0.2861] | 0.2715 | 0.1275 | 0.0887 | +0.017 | yes | 4.3 % |

**lambda_rate** (per second; SEL 2.000; prior [1.000, 10.00])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.409 | 1.461 | [0.7952, 4.419] | 3.624 | 1.650 | 1.416 | -0.002 | yes | 20.9 % |
| 1 | 1.450 | 1.381 | [0.8410, 3.787] | 2.946 | 1.511 | 1.430 | +0.006 | yes | 15.7 % |
| 2 | 1.836 | 1.935 | [1.118, 4.002] | 2.884 | 1.928 | 1.842 | -0.002 | yes | 2.0 % |
| 3 | 1.565 | 1.662 | [0.9382, 3.689] | 2.751 | 1.649 | 1.566 | -0.000 | yes | 8.1 % |
| 4 | 1.959 | 1.982 | [1.193, 4.228] | 3.035 | 2.157 | 1.955 | +0.001 | yes | 1.0 % |
| 5 | 1.633 | 1.624 | [0.9829, 3.948] | 2.965 | 1.822 | 1.628 | +0.001 | yes | 5.7 % |
| 6 | 1.385 | 1.410 | [0.8119, 3.874] | 3.062 | 1.547 | 1.377 | +0.003 | yes | 19.0 % |
| 7 | 1.393 | 1.447 | [0.8156, 4.000] | 3.184 | 1.589 | 1.394 | -0.000 | yes | 19.5 % |
| 8 | 1.350 | 1.430 | [0.7973, 3.865] | 3.068 | 1.591 | 1.368 | -0.006 | yes | 21.3 % |
| 9 | 1.346 | 1.421 | [0.7631, 4.390] | 3.627 | 1.570 | 1.352 | -0.002 | yes | 25.0 % |

### Cell 11: joint depth of the reference vectors

| chunk | reference | prior-scaled distance to the fresh SGM | share of draws farther | inside every marginal 90 % |
|---|---|---|---|---|
| 0 | SEL | 0.432 | 0.261 | no |
| 0 | source MAP | 0.077 | 0.995 | yes |
| 0 | source SGM | 0.032 | 1.000 | yes |
| 0 | fresh median vector | 0.038 | 1.000 | yes |
| 1 | SEL | 0.290 | 0.556 | yes |
| 1 | source MAP | 0.050 | 0.999 | yes |
| 1 | source SGM | 0.042 | 1.000 | yes |
| 1 | fresh median vector | 0.040 | 1.000 | yes |
| 2 | SEL | 0.397 | 0.289 | no |
| 2 | source MAP | 0.050 | 0.999 | yes |
| 2 | source SGM | 0.034 | 1.000 | yes |
| 2 | fresh median vector | 0.038 | 1.000 | yes |
| 3 | SEL | 0.278 | 0.606 | yes |
| 3 | source MAP | 0.037 | 1.000 | yes |
| 3 | source SGM | 0.031 | 1.000 | yes |
| 3 | fresh median vector | 0.034 | 1.000 | yes |
| 4 | SEL | 0.438 | 0.205 | no |
| 4 | source MAP | 0.059 | 0.998 | yes |
| 4 | source SGM | 0.058 | 0.998 | yes |
| 4 | fresh median vector | 0.016 | 1.000 | yes |
| 5 | SEL | 0.339 | 0.417 | no |
| 5 | source MAP | 0.077 | 0.994 | yes |
| 5 | source SGM | 0.068 | 0.996 | yes |
| 5 | fresh median vector | 0.025 | 1.000 | yes |
| 6 | SEL | 0.254 | 0.646 | yes |
| 6 | source MAP | 0.086 | 0.986 | yes |
| 6 | source SGM | 0.044 | 0.999 | yes |
| 6 | fresh median vector | 0.034 | 1.000 | yes |
| 7 | SEL | 0.208 | 0.788 | yes |
| 7 | source MAP | 0.139 | 0.931 | yes |
| 7 | source SGM | 0.041 | 0.999 | yes |
| 7 | fresh median vector | 0.056 | 0.999 | yes |
| 8 | SEL | 0.236 | 0.715 | yes |
| 8 | source MAP | 0.120 | 0.952 | yes |
| 8 | source SGM | 0.038 | 0.999 | yes |
| 8 | fresh median vector | 0.041 | 0.999 | yes |
| 9 | SEL | 0.324 | 0.514 | no |
| 9 | source MAP | 0.128 | 0.948 | yes |
| 9 | source SGM | 0.042 | 0.999 | yes |
| 9 | fresh median vector | 0.049 | 0.999 | yes |

The share of draws farther from the window's SGM than the reference is a distance rank relative to the SGM in prior-scaled estimator coordinates: near 1 means most draws lie farther from the SGM than the reference does; near 0 means almost every draw is closer to the SGM than the reference is. It is a distance rank, not a posterior-density or joint-credible-region statement.

### Cell 11: within-window correlations (Pearson, estimator coordinates), chunks (0, 4, 9)

chunk 0:

|  | mu_r | sigma_r | mu_pc | sigma_pc | prob_photo_bleach | lambda_rate |
|---|---|---|---|---|---|---|
| mu_r | +1.00 / +1.00 | +0.14 / +0.12 | +0.16 / +0.11 | -0.28 / -0.29 | +0.06 / +0.09 | +0.29 / +0.31 |
| sigma_r | +0.14 / +0.12 | +1.00 / +1.00 | +0.11 / +0.07 | -0.20 / -0.19 | -0.19 / -0.19 | +0.17 / +0.17 |
| mu_pc | +0.16 / +0.11 | +0.11 / +0.07 | +1.00 / +1.00 | -0.08 / -0.06 | -0.39 / -0.40 | -0.03 / -0.05 |
| sigma_pc | -0.28 / -0.29 | -0.20 / -0.19 | -0.08 / -0.06 | +1.00 / +1.00 | +0.27 / +0.26 | -0.97 / -0.96 |
| prob_photo_bleach | +0.06 / +0.09 | -0.19 / -0.19 | -0.39 / -0.40 | +0.27 / +0.26 | +1.00 / +1.00 | -0.21 / -0.21 |
| lambda_rate | +0.29 / +0.31 | +0.17 / +0.17 | -0.03 / -0.05 | -0.97 / -0.96 | -0.21 / -0.21 | +1.00 / +1.00 |
(Pearson / Spearman)

chunk 4:

|  | mu_r | sigma_r | mu_pc | sigma_pc | prob_photo_bleach | lambda_rate |
|---|---|---|---|---|---|---|
| mu_r | +1.00 / +1.00 | +0.23 / +0.22 | +0.25 / +0.23 | -0.04 / -0.04 | +0.14 / +0.19 | +0.04 / +0.05 |
| sigma_r | +0.23 / +0.22 | +1.00 / +1.00 | +0.29 / +0.26 | -0.30 / -0.29 | -0.12 / -0.11 | +0.19 / +0.18 |
| mu_pc | +0.25 / +0.23 | +0.29 / +0.26 | +1.00 / +1.00 | -0.69 / -0.71 | -0.37 / -0.36 | +0.61 / +0.63 |
| sigma_pc | -0.04 / -0.04 | -0.30 / -0.29 | -0.69 / -0.71 | +1.00 / +1.00 | +0.28 / +0.26 | -0.92 / -0.91 |
| prob_photo_bleach | +0.14 / +0.19 | -0.12 / -0.11 | -0.37 / -0.36 | +0.28 / +0.26 | +1.00 / +1.00 | -0.20 / -0.17 |
| lambda_rate | +0.04 / +0.05 | +0.19 / +0.18 | +0.61 / +0.63 | -0.92 / -0.91 | -0.20 / -0.17 | +1.00 / +1.00 |
(Pearson / Spearman)

chunk 9:

|  | mu_r | sigma_r | mu_pc | sigma_pc | prob_photo_bleach | lambda_rate |
|---|---|---|---|---|---|---|
| mu_r | +1.00 / +1.00 | +0.14 / +0.14 | +0.05 / +0.01 | -0.24 / -0.23 | +0.04 / +0.05 | +0.26 / +0.25 |
| sigma_r | +0.14 / +0.14 | +1.00 / +1.00 | +0.11 / +0.06 | -0.16 / -0.15 | -0.16 / -0.16 | +0.15 / +0.15 |
| mu_pc | +0.05 / +0.01 | +0.11 / +0.06 | +1.00 / +1.00 | -0.27 / -0.26 | -0.35 / -0.36 | +0.18 / +0.18 |
| sigma_pc | -0.24 / -0.23 | -0.16 / -0.15 | -0.27 / -0.26 | +1.00 / +1.00 | +0.13 / +0.13 | -0.98 / -0.97 |
| prob_photo_bleach | +0.04 / +0.05 | -0.16 / -0.16 | -0.35 / -0.36 | +0.13 / +0.13 | +1.00 / +1.00 | -0.10 / -0.11 |
| lambda_rate | +0.26 / +0.25 | +0.15 / +0.15 | +0.18 / +0.18 | -0.98 / -0.97 | -0.10 / -0.11 | +1.00 / +1.00 |
(Pearson / Spearman)

Figures: `figures/joint_cell_11_chunk_0.png`, `figures/joint_cell_11_chunk_4.png`, `figures/joint_cell_11_chunk_9.png`, `figures/marginals_cell_11.png`, `figures/correlations_cell_11.png`.

## Files

- `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_Window_Posterior_Draws.npz`: posterior_samples_cloud (windows x draws x parameters, float32, log10), kind_index/cell/chunk, fresh_quantiles, fresh_sgm, fresh_sgm_index, draws_outside_prior, source_map_estimate / source_posterior_quantiles / source_posterior_sgm / source_scores / source_row_index (joined references), prior box, sgm_scale, collection_manifest_json.
- `window_summaries.csv`, `correlations.csv`, `joint_depth.csv`, `figures/`, `provenance.json`.

## Reading rules

- Every row of the cloud is one complete joint draw; columns are never resampled or rank-matched independently.
- Outside-prior draws are retained and counted (unrestricted flow draws, as in the source).
- Correlations are reported in estimator coordinates (log10) with Spearman beside Pearson; the physical-coordinate Pearson is in `correlations.csv`.
- Fresh and source summaries come from different Monte Carlo samples; their differences (dex) measure sampling reproducibility, not a change of the posterior.

## Erratum (2026-11-03, after review)

The explanatory sentence under "joint depth of the reference vectors" was reworded from the saved text: the share of draws farther than the reference is a distance rank relative to the window SGM, not a statement about posterior density ("dense core"). Numbers, draws, figures and provenance are unchanged; nothing was resampled.
