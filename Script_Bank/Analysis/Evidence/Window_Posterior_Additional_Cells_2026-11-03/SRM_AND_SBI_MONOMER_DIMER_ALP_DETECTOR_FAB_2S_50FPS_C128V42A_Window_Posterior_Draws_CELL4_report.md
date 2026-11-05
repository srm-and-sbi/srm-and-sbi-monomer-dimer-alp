# Window-posterior draws (FAB, 2S_50FPS_C128V42A)

Fresh joint draws for cells [4], ten windows each, 10000 complete draws per window (flow-draws, pool mode unrestricted as in the source), with the adopted estimator (weights sha256 01a2d0c17b99105132be2aad12fcf3e0085f6451577958608886bbd81097a6d7) on cuda:0 (GPU), seed 20261103. Source references joined from `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_MAP_Experiment.npz` (sha256 addc6fc1395aed08b83bec1bbfde93a1a6c84843003ec6ae3541ed2d019d9737). Reference SEL: `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_SEL_Nuisance_DLI.npz`. Stored coordinates are the estimator's (log10); physical values are ten to the power of the stored values. The fresh summaries are computed from the retained cloud; the source MAP is a reference from the source run, not a result of this cloud. Package 0.1.44; torch 2.7.1+cu118. Collection label `CELL4` (output-only; same estimator and source as the unlabeled collection).

## Checks

| check | result | detail |
|---|---|---|
| checkpoint_matches_source | PASS | loaded weights sha256 01a2d0c17b99105132be2aad12fcf3e0085f6451577958608886bbd81097a6d7; source product recorded 01a2d0c17b99105132be2aad12fcf3e0085f6451577958608886bbd81097a6d7 |
| cloud_shape | PASS | (10, 10000, 6) float32; expected (10, 10000, 6) float32 |
| cloud_finite | PASS | every draw finite |
| window_identities | PASS | 10 windows; each cell carries chunks 0..9 once; cells [4] |
| parameter_order_and_transform | PASS | ['mu_r', 'sigma_r', 'mu_pc', 'sigma_pc', 'prob_photo_bleach', 'lambda_rate']; parameterization.to_flow (log10 for log rows, linear rows as-is) |
| quantiles_recompute | PASS | quantiles recomputed from the saved cloud agree with the saved ones (max |diff| 0.0e+00 in log10; the saved cloud is float32, the summary was taken on the same float32 draws) |
| sgm_is_a_cloud_row | PASS | the saved SGM equals the cloud row at its index, and recomputing the medoid with the prior-width scale returns the same row |
| physical_endpoints_reproduce | PASS | physical endpoints = 10 ** stored log10 quantiles; widths are computed inside each window from these |
| source_join_by_identity | PASS | source MAP/quantiles/SGM joined by (kind_index, cell, chunk); row indices recorded |
| checkpoint_hash_recorded | PASS | 01a2d0c17b99105132be2aad12fcf3e0085f6451577958608886bbd81097a6d7 |
| source_product_unchanged | PASS | source npz sha256 addc6fc1395aed08b83bec1bbfde93a1a6c84843003ec6ae3541ed2d019d9737 |
| reference_artifact_unchanged | PASS | SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_SEL_Nuisance_DLI.npz sha256 fe9cb5bae47eb0fc88cc70c195d6b46dd7804e4d006bf973c33028846a9159ad |
| code_unchanged_during_run | PASS | implementation hash c86416260f1f8b85 |

## Runtime and storage

- seconds per window: 3.2, 2.3, 2.2, 2.3, 2.2, 2.2, 2.3, 2.3, 2.2, 2.2 (total sampling 24 s; wall 44 s)
- peak GPU memory 914 MiB
- `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Draws_CELL4.npz`: 2.1 MiB compressed (2.3 MiB of float32 draws)

## Cell 4: fresh summaries per window (physical units) and the source references

**mu_r** (pixels (sqrt(2) times Gaussian sigma); SEL 1.734; prior [1.000, 1.995])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.020 | 2.000 | [1.918, 2.099] | 0.1812 | 2.032 | 2.020 | +0.000 | no | 67.3 % |
| 1 | 1.931 | 1.926 | [1.840, 2.011] | 0.1714 | 1.933 | 1.931 | -0.000 | no | 8.9 % |
| 2 | 1.860 | 1.837 | [1.771, 1.949] | 0.1781 | 1.861 | 1.860 | -0.000 | no | 1.0 % |
| 3 | 1.819 | 1.818 | [1.729, 1.900] | 0.1710 | 1.819 | 1.819 | -0.000 | yes | 0.1 % |
| 4 | 1.810 | 1.801 | [1.727, 1.889] | 0.1623 | 1.811 | 1.809 | +0.000 | yes | 0.1 % |
| 5 | 1.784 | 1.785 | [1.697, 1.867] | 0.1702 | 1.786 | 1.784 | -0.000 | yes | 0.1 % |
| 6 | 1.807 | 1.810 | [1.720, 1.891] | 0.1714 | 1.808 | 1.807 | -0.000 | yes | 0.1 % |
| 7 | 1.823 | 1.825 | [1.730, 1.912] | 0.1817 | 1.825 | 1.824 | -0.000 | yes | 0.3 % |
| 8 | 1.810 | 1.793 | [1.717, 1.898] | 0.1814 | 1.811 | 1.810 | +0.000 | yes | 0.3 % |
| 9 | 1.792 | 1.773 | [1.697, 1.877] | 0.1800 | 1.791 | 1.791 | +0.000 | yes | 0.1 % |

**sigma_r** (dimensionless ln-spread; SEL 0.1770; prior [0.1000, 0.5623])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.3081 | 0.3088 | [0.2496, 0.3833] | 0.1337 | 0.3025 | 0.3077 | +0.001 | no | 0.0 % |
| 1 | 0.2144 | 0.2125 | [0.1657, 0.2772] | 0.1115 | 0.2128 | 0.2150 | -0.001 | yes | 0.0 % |
| 2 | 0.1905 | 0.1909 | [0.1450, 0.2514] | 0.1064 | 0.1910 | 0.1908 | -0.001 | yes | 0.1 % |
| 3 | 0.1614 | 0.1593 | [0.1207, 0.2125] | 0.0918 | 0.1636 | 0.1611 | +0.001 | yes | 0.5 % |
| 4 | 0.1486 | 0.1493 | [0.1123, 0.1942] | 0.0819 | 0.1514 | 0.1487 | -0.000 | yes | 1.5 % |
| 5 | 0.1552 | 0.1519 | [0.1155, 0.2056] | 0.0901 | 0.1578 | 0.1556 | -0.001 | yes | 1.1 % |
| 6 | 0.1554 | 0.1509 | [0.1166, 0.2061] | 0.0896 | 0.1582 | 0.1556 | -0.001 | yes | 1.1 % |
| 7 | 0.1835 | 0.1800 | [0.1380, 0.2434] | 0.1053 | 0.1839 | 0.1840 | -0.001 | yes | 0.1 % |
| 8 | 0.1695 | 0.1711 | [0.1261, 0.2266] | 0.1006 | 0.1715 | 0.1694 | +0.000 | yes | 0.4 % |
| 9 | 0.1511 | 0.1486 | [0.1117, 0.2044] | 0.0926 | 0.1538 | 0.1508 | +0.001 | yes | 1.6 % |

**mu_pc** (photons per active dye per frame; SEL 105.00; prior [100.00, 562.34])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 105.52 | 105.01 | [93.03, 120.19] | 27.16 | 106.53 | 105.29 | +0.001 | yes | 23.8 % |
| 1 | 105.79 | 105.21 | [93.29, 121.21] | 27.92 | 106.36 | 105.51 | +0.001 | yes | 23.6 % |
| 2 | 106.12 | 108.36 | [93.78, 121.33] | 27.55 | 106.77 | 106.23 | -0.000 | yes | 21.0 % |
| 3 | 105.08 | 102.26 | [93.67, 119.18] | 25.51 | 105.69 | 105.14 | -0.000 | yes | 23.5 % |
| 4 | 106.46 | 108.88 | [94.43, 121.30] | 26.88 | 107.05 | 106.34 | +0.000 | yes | 19.3 % |
| 5 | 104.14 | 101.13 | [93.42, 117.21] | 23.79 | 104.10 | 103.98 | +0.001 | yes | 26.1 % |
| 6 | 103.11 | 104.31 | [93.14, 115.66] | 22.53 | 103.32 | 103.24 | -0.001 | yes | 31.2 % |
| 7 | 103.07 | 104.64 | [92.48, 116.12] | 23.65 | 103.29 | 103.09 | -0.000 | yes | 31.5 % |
| 8 | 101.92 | 103.06 | [92.02, 113.96] | 21.94 | 101.90 | 102.21 | -0.001 | yes | 37.6 % |
| 9 | 100.05 | 101.49 | [90.52, 111.26] | 20.74 | 99.72 | 99.90 | +0.001 | yes | 49.7 % |

**sigma_pc** (dimensionless ln-spread; SEL 0.5848; prior [0.1778, 1.000])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.5887 | 0.5906 | [0.5375, 0.6769] | 0.1394 | 0.5856 | 0.5892 | -0.000 | yes | 0.0 % |
| 1 | 0.5984 | 0.6018 | [0.5403, 0.6998] | 0.1595 | 0.5984 | 0.5980 | +0.000 | yes | 0.0 % |
| 2 | 0.5861 | 0.5765 | [0.5268, 0.6959] | 0.1691 | 0.5845 | 0.5860 | +0.000 | yes | 0.0 % |
| 3 | 0.5553 | 0.5538 | [0.4985, 0.6519] | 0.1535 | 0.5535 | 0.5546 | +0.001 | yes | 0.0 % |
| 4 | 0.5534 | 0.5411 | [0.4923, 0.6700] | 0.1777 | 0.5493 | 0.5537 | -0.000 | yes | 0.0 % |
| 5 | 0.5439 | 0.5534 | [0.4915, 0.6328] | 0.1414 | 0.5438 | 0.5438 | +0.000 | yes | 0.0 % |
| 6 | 0.5224 | 0.5306 | [0.4702, 0.6023] | 0.1320 | 0.5234 | 0.5218 | +0.000 | yes | 0.0 % |
| 7 | 0.5602 | 0.5585 | [0.5090, 0.6400] | 0.1310 | 0.5615 | 0.5607 | -0.000 | yes | 0.0 % |
| 8 | 0.5444 | 0.5515 | [0.4953, 0.6177] | 0.1224 | 0.5480 | 0.5448 | -0.000 | yes | 0.0 % |
| 9 | 0.5518 | 0.5598 | [0.5028, 0.6189] | 0.1161 | 0.5551 | 0.5524 | -0.001 | yes | 0.0 % |

**prob_photo_bleach** (probability per 100 frames; SEL 0.0500; prior [0.0100, 0.3162])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0947 | 0.0949 | [0.0152, 0.2964] | 0.2812 | 0.1458 | 0.0956 | -0.004 | yes | 3.8 % |
| 1 | 0.0919 | 0.0873 | [0.0145, 0.3041] | 0.2896 | 0.1432 | 0.0894 | +0.012 | yes | 4.8 % |
| 2 | 0.0856 | 0.0873 | [0.0135, 0.3013] | 0.2878 | 0.1393 | 0.0840 | +0.008 | yes | 4.9 % |
| 3 | 0.0885 | 0.0794 | [0.0140, 0.3130] | 0.2990 | 0.1442 | 0.0901 | -0.008 | yes | 5.7 % |
| 4 | 0.0788 | 0.0767 | [0.0134, 0.3012] | 0.2878 | 0.1323 | 0.0803 | -0.008 | yes | 4.7 % |
| 5 | 0.0968 | 0.0937 | [0.0150, 0.3250] | 0.3099 | 0.1528 | 0.0960 | +0.003 | yes | 6.5 % |
| 6 | 0.1025 | 0.0920 | [0.0155, 0.3320] | 0.3165 | 0.1582 | 0.1036 | -0.005 | yes | 7.0 % |
| 7 | 0.1074 | 0.1053 | [0.0164, 0.3282] | 0.3117 | 0.1615 | 0.1073 | +0.001 | yes | 6.7 % |
| 8 | 0.1152 | 0.1093 | [0.0173, 0.3442] | 0.3268 | 0.1700 | 0.1152 | -0.000 | yes | 8.1 % |
| 9 | 0.1339 | 0.1311 | [0.0202, 0.3742] | 0.3540 | 0.1851 | 0.1374 | -0.011 | yes | 11.0 % |

**lambda_rate** (per second; SEL 8.000; prior [1.000, 10.00])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 8.461 | 8.004 | [5.317, 11.09] | 5.771 | 8.566 | 8.439 | +0.001 | yes | 17.4 % |
| 1 | 7.884 | 7.542 | [4.780, 10.69] | 5.914 | 7.894 | 7.891 | -0.000 | yes | 11.4 % |
| 2 | 7.634 | 7.391 | [4.409, 10.59] | 6.183 | 7.725 | 7.632 | +0.000 | yes | 10.7 % |
| 3 | 7.761 | 7.737 | [4.547, 10.90] | 6.351 | 7.849 | 7.761 | -0.000 | yes | 13.1 % |
| 4 | 7.374 | 7.326 | [4.025, 10.70] | 6.674 | 7.615 | 7.429 | -0.003 | yes | 10.2 % |
| 5 | 8.044 | 8.102 | [4.812, 11.08] | 6.270 | 8.110 | 8.063 | -0.001 | yes | 15.8 % |
| 6 | 8.104 | 7.650 | [4.944, 11.28] | 6.338 | 8.081 | 8.134 | -0.002 | yes | 16.6 % |
| 7 | 8.237 | 7.802 | [5.210, 11.14] | 5.930 | 8.222 | 8.208 | +0.002 | yes | 16.7 % |
| 8 | 8.422 | 8.320 | [5.435, 11.41] | 5.978 | 8.323 | 8.394 | +0.001 | yes | 19.3 % |
| 9 | 8.631 | 8.666 | [5.748, 11.49] | 5.744 | 8.566 | 8.575 | +0.003 | yes | 22.2 % |

### Cell 4: joint depth of the reference vectors

| chunk | reference | prior-scaled distance to the fresh SGM | share of draws farther | inside every marginal 90 % |
|---|---|---|---|---|
| 0 | SEL | 0.426 | 0.147 | no |
| 0 | source MAP | 0.131 | 0.872 | yes |
| 0 | source SGM | 0.044 | 0.999 | yes |
| 0 | fresh median vector | 0.028 | 1.000 | yes |
| 1 | SEL | 0.248 | 0.588 | no |
| 1 | source MAP | 0.145 | 0.868 | yes |
| 1 | source SGM | 0.022 | 1.000 | yes |
| 1 | fresh median vector | 0.025 | 1.000 | yes |
| 2 | SEL | 0.191 | 0.780 | no |
| 2 | source MAP | 0.138 | 0.906 | yes |
| 2 | source SGM | 0.043 | 0.999 | yes |
| 2 | fresh median vector | 0.028 | 1.000 | yes |
| 3 | SEL | 0.167 | 0.837 | yes |
| 3 | source MAP | 0.175 | 0.820 | yes |
| 3 | source SGM | 0.045 | 0.999 | yes |
| 3 | fresh median vector | 0.036 | 0.999 | yes |
| 4 | SEL | 0.179 | 0.830 | yes |
| 4 | source MAP | 0.160 | 0.874 | yes |
| 4 | source SGM | 0.024 | 1.000 | yes |
| 4 | fresh median vector | 0.021 | 1.000 | yes |
| 5 | SEL | 0.210 | 0.713 | yes |
| 5 | source MAP | 0.145 | 0.879 | yes |
| 5 | source SGM | 0.032 | 1.000 | yes |
| 5 | fresh median vector | 0.025 | 1.000 | yes |
| 6 | SEL | 0.217 | 0.693 | yes |
| 6 | source MAP | 0.161 | 0.841 | yes |
| 6 | source SGM | 0.041 | 0.999 | yes |
| 6 | fresh median vector | 0.045 | 0.999 | yes |
| 7 | SEL | 0.230 | 0.624 | yes |
| 7 | source MAP | 0.127 | 0.904 | yes |
| 7 | source SGM | 0.038 | 0.999 | yes |
| 7 | fresh median vector | 0.028 | 1.000 | yes |
| 8 | SEL | 0.236 | 0.593 | yes |
| 8 | source MAP | 0.129 | 0.897 | yes |
| 8 | source SGM | 0.034 | 1.000 | yes |
| 8 | fresh median vector | 0.024 | 1.000 | yes |
| 9 | SEL | 0.302 | 0.361 | yes |
| 9 | source MAP | 0.104 | 0.939 | yes |
| 9 | source SGM | 0.028 | 1.000 | yes |
| 9 | fresh median vector | 0.023 | 1.000 | yes |

The share of draws farther from the window's SGM than the reference is a distance rank relative to the SGM in prior-scaled estimator coordinates: near 1 means most draws lie farther from the SGM than the reference does; near 0 means almost every draw is closer to the SGM than the reference is. It is a distance rank, not a posterior-density or joint-credible-region statement.

### Cell 4: within-window correlations (Pearson, estimator coordinates), chunks (0, 4, 9)

chunk 0:

|  | mu_r | sigma_r | mu_pc | sigma_pc | prob_photo_bleach | lambda_rate |
|---|---|---|---|---|---|---|
| mu_r | +1.00 / +1.00 | -0.09 / -0.06 | +0.05 / +0.04 | -0.21 / -0.16 | +0.12 / +0.10 | +0.09 / +0.05 |
| sigma_r | -0.09 / -0.06 | +1.00 / +1.00 | -0.12 / -0.09 | +0.12 / +0.12 | -0.03 / -0.02 | -0.12 / -0.12 |
| mu_pc | +0.05 / +0.04 | -0.12 / -0.09 | +1.00 / +1.00 | -0.45 / -0.45 | +0.08 / +0.11 | +0.20 / +0.19 |
| sigma_pc | -0.21 / -0.16 | +0.12 / +0.12 | -0.45 / -0.45 | +1.00 / +1.00 | -0.08 / -0.09 | -0.87 / -0.82 |
| prob_photo_bleach | +0.12 / +0.10 | -0.03 / -0.02 | +0.08 / +0.11 | -0.08 / -0.09 | +1.00 / +1.00 | +0.02 / +0.02 |
| lambda_rate | +0.09 / +0.05 | -0.12 / -0.12 | +0.20 / +0.19 | -0.87 / -0.82 | +0.02 / +0.02 | +1.00 / +1.00 |
(Pearson / Spearman)

chunk 4:

|  | mu_r | sigma_r | mu_pc | sigma_pc | prob_photo_bleach | lambda_rate |
|---|---|---|---|---|---|---|
| mu_r | +1.00 / +1.00 | -0.11 / -0.09 | -0.08 / -0.10 | +0.09 / +0.11 | +0.02 / +0.02 | -0.06 / -0.08 |
| sigma_r | -0.11 / -0.09 | +1.00 / +1.00 | -0.07 / -0.03 | +0.05 / +0.04 | +0.06 / +0.07 | -0.05 / -0.04 |
| mu_pc | -0.08 / -0.10 | -0.07 / -0.03 | +1.00 / +1.00 | -0.41 / -0.42 | +0.10 / +0.12 | +0.21 / +0.22 |
| sigma_pc | +0.09 / +0.11 | +0.05 / +0.04 | -0.41 / -0.42 | +1.00 / +1.00 | +0.04 / +0.03 | -0.92 / -0.89 |
| prob_photo_bleach | +0.02 / +0.02 | +0.06 / +0.07 | +0.10 / +0.12 | +0.04 / +0.03 | +1.00 / +1.00 | -0.11 / -0.11 |
| lambda_rate | -0.06 / -0.08 | -0.05 / -0.04 | +0.21 / +0.22 | -0.92 / -0.89 | -0.11 / -0.11 | +1.00 / +1.00 |
(Pearson / Spearman)

chunk 9:

|  | mu_r | sigma_r | mu_pc | sigma_pc | prob_photo_bleach | lambda_rate |
|---|---|---|---|---|---|---|
| mu_r | +1.00 / +1.00 | -0.14 / -0.12 | -0.11 / -0.11 | +0.04 / +0.05 | -0.02 / -0.02 | -0.02 / -0.03 |
| sigma_r | -0.14 / -0.12 | +1.00 / +1.00 | -0.13 / -0.11 | +0.01 / +0.01 | +0.11 / +0.13 | -0.00 / -0.01 |
| mu_pc | -0.11 / -0.11 | -0.13 / -0.11 | +1.00 / +1.00 | -0.11 / -0.14 | -0.10 / -0.07 | -0.06 / -0.03 |
| sigma_pc | +0.04 / +0.05 | +0.01 / +0.01 | -0.11 / -0.14 | +1.00 / +1.00 | -0.09 / -0.09 | -0.78 / -0.75 |
| prob_photo_bleach | -0.02 / -0.02 | +0.11 / +0.13 | -0.10 / -0.07 | -0.09 / -0.09 | +1.00 / +1.00 | +0.05 / +0.05 |
| lambda_rate | -0.02 / -0.03 | -0.00 / -0.01 | -0.06 / -0.03 | -0.78 / -0.75 | +0.05 / +0.05 | +1.00 / +1.00 |
(Pearson / Spearman)

Figures: `figures/joint_cell_4_chunk_0.png`, `figures/joint_cell_4_chunk_4.png`, `figures/joint_cell_4_chunk_9.png`, `figures/marginals_cell_4.png`, `figures/correlations_cell_4.png`.

## Files

- `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Draws_CELL4.npz`: posterior_samples_cloud (windows x draws x parameters, float32, log10), kind_index/cell/chunk, fresh_quantiles, fresh_sgm, fresh_sgm_index, draws_outside_prior, source_map_estimate / source_posterior_quantiles / source_posterior_sgm / source_scores / source_row_index (joined references), prior box, sgm_scale, collection_manifest_json.
- `window_summaries.csv`, `correlations.csv`, `joint_depth.csv`, `figures/`, `provenance.json`.

## Reading rules

- Every row of the cloud is one complete joint draw; columns are never resampled or rank-matched independently.
- Outside-prior draws are retained and counted (unrestricted flow draws, as in the source).
- Correlations are reported in estimator coordinates (log10) with Spearman beside Pearson; the physical-coordinate Pearson is in `correlations.csv`.
- Fresh and source summaries come from different Monte Carlo samples; their differences (dex) measure sampling reproducibility, not a change of the posterior.

## Erratum (2026-11-03, after review)

The explanatory sentence under "joint depth of the reference vectors" was reworded from the saved text: the share of draws farther than the reference is a distance rank relative to the window SGM, not a statement about posterior density ("dense core"). Numbers, draws, figures and provenance are unchanged; nothing was resampled.
