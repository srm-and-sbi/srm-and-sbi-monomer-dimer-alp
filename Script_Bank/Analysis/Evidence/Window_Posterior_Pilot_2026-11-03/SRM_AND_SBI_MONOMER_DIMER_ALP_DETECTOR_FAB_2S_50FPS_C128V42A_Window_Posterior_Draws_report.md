# Window-posterior draws (FAB, 2S_50FPS_C128V42A)

Fresh joint draws for cells [26], ten windows each, 10000 complete draws per window (flow-draws, pool mode unrestricted as in the source), with the adopted estimator (weights sha256 01a2d0c17b99105132be2aad12fcf3e0085f6451577958608886bbd81097a6d7) on cuda:0 (GPU), seed 20261103. Source references joined from `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_MAP_Experiment.npz` (sha256 addc6fc1395aed08b83bec1bbfde93a1a6c84843003ec6ae3541ed2d019d9737). Reference SEL: `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_SEL_Nuisance_DLI.npz`. Stored coordinates are the estimator's (log10); physical values are ten to the power of the stored values. The fresh summaries are computed from the retained cloud; the source MAP is a reference from the source run, not a result of this cloud. Package 0.1.44; torch 2.7.1+cu118.

## Checks

| check | result | detail |
|---|---|---|
| checkpoint_matches_source | PASS | loaded weights sha256 01a2d0c17b99105132be2aad12fcf3e0085f6451577958608886bbd81097a6d7; source product recorded 01a2d0c17b99105132be2aad12fcf3e0085f6451577958608886bbd81097a6d7 |
| cloud_shape | PASS | (10, 10000, 6) float32; expected (10, 10000, 6) float32 |
| cloud_finite | PASS | every draw finite |
| window_identities | PASS | 10 windows; each cell carries chunks 0..9 once; cells [26] |
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

- seconds per window: 4.6, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3 (total sampling 25 s; wall 47 s)
- peak GPU memory 914 MiB
- `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Draws.npz`: 2.1 MiB compressed (2.3 MiB of float32 draws)

## Cell 26: fresh summaries per window (physical units) and the source references

**mu_r** (pixels (sqrt(2) times Gaussian sigma); SEL 1.734; prior [1.000, 1.995])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.743 | 1.752 | [1.641, 1.845] | 0.2038 | 1.746 | 1.742 | +0.000 | yes | 0.1 % |
| 1 | 1.673 | 1.668 | [1.587, 1.757] | 0.1704 | 1.679 | 1.674 | -0.000 | yes | 0.0 % |
| 2 | 1.653 | 1.641 | [1.569, 1.737] | 0.1676 | 1.657 | 1.652 | +0.000 | yes | 0.0 % |
| 3 | 1.671 | 1.684 | [1.588, 1.746] | 0.1586 | 1.675 | 1.671 | +0.000 | yes | 0.0 % |
| 4 | 1.656 | 1.642 | [1.569, 1.740] | 0.1711 | 1.660 | 1.655 | +0.000 | yes | 0.0 % |
| 5 | 1.673 | 1.676 | [1.587, 1.755] | 0.1671 | 1.677 | 1.673 | +0.000 | yes | 0.0 % |
| 6 | 1.660 | 1.663 | [1.580, 1.735] | 0.1551 | 1.664 | 1.660 | -0.000 | yes | 0.0 % |
| 7 | 1.644 | 1.641 | [1.561, 1.720] | 0.1597 | 1.649 | 1.645 | -0.000 | no | 0.0 % |
| 8 | 1.652 | 1.632 | [1.560, 1.735] | 0.1748 | 1.656 | 1.652 | -0.000 | yes | 0.0 % |
| 9 | 1.663 | 1.671 | [1.576, 1.740] | 0.1636 | 1.666 | 1.662 | +0.000 | yes | 0.0 % |

**sigma_r** (dimensionless ln-spread; SEL 0.1770; prior [0.1000, 0.5623])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.2085 | 0.2025 | [0.1604, 0.2722] | 0.1118 | 0.2068 | 0.2083 | +0.000 | yes | 0.0 % |
| 1 | 0.1600 | 0.1598 | [0.1213, 0.2097] | 0.0884 | 0.1622 | 0.1605 | -0.001 | yes | 0.7 % |
| 2 | 0.1486 | 0.1439 | [0.1119, 0.1966] | 0.0848 | 0.1517 | 0.1492 | -0.002 | yes | 1.6 % |
| 3 | 0.1425 | 0.1434 | [0.1073, 0.1858] | 0.0785 | 0.1454 | 0.1428 | -0.001 | yes | 2.1 % |
| 4 | 0.1545 | 0.1552 | [0.1166, 0.2035] | 0.0869 | 0.1568 | 0.1544 | +0.000 | yes | 1.0 % |
| 5 | 0.1492 | 0.1503 | [0.1114, 0.1971] | 0.0857 | 0.1521 | 0.1500 | -0.002 | yes | 1.6 % |
| 6 | 0.1347 | 0.1314 | [0.1024, 0.1757] | 0.0733 | 0.1382 | 0.1356 | -0.003 | no | 3.8 % |
| 7 | 0.1407 | 0.1383 | [0.1066, 0.1849] | 0.0783 | 0.1441 | 0.1409 | -0.001 | yes | 2.7 % |
| 8 | 0.1610 | 0.1570 | [0.1206, 0.2130] | 0.0925 | 0.1633 | 0.1614 | -0.001 | yes | 0.7 % |
| 9 | 0.1382 | 0.1396 | [0.1040, 0.1825] | 0.0785 | 0.1413 | 0.1381 | +0.000 | yes | 3.2 % |

**mu_pc** (photons per active dye per frame; SEL 105.00; prior [100.00, 562.34])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 110.03 | 110.82 | [94.40, 128.78] | 34.38 | 111.59 | 109.87 | +0.001 | yes | 15.2 % |
| 1 | 107.24 | 106.65 | [94.59, 122.59] | 27.99 | 107.38 | 106.98 | +0.001 | yes | 18.1 % |
| 2 | 104.13 | 105.25 | [93.66, 117.07] | 23.41 | 104.16 | 104.38 | -0.001 | yes | 26.1 % |
| 3 | 106.07 | 103.43 | [94.32, 120.45] | 26.13 | 106.49 | 106.14 | -0.000 | yes | 19.7 % |
| 4 | 105.31 | 107.58 | [94.33, 118.63] | 24.30 | 105.36 | 105.16 | +0.001 | yes | 21.8 % |
| 5 | 104.65 | 101.33 | [93.87, 117.66] | 23.79 | 104.53 | 104.59 | +0.000 | yes | 23.8 % |
| 6 | 103.90 | 105.44 | [93.44, 117.23] | 23.79 | 104.02 | 104.07 | -0.001 | yes | 27.6 % |
| 7 | 103.35 | 105.11 | [93.26, 115.93] | 22.67 | 103.22 | 103.28 | +0.000 | yes | 29.5 % |
| 8 | 104.07 | 104.44 | [93.41, 117.10] | 23.69 | 104.08 | 104.22 | -0.001 | yes | 26.8 % |
| 9 | 102.95 | 102.13 | [92.78, 115.22] | 22.44 | 102.66 | 102.76 | +0.001 | yes | 31.5 % |

**sigma_pc** (dimensionless ln-spread; SEL 0.5848; prior [0.1778, 1.000])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6327 | 0.6277 | [0.5546, 0.7817] | 0.2271 | 0.6196 | 0.6350 | -0.002 | yes | 0.0 % |
| 1 | 0.5974 | 0.5979 | [0.5359, 0.7131] | 0.1772 | 0.5905 | 0.5971 | +0.000 | yes | 0.0 % |
| 2 | 0.5808 | 0.5973 | [0.5303, 0.6687] | 0.1384 | 0.5784 | 0.5805 | +0.000 | yes | 0.0 % |
| 3 | 0.5840 | 0.5824 | [0.5269, 0.6873] | 0.1604 | 0.5784 | 0.5839 | +0.000 | yes | 0.0 % |
| 4 | 0.5674 | 0.5587 | [0.5132, 0.6649] | 0.1518 | 0.5640 | 0.5676 | -0.000 | yes | 0.0 % |
| 5 | 0.5513 | 0.5518 | [0.4987, 0.6436] | 0.1450 | 0.5491 | 0.5510 | +0.000 | yes | 0.0 % |
| 6 | 0.5851 | 0.5911 | [0.5327, 0.6689] | 0.1363 | 0.5828 | 0.5847 | +0.000 | yes | 0.0 % |
| 7 | 0.5653 | 0.5638 | [0.5146, 0.6464] | 0.1318 | 0.5640 | 0.5661 | -0.001 | yes | 0.0 % |
| 8 | 0.5850 | 0.5877 | [0.5334, 0.6710] | 0.1376 | 0.5835 | 0.5855 | -0.000 | yes | 0.0 % |
| 9 | 0.5729 | 0.5880 | [0.5233, 0.6521] | 0.1287 | 0.5725 | 0.5732 | -0.000 | yes | 0.0 % |

**prob_photo_bleach** (probability per 100 frames; SEL 0.0500; prior [0.0100, 0.3162])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0736 | 0.0667 | [0.0127, 0.2884] | 0.2757 | 0.1300 | 0.0755 | -0.011 | yes | 4.1 % |
| 1 | 0.0823 | 0.0733 | [0.0135, 0.3040] | 0.2905 | 0.1326 | 0.0804 | +0.010 | yes | 5.2 % |
| 2 | 0.0963 | 0.0907 | [0.0145, 0.3230] | 0.3086 | 0.1510 | 0.0957 | +0.002 | yes | 6.4 % |
| 3 | 0.0841 | 0.0841 | [0.0134, 0.3096] | 0.2962 | 0.1378 | 0.0843 | -0.001 | yes | 5.5 % |
| 4 | 0.0902 | 0.0852 | [0.0144, 0.3144] | 0.2999 | 0.1450 | 0.0918 | -0.007 | yes | 5.5 % |
| 5 | 0.0940 | 0.0967 | [0.0146, 0.3225] | 0.3079 | 0.1488 | 0.0949 | -0.004 | yes | 6.3 % |
| 6 | 0.0973 | 0.0947 | [0.0150, 0.3265] | 0.3115 | 0.1501 | 0.0960 | +0.006 | yes | 6.6 % |
| 7 | 0.1020 | 0.0992 | [0.0155, 0.3306] | 0.3150 | 0.1538 | 0.1001 | +0.008 | yes | 7.0 % |
| 8 | 0.0998 | 0.0952 | [0.0151, 0.3245] | 0.3094 | 0.1534 | 0.0991 | +0.003 | yes | 6.3 % |
| 9 | 0.1063 | 0.0935 | [0.0161, 0.3397] | 0.3235 | 0.1605 | 0.1071 | -0.003 | yes | 7.6 % |

**lambda_rate** (per second; SEL 8.000; prior [1.000, 10.00])

| chunk | fresh median | fresh SGM | fresh 90 % | fresh width90 | source MAP | source median | median fresh-source (dex) | SEL inside fresh 90 % | draws outside prior |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 6.970 | 6.997 | [3.704, 10.24] | 6.532 | 7.446 | 6.912 | +0.004 | yes | 6.8 % |
| 1 | 7.647 | 7.477 | [4.407, 10.69] | 6.282 | 8.015 | 7.672 | -0.001 | yes | 10.8 % |
| 2 | 8.323 | 8.068 | [5.218, 11.16] | 5.941 | 8.521 | 8.366 | -0.002 | yes | 17.5 % |
| 3 | 7.855 | 7.635 | [4.661, 10.93] | 6.268 | 8.175 | 7.895 | -0.002 | yes | 13.9 % |
| 4 | 8.056 | 7.755 | [4.782, 11.14] | 6.358 | 8.274 | 8.050 | +0.000 | yes | 15.8 % |
| 5 | 8.124 | 7.932 | [4.834, 11.17] | 6.331 | 8.299 | 8.148 | -0.001 | yes | 17.0 % |
| 6 | 8.360 | 8.041 | [5.336, 11.28] | 5.945 | 8.540 | 8.332 | +0.001 | yes | 18.1 % |
| 7 | 8.417 | 7.965 | [5.331, 11.36] | 6.027 | 8.584 | 8.437 | -0.001 | yes | 19.7 % |
| 8 | 8.378 | 8.311 | [5.313, 11.31] | 5.997 | 8.544 | 8.369 | +0.000 | yes | 18.6 % |
| 9 | 8.559 | 8.224 | [5.518, 11.38] | 5.858 | 8.650 | 8.546 | +0.001 | yes | 20.5 % |

### Cell 26: joint depth of the reference vectors

| chunk | reference | prior-scaled distance to the fresh SGM | share of draws farther | inside every marginal 90 % |
|---|---|---|---|---|
| 0 | SEL | 0.139 | 0.922 | yes |
| 0 | source MAP | 0.195 | 0.808 | yes |
| 0 | source SGM | 0.044 | 0.999 | yes |
| 0 | fresh median vector | 0.034 | 1.000 | yes |
| 1 | SEL | 0.142 | 0.896 | yes |
| 1 | source MAP | 0.175 | 0.821 | yes |
| 1 | source SGM | 0.032 | 1.000 | yes |
| 1 | fresh median vector | 0.035 | 1.000 | yes |
| 2 | SEL | 0.225 | 0.663 | yes |
| 2 | source MAP | 0.154 | 0.857 | yes |
| 2 | source SGM | 0.030 | 1.000 | yes |
| 2 | fresh median vector | 0.035 | 1.000 | yes |
| 3 | SEL | 0.200 | 0.756 | yes |
| 3 | source MAP | 0.147 | 0.883 | yes |
| 3 | source SGM | 0.021 | 1.000 | yes |
| 3 | fresh median vector | 0.022 | 1.000 | yes |
| 4 | SEL | 0.192 | 0.770 | yes |
| 4 | source MAP | 0.158 | 0.859 | yes |
| 4 | source SGM | 0.033 | 1.000 | yes |
| 4 | fresh median vector | 0.031 | 1.000 | yes |
| 5 | SEL | 0.222 | 0.679 | yes |
| 5 | source MAP | 0.128 | 0.913 | yes |
| 5 | source SGM | 0.014 | 1.000 | yes |
| 5 | fresh median vector | 0.023 | 1.000 | yes |
| 6 | SEL | 0.260 | 0.553 | no |
| 6 | source MAP | 0.139 | 0.878 | yes |
| 6 | source SGM | 0.046 | 0.998 | yes |
| 6 | fresh median vector | 0.026 | 1.000 | yes |
| 7 | SEL | 0.258 | 0.550 | no |
| 7 | source MAP | 0.134 | 0.894 | yes |
| 7 | source SGM | 0.042 | 0.999 | yes |
| 7 | fresh median vector | 0.029 | 1.000 | yes |
| 8 | SEL | 0.218 | 0.675 | yes |
| 8 | source MAP | 0.142 | 0.875 | yes |
| 8 | source SGM | 0.041 | 0.999 | yes |
| 8 | fresh median vector | 0.027 | 1.000 | yes |
| 9 | SEL | 0.235 | 0.630 | yes |
| 9 | source MAP | 0.159 | 0.834 | yes |
| 9 | source SGM | 0.031 | 1.000 | yes |
| 9 | fresh median vector | 0.045 | 0.998 | yes |

The share of draws farther from the window's SGM than the reference is a distance rank relative to the SGM in prior-scaled estimator coordinates: near 1 means most draws lie farther from the SGM than the reference does; near 0 means almost every draw is closer to the SGM than the reference is. It is a distance rank, not a posterior-density or joint-credible-region statement.

### Cell 26: within-window correlations (Pearson, estimator coordinates), chunks (0, 4, 9)

chunk 0:

|  | mu_r | sigma_r | mu_pc | sigma_pc | prob_photo_bleach | lambda_rate |
|---|---|---|---|---|---|---|
| mu_r | +1.00 / +1.00 | -0.07 / -0.04 | -0.07 / -0.07 | +0.05 / +0.07 | +0.02 / +0.02 | -0.03 / -0.05 |
| sigma_r | -0.07 / -0.04 | +1.00 / +1.00 | -0.04 / -0.02 | +0.06 / +0.06 | +0.03 / +0.05 | -0.07 / -0.07 |
| mu_pc | -0.07 / -0.07 | -0.04 / -0.02 | +1.00 / +1.00 | -0.58 / -0.59 | +0.11 / +0.12 | +0.38 / +0.39 |
| sigma_pc | +0.05 / +0.07 | +0.06 / +0.06 | -0.58 / -0.59 | +1.00 / +1.00 | +0.02 / +0.01 | -0.94 / -0.92 |
| prob_photo_bleach | +0.02 / +0.02 | +0.03 / +0.05 | +0.11 / +0.12 | +0.02 / +0.01 | +1.00 / +1.00 | -0.08 / -0.08 |
| lambda_rate | -0.03 / -0.05 | -0.07 / -0.07 | +0.38 / +0.39 | -0.94 / -0.92 | -0.08 / -0.08 | +1.00 / +1.00 |
(Pearson / Spearman)

chunk 4:

|  | mu_r | sigma_r | mu_pc | sigma_pc | prob_photo_bleach | lambda_rate |
|---|---|---|---|---|---|---|
| mu_r | +1.00 / +1.00 | -0.11 / -0.09 | -0.09 / -0.11 | +0.06 / +0.09 | -0.02 / -0.02 | -0.02 / -0.03 |
| sigma_r | -0.11 / -0.09 | +1.00 / +1.00 | -0.06 / -0.03 | +0.03 / +0.03 | +0.07 / +0.08 | -0.04 / -0.04 |
| mu_pc | -0.09 / -0.11 | -0.06 / -0.03 | +1.00 / +1.00 | -0.32 / -0.33 | +0.07 / +0.09 | +0.10 / +0.11 |
| sigma_pc | +0.06 / +0.09 | +0.03 / +0.03 | -0.32 / -0.33 | +1.00 / +1.00 | -0.00 / -0.00 | -0.89 / -0.86 |
| prob_photo_bleach | -0.02 / -0.02 | +0.07 / +0.08 | +0.07 / +0.09 | -0.00 / -0.00 | +1.00 / +1.00 | -0.07 / -0.08 |
| lambda_rate | -0.02 / -0.03 | -0.04 / -0.04 | +0.10 / +0.11 | -0.89 / -0.86 | -0.07 / -0.08 | +1.00 / +1.00 |
(Pearson / Spearman)

chunk 9:

|  | mu_r | sigma_r | mu_pc | sigma_pc | prob_photo_bleach | lambda_rate |
|---|---|---|---|---|---|---|
| mu_r | +1.00 / +1.00 | -0.13 / -0.11 | -0.11 / -0.12 | +0.05 / +0.06 | -0.04 / -0.04 | -0.00 / -0.01 |
| sigma_r | -0.13 / -0.11 | +1.00 / +1.00 | -0.11 / -0.08 | +0.03 / +0.03 | +0.11 / +0.11 | -0.02 / -0.03 |
| mu_pc | -0.11 / -0.12 | -0.11 / -0.08 | +1.00 / +1.00 | -0.23 / -0.26 | -0.01 / +0.02 | +0.02 / +0.04 |
| sigma_pc | +0.05 / +0.06 | +0.03 / +0.03 | -0.23 / -0.26 | +1.00 / +1.00 | -0.06 / -0.06 | -0.83 / -0.80 |
| prob_photo_bleach | -0.04 / -0.04 | +0.11 / +0.11 | -0.01 / +0.02 | -0.06 / -0.06 | +1.00 / +1.00 | -0.01 / -0.01 |
| lambda_rate | -0.00 / -0.01 | -0.02 / -0.03 | +0.02 / +0.04 | -0.83 / -0.80 | -0.01 / -0.01 | +1.00 / +1.00 |
(Pearson / Spearman)

Figures: `figures/joint_cell_26_chunk_0.png`, `figures/joint_cell_26_chunk_4.png`, `figures/joint_cell_26_chunk_9.png`, `figures/marginals_cell_26.png`, `figures/correlations_cell_26.png`.

## Files

- `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Draws.npz`: posterior_samples_cloud (windows x draws x parameters, float32, log10), kind_index/cell/chunk, fresh_quantiles, fresh_sgm, fresh_sgm_index, draws_outside_prior, source_map_estimate / source_posterior_quantiles / source_posterior_sgm / source_scores / source_row_index (joined references), prior box, sgm_scale, collection_manifest_json.
- `window_summaries.csv`, `correlations.csv`, `joint_depth.csv`, `figures/`, `provenance.json`.

## Reading rules

- Every row of the cloud is one complete joint draw; columns are never resampled or rank-matched independently.
- Outside-prior draws are retained and counted (unrestricted flow draws, as in the source).
- Correlations are reported in estimator coordinates (log10) with Spearman beside Pearson; the physical-coordinate Pearson is in `correlations.csv`.
- Fresh and source summaries come from different Monte Carlo samples; their differences (dex) measure sampling reproducibility, not a change of the posterior.

## Erratum (2026-11-03, after review)

The explanatory sentence under "joint depth of the reference vectors" was reworded from the saved text: the share of draws farther than the reference is a distance rank relative to the window SGM, not a statement about posterior density ("dense core"). Numbers, draws, figures and provenance are unchanged; nothing was resampled.
