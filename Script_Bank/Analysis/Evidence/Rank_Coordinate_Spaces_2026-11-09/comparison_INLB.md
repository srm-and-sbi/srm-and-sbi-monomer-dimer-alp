# INLB: member weights in the two rank spaces

| quantity | all six coordinates | four retained coordinates |
|---|---|---|
| effective rows (baseline 600) | 507.73 | 380.05 |
| zero-probability rows | 4 [(34, 8), (34, 9), (54, 0), (54, 1)] | 24 [(9, 9), (13, 0), (14, 0), (15, 0), (16, 0), (24, 0), (25, 0), (29, 0), (32, 0), (34, 0), (34, 4), (34, 5), (34, 6), (34, 8), (34, 9), (36, 0), (38, 0), (49, 0), (49, 1), (50, 0), (54, 0), (54, 1), (55, 0), (55, 9)] |
| within-cell effective windows min / median / max | 4.29 / 8.94 / 9.98 | 2.76 / 7.33 / 9.62 |
| cells below 8 effective windows | 12 | 41 |
| ranked coordinates disjoint from the overrides | False | True |

Spearman rank correlation of the 600 row probabilities: 0.8282; within-cell share changes: max 0.3520, median 0.0286, 366 rows change by more than 0.02 (equal weighting would give every window 1/10).

| parameter | median (all) | median (four) | 90 % range (all) | 90 % range (four) |
|---|---|---|---|---|
| mu_r | 2.2572 | 2.2544 | 1.9736 – 2.3384 | 1.9632 – 2.3375 |
| sigma_r | 0.40435 | 0.40559 | 0.31962 – 0.44558 | 0.32637 – 0.44616 |
| mu_pc | 143.27 | 143.17 | 66.128 – 381.36 | 67.117 – 381 |
| sigma_pc | 0.85966 | 0.85848 | 0.63145 – 1.0073 | 0.67068 – 1.0057 |
| prob_photo_bleach | 0.05 | 0.05 | 0.05 – 0.05 | 0.05 – 0.05 |
| lambda_rate | 2 | 2 | 2 – 2 | 2 – 2 |

Largest within-cell share changes (four-coordinate minus all-coordinate):

| cell | window | share (all) | share (four) | score (all) | score (four) |
|---|---|---|---|---|---|
| 34 | 4 | 0.3520 | 0.0000 | 0.1825 | 0.0000 |
| 25 | 5 | 0.2246 | 0.5561 | 0.2676 | 0.0228 |
| 15 | 5 | 0.1962 | 0.4530 | 0.7343 | 0.1508 |
| 14 | 8 | 0.2520 | 0.5028 | 0.5653 | 0.0265 |
| 55 | 5 | 0.2058 | 0.4403 | 0.7165 | 0.0630 |
| 34 | 3 | 0.0924 | 0.3103 | 0.0479 | 0.0009 |
| 22 | 3 | 0.1585 | 0.3392 | 0.8728 | 0.2888 |
| 34 | 2 | 0.1686 | 0.3448 | 0.0874 | 0.0010 |
| 49 | 5 | 0.1688 | 0.3425 | 0.8657 | 0.3113 |
| 28 | 8 | 0.1391 | 0.2975 | 0.6601 | 0.1604 |
| 16 | 6 | 0.1634 | 0.3195 | 0.9087 | 0.4439 |
| 25 | 3 | 0.2009 | 0.0488 | 0.2394 | 0.0020 |
| 49 | 6 | 0.1467 | 0.2942 | 0.7524 | 0.2674 |
| 38 | 5 | 0.1394 | 0.2794 | 0.1545 | 0.0140 |
| 7 | 7 | 0.1224 | 0.2467 | 0.9382 | 0.6162 |
