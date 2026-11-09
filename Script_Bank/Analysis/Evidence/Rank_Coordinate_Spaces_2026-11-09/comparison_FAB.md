# FAB: member weights in the two rank spaces

| quantity | all six coordinates | four retained coordinates |
|---|---|---|
| effective rows (baseline 600) | 572.52 | 483.55 |
| zero-probability rows | 0 [] | 1 [(31, 6)] |
| within-cell effective windows min / median / max | 8.43 / 9.65 / 9.99 | 5.98 / 8.29 / 9.75 |
| cells below 8 effective windows | 0 | 22 |
| ranked coordinates disjoint from the overrides | False | True |

Spearman rank correlation of the 600 row probabilities: 0.8016; within-cell share changes: max 0.1191, median 0.0205, 305 rows change by more than 0.02 (equal weighting would give every window 1/10).

| parameter | median (all) | median (four) | 90 % range (all) | 90 % range (four) |
|---|---|---|---|---|
| mu_r | 1.7308 | 1.7208 | 1.6139 – 2.0014 | 1.6124 – 1.9568 |
| sigma_r | 0.17182 | 0.16771 | 0.13595 – 0.275 | 0.13601 – 0.25011 |
| mu_pc | 104.62 | 104.28 | 98.048 – 115.86 | 98.048 – 113.76 |
| sigma_pc | 0.58138 | 0.57883 | 0.51162 – 0.67833 | 0.51142 – 0.66844 |
| prob_photo_bleach | 0.05 | 0.05 | 0.05 – 0.05 | 0.05 – 0.05 |
| lambda_rate | 8 | 8 | 8 – 8 | 8 – 8 |

Largest within-cell share changes (four-coordinate minus all-coordinate):

| cell | window | share (all) | share (four) | score (all) | score (four) |
|---|---|---|---|---|---|
| 50 | 6 | 0.1187 | 0.2379 | 0.9515 | 0.5435 |
| 19 | 8 | 0.1217 | 0.2372 | 0.9004 | 0.3803 |
| 17 | 8 | 0.1279 | 0.2416 | 0.9553 | 0.6239 |
| 50 | 5 | 0.1184 | 0.2242 | 0.9490 | 0.5122 |
| 30 | 6 | 0.1109 | 0.2067 | 0.9849 | 0.8013 |
| 18 | 7 | 0.1240 | 0.2121 | 0.9869 | 0.8101 |
| 53 | 6 | 0.1226 | 0.2104 | 0.9851 | 0.7098 |
| 16 | 4 | 0.1111 | 0.1977 | 0.8658 | 0.6257 |
| 32 | 6 | 0.1391 | 0.2221 | 0.8981 | 0.3298 |
| 31 | 7 | 0.1212 | 0.2036 | 0.9058 | 0.6173 |
| 2 | 8 | 0.1141 | 0.1960 | 0.9033 | 0.5563 |
| 2 | 5 | 0.1122 | 0.1936 | 0.8878 | 0.5495 |
| 30 | 7 | 0.1090 | 0.1893 | 0.9677 | 0.7339 |
| 32 | 8 | 0.1065 | 0.0296 | 0.6873 | 0.0440 |
| 16 | 2 | 0.1004 | 0.0238 | 0.7826 | 0.0752 |
