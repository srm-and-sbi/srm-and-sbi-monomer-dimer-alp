# The ten Fab recordings on the PC: experimental opening- and closing-window summaries

Read from the REF-era companions' records (`Posit/..._Posterior_Predictive_Video_Spot_Count_REF/spot_count.json`,
`..._Tail_Structure_REF/tail_structure.json`, package 0.1.26, 2026-09-29): the experimental columns only, which do not
depend on the render beside them. Detected spots per frame by the count-match rule (`measure_spot_widths` detection on
the 8-bit conversion, n_sigma 4, SCOPE box center); ADU quantiles of the opening 100 frames (2 s). Labels for the
pilot sets follow these direct summaries: dim = cells 0, 4 (lowest density, lowest tail); typical = 26, 45; dense = 32.

| cell | opening spots/frame | closing spots/frame | closing/opening | opening median ADU | opening p99 | opening p99.9 | per-frame max, median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 44.3 | 29.5 | 0.66 | 1241 | 2059 | 3521 | 9836 |
| 5 | 106.3 | 62.5 | 0.59 | 1278 | 2361 | 5041 | 11272 |
| 4 | 107.5 | 55.4 | 0.52 | 1293 | 2365 | 4202 | 7992 |
| 45 | 120.5 | 63.0 | 0.52 | 1250 | 2516 | 5147 | 11780 |
| 49 | 132.2 | 179.7 | 1.36 | 1407 | 4602 | 8190 | 21010 |
| 43 | 161.7 | 89.6 | 0.55 | 1279 | 2661 | 4903 | 9494 |
| 26 | 188.0 | 116.2 | 0.62 | 1242 | 3247 | 6223 | 12603 |
| 53 | 222.4 | 237.9 | 1.07 | 1397 | 4392 | 7542 | 13752 |
| 16 | 259.4 | 165.9 | 0.64 | 1401 | 3926 | 6655 | 11286 |
| 32 | 301.1 | 316.3 | 1.05 | 1505 | 4854 | 7862 | 13267 |

The inferred per-dye brightness of the adopted Fab estimator (window medians, `..._FAB_2S_50FPS_C128V42A_MAP_Experiment`)
spans only 97 to 113 photons over the sixty recordings, so it does not separate these cells; the direct summaries do.
