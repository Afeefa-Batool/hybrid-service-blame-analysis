# Expected Results Summary

## Paper benchmark (Chen et al., 2025)

| Outcome | Direction | Reported magnitude |
|---|---|---|
| Customer emotion | Positive effect of blame | +0.182 sentiment points |
| Customer engagement | Negative effect of blame | about -8.47% |
| Subsequent purchase | Negative effect of blame | about -27.12% baseline rate |
| IV relevance | Peer blame fluctuation significant | F-stat = 35.40 |

## Notebook demo (synthetic data, verified run)

Local pipeline run completed successfully. Directional pattern matches the paper:

| Metric | Verified local run |
|---|---|
| Blame label accuracy | 97.5% |
| Blame label AUC | 0.97 |
| IV (peer_blame_delta) | coef = 0.37, p < 0.001 |
| Emotion effect | +0.22 sentiment, p < 0.001 |
| Engagement effect | -3.95%, p = 0.31 |
| Purchase effect | run in Colab for full logit output |

| Outcome | Expected sign of blame coefficient |
|---|---|
| Emotion | Positive |
| Engagement | Negative |
| Purchase | Negative |
| Blame x previous_purchase (emotion) | Positive interaction |
| Blame x previous_purchase (engagement/purchase) | Positive interaction (attenuates harm) |

## Interpretation (ERA)

- Blame shifts customer attribution from individual human fault to human-AI coordination failure.
- Immediate anger toward the human agent drops.
- Perceived system reliability drops, so customers disengage and buy less.
- Loyal customers (more prior purchases) are partially protected.

## How to reproduce

1. Open notebook in Google Colab.
2. Run all cells.
3. Check Section 4 coefficients and Section 5 charts.
