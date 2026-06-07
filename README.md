# Hybrid Service Blame Analysis

Replication and extension of Chen et al. (2025), *Agentic AI as A Scapegoat: The Consequences of Blame in Hybrid Service Systems* (SSRN-6032194).

This repository gives you a practical Python pipeline that mirrors the paper's logic:
- detect service failures in chat sessions,
- identify human-agent blame toward AI,
- measure customer emotion and behavioral outcomes,
- estimate effects with a two-stage control-function design.

## Why this project exists

The original study uses proprietary e-commerce chat logs. This repo provides a transparent, runnable version you can open in Google Colab to demonstrate methodology and coding ability for PhD outreach.

## Repository structure

```text
hybrid-service-blame-analysis/
|-- Hybrid_Service_Blame_Analysis.ipynb   # main Colab notebook
|-- PROJECT_DESCRIPTION.md                  # one-page project summary
|-- README.md
|-- requirements.txt
|-- assets/
|   `-- workflow_pipeline.svg             # pipeline flowchart
`-- src/
    |-- data_generator.py
    |-- nlp_pipeline.py
    |-- econometrics.py
    `-- workflow_diagram.py
```

## Quick start (Google Colab)

1. Upload `Hybrid_Service_Blame_Analysis.ipynb` to Google Colab.
2. Run all cells in order.
3. Review:
   - blame detection quality table,
   - first-stage IV relevance,
   - second-stage outcome effects,
   - charts and policy summary.

No API key is required for the default demo.

## Quick start (local)

```bash
cd hybrid-service-blame-analysis
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/workflow_diagram.py
jupyter notebook Hybrid_Service_Blame_Analysis.ipynb
```

## Pipeline overview

1. Load or generate session-level hybrid service chats.
2. Keep only failure sessions.
3. NLP labeling:
   - service failure flag,
   - blame flag,
   - customer sentiment score.
4. Build outcome variables:
   - emotion,
   - log engagement,
   - purchase indicator.
5. First-stage blame model with peer blame instrument.
6. Second-stage outcome models with control-function residual.
7. Report effect sizes and visualize results.

See `assets/workflow_pipeline.png` for the full flow.

## Modern Python stack used

- `pandas`, `numpy` for data handling
- `scikit-learn` for label evaluation
- `statsmodels` for logit/OLS estimation
- `matplotlib`, `seaborn`, `plotly` for visualization
- Optional extension path: `transformers` and `sentence-transformers`

## Mapping to paper variables

| Paper variable | Project variable |
|---|---|
| Emotion_ijt | customer_emotion / sentiment_score |
| Engagement_ijt (log) | log_engagement |
| Purchase_ijt | purchase |
| Blame_ijt | blame_detected |
| Delta PeerBlameRate_t-1 | peer_blame_delta |

## Key expected results (directional)

- Blame increases short-term customer sentiment.
- Blame reduces engagement.
- Blame reduces purchase probability.
- Prior purchase history moderates negative behavioral impact.

Exact magnitudes in this demo come from synthetic data and are for illustration.

## Email-ready one-liner

"I built a reproducible Colab pipeline extending your hybrid-service blame framework with modern Python NLP and econometric modules, including endogeneity correction and result visualization."

## Citation

Chen, J., Yang, C., Qiu, L., and Fu, X. (2025). *Agentic AI as A Scapegoat: The Consequences of Blame in Hybrid Service Systems*. SSRN-6032194.

## License

Educational and outreach use.
