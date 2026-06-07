# Project Description (One Page)

## Title
Replicating and Extending "Agentic AI as a Scapegoat" with a Modern Python Analytics Pipeline

## Research Context
Chen et al. (2025) study hybrid customer service systems where human agents and agentic AI chatbots serve customers through one interface. When service failures happen, human agents sometimes blame the AI. The paper shows a trade-off: blaming reduces immediate customer negative emotion, but lowers in-session engagement and subsequent purchase probability. The mechanism is explained by External Relational Attribution (ERA): customers interpret blame as a sign of poor human-AI coordination, which weakens trust in the whole service system.

## Objective of This Project
This project demonstrates how to operationalize that research design using reproducible Python tooling. It is built for a PhD outreach conversation: it shows initiative, technical fluency, and respect for the original empirical strategy without claiming access to proprietary firm data.

## What Was Implemented
1. Session-level dataset generation that mirrors the paper variables (failure sessions, blame indicator, customer emotion, log engagement, purchase, controls, and peer blame instrument).
2. NLP labeling layer for failure detection, blame detection, and customer sentiment scoring using lightweight modern libraries (regex + lexicon baseline, extendable to HuggingFace transformers in Colab).
3. Two-stage econometric workflow:
   - Stage 1: logistic model of blame with peer blame fluctuation instrument.
   - Stage 2: outcome models for emotion (OLS), engagement (OLS on log messages), and purchase (logit), with control-function residual.
4. Result visualization and a policy-oriented summary aligned with ERA implications.

## Expected Directional Findings
Consistent with the paper:
- Blame -> more positive short-term customer emotion.
- Blame -> lower engagement.
- Blame -> lower purchase likelihood.
- Stronger prior customer relationship weakens negative behavioral effects.

## Why This Matters for Hybrid AI Service Design
The pipeline supports practical recommendations from the paper:
- Train agents to acknowledge failures without scapegoating AI.
- Monitor coordination signals visible to customers.
- Include human-AI collaboration quality in performance management.

## Deliverables
- Google Colab notebook (`Hybrid_Service_Blame_Analysis.ipynb`)
- README with setup and run instructions
- Workflow diagram (`assets/workflow_pipeline.png`)
- Modular Python source (`src/`)

## Limitations (Transparent)
- Uses synthetic data for demonstration because the original Taobao chat logs are not public.
- NLP labels are proxy implementations; production replication should use audited LLM prompts as in the paper appendices.
- Full fixed-effects structure from the paper is approximated with rich controls in this demo.

## Suggested Next Step with Professor
If field data access is possible, this pipeline can be swapped from synthetic sessions to real chat logs with minimal code changes, then validated against the paper's reported effect sizes.
