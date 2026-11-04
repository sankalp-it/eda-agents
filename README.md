 the # EDA → Strands Agents → Model Evaluation Report

A VS Code project in two phases:

1. **Phase 1 – Notebook EDA** (`notebooks/01_eda.ipynb`): manual EDA and baseline modeling.
2. **Phase 2 – Strands agents on Amazon Bedrock** (`run_agents.py`): an orchestrator delegates
   to an *EDA specialist* and an *Evaluation specialist*, then writes a
   **Model Evaluation Report** (`reports/model_evaluation_report_latest.md`).

Both phases call the same code in `src/`, so the agents' numbers match the notebook's.

## Layout

```
src/            config, data loading, EDA helpers, modeling/evaluation (shared)
notebooks/      01_eda.ipynb  (regenerate with scripts/build_notebook.py)
agents/         tools.py (Strands @tool wrappers), specialists.py, models.py (Bedrock)
reports/        generated report + figures/
run_agents.py   Phase 2 entry point
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then edit
```

In VS Code: install the recommended extensions (Python, Jupyter), then
**Python: Select Interpreter → `.venv`**. Open the notebook and pick the `.venv` kernel.

## Data

By default a synthetic auto-loan default dataset (5,000 rows, ~10% default rate, with injected
missing values and outliers) is generated. To use your own data set in `.env`:

```
DATA_PATH=data/my_data.csv
TARGET_COLUMN=my_binary_target
```

The target must be 0/1. Columns ending in `_id` are ignored as features. Segment checks use
`vehicle_type` and `region` if present; edit `agents/tools.py` / the notebook to change them.

## Phase 1 – Notebook

Open `notebooks/01_eda.ipynb` and Run All. Sections: overview, missingness, numeric summary,
categoricals vs. target, outliers, correlations, charts, takeaways, then baseline modeling
(logistic regression, random forest, gradient boosting) with ROC/PR, calibration,
permutation importance and segment performance.

## Phase 2 – Agents

Needs AWS credentials with Bedrock model access in the region/model set in `.env`
(`aws sso login --profile <profile>` if you use SSO).

```bash
python run_agents.py          # orchestrator → EDA + eval specialists → report
python run_agents.py --eda    # EDA specialist only
python run_agents.py --eval   # evaluation specialist only
```

Architecture (agents-as-tools):

```
orchestrator ──► run_eda_specialist ──► EDA tools (overview, quality, features, figures)
             ├─► run_evaluation_specialist ──► train_and_compare_models, evaluate_best_model
             └─► save_report
```

The LLMs never compute metrics themselves; they call deterministic tools and write the
narrative from tool output, which keeps the report reproducible and auditable.

## Notes

- Report sections: executive summary, data quality, EDA findings, model comparison,
  best-model evaluation, risks/limitations, recommendations.
- Metrics include ROC AUC, PR AUC, Gini, KS, Brier, precision/recall/F1 and 5-fold CV AUC.
- Ideas for next steps: add Bedrock Guardrails, swap in SHAP, add a challenger-model tool,
  add a fairness/adverse-impact tool, render the report to PDF.
