"""Regenerates notebooks/01_eda.ipynb. Run: python scripts/build_notebook.py"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md(
        "# Phase 1 — Exploratory Data Analysis\n"
        "Manual EDA and baseline modeling. Phase 2 (`agents/`) automates the same steps with "
        "Strands agents on Bedrock, reusing the `src/` modules called here.\n\n"
        "Select the `.venv` kernel in VS Code (top-right of the notebook)."
    ),
    code(
        "import sys, pathlib\n"
        "ROOT = pathlib.Path.cwd()\n"
        "if ROOT.name == 'notebooks':\n"
        "    ROOT = ROOT.parent\n"
        "sys.path.insert(0, str(ROOT))\n\n"
        "%matplotlib inline\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from IPython.display import Image, display\n"
        "from src import config, data, eda, modeling\n\n"
        "pd.set_option('display.max_columns', 50)\n"
        "TARGET = config.TARGET_COLUMN\n"
        "df = data.load_data()\n"
        "df.head()"
    ),
    md("## 1. Overview"),
    code("eda.overview(df, TARGET)"),
    md("## 2. Missing values"),
    code("eda.missing_report(df)"),
    md("## 3. Numeric summary"),
    code("eda.numeric_summary(df)"),
    md("## 4. Categorical features vs. target"),
    code(
        "import json\n"
        "print(json.dumps(eda.categorical_summary(df, TARGET), indent=2, default=str))"
    ),
    md("## 5. Outliers (IQR rule)"),
    code("eda.outlier_report(df, TARGET)"),
    md("## 6. Correlation with target"),
    code("eda.target_correlations(df, TARGET)"),
    md("## 7. Visuals"),
    code(
        "figs = {\n"
        "    'balance': eda.plot_target_balance(df, TARGET),\n"
        "    'dist': eda.plot_distributions(df, TARGET),\n"
        "    'corr': eda.plot_correlation_heatmap(df),\n"
        "    'cat': eda.plot_target_rate_by_category(df, TARGET),\n"
        "}\n"
        "for p in figs.values():\n"
        "    display(Image(p))"
    ),
    md(
        "## 8. EDA takeaways\n"
        "_Edit these after reviewing the output above._\n\n"
        "- Class imbalance: _?_\n"
        "- Data quality (missingness, outliers): _?_\n"
        "- Strongest predictors: _?_\n"
        "- Planned preprocessing: median impute + scale numerics, one-hot categoricals; "
        "consider capping income outliers."
    ),
    md("## 9. Baseline modeling & evaluation"),
    code(
        "cmp = modeling.compare_models(df, TARGET)\n"
        "pd.DataFrame(cmp['results']).T.drop(columns='confusion')"
    ),
    code(
        "best = cmp['best']\n"
        "X_tr, X_te, y_tr, y_te = cmp['split']\n"
        "plots = modeling.evaluation_plots(cmp['fitted'][best], X_te, y_te, best)\n"
        "print('Best model:', best)\n"
        "for k in ('roc_pr', 'calibration', 'importance'):\n"
        "    display(Image(plots[k]))"
    ),
    code(
        "# Segment fairness / stability check\n"
        "for col in ['vehicle_type', 'region']:\n"
        "    if col in X_te.columns:\n"
        "        display(modeling.segment_performance(cmp['fitted'][best], X_te, y_te, col))"
    ),
]

nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

out = Path(__file__).resolve().parent.parent / "notebooks" / "01_eda.ipynb"
nbf.write(nb, out)
print("wrote", out)
