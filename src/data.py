"""Dataset loading. Defaults to a synthetic auto-loan default dataset."""
import numpy as np
import pandas as pd

from src import config


def make_synthetic_auto_loans(n: int = 5000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Synthetic auto-loan portfolio with a realistic-ish default signal.

    Includes some missing values and a few outliers so EDA has something to find.
    """
    rng = np.random.default_rng(seed)

    credit_score = np.clip(rng.normal(690, 70, n), 450, 850).round()
    income = np.clip(rng.lognormal(10.9, 0.45, n), 18_000, 400_000).round(-2)
    loan_amount = np.clip(rng.normal(27_000, 9_000, n), 5_000, 75_000).round(-2)
    term_months = rng.choice([36, 48, 60, 72, 84], n, p=[0.1, 0.15, 0.35, 0.3, 0.1])
    vehicle_type = rng.choice(["new", "used", "certified"], n, p=[0.45, 0.4, 0.15])
    employment_years = np.clip(rng.gamma(2.5, 3.0, n), 0, 40).round(1)
    region = rng.choice(["South", "West", "Midwest", "Northeast"], n, p=[0.38, 0.27, 0.2, 0.15])
    apr = np.clip(3 + (760 - credit_score) / 28 + rng.normal(0, 0.8, n), 1.9, 24).round(2)

    dti = np.clip(0.12 + 0.25 * loan_amount / income + rng.normal(0, 0.06, n), 0.03, 0.8).round(3)
    ltv = np.clip(rng.normal(1.0, 0.18, n) + (vehicle_type == "used") * 0.05, 0.4, 1.6).round(3)

    logit = (
        -2.2
        - 0.012 * (credit_score - 690)
        + 4.0 * (dti - 0.30)
        + 1.5 * (ltv - 1.0)
        + 0.015 * (term_months - 60)
        - 0.03 * employment_years
        + 0.04 * (apr - 8)
    )
    p_default = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, p_default)

    df = pd.DataFrame(
        {
            "loan_id": np.arange(1, n + 1),
            "credit_score": credit_score,
            "annual_income": income,
            "loan_amount": loan_amount,
            "term_months": term_months,
            "apr": apr,
            "dti": dti,
            "ltv": ltv,
            "employment_years": employment_years,
            "vehicle_type": vehicle_type,
            "region": region,
            "default": default,
        }
    )

    # Inject messiness: missing values and a few income outliers
    for col, frac in [("employment_years", 0.04), ("dti", 0.02), ("credit_score", 0.01)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    df.loc[rng.choice(n, 8, replace=False), "annual_income"] *= 8
    return df


def load_data() -> pd.DataFrame:
    """Load the user CSV from DATA_PATH, or fall back to synthetic data."""
    if config.DATA_PATH:
        df = pd.read_csv(config.DATA_PATH)
    else:
        df = make_synthetic_auto_loans()
    if config.TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{config.TARGET_COLUMN}' not in data. Columns: {list(df.columns)}"
        )
    return df
