"""Central configuration, read from environment / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATA_DIR = ROOT / "data"

DATA_PATH = os.getenv("DATA_PATH", "").strip()
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "default").strip() or "default"

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

RANDOM_STATE = 42

for _d in (REPORTS_DIR, FIGURES_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)
