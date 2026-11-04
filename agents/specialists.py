"""Specialist agents and the orchestrator (agents-as-tools pattern)."""
from strands import Agent, tool

from agents.models import bedrock_model
from agents.tools import EDA_TOOLS, EVAL_TOOLS, save_report
from src import config

EDA_PROMPT = """You are a senior data scientist performing exploratory data analysis on a
binary-classification dataset (target column: {target}).
Use your tools to inspect the data, check data quality, analyse features and generate figures.
Return a concise, factual findings summary: dataset size, class balance, missing data,
outliers, strongest predictors, and recommended preprocessing. Include the report-relative
figure paths returned by generate_eda_figures. Never invent numbers; only cite tool output."""

EVAL_PROMPT = """You are a model-validation specialist (credit-risk style).
Use your tools to train and compare candidate models, then evaluate the best one.
Return: a metrics table (ROC AUC, PR AUC, Gini, KS, Brier, precision, recall, F1, CV AUC),
which model is best and why, calibration/segment observations, top features by permutation
importance, and the report-relative figure paths. Flag overfitting (CV vs hold-out gap),
calibration problems, and threshold/recall trade-offs. Never invent numbers."""

REPORT_PROMPT = """You are the lead analyst. Produce a Model Evaluation Report.
Steps:
1. Call run_eda_specialist.
2. Call run_evaluation_specialist.
3. Write the full report in markdown with these sections:
   # Model Evaluation Report, ## 1. Executive Summary, ## 2. Data Overview & Quality,
   ## 3. Exploratory Findings, ## 4. Model Comparison (table), ## 5. Best Model Evaluation
   (discrimination, calibration, importance, segments), ## 6. Risks & Limitations,
   ## 7. Recommendations / Next Steps.
   Embed figures as ![caption](figures/<file>.png) using the exact paths provided.
4. Call save_report with the complete markdown, then reply with one short sentence
   confirming where it was saved.
Use only numbers from the specialists' outputs. State clearly that the data may be synthetic
if the specialists indicate it."""


def build_eda_agent() -> Agent:
    return Agent(model=bedrock_model(), tools=EDA_TOOLS,
                 system_prompt=EDA_PROMPT.format(target=config.TARGET_COLUMN),
                 callback_handler=None)


def build_evaluation_agent() -> Agent:
    return Agent(model=bedrock_model(), tools=EVAL_TOOLS,
                 system_prompt=EVAL_PROMPT, callback_handler=None)


@tool
def run_eda_specialist(request: str = "Run a full EDA and summarise findings.") -> str:
    """Delegate exploratory data analysis to the EDA specialist agent.

    Args:
        request: Instruction for the EDA specialist.
    """
    return str(build_eda_agent()(request))


@tool
def run_evaluation_specialist(request: str = "Train, compare and evaluate models.") -> str:
    """Delegate model training and evaluation to the evaluation specialist agent.

    Args:
        request: Instruction for the evaluation specialist.
    """
    return str(build_evaluation_agent()(request))


def build_orchestrator() -> Agent:
    return Agent(
        model=bedrock_model(temperature=0.3),
        tools=[run_eda_specialist, run_evaluation_specialist, save_report],
        system_prompt=REPORT_PROMPT,
    )
