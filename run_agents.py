"""Phase 2 entry point: agents run EDA + modeling and write the evaluation report.

    python run_agents.py            # full pipeline via orchestrator
    python run_agents.py --eda      # EDA specialist only
    python run_agents.py --eval     # evaluation specialist only
"""
import argparse

from agents.specialists import build_eda_agent, build_evaluation_agent, build_orchestrator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda", action="store_true", help="run only the EDA specialist")
    ap.add_argument("--eval", action="store_true", help="run only the evaluation specialist")
    args = ap.parse_args()

    if args.eda:
        print(build_eda_agent()("Run a full EDA and summarise the findings."))
    elif args.eval:
        print(build_evaluation_agent()("Train, compare and evaluate models."))
    else:
        build_orchestrator()("Create the Model Evaluation Report for the dataset.")
        print("\nDone. See reports/model_evaluation_report_latest.md")


if __name__ == "__main__":
    main()
