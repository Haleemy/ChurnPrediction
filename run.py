"""
run.py — CLI entry point for training and managing the project.

Usage:
  python run.py train       # Train and save the model
  python run.py inspect     # Inspect the dataset
  python run.py test-agent  # Run a quick agent smoke test
  python run.py ui          # Launch Streamlit UI
"""
import sys
import os
import logging
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("run")


def cmd_train():
    print("=" * 60)
    print("TRAINING CHURN PREDICTION MODEL")
    print("=" * 60)
    from app.model.train import train_and_save
    metadata = train_and_save()
    print("\n✅ Training complete!")
    print(f"   Model: {metadata['model_name']}")
    print(f"   Train samples: {metadata['n_train']}")
    print(f"   Test samples: {metadata['n_test']}")
    m = metadata["test_metrics"]
    print("\n   Test Metrics:")
    print(f"   |-- ROC-AUC:  {m['roc_auc']:.4f}")
    print(f"   |-- PR-AUC:   {m['pr_auc']:.4f}")
    print(f"   |-- F1@0.5:   {m['f1_at_0.5']:.4f}")
    print(f"   |-- Recall@optimal: {m.get('recall_at_optimal', 'N/A')}")
    print(f"   +-- Optimal threshold: {m['optimal_threshold']:.3f}")
    print(f"\n   CV Results:")
    for model_name, cv in metadata.get("cv_results", {}).items():
        print(f"   |-- {model_name}: ROC-AUC={cv['roc_auc_mean']:.4f} ± {cv['roc_auc_std']:.4f}")
    print(f"\n   Model saved to: {metadata['model_path']}")


def cmd_inspect():
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    from app.data.loader import load_dataset, get_dataset_info
    df = load_dataset()
    info = get_dataset_info(df)

    print(f"\n  Shape: {info['n_rows']} rows x {info['n_cols']} columns")
    print(f"  Customer ID unique: {info['is_customer_id_unique']}")
    print(f"  Duplicate rows: {info['n_duplicate_rows']}")
    print(f"\n  Target distribution:")
    print(f"  |-- Churned (Yes): {info['churn_yes']} ({info['churn_rate_pct']:.2f}%)")
    print(f"  +-- Not Churned (No): {info['churn_no']} ({100-info['churn_rate_pct']:.2f}%)")
    print(f"\n  Missing values: {info['n_missing_total']} total")
    if info['missing_columns']:
        for col, n in info['missing_columns'].items():
            print(f"    +-- {col}: {n}")
    else:
        print("    +-- None")
    print(f"\n  Numeric columns: {', '.join(info['numeric_columns'])}")
    print(f"  Categorical columns: {', '.join(info['categorical_columns'])}")
    print("\n  Sample rows:")
    print(df.head(3).to_string())


def cmd_test_agent():
    print("=" * 60)
    print("AGENT SMOKE TEST")
    print("=" * 60)
    from app.llm.provider import create_provider
    from app.agent.agent import ChurnAnalystAgent

    llm = create_provider()
    agent = ChurnAnalystAgent(llm)

    test_questions = [
        "What is the overall churn rate?",
        "Which contract type has the highest churn?",
        "What is the churn risk of customer 7590-VHVEG?",
        "Show me the top 5 highest risk customers",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        try:
            result = agent.answer(q)
            print(f"[+] Answer: {result['answer'][:200]}...")
            print(f"    Tools used: {[tr['tool'] for tr in result['tool_results']]}")
            print(f"    Charts generated: {len(result.get('charts', []))}")
            if result.get('warnings'):
                print(f"    [!] Warnings: {result['warnings']}")
        except Exception as e:
            print(f"[-] Error: {e}")


def cmd_ui():
    import subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/ui/streamlit_app.py",
        "--server.port", "8501",
    ], env=env)


if __name__ == "__main__":
    commands = {
        "train": cmd_train,
        "inspect": cmd_inspect,
        "test-agent": cmd_test_agent,
        "ui": cmd_ui,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python run.py [{' | '.join(commands.keys())}]")
        print("\nCommands:")
        print("  train       - Train and save the ML model")
        print("  inspect     - Inspect the dataset")
        print("  test-agent  - Quick smoke test of the agent")
        print("  ui          - Launch Streamlit interface")
        sys.exit(1)

    commands[sys.argv[1]]()
