import pandas as pd
from pathlib import Path
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred, metric_name):
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return {
            f"{metric_name}_accuracy": accuracy,
            f"{metric_name}_precision": precision,
            f"{metric_name}_recall": recall,
            f"{metric_name}_f1": f1
        }
    except Exception as e:
        logger.error(f"Error calculating metrics for {metric_name}: {e}")
        return {}

def generate_evaluation_report(dataset_path: Path, operational_metrics: dict):
    report_path = dataset_path.parent / "evaluation" / "evaluation_report.md"
    
    try:
        sample_claims_df = pd.read_csv(dataset_path / "sample_claims.csv")
        output_df = pd.read_csv(dataset_path / "output.csv")
    except FileNotFoundError as e:
        logger.error(f"Required file not found for evaluation: {e}")
        return

    # Ensure both dataframes have the same user_id for merging
    sample_claims_df["user_id"] = sample_claims_df["user_id"].astype(str)
    output_df["user_id"] = output_df["user_id"].astype(str)

    # Merge dataframes on user_id to compare
    merged_df = pd.merge(sample_claims_df, output_df, on="user_id", suffixes=('_true', '_pred'))

    metrics_results = {}
    fields_to_evaluate = [
        "claim_status", "issue_type", "object_part", "severity", "evidence_standard_met", "valid_image"
    ]

    for field in fields_to_evaluate:
        if f"{field}_true" in merged_df.columns and f"{field}_pred" in merged_df.columns:
            y_true = merged_df[f"{field}_true"].astype(str)
            y_pred = merged_df[f"{field}_pred"].astype(str)
            metrics_results.update(calculate_metrics(y_true, y_pred, field))

    report_content = "# TruthFrame Evaluation Report\n\n"
    report_content += "## Operational Command Center (Metrics)\n\n"
    
    # Organize metrics into a table for better readability (Operationally Superior)
    report_content += "| Metric | Value |\n"
    report_content += "| :--- | :--- |\n"
    for key, value in operational_metrics.items():
        report_content += f"| {key.replace('_',' ').title()} | {value} |\n"
    report_content += "\n"

    report_content += "## Model Performance (Accuracy vs Sample)\n\n"
    report_content += "| Field | Accuracy | Precision | Recall | F1-Score |\n"
    report_content += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    # Group metrics by field for the table
    for field in fields_to_evaluate:
        acc = metrics_results.get(f"{field}_accuracy", 0)
        prec = metrics_results.get(f"{field}_precision", 0)
        rec = metrics_results.get(f"{field}_recall", 0)
        f1 = metrics_results.get(f"{field}_f1", 0)
        report_content += f"| {field.replace('_',' ').title()} | {acc:.2%} | {prec:.2%} | {rec:.2%} | {f1:.2%} |\n"
    report_content += "\n"
    
    report_content += "## Cost Efficiency Summary\n"
    total_tokens = operational_metrics.get("gemini_tokens", 0) + operational_metrics.get("groq_tokens", 0)
    report_content += f"- **Total Tokens Consumed**: {total_tokens:,}\n"
    report_content += f"- **Average Tokens per Claim**: {total_tokens / len(output_df) if len(output_df) > 0 else 0:.1f}\n"
    report_content += "\n"

    with open(report_path, "w") as f:
        f.write(report_content)
    logger.info(f"Evaluation report generated at {report_path}")

if __name__ == "__main__":
    # This part will be called from main.py, passing operational metrics
    # For standalone testing, you might mock operational_metrics
    mock_operational_metrics = {
        "gemini_calls": 100,
        "groq_calls": 50,
        "gemini_tokens": 10000,
        "groq_tokens": 5000,
        "images_processed": 200,
        "self_corrections": 5
    }
    # Use absolute path relative to project root (hackerrank-orchestrate-june26)
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_root = project_root / "dataset"
    generate_evaluation_report(dataset_root, mock_operational_metrics)
