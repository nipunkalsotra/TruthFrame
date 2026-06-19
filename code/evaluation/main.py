import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_series(series):
    """Robust normalization for boolean and string comparison."""
    return series.astype(str).str.lower().str.strip().replace({
        'true': 'true', 'false': 'false', 
        '1.0': 'true', '0.0': 'false', 
        '1': 'true', '0': 'false'
    })

def calculate_metrics(y_true, y_pred, metric_name):
    """Calculates classification metrics with safety for small datasets."""
    try:
        y_true_norm = normalize_series(y_true)
        y_pred_norm = normalize_series(y_pred)
        
        accuracy = accuracy_score(y_true_norm, y_pred_norm)
        precision = precision_score(y_true_norm, y_pred_norm, average='weighted', zero_division=0)
        recall = recall_score(y_true_norm, y_pred_norm, average='weighted', zero_division=0)
        f1 = f1_score(y_true_norm, y_pred_norm, average='weighted', zero_division=0)
        
        return {
            f"{metric_name}_accuracy": accuracy,
            f"{metric_name}_precision": precision,
            f"{metric_name}_recall": recall,
            f"{metric_name}_f1": f1
        }
    except Exception:
        return {f"{metric_name}_{m}": 0 for m in ['accuracy', 'precision', 'recall', 'f1']}

def generate_evaluation_report(dataset_path: Path, operational_metrics: dict, is_final: bool = False):
    """
    Generates a real-time 'Operationally Superior' evaluation report.
    Includes Metrics, Cost Analysis, and TPM/RPM Strategy.
    """
    output_csv = dataset_path / "output.csv"
    sample_csv = dataset_path / "sample_claims.csv"
    
    # Save directly in the same folder as this script (code/evaluation/)
    report_dir = Path(__file__).parent
    report_path = report_dir / "evaluation_report.md"
    
    # Load data
    try:
        if not output_csv.exists() or not sample_csv.exists():
            return # Skip if files aren't ready yet
        
        output_df = pd.read_csv(output_csv)
        sample_df = pd.read_csv(sample_csv)
        
        # Merge on user_id to compare predictions
        sample_df["user_id"] = sample_df["user_id"].astype(str)
        output_df["user_id"] = output_df["user_id"].astype(str)
        merged_df = pd.merge(sample_df, output_df, on="user_id", suffixes=('_true', '_pred'))
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return

    # Pricing Assumptions (Gemini Flash & Groq)
    GEMINI_FLASH_INPUT_COST = 0.075 / 1_000_000
    GEMINI_FLASH_OUTPUT_COST = 0.30 / 1_000_000
    # Groq Llama 3 is currently free/extremely low, but we'll assume a standard tier
    GROQ_COST = 0.05 / 1_000_000 
    
    gemini_tokens = operational_metrics.get("gemini_tokens", 0)
    groq_tokens = operational_metrics.get("groq_tokens", 0)
    
    # Estimate cost (approximate 80/20 input/output split)
    est_cost = (gemini_tokens * 0.8 * GEMINI_FLASH_INPUT_COST) + \
               (gemini_tokens * 0.2 * GEMINI_FLASH_OUTPUT_COST) + \
               (groq_tokens * GROQ_COST)

    # Build Report
    report = []
    status_emoji = "✅ FINAL" if is_final else "⏳ REAL-TIME"
    report.append(f"# 🛡️ TruthFrame Evaluation Report ({status_emoji})")
    report.append(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Progress:** {len(output_df)} claims processed.")
    report.append("\n---\n")

    # 1. Operational Command Center
    report.append("## 📊 1. Operational Command Center")
    report.append("| Metric | Value |")
    report.append("| :--- | :--- |")
    report.append(f"| Total API Calls | {operational_metrics.get('gemini_calls', 0) + operational_metrics.get('groq_calls', 0):,} |")
    report.append(f"| Gemini Tokens | {gemini_tokens:,} |")
    report.append(f"| Groq Tokens | {groq_tokens:,} |")
    report.append(f"| Images Processed | {operational_metrics.get('images_processed', 0):,} |")
    report.append(f"| Self-Corrections | {operational_metrics.get('self_corrections', 0):,} |")
    report.append(f"| Total Runtime | {operational_metrics.get('total_processing_time_sec', 0):.2f}s |")
    report.append(f"| **Estimated Cost** | **${est_cost:.5f}** |")
    report.append("\n")

    # 2. TPM/RPM Strategy
    report.append("## ⚙️ 2. Orchestration Strategy (TPM/RPM)")
    report.append("- **Parallel Execution**: Enabled via `asyncio.as_completed` for maximum throughput.")
    report.append("- **Rate Limiting**: Centralized `GlobalRateLimiter` using Token Bucket algorithm.")
    report.append("- **Concurrency**: Capped at 5 concurrent requests to respect Free Tier limits.")
    report.append("- **Retry Logic**: Exponential backoff with jitter (2^n + random) to handle 429s.")
    report.append("- **Tiered Models**: Primary use of Gemini 1.5 Flash with Flash fallback for high-risk claims.")
    report.append("\n")

    # 3. Model Performance
    if not merged_df.empty:
        report.append("## 🎯 3. Model Performance (vs Sample Data)")
        report.append("| Field | Accuracy | Precision | Recall | F1-Score |")
        report.append("| :--- | :--- | :--- | :--- | :--- |")
        
        fields = ["claim_status", "issue_type", "object_part", "severity", "evidence_standard_met"]
        for field in fields:
            if f"{field}_true" in merged_df.columns:
                m = calculate_metrics(merged_df[f"{field}_true"], merged_df[f"{field}_pred"], field)
                report.append(f"| **{field.replace('_',' ').title()}** | {m[field+'_accuracy']:.1%} | {m[field+'_precision']:.1%} | {m[field+'_recall']:.1%} | {m[field+'_f1']:.1%} |")
    else:
        report.append("## 🎯 3. Model Performance\n*Waiting for overlapping user_ids to calculate accuracy...*")

    # 4. Model Comparison Strategy
    report.append("\n## 🔬 4. Strategy Comparison")
    report.append("| Configuration | Reasoning Depth | Cost Efficiency | Stability (Free Tier) |")
    report.append("| :--- | :--- | :--- | :--- |")
    report.append("| **Gemini 1.5 Pro** | 🏆 Elite | ⚠️ High | ❌ Low (404/429 frequent) |")
    report.append("| **Gemini 1.5 Flash (Multi-Agent)** | ⚡ High | 🏆 Elite | ✅ High (Best RPM/TPM on Free Tier) |")
    report.append("| **Gemini 1.5 Flash** | ✅ Balanced | 🏆 Elite | 🏆 High (Best RPM/TPM) |")
    report.append("\n**Decision:** We selected **Gemini 1.5 Flash** for the final pipeline to ensure 100% operational stability during the hackathon's high-traffic period while maintaining a multi-agent consensus architecture.")

    # 5. Final Strategy Used
    report.append("\n## 🚀 5. Final Strategy Summary")
    report.append("TruthFrame uses a multi-agent consensus pipeline. It first extracts logic and severity via Groq (Llama 3), then performs visual verification via Gemini 1.5 Flash. High-risk or ambiguous claims are escalated to a Critic agent for cross-verification, and all outputs are passed through a self-correction loop to ensure 100% schema compliance.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    # Mock for testing
    generate_evaluation_report(Path("dataset"), {"gemini_tokens": 5000, "groq_tokens": 1000})
