# 🛡️ TruthFrame Evaluation Report (⏳ REAL-TIME)
**Last Updated:** 2026-06-20 01:43:26
**Progress:** 15 claims processed.

---

## 📊 1. Operational Command Center
| Metric | Value |
| :--- | :--- |
| Total API Calls | 23 |
| Gemini Tokens | 0 |
| Groq Tokens | 1,359 |
| Images Processed | 12 |
| Self-Corrections | 0 |
| Total Runtime | 36.00s |
| **Estimated Cost** | **$0.00007** |


## ⚙️ 2. Orchestration Strategy (TPM/RPM)
- **Parallel Execution**: Enabled via `asyncio.as_completed` for maximum throughput.
- **Rate Limiting**: Centralized `GlobalRateLimiter` using Token Bucket algorithm.
- **Concurrency**: Capped at 5 concurrent requests to respect Free Tier limits.
- **Retry Logic**: Exponential backoff with jitter (2^n + random) to handle 429s.
- **Tiered Models**: Primary use of Gemini 2.0 Flash with Pro fallback for high-risk claims.


## 🎯 3. Model Performance (vs Sample Data)
| Field | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Claim Status** | 16.7% | 2.8% | 16.7% | 4.8% |
| **Issue Type** | 0.0% | 0.0% | 0.0% | 0.0% |
| **Object Part** | 0.0% | 0.0% | 0.0% | 0.0% |
| **Severity** | 16.7% | 3.3% | 16.7% | 5.6% |
| **Evidence Standard Met** | 16.7% | 2.8% | 16.7% | 4.8% |

## 🚀 4. Final Strategy Summary
TruthFrame uses a multi-agent consensus pipeline. It first extracts logic and severity via Groq (Llama 3), then performs visual verification via Gemini 2.0 Flash. High-risk or ambiguous claims are escalated to a Critic agent for cross-verification, and all outputs are passed through a self-correction loop to ensure 100% schema compliance.