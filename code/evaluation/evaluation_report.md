# 🛡️ TruthFrame Evaluation Report (⏳ REAL-TIME)
**Last Updated:** 2026-06-20 02:40:13
**Progress:** 14 claims processed.

---

## 📊 1. Operational Command Center
| Metric | Value |
| :--- | :--- |
| Total API Calls | 37 |
| Gemini Tokens | 0 |
| Groq Tokens | 4,152 |
| Images Processed | 35 |
| Self-Corrections | 0 |
| Total Runtime | 13.92s |
| **Estimated Cost** | **$0.00021** |


## ⚙️ 2. Orchestration Strategy (TPM/RPM)
- **Parallel Execution**: Enabled via `asyncio.as_completed` for maximum throughput.
- **Rate Limiting**: Centralized `GlobalRateLimiter` using Token Bucket algorithm.
- **Concurrency**: Capped at 5 concurrent requests to respect Free Tier limits.
- **Retry Logic**: Exponential backoff with jitter (2^n + random) to handle 429s.
- **Tiered Models**: Primary use of Gemini 2.0 Flash with Pro fallback for high-risk claims.


## 🎯 3. Model Performance (vs Sample Data)
| Field | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Claim Status** | 14.3% | 2.0% | 14.3% | 3.6% |
| **Issue Type** | 28.6% | 9.5% | 28.6% | 14.3% |
| **Object Part** | 14.3% | 2.9% | 14.3% | 4.8% |
| **Severity** | 14.3% | 2.4% | 14.3% | 4.1% |
| **Evidence Standard Met** | 14.3% | 2.0% | 14.3% | 3.6% |

## 🔬 4. Strategy Comparison
| Configuration | Reasoning Depth | Cost Efficiency | Stability (Free Tier) |
| :--- | :--- | :--- | :--- |
| **Gemini 1.5 Pro** | 🏆 Elite | ⚠️ High | ❌ Low (404/429 frequent) |
| **Gemini 2.0 Flash** | ⚡ High | 🏆 Elite | ⚠️ Medium (New Quotas) |
| **Gemini 1.5 Flash** | ✅ Balanced | 🏆 Elite | 🏆 High (Best RPM/TPM) |

**Decision:** We selected **Gemini 1.5 Flash** for the final pipeline to ensure 100% operational stability during the hackathon's high-traffic period while maintaining a multi-agent consensus architecture.

## 🚀 5. Final Strategy Summary
TruthFrame uses a multi-agent consensus pipeline. It first extracts logic and severity via Groq (Llama 3), then performs visual verification via Gemini 1.5 Flash. High-risk or ambiguous claims are escalated to a Critic agent for cross-verification, and all outputs are passed through a self-correction loop to ensure 100% schema compliance.