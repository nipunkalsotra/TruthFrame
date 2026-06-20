# 🛡️ TruthFrame Evaluation Report (⏳ REAL-TIME)
**Last Updated:** 2026-06-20 11:06:26
**Progress:** 15 claims processed.

---

## 📊 1. Operational Command Center
| Metric | Value |
| :--- | :--- |
| Total API Calls | 70 |
| Gemini Tokens | 1,629 |
| Groq Tokens | 4,673 |
| Images Processed | 37 |
| Self-Corrections | 0 |
| Total Runtime | 157.64s |
| **Estimated Cost** | **$0.00314** |


## ⚙️ 2. Orchestration Strategy (TPM/RPM)
- **Parallel Execution**: Enabled via `asyncio.as_completed` for maximum throughput.
- **Rate Limiting**: Centralized `GlobalRateLimiter` using Token Bucket algorithm.
- **Concurrency**: Capped at 5 concurrent requests to respect Free Tier limits.
- **Retry Logic**: Exponential backoff with jitter (2^n + random) to handle 429s.
- **Tiered Models**: Primary use of Gemini 1.5 Flash with Flash fallback for high-risk claims.


## 🎯 3. Model Performance (vs Sample Data)
| Field | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Claim Status** | 16.7% | 3.3% | 16.7% | 5.6% |
| **Issue Type** | 16.7% | 3.3% | 16.7% | 5.6% |
| **Object Part** | 33.3% | 33.3% | 33.3% | 33.3% |
| **Severity** | 16.7% | 4.2% | 16.7% | 6.7% |
| **Evidence Standard Met** | 33.3% | 86.7% | 33.3% | 33.3% |

## 🔬 4. Strategy Comparison
| Configuration | Reasoning Depth | Cost Efficiency | Stability (Free Tier) |
| :--- | :--- | :--- | :--- |
| **Gemini 1.5 Pro** | 🏆 Elite | ⚠️ High | ❌ Low (404/429 frequent) |
| **Gemini 1.5 Flash (Multi-Agent)** | ⚡ High | 🏆 Elite | ✅ High (Best RPM/TPM on Free Tier) |
| **Gemini 1.5 Flash** | ✅ Balanced | 🏆 Elite | 🏆 High (Best RPM/TPM) |

**Decision:** We selected **Gemini 1.5 Flash** for the final pipeline to ensure 100% operational stability during the hackathon's high-traffic period while maintaining a multi-agent consensus architecture.

## 🚀 5. Final Strategy Summary
TruthFrame uses a multi-agent consensus pipeline. It first extracts logic and severity via Groq (Llama 3), then performs visual verification via Gemini 1.5 Flash. High-risk or ambiguous claims are escalated to a Critic agent for cross-verification, and all outputs are passed through a self-correction loop to ensure 100% schema compliance.