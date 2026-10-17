# SLOs and cost model

Status: **deploy-ready; not applied (no cloud account connected).** The targets below are design targets; the
latency numbers they are based on were measured locally (RTX 3060 Laptop 6 GB, Docker Desktop/WSL2), not on Cloud Run.

## SLIs and SLOs (router service, 28-day window)

| SLI | Measurement | SLO |
|---|---|---|
| Availability | share of `/v1/classify` requests answered non-5xx (Cloud Run request logs) | 99.5% |
| Latency, student path (`source=student`) | Cloud Run request latency | p50 < 150 ms, p95 < 400 ms, p99 < 1 s |
| Latency, escalated path (`source=teacher`) | same, filtered by log field `source` | p95 < 3 s, p99 < 6 s (teacher API bound) |
| Quality | weekly audit: 200 sampled requests labelled by the teacher (or humans); agreement of the routed label | >= 99% of the teacher's agreement on the frozen test set (see README) |
| Escalation rate | share of requests with `source=teacher` | within +/- 5 points of the validation-tuned rate; alert outside |
| Degraded answers | share with `source=student_degraded` | < 1% outside a declared kill-switch window |
| Teacher spend | ledger / billing | below the configured cap; alert at 80% |

Error budget policy: if availability or student-path p95 burns > 50% of the monthly budget in a week, freeze deploys
and lower `vllm_concurrency` (scale out earlier) before adding features.

Alerting (Cloud Monitoring, log-based metrics on the JSON logs): 5xx ratio > 1% for 10 min; p95 > 400 ms for 15 min;
escalation rate outside band for 1 h; any `BudgetExceeded`; vLLM `/health` startup-probe failures.

## Monthly cost estimate (estimate - verify against the pricing pages)

Prices: Cloud Run GPU pricing, https://cloud.google.com/run/pricing (retrieved 2026-10-17, Tier 1, instance-based
billing, L4 without zonal redundancy): GPU $0.0001867/s, vCPU $0.000018/s, memory $0.000002/GiB-s. An L4 instance
needs >= 4 vCPU and 16 GiB, so one always-on instance costs (0.0001867 + 4 x 0.000018 + 16 x 0.000002) x 3600
= **$1.047/hour = ~$764/month** (730 h). Teacher: Groq gpt-oss-120b $0.15 / $0.60 per 1M input/output tokens
(https://console.groq.com/docs/model/openai/gpt-oss-120b).

| Driver | prod | staging |
|---|---|---|
| vLLM L4 instances (min 1 always on in prod; staging scales to zero) | ~$764 / month per instance | pay per use (~$1.05 per instance-hour) |
| Router (1 vCPU / 512 MiB, request-based billing, scales to zero) | a few $ at <= 1M requests/day; mostly within free tier at low volume | ~$0 |
| Teacher escalations | escalation rate x volume x measured cost per teacher call (README break-even section) | ~$0 |
| Artifact Registry (vLLM image ~10 GB + router) | ~$1 / month storage | shared |
| GCS model bucket (~1-3 GB, versioned) | < $0.10 / month | < $0.10 |
| Secret Manager (1 secret, few accesses) | < $0.10 / month | < $0.10 |

The fixed GPU cost dominates: below the break-even volume in the README, calling the teacher API directly is cheaper.
Cost levers: Cloud Run committed-use discounts do not apply to GPUs on this price page; zonal-redundancy-disabled is
already the cheaper SKU; scale-to-zero (`vllm_min_instances = 0`) trades multi-minute cold starts for idle cost.
