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

Alerting is code: `infra/terraform/modules/distill/monitoring.tf` (verified against hashicorp/google 8.7.0; not
applied). Notification channel: email from `alert_email`.

| Alert | Source | Threshold |
|---|---|---|
| Uptime | uptime check on the router `/healthz` (Monitoring service agent ID token) | failing from > 1 region |
| 5xx ratio | `run.googleapis.com/request_count` 5xx / all | > 1% for 10 min |
| Error-budget burn (multi-window) | SLO `availability` (99.5%, 28 d rolling, request-based) | fast: 14.4x over 1 h AND 5 min; slow: 6x over 6 h AND 30 min |
| Latency, all paths | `run.googleapis.com/request_latencies` p95 | > 3 s for 15 min |
| Latency, student path | log-based distribution of `latency_ms` where `source=student` | p95 > 400 ms for 15 min |
| Saturation | instance_count vs max (router, vLLM), per-instance concurrency p95 | >= 80% of max for 15 min / vLLM at ceiling |
| Teacher unavailable | log-based counter `teacher_unavailable` (label `cause`) | > 0 for 5 min |
| Spend cap reached | same counter, `cause=budget_exceeded` | any |
| Load shedding | log-based counter `request_shed` (429 overloaded) | > 30/min for 10 min |
| Escalation rate | log-based counter `classified` by `source` | teacher share > 40% over 1 h |

Smoke test after every deploy (staging, prod candidate, prod 10% canary): `scripts/smoke.py`, see
[RUNBOOK section 3](RUNBOOK.md#3-deploy-and-verify).

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
| Firestore (shared spend cap: ~2 reads + 1-2 writes per $0.05 budget lease) | ~$0 (daily free tier; list price $0.03 / 100K reads, $0.09 / 100K writes, https://cloud.google.com/firestore/pricing) | ~$0 |
| Cloud Monitoring (1 uptime check, 4 log-based metrics, 10 alert policies) | within the free allotments at this volume (https://cloud.google.com/stackdriver/pricing) | same |

The fixed GPU cost dominates: below the break-even volume in the README, calling the teacher API directly is cheaper.
Cost levers: Cloud Run committed-use discounts do not apply to GPUs on this price page; zonal-redundancy-disabled is
already the cheaper SKU; scale-to-zero (`vllm_min_instances = 0`) trades multi-minute cold starts for idle cost.
