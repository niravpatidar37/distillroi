# Runbook: distillroi on Cloud Run

Status: **deploy-ready; not applied (no cloud account connected).** Every command below is what an operator would
run; none of it has been executed against a real project.

Services per environment (`staging`, `prod`):

| Service | What | Scaling | Notes |
|---|---|---|---|
| `distill-<env>-vllm` | vLLM serving the merged student (NVIDIA L4) | min 1 (prod) / 0 (staging), max 2 | IAM-gated: only the router SA may invoke |
| `distill-<env>-router` | FastAPI router (student first, teacher on low confidence) | 0..4 | callers need `roles/run.invoker` |

Weights: `gs://<project>-distill-<env>-models/student/` mounted read-only at `/models` (Cloud Run GCS volume).
Nothing is downloaded from the internet at request time (`HF_HUB_OFFLINE=1`).

## 0. One-time bootstrap (per project)

```bash
PROJECT=my-proj; REGION=us-central1; REPO=niravpatidar37/distillroi
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com \
  iamcredentials.googleapis.com storage.googleapis.com firestore.googleapis.com monitoring.googleapis.com \
  logging.googleapis.com --project $PROJECT
# Cloud Run L4 GPU quota (request if 0): Console > IAM & Admin > Quotas > "Total Nvidia L4 GPU allocation, per project per region"
gcloud storage buckets create gs://$PROJECT-tfstate --location=$REGION --uniform-bucket-level-access --public-access-prevention

# Workload Identity Federation for GitHub Actions (keyless)
gcloud iam workload-identity-pools create github --location=global --project $PROJECT
gcloud iam workload-identity-pools providers create-oidc github --location=global --workload-identity-pool=github \
  --issuer-uri=https://token.actions.githubusercontent.com \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='$REPO'" --project $PROJECT
gcloud iam service-accounts create gha-deployer --project $PROJECT
for role in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do  # + whoever runs terraform needs datastore/monitoring/logging admin
  gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:gha-deployer@$PROJECT.iam.gserviceaccount.com --role=$role
done
PN=$(gcloud projects describe $PROJECT --format='value(projectNumber)')
gcloud iam service-accounts add-iam-policy-binding gha-deployer@$PROJECT.iam.gserviceaccount.com \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PN/locations/global/workloadIdentityPools/github/attribute.repository/$REPO"

# GitHub repo variables (not secrets) + environments
gh variable set GCP_PROJECT_ID --body $PROJECT
gh variable set GCP_REGION --body $REGION
gh variable set GCP_WIF_PROVIDER --body projects/$PN/locations/global/workloadIdentityPools/github/providers/github
gh variable set GCP_DEPLOY_SA --body gha-deployer@$PROJECT.iam.gserviceaccount.com
# Create environments "staging" and "production" in GitHub; add required reviewers to "production".
```

## 1. Provision (Terraform)

```bash
cd infra/terraform/envs/staging            # or prod
terraform init -backend-config="bucket=$PROJECT-tfstate" -backend-config="prefix=distill/staging"
terraform plan -out tf.plan -var project_id=$PROJECT   # edit terraform.tfvars first: images by digest, threshold, knee
terraform apply tf.plan
```

Set `router_threshold` from `results/router_eval.json` (tuned on validation) and `vllm_concurrency` from the knee in
`results/vllm_sweep.json`. Set `alert_email` (notification channel for every alert policy) and keep
`spend_store = "firestore"` (shared teacher spend cap, section 8). Terraform also creates the uptime check, the SLO,
the burn-rate alerts and the log-based metrics (`infra/terraform/modules/distill/monitoring.tf`).

## 2. Upload the model and the teacher key (out of band, never via Terraform)

```bash
uv run python scripts/eval_student_hf.py <config> --merge            # -> artifacts/merged/<config>
gcloud storage cp -r artifacts/merged/<config>/* gs://$PROJECT-distill-staging-models/student/
printf '%s' "$GROQ_API_KEY" | gcloud secrets versions add distill-staging-groq-api-key --data-file=-
```

## 3. Deploy and verify

Push to `main` (or run the `deploy` workflow manually). The workflow builds the router image, signs it keyless with
cosign (Sigstore, the workflow's GitHub OIDC identity), mirrors the pinned vLLM image into Artifact Registry,
**verifies the signature**, deploys **by digest** to staging, runs `scripts/smoke.py` (staging honours the fault hooks,
so the escalation, teacher-unavailable and kill-switch checks run for real), then waits for approval on the
`production` environment, verifies the signature again, and does candidate tag (smoke) -> 10% canary (smoke) -> 100%,
rolling back automatically if any smoke run fails. Each smoke JSON summary is uploaded as a workflow artifact.

Manual verification (the same script; `SMOKE_*` credentials come from the environment, never argv):

```bash
URL=$(gcloud run services describe distill-prod-router --region $REGION --format='value(status.url)')
SMOKE_ID_TOKEN=$(gcloud auth print-identity-token --audiences=$URL) \
  python3 scripts/smoke.py --base-url "$URL" --skip-rate-limit --json-out /tmp/smoke.json
gcloud run services logs read distill-prod-router --region $REGION --limit 50     # JSON logs, no query text
```

`scripts/smoke.py` checks (PASS/FAIL/SKIP per line, exit 1 on any FAIL, < 60 s): liveness, readiness (student
reachable, teacher configured, spend store reachable), security headers, no stack traces / no `/docs`, auth
rejected without or with a wrong key and accepted with one, 422 on bad input, 422/413 on oversize input, a real
classification (banking77 label + confidence), low confidence -> teacher, invalid student output -> teacher,
teacher-unavailable policy, the runtime kill switch (student-only path, zero teacher calls), and the rate limiter
(429 + Retry-After). Fault-driven checks need `fault_injection=true` on the target (never in prod: they SKIP).

## 4. Rollback

```bash
gcloud run revisions list --service distill-prod-router --region $REGION
gcloud run services update-traffic distill-prod-router --region $REGION --to-revisions <previous-revision>=100
# model rollback: GCS versioning is on; restore the previous object generation, then roll the vLLM service
gcloud storage ls -a gs://$PROJECT-distill-prod-models/student/model.safetensors
gcloud storage cp gs://$PROJECT-distill-prod-models/student/model.safetensors#<generation> gs://$PROJECT-distill-prod-models/student/model.safetensors
gcloud run services update distill-prod-vllm --region $REGION --update-labels=rolled=$(date +%s)   # new revision re-reads weights
```

## 5. Rotate the teacher key

```bash
printf '%s' "$NEW_KEY" | gcloud secrets versions add distill-prod-groq-api-key --data-file=-
gcloud run services update distill-prod-router --region $REGION --update-labels=rotated=$(date +%s)  # pick up "latest"
gcloud secrets versions disable <old-version> --secret distill-prod-groq-api-key
```

## 6. Kill switch (stop all paid teacher calls)

```bash
gcloud run services update distill-prod-router --region $REGION --update-env-vars TEACHER_ENABLED=false
```

This rolls a new revision, so it covers every instance. The router keeps serving student answers; escalations come
back as `source: student_degraded` with `low_confidence: true`, and 503 only when the student output is not a valid
label. Re-enable with `TEACHER_ENABLED=true`. Locally: `TEACHER_ENABLED=false docker compose up -d router`.

Runtime switch (no restart): `POST /admin/kill-switch {"teacher_enabled": false}` with `X-Admin-Key`. It exists only
when `ADMIN_API_KEY` is set (404 otherwise) and is **per instance**, so on Cloud Run (no admin key is configured by
Terraform) use the env var above; the runtime switch is for the local stack and the smoke test.

## 7. Common incidents

| Symptom | Likely cause | Action |
|---|---|---|
| `/readyz` 503, `student: false` | vLLM cold start (model load + CUDA graphs take minutes) or GPU quota | check `distill-*-vllm` logs; keep `min_instance_count = 1` in prod |
| escalation rate jumps | input drift or a bad model upload | compare with `results/router_eval.json`; roll back the model (section 4); collect new teacher labels |
| teacher 429 / `teacher_unavailable` logs | Groq rate limit or spend cap reached | router degrades to student answers; raise quota or cap deliberately |
| p95 latency above SLO | concurrency above the measured knee | lower `vllm_concurrency` so Cloud Run scales out earlier, or raise `vllm_max_instances` |
| `teacher_unavailable` with `cause=budget_exceeded` | shared spend cap reached (section 8) | intended fail-closed behaviour; investigate escalation volume before raising the cap |
| 429 `overloaded, retry later` / `request_shed` logs | more concurrent `/v1/*` requests than `MAX_INFLIGHT` + `MAX_QUEUE` on an instance | expected under overload; raise `router_max_instances` or vLLM capacity (section 9) |
| 429 `rate limit exceeded` | one caller above `RATE_LIMIT_RPS` / `RATE_LIMIT_BURST` | caller backs off per `Retry-After`; raise per-client limits only deliberately |

Audit: enable Data Access audit logs for Cloud Storage on the model bucket (checkov CKV_GCP_62 skip rationale).

## 8. Shared teacher spend cap (Firestore budget leases)

Cloud Run runs several router instances; a per-instance ledger would either split the cap or multiply it. With
`spend_store = "firestore"` (default) each instance leases budget (default $0.05) from one document,
`spend_caps/p5-distill` in the `distill-<env>-spend` database, with a compare-and-swap on the document's
`updateTime`, and spends locally from the lease. The sum of teacher spend over all instances can never exceed
`teacher_spend_cap_usd`. A crashed instance strands at most one unused lease (the cap gets stricter, not looser); a
graceful shutdown returns it. If Firestore is unreachable, new leases are refused (fail closed -> teacher
unavailable path). Fallback: `spend_store = "file"` sets a hard per-instance cap of `teacher_spend_cap_usd /
router_max_instances`.

```bash
# inspect (granted = handed to instances, spent = reported back), micro-USD
curl -s -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://firestore.googleapis.com/v1/projects/$PROJECT/databases/distill-prod-spend/documents/spend_caps/p5-distill"
# new budget period: raise teacher_spend_cap_usd in terraform.tfvars and apply, or delete the document (all
# instances re-lease from zero). Never edit it while you suspect a runaway escalation loop.
```

## 9. Overload, teacher outages, alerts

- **Overload:** each instance admits `MAX_INFLIGHT` (64) concurrent `/v1/*` requests, queues `MAX_QUEUE` more for at
  most `QUEUE_TIMEOUT_S` (2 s), and rejects the rest with **429 + `Retry-After: 1`** (`request_shed` log). Health
  probes are never shed. Per-client token buckets (`RATE_LIMIT_RPS`/`RATE_LIMIT_BURST`) return 429 + `Retry-After`.
- **Teacher down / out of quota / over budget:** a circuit breaker opens after 3 consecutive failures (30 s, or until
  Groq's quota-reset hint for daily-quota errors) so escalations fail fast. `TEACHER_UNAVAILABLE_POLICY=degrade`
  (default) returns the student's answer with `low_confidence: true, teacher_unavailable: true`; `fail` returns
  503 + `Retry-After: 30`. Each event logs `teacher_unavailable` with a `cause`.

| Alert policy (Terraform) | Fires when | First response |
|---|---|---|
| router /healthz uptime check failing | uptime check fails from more than one region | `gcloud run revisions list`; roll back (section 4) |
| router 5xx ratio > 1% | 10 min | check teacher-unavailable and vLLM health; roll back if it follows a deploy |
| error budget fast burn (14.4x 1h + 5m) / slow burn (6x 6h + 30m) | availability SLO (99.5%, 28 d) burning | freeze deploys (error budget policy in docs/SLO.md) |
| router p95 > 3 s (all paths) / student-path p95 > 400 ms | 15 min | vLLM over its knee: lower `vllm_concurrency`, raise `vllm_max_instances` |
| instance saturation | router instances >= 80% of max, per-instance concurrency p95 >= 80%, or vLLM at its ceiling | scale limits / GPU quota |
| teacher unavailable | any `teacher_unavailable` for 5 min (label `cause`) | Groq status, quota, key rotation (section 5) |
| teacher spend cap reached | `cause=budget_exceeded` | section 8; check escalation rate first |
| router shedding load | `request_shed` > 30/min for 10 min | capacity (above) |
| escalation rate above 40% | teacher share of classified requests over 1 h | input drift or bad model upload: compare with `results/router_eval.json`, roll back the model |

## 10. On-call checklist

1. Acknowledge the alert; open the Cloud Run service page and the router logs (`jsonPayload.msg`).
2. Run the smoke script against the affected environment (section 3) and keep its JSON output for the incident.
3. Did it start with a deploy? `gcloud run revisions list` -> roll back traffic (section 4) first, debug second.
4. Teacher-related? Check `teacher_unavailable` `cause`: `quota_exhausted` (wait or raise quota), `budget_exceeded`
   (section 8), `error`/`deadline` (provider outage: the router already degrades; consider policy `fail` if callers
   must know). Spend runaway: kill switch (section 6).
5. Capacity? Saturation/shedding alerts -> raise max instances within the GPU quota.
6. Close with a note: what fired, smoke result, action taken, follow-ups.
