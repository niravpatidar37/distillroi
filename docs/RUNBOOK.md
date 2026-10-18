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
  iamcredentials.googleapis.com storage.googleapis.com --project $PROJECT
# Cloud Run L4 GPU quota (request if 0): Console > IAM & Admin > Quotas > "Total Nvidia L4 GPU allocation, per project per region"
gcloud storage buckets create gs://$PROJECT-tfstate --location=$REGION --uniform-bucket-level-access --public-access-prevention

# Workload Identity Federation for GitHub Actions (keyless)
gcloud iam workload-identity-pools create github --location=global --project $PROJECT
gcloud iam workload-identity-pools providers create-oidc github --location=global --workload-identity-pool=github \
  --issuer-uri=https://token.actions.githubusercontent.com \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='$REPO'" --project $PROJECT
gcloud iam service-accounts create gha-deployer --project $PROJECT
for role in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
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
`results/vllm_sweep.json`.

## 2. Upload the model and the teacher key (out of band, never via Terraform)

```bash
uv run python scripts/eval_student_hf.py <config> --merge            # -> artifacts/merged/<config>
gcloud storage cp -r artifacts/merged/<config>/* gs://$PROJECT-distill-staging-models/student/
printf '%s' "$GROQ_API_KEY" | gcloud secrets versions add distill-staging-groq-api-key --data-file=-
```

## 3. Deploy and verify

Push to `main` (or run the `deploy` workflow manually). The workflow builds the router image, mirrors the pinned vLLM
image into Artifact Registry, deploys **by digest** to staging, runs `scripts/smoke.sh`, then waits for approval on
the `production` environment and does candidate tag -> 10% canary -> 100%, rolling back automatically if a smoke test
fails. Manual verification:

```bash
URL=$(gcloud run services describe distill-prod-router --region $REGION --format='value(status.url)')
bash scripts/smoke.sh "$URL" "$(gcloud auth print-identity-token --audiences=$URL)"
gcloud run services logs read distill-prod-router --region $REGION --limit 50     # JSON logs, no query text
```

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

The router keeps serving student answers (`source: student_degraded`) and returns 503 only when the student output is
invalid. Re-enable with `TEACHER_ENABLED=true`. Locally: `TEACHER_ENABLED=false docker compose up -d router`.

## 7. Common incidents

| Symptom | Likely cause | Action |
|---|---|---|
| `/readyz` 503, `student: false` | vLLM cold start (model load + CUDA graphs take minutes) or GPU quota | check `distill-*-vllm` logs; keep `min_instance_count = 1` in prod |
| escalation rate jumps | input drift or a bad model upload | compare with `results/router_eval.json`; roll back the model (section 4); collect new teacher labels |
| teacher 429 / `teacher_unavailable` logs | Groq rate limit or spend cap reached | router degrades to student answers; raise quota or cap deliberately |
| p95 latency above SLO | concurrency above the measured knee | lower `vllm_concurrency` so Cloud Run scales out earlier, or raise `vllm_max_instances` |
| `BudgetExceeded` in logs | per-instance spend cap hit | intended behaviour; investigate escalation volume before raising the cap |

Audit: enable Data Access audit logs for Cloud Storage on the model bucket (checkov CKV_GCP_62 skip rationale).
