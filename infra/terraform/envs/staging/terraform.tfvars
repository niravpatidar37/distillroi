# Non-secret settings only. Secrets live in Secret Manager; images are pinned by digest by the CD workflow.
project_id            = "REPLACE-with-staging-project-id"
region                = "us-central1"
router_image          = "us-central1-docker.pkg.dev/REPLACE/distill-staging/router@sha256:0000000000000000000000000000000000000000000000000000000000000000"
vllm_image            = "us-central1-docker.pkg.dev/REPLACE/distill-staging/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
router_threshold      = 0.5 # set from results/router_eval.json (tuned on VAL)
teacher_enabled       = true
vllm_min_instances    = 0
vllm_max_instances    = 1
vllm_concurrency      = 32 # set from results/vllm_sweep.json (latency knee)
router_max_instances  = 2
teacher_spend_cap_usd = 2
router_invokers       = []
deletion_protection   = false
