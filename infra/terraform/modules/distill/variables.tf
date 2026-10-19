variable "project_id" {
  description = "GCP project id."
  type        = string
}

variable "region" {
  description = "Region with Cloud Run L4 GPU capacity (e.g. us-central1, europe-west4)."
  type        = string
}

variable "env" {
  description = "Environment name (staging | prod)."
  type        = string
  validation {
    condition     = contains(["staging", "prod"], var.env)
    error_message = "env must be staging or prod."
  }
}

variable "router_image" {
  description = "Router image BY DIGEST (…/router@sha256:…). CD updates it; Terraform ignores later drift."
  type        = string
}

variable "vllm_image" {
  description = "vllm/vllm-openai mirrored into Artifact Registry, BY DIGEST."
  type        = string
}

variable "model_bucket_name" {
  description = "GCS bucket holding the merged student weights under student/ (uploaded out of band)."
  type        = string
}

variable "student_model_path" {
  description = "Path of the merged model inside the bucket mount."
  type        = string
  default     = "student"
}

variable "router_threshold" {
  description = "Escalation threshold tuned on the validation split (results/router_eval.json)."
  type        = number
}

variable "teacher_enabled" {
  description = "Kill switch for all paid teacher calls."
  type        = bool
  default     = true
}

variable "vllm_min_instances" {
  description = "Always-on GPU instances. 1 avoids multi-minute GPU cold starts (and is what the cost model assumes)."
  type        = number
  default     = 1
}

variable "vllm_max_instances" {
  description = "GPU autoscaling ceiling (bounded by the project's Cloud Run L4 quota)."
  type        = number
  default     = 2
}

variable "vllm_concurrency" {
  description = "Requests per vLLM instance; set to the measured latency knee (results/vllm_sweep.json)."
  type        = number
  default     = 32
}

variable "router_min_instances" {
  description = "Router minimum instances."
  type        = number
  default     = 0
}

variable "router_max_instances" {
  description = "Router max instances. Without the shared spend store the per-instance cap is cap/max."
  type        = number
  default     = 4
}

variable "teacher_spend_cap_usd" {
  description = "Total teacher spend cap across ALL router instances (shared via Firestore, or split per instance)."
  type        = number
  default     = 7
}

variable "router_invokers" {
  description = "IAM members allowed to call the router (e.g. serviceAccount:frontend@…). No allUsers."
  type        = list(string)
  default     = []
}

variable "deletion_protection" {
  description = "Protect Cloud Run services from terraform destroy."
  type        = bool
  default     = true
}

variable "spend_store" {
  description = "Where the cross-instance teacher spend cap lives: firestore (shared, exact) or file (per-instance cap = cap / router_max_instances)."
  type        = string
  default     = "firestore"
  validation {
    condition     = contains(["firestore", "file"], var.spend_store)
    error_message = "spend_store must be firestore or file."
  }
}

variable "firestore_location" {
  description = "Firestore location for the spend-cap database (a region such as us-central1, or nam5)."
  type        = string
  default     = "us-central1"
}

variable "teacher_unavailable_policy" {
  description = "When the teacher is down/out of quota: degrade (student answer flagged low_confidence) or fail (503)."
  type        = string
  default     = "degrade"
  validation {
    condition     = contains(["degrade", "fail"], var.teacher_unavailable_policy)
    error_message = "teacher_unavailable_policy must be degrade or fail."
  }
}

variable "router_concurrency" {
  description = "Cloud Run max concurrent requests per router instance."
  type        = number
  default     = 80
}

variable "router_max_inflight" {
  description = "App-level admission limit per instance (MAX_INFLIGHT); excess waits in a bounded queue, then 429."
  type        = number
  default     = 64
}

variable "router_max_queue" {
  description = "Requests allowed to wait for an in-flight slot per instance (MAX_QUEUE)."
  type        = number
  default     = 16
}

variable "rate_limit_rps" {
  description = "Per-client token refill rate on /v1/* (requests/s, per instance); 0 disables."
  type        = number
  default     = 50
}

variable "rate_limit_burst" {
  description = "Per-client burst capacity on /v1/* (per instance)."
  type        = number
  default     = 100
}

variable "staging_fault_injection" {
  description = "Honour X-Distill-Fault smoke hooks in staging (never in prod: the app refuses it)."
  type        = bool
  default     = true
}

variable "alert_email" {
  description = "On-call email for the Cloud Monitoring notification channel."
  type        = string
}

variable "availability_slo" {
  description = "Availability SLO (non-5xx share, 28-day rolling) from docs/SLO.md."
  type        = number
  default     = 0.995
}

variable "error_ratio_threshold" {
  description = "Alert when the 5xx ratio exceeds this for 10 minutes (docs/SLO.md: 1%)."
  type        = number
  default     = 0.01
}

variable "latency_p95_all_ms" {
  description = "All-path p95 alert threshold in ms (escalated-path SLO: p95 < 3 s)."
  type        = number
  default     = 3000
}

variable "latency_p95_student_ms" {
  description = "Student-path p95 alert threshold in ms (docs/SLO.md: 400 ms)."
  type        = number
  default     = 400
}

variable "shed_alert_per_min" {
  description = "Alert when admission control sheds more than this many requests per minute for 10 minutes."
  type        = number
  default     = 30
}

variable "escalation_alert_ratio" {
  description = "Alert when the teacher share of classified requests exceeds this over 1 h (validation-tuned rate ~0.29)."
  type        = number
  default     = 0.4
}
