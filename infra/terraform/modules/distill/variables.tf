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
  description = "Router max instances. The spend ledger is per instance, so the per-instance cap is cap/max."
  type        = number
  default     = 4
}

variable "teacher_spend_cap_usd" {
  description = "Total teacher spend cap across router instances (split evenly per instance)."
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
