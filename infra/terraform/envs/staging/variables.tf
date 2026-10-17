variable "project_id" {
  description = "GCP project id."
  type        = string
}

variable "region" {
  description = "Region with Cloud Run L4 GPU availability."
  type        = string
}

variable "router_image" {
  description = "Router image by digest."
  type        = string
}

variable "vllm_image" {
  description = "Mirrored vLLM image by digest."
  type        = string
}

variable "router_threshold" {
  description = "Escalation threshold tuned on validation data."
  type        = number
}

variable "teacher_enabled" {
  description = "Kill switch for paid teacher calls."
  type        = bool
  default     = true
}

variable "vllm_min_instances" {
  description = "Always-on GPU instances."
  type        = number
}

variable "vllm_max_instances" {
  description = "GPU autoscaling ceiling."
  type        = number
}

variable "vllm_concurrency" {
  description = "Per-instance request concurrency (measured knee)."
  type        = number
}

variable "router_max_instances" {
  description = "Router autoscaling ceiling."
  type        = number
}

variable "teacher_spend_cap_usd" {
  description = "Total teacher spend cap."
  type        = number
}

variable "router_invokers" {
  description = "IAM members allowed to invoke the router."
  type        = list(string)
  default     = []
}

variable "deletion_protection" {
  description = "Protect services from destroy."
  type        = bool
}
