provider "google" {
  project = var.project_id
  region  = var.region
}

module "distill" {
  source = "../../modules/distill"

  project_id            = var.project_id
  region                = var.region
  env                   = "prod"
  router_image          = var.router_image
  vllm_image            = var.vllm_image
  model_bucket_name     = "${var.project_id}-distill-prod-models"
  router_threshold      = var.router_threshold
  teacher_enabled       = var.teacher_enabled
  vllm_min_instances    = var.vllm_min_instances
  vllm_max_instances    = var.vllm_max_instances
  vllm_concurrency      = var.vllm_concurrency
  router_max_instances  = var.router_max_instances
  teacher_spend_cap_usd = var.teacher_spend_cap_usd
  router_invokers       = var.router_invokers
  deletion_protection   = var.deletion_protection
}

output "router_url" {
  description = "Router API URL."
  value       = module.distill.router_url
}

output "vllm_url" {
  description = "vLLM student URL."
  value       = module.distill.vllm_url
}

output "artifact_registry" {
  description = "Artifact Registry docker repo."
  value       = module.distill.artifact_registry
}

output "model_bucket" {
  description = "Model weights bucket."
  value       = module.distill.model_bucket
}

output "teacher_secret_id" {
  description = "Secret Manager secret for the Groq key."
  value       = module.distill.teacher_secret_id
}
