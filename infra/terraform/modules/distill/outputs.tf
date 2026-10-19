output "router_url" {
  description = "Router API URL (IAM-authenticated)."
  value       = google_cloud_run_v2_service.router.uri
}

output "vllm_url" {
  description = "vLLM student URL (only the router SA may invoke it)."
  value       = google_cloud_run_v2_service.vllm.uri
}

output "artifact_registry" {
  description = "Docker repository path for router and vLLM images."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "model_bucket" {
  description = "Bucket to upload the merged student into (gs://<bucket>/student/)."
  value       = google_storage_bucket.models.name
}

output "teacher_secret_id" {
  description = "Add the Groq key as a version of this secret (never via Terraform)."
  value       = google_secret_manager_secret.groq_api_key.secret_id
}

output "project_number" {
  description = "Project number (for Workload Identity Federation principals)."
  value       = data.google_project.this.number
}

output "spend_database" {
  description = "Firestore database holding the shared teacher spend cap (empty when spend_store = file)."
  value       = local.shared_spend ? google_firestore_database.spend[0].name : ""
}

output "uptime_check_id" {
  description = "Cloud Monitoring uptime check on the router /healthz."
  value       = google_monitoring_uptime_check_config.router_healthz.uptime_check_id
}
