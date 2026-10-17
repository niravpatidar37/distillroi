locals {
  name   = "distill-${var.env}"
  labels = { app = "model-distillation-pipeline", env = var.env, managed_by = "terraform" }
}

data "google_project" "this" {
  project_id = var.project_id
}

# ---------------------------------------------------------------- registry + model storage
resource "google_artifact_registry_repository" "images" {
  #checkov:skip=CKV_GCP_84:Google-managed encryption is accepted for public-base images; add kms_key_name if policy requires CMEK.
  project       = var.project_id
  location      = var.region
  repository_id = local.name
  format        = "DOCKER"
  description   = "Router image and the mirrored, digest-pinned vLLM image."
  labels        = local.labels

  docker_config {
    immutable_tags = true
  }
}

# Merged student weights live here (uploaded by the runbook, never pulled from the internet at runtime).
resource "google_storage_bucket" "models" {
  #checkov:skip=CKV_GCP_62:No separate access-log bucket in this MVP; enable Data Access audit logs for GCS instead (docs/RUNBOOK.md).
  project                     = var.project_id
  name                        = var.model_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false
  labels                      = local.labels

  versioning {
    enabled = true
  }
}

# ---------------------------------------------------------------- secrets (values never in TF/state)
resource "google_secret_manager_secret" "groq_api_key" {
  project   = var.project_id
  secret_id = "${local.name}-groq-api-key"
  labels    = local.labels

  replication {
    auto {}
  }
}

# ---------------------------------------------------------------- service accounts (least privilege)
resource "google_service_account" "vllm" {
  project      = var.project_id
  account_id   = "${local.name}-vllm"
  display_name = "distill ${var.env} vLLM student (reads model bucket only)"
}

resource "google_service_account" "router" {
  project      = var.project_id
  account_id   = "${local.name}-router"
  display_name = "distill ${var.env} router (invokes vLLM, reads teacher key)"
}

resource "google_storage_bucket_iam_member" "vllm_reads_models" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.vllm.email}"
}

resource "google_secret_manager_secret_iam_member" "router_reads_key" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.groq_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.router.email}"
}

# ---------------------------------------------------------------- vLLM student on an L4 GPU
resource "google_cloud_run_v2_service" "vllm" {
  project             = var.project_id
  name                = "${local.name}-vllm"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL" # IAM-gated: only the router SA holds run.invoker
  deletion_protection = var.deletion_protection
  labels              = local.labels

  scaling {
    min_instance_count = var.vllm_min_instances
    max_instance_count = var.vllm_max_instances
  }

  template {
    service_account                  = google_service_account.vllm.email
    max_instance_request_concurrency = var.vllm_concurrency
    timeout                          = "60s"
    gpu_zonal_redundancy_disabled    = true # cheaper L4 SKU; see docs/SLO.md for the availability trade-off

    node_selector {
      accelerator = "nvidia-l4"
    }

    volumes {
      name = "models"
      gcs {
        bucket    = google_storage_bucket.models.name
        read_only = true
      }
    }

    containers {
      image = var.vllm_image
      args = [
        "/models/${var.student_model_path}",
        "--served-model-name=distill-student",
        "--host=0.0.0.0",
        "--port=8080",
        "--dtype=bfloat16",
        "--max-model-len=512",
        "--gpu-memory-utilization=0.85",
        "--max-num-seqs=128",
        "--max-logprobs=5",
      ]

      ports {
        container_port = 8080
      }

      env {
        name  = "HF_HUB_OFFLINE"
        value = "1"
      }
      env {
        name  = "VLLM_NO_USAGE_STATS"
        value = "1"
      }

      resources {
        limits = {
          "cpu"            = "4"
          "memory"         = "16Gi"
          "nvidia.com/gpu" = "1"
        }
        cpu_idle          = false # GPU services use instance-based billing
        startup_cpu_boost = true
      }

      volume_mounts {
        name       = "models"
        mount_path = "/models"
      }

      startup_probe {
        initial_delay_seconds = 10
        period_seconds        = 10
        timeout_seconds       = 5
        failure_threshold     = 60 # model load + CUDA graph capture can take minutes
        http_get {
          path = "/health"
          port = 8080
        }
      }

      liveness_probe {
        period_seconds    = 30
        timeout_seconds   = 5
        failure_threshold = 3
        http_get {
          path = "/health"
          port = 8080
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [client, client_version]
  }
}

resource "google_cloud_run_v2_service_iam_member" "router_invokes_vllm" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.vllm.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.router.email}"
}

# ---------------------------------------------------------------- router API (CPU)
resource "google_cloud_run_v2_service" "router" {
  project             = var.project_id
  name                = "${local.name}-router"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL" # callers need IAM run.invoker (no allUsers binding)
  deletion_protection = var.deletion_protection
  labels              = local.labels

  scaling {
    min_instance_count = var.router_min_instances
    max_instance_count = var.router_max_instances
  }

  template {
    service_account                  = google_service_account.router.email
    max_instance_request_concurrency = 80
    timeout                          = "30s"

    containers {
      image = var.router_image

      ports {
        container_port = 8080
      }

      env {
        name  = "STUDENT_BASE_URL"
        value = "${google_cloud_run_v2_service.vllm.uri}/v1"
      }
      env {
        name  = "STUDENT_GCP_ID_TOKEN"
        value = "true"
      }
      env {
        name  = "ROUTER_THRESHOLD"
        value = tostring(var.router_threshold)
      }
      env {
        name  = "TEACHER_ENABLED"
        value = tostring(var.teacher_enabled)
      }
      env {
        name  = "PROJECT_SPEND_CAP_USD"
        value = tostring(var.teacher_spend_cap_usd / var.router_max_instances)
      }
      env {
        name  = "LLM_SPEND_CAP_USD"
        value = tostring(var.teacher_spend_cap_usd / var.router_max_instances)
      }
      env {
        name  = "LLM_SPEND_LEDGER"
        value = "/tmp/ledger.jsonl"
      }
      env {
        name = "GROQ_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.groq_api_key.secret_id
            version = "latest"
          }
        }
      }

      resources {
        limits = {
          "cpu"    = "1"
          "memory" = "512Mi"
        }
        cpu_idle = true
      }

      startup_probe {
        period_seconds    = 5
        timeout_seconds   = 2
        failure_threshold = 12
        http_get {
          path = "/healthz"
          port = 8080
        }
      }

      liveness_probe {
        period_seconds    = 15
        timeout_seconds   = 2
        failure_threshold = 3
        http_get {
          path = "/healthz"
          port = 8080
        }
      }
    }
  }

  lifecycle {
    # CD deploys new image digests with gcloud and shifts traffic; Terraform owns everything else.
    ignore_changes = [template[0].containers[0].image, traffic, client, client_version]
  }

  depends_on = [google_secret_manager_secret_iam_member.router_reads_key]
}

resource "google_cloud_run_v2_service_iam_member" "router_invokers" {
  for_each = toset(var.router_invokers)
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.router.name
  role     = "roles/run.invoker"
  member   = each.value
}
