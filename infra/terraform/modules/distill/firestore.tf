# ---------------------------------------------------------------- shared teacher spend cap (gap: per-instance ledger)
# One small Firestore (Native mode) database holds a single document per project with the budget granted to router
# instances (compare-and-swap leases, src/distill/shared_spend.py). Chosen over Memorystore because it is
# serverless (no VPC connector, no always-on instance) and the workload is a handful of document reads/writes per
# lease: list price $0.03 / 100K reads and $0.09 / 100K writes (us-central1), and the first database in a project
# gets the daily free tier (https://cloud.google.com/firestore/pricing). Resource/argument names verified against
# hashicorp/google 8.7.0 (`terraform providers schema -json`).

resource "google_firestore_database" "spend" {
  count = local.shared_spend ? 1 : 0

  project                           = var.project_id
  name                              = local.spend_database_id
  location_id                       = var.firestore_location
  type                              = "FIRESTORE_NATIVE"
  concurrency_mode                  = "OPTIMISTIC" # matches the app's updateTime-precondition CAS
  app_engine_integration_mode       = "DISABLED"
  point_in_time_recovery_enablement = "POINT_IN_TIME_RECOVERY_ENABLED"
  delete_protection_state           = var.deletion_protection ? "DELETE_PROTECTION_ENABLED" : "DELETE_PROTECTION_DISABLED"
  deletion_policy                   = var.deletion_protection ? "PREVENT" : "DELETE"
}

# Least privilege: datastore.user on THIS database only (IAM condition on the database resource name, per
# https://firebase.google.com/docs/firestore/manage-databases#configure_per-database_access_permissions).
resource "google_project_iam_member" "router_spend_store" {
  count = local.shared_spend ? 1 : 0

  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.router.email}"

  condition {
    title       = "only-${local.spend_database_id}"
    description = "Router may read/write only the spend-cap database."
    expression  = "resource.name == \"projects/${var.project_id}/databases/${local.spend_database_id}\""
  }
}
