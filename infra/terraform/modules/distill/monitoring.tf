# ---------------------------------------------------------------- monitoring as code
# Uptime check, SLO + multi-window burn-rate alerts, 5xx ratio, latency, saturation, and app-specific alerts from
# log-based metrics on the router's JSON logs (field names asserted in tests/test_hardening.py). Every resource and
# argument here was checked against the hashicorp/google 8.7.0 provider schema (`terraform providers schema -json`).
# Thresholds come from docs/SLO.md. Nothing here has been applied (no cloud account connected).

locals {
  router_service = google_cloud_run_v2_service.router.name
  vllm_service   = google_cloud_run_v2_service.vllm.name
  router_host    = trimprefix(google_cloud_run_v2_service.router.uri, "https://")
  run_filter     = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${local.router_service}\""
  log_filter     = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${local.router_service}\""
  channels       = [google_monitoring_notification_channel.email.name]
  runbook        = "Runbook: docs/RUNBOOK.md (section 7, common incidents) in github.com/niravpatidar37/distillroi"
}

resource "google_monitoring_notification_channel" "email" {
  project      = var.project_id
  display_name = "distill ${var.env} on-call (email)"
  type         = "email"
  labels = {
    email_address = var.alert_email
  }
}

# ---------------------------------------------------------------- uptime: /healthz on the IAM-gated router
resource "google_monitoring_uptime_check_config" "router_healthz" {
  project      = var.project_id
  display_name = "${local.name}-router /healthz"
  timeout      = "10s"
  period       = "60s"

  http_check {
    path           = "/healthz"
    port           = 443
    use_ssl        = true
    validate_ssl   = true
    request_method = "GET"
    service_agent_authentication {
      type = "OIDC_TOKEN" # ID token of the Monitoring service agent (granted run.invoker below)
    }
    accepted_response_status_codes {
      status_class = "STATUS_CLASS_2XX"
    }
  }

  content_matchers {
    content = "\"status\":\"ok\""
    matcher = "CONTAINS_STRING"
  }

  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = local.router_host
    }
  }
}

resource "google_cloud_run_v2_service_iam_member" "uptime_invokes_router" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.router.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:service-${data.google_project.this.number}@gcp-sa-monitoring-notification.iam.gserviceaccount.com"
}

resource "google_monitoring_alert_policy" "uptime" {
  project               = var.project_id
  display_name          = "${local.name}: router /healthz uptime check failing"
  combiner              = "OR"
  severity              = "CRITICAL"
  notification_channels = local.channels

  conditions {
    display_name = "uptime check failed from >1 region"
    condition_threshold {
      filter          = "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" AND resource.type=\"uptime_url\" AND metric.labels.check_id=\"${google_monitoring_uptime_check_config.router_healthz.uptime_check_id}\""
      comparison      = "COMPARISON_GT"
      threshold_value = 1
      duration        = "300s"
      aggregations {
        alignment_period     = "1200s"
        per_series_aligner   = "ALIGN_NEXT_OLDER"
        cross_series_reducer = "REDUCE_COUNT_FALSE"
        group_by_fields      = ["resource.label.*"]
      }
      trigger {
        count = 1
      }
    }
  }

  documentation {
    content = "Router /healthz fails. Check `gcloud run services describe ${local.router_service}` and recent revisions; roll back (RUNBOOK section 4). ${local.runbook}"
  }
}

# ---------------------------------------------------------------- 5xx ratio
resource "google_monitoring_alert_policy" "error_ratio" {
  project               = var.project_id
  display_name          = "${local.name}: router 5xx ratio > ${var.error_ratio_threshold * 100}%"
  combiner              = "OR"
  severity              = "ERROR"
  notification_channels = local.channels

  conditions {
    display_name = "5xx / all requests"
    condition_threshold {
      filter             = "${local.run_filter} AND metric.type = \"run.googleapis.com/request_count\" AND metric.labels.response_code_class = \"5xx\""
      denominator_filter = "${local.run_filter} AND metric.type = \"run.googleapis.com/request_count\""
      comparison         = "COMPARISON_GT"
      threshold_value    = var.error_ratio_threshold
      duration           = "600s"
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }
      denominator_aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  documentation {
    content = "5xx include 503 'no valid label' (student invalid + teacher off/unavailable) and teacher_unavailable under policy=fail. Check the teacher_unavailable alert and vLLM health. ${local.runbook}"
  }
}

# ---------------------------------------------------------------- latency
resource "google_monitoring_alert_policy" "latency_p95" {
  project               = var.project_id
  display_name          = "${local.name}: router p95 latency > ${var.latency_p95_all_ms} ms (all paths)"
  combiner              = "OR"
  severity              = "WARNING"
  notification_channels = local.channels

  conditions {
    display_name = "request_latencies p95"
    condition_threshold {
      filter          = "${local.run_filter} AND metric.type = \"run.googleapis.com/request_latencies\""
      comparison      = "COMPARISON_GT"
      threshold_value = var.latency_p95_all_ms
      duration        = "900s"
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_PERCENTILE_95"
      }
    }
  }

  documentation {
    content = "All-path p95 (escalations included; SLO for the escalated path is p95 < 3 s). If only escalations are slow, the teacher API is the cause. ${local.runbook}"
  }
}

# Student-path latency SLO (p95 < 400 ms) needs the `source` field, so it comes from the router's own logs.
resource "google_logging_metric" "student_latency" {
  project         = var.project_id
  name            = "${local.name}-student-latency-ms"
  description     = "Router-measured latency of requests answered by the student (source=student)."
  filter          = "${local.log_filter} AND jsonPayload.msg=\"classified\" AND jsonPayload.source=\"student\""
  value_extractor = "EXTRACT(jsonPayload.latency_ms)"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "ms"
  }

  bucket_options {
    exponential_buckets {
      num_finite_buckets = 40
      growth_factor      = 1.3
      scale              = 5
    }
  }
}

resource "google_monitoring_alert_policy" "student_latency_p95" {
  project               = var.project_id
  display_name          = "${local.name}: student-path p95 > ${var.latency_p95_student_ms} ms"
  combiner              = "OR"
  severity              = "WARNING"
  notification_channels = local.channels

  conditions {
    display_name = "student latency p95"
    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.student_latency.name}\" AND resource.type=\"cloud_run_revision\""
      comparison      = "COMPARISON_GT"
      threshold_value = var.latency_p95_student_ms
      duration        = "900s"
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_PERCENTILE_95"
      }
    }
  }

  documentation {
    content = "Student path slower than the SLO: vLLM over its concurrency knee. Lower vllm_concurrency so Cloud Run scales out earlier, or raise vllm_max_instances. ${local.runbook}"
  }
}

# ---------------------------------------------------------------- saturation
resource "google_monitoring_alert_policy" "saturation" {
  project               = var.project_id
  display_name          = "${local.name}: instance saturation (router or vLLM near max instances / concurrency)"
  combiner              = "OR"
  severity              = "WARNING"
  notification_channels = local.channels

  conditions {
    display_name = "router instances >= 80% of max"
    condition_threshold {
      filter          = "${local.run_filter} AND metric.type = \"run.googleapis.com/container/instance_count\""
      comparison      = "COMPARISON_GT"
      threshold_value = var.router_max_instances * 0.8
      duration        = "900s"
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_MAX"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  conditions {
    display_name = "router per-instance concurrency p95 >= 80% of the limit"
    condition_threshold {
      filter          = "${local.run_filter} AND metric.type = \"run.googleapis.com/container/max_request_concurrencies\""
      comparison      = "COMPARISON_GT"
      threshold_value = var.router_concurrency * 0.8
      duration        = "600s"
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_PERCENTILE_95"
      }
    }
  }

  conditions {
    display_name = "vLLM at its instance ceiling"
    condition_threshold {
      filter          = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${local.vllm_service}\" AND metric.type = \"run.googleapis.com/container/instance_count\""
      comparison      = "COMPARISON_GT"
      threshold_value = var.vllm_max_instances - 0.5
      duration        = "900s"
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_MAX"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  documentation {
    content = "Capacity is close to the configured ceiling; the router will start shedding with 429 (request_shed alert). Raise max instances (GPU quota) or accept shedding. ${local.runbook}"
  }
}

# ---------------------------------------------------------------- availability SLO + multi-window burn rate
resource "google_monitoring_custom_service" "router" {
  project      = var.project_id
  service_id   = "${local.name}-router"
  display_name = "distill ${var.env} router"
}

resource "google_monitoring_slo" "availability" {
  project             = var.project_id
  service             = google_monitoring_custom_service.router.service_id
  slo_id              = "availability"
  display_name        = "${var.availability_slo * 100}% of router requests are non-5xx (28 days)"
  goal                = var.availability_slo
  rolling_period_days = 28

  request_based_sli {
    good_total_ratio {
      total_service_filter = "${local.run_filter} AND metric.type = \"run.googleapis.com/request_count\""
      bad_service_filter   = "${local.run_filter} AND metric.type = \"run.googleapis.com/request_count\" AND metric.labels.response_code_class = \"5xx\""
    }
  }
}

# Google SRE workbook multi-window, multi-burn-rate: page at 14.4x (2% of a 30-day budget in 1 h) confirmed by the
# 5 min window; ticket at 6x (5% in 6 h) confirmed by 30 min.
resource "google_monitoring_alert_policy" "burn_fast" {
  project               = var.project_id
  display_name          = "${local.name}: error budget fast burn (14.4x over 1h and 5m)"
  combiner              = "AND"
  severity              = "CRITICAL"
  notification_channels = local.channels

  conditions {
    display_name = "burn rate 1h > 14.4"
    condition_threshold {
      filter          = "select_slo_burn_rate(\"${google_monitoring_slo.availability.name}\", \"60m\")"
      comparison      = "COMPARISON_GT"
      threshold_value = 14.4
      duration        = "0s"
    }
  }

  conditions {
    display_name = "burn rate 5m > 14.4"
    condition_threshold {
      filter          = "select_slo_burn_rate(\"${google_monitoring_slo.availability.name}\", \"5m\")"
      comparison      = "COMPARISON_GT"
      threshold_value = 14.4
      duration        = "0s"
    }
  }

  documentation {
    content = "Fast error-budget burn. Freeze deploys; roll back the last revision if it correlates (RUNBOOK section 4). ${local.runbook}"
  }
}

resource "google_monitoring_alert_policy" "burn_slow" {
  project               = var.project_id
  display_name          = "${local.name}: error budget slow burn (6x over 6h and 30m)"
  combiner              = "AND"
  severity              = "WARNING"
  notification_channels = local.channels

  conditions {
    display_name = "burn rate 6h > 6"
    condition_threshold {
      filter          = "select_slo_burn_rate(\"${google_monitoring_slo.availability.name}\", \"360m\")"
      comparison      = "COMPARISON_GT"
      threshold_value = 6
      duration        = "0s"
    }
  }

  conditions {
    display_name = "burn rate 30m > 6"
    condition_threshold {
      filter          = "select_slo_burn_rate(\"${google_monitoring_slo.availability.name}\", \"30m\")"
      comparison      = "COMPARISON_GT"
      threshold_value = 6
      duration        = "0s"
    }
  }

  documentation {
    content = "Slow error-budget burn: open a ticket, follow the error budget policy in docs/SLO.md. ${local.runbook}"
  }
}

# ---------------------------------------------------------------- app-specific (log-based metrics)
resource "google_logging_metric" "teacher_unavailable" {
  project     = var.project_id
  name        = "${local.name}-teacher-unavailable"
  description = "Escalations that could not reach the teacher (breaker open, 5xx, quota, spend cap, deadline)."
  filter      = "${local.log_filter} AND jsonPayload.msg=\"teacher_unavailable\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
    labels {
      key         = "cause"
      value_type  = "STRING"
      description = "budget_exceeded | quota_exhausted | deadline | error"
    }
  }

  label_extractors = {
    cause = "EXTRACT(jsonPayload.cause)"
  }
}

resource "google_monitoring_alert_policy" "teacher_unavailable" {
  project               = var.project_id
  display_name          = "${local.name}: teacher unavailable (escalations served degraded or 503)"
  combiner              = "OR"
  severity              = "ERROR"
  notification_channels = local.channels

  conditions {
    display_name = "teacher_unavailable events for 5 min"
    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.teacher_unavailable.name}\" AND resource.type=\"cloud_run_revision\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "300s"
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["metric.label.cause"]
      }
    }
  }

  documentation {
    content = "The teacher (Groq) is down, out of quota, over the spend cap, or too slow. Answers are the student's, flagged low_confidence (policy=degrade) or 503 (policy=fail). The `cause` label says which. ${local.runbook}"
  }
}

resource "google_monitoring_alert_policy" "spend_cap" {
  project               = var.project_id
  display_name          = "${local.name}: teacher spend cap reached (BudgetExceeded)"
  combiner              = "OR"
  severity              = "WARNING"
  notification_channels = local.channels

  conditions {
    display_name = "budget_exceeded"
    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.teacher_unavailable.name}\" AND resource.type=\"cloud_run_revision\" AND metric.labels.cause=\"budget_exceeded\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "0s"
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  documentation {
    content = "The shared teacher spend cap (Firestore budget leases) is exhausted. Intended fail-closed behaviour; investigate escalation volume before raising teacher_spend_cap_usd. ${local.runbook}"
  }
}

resource "google_logging_metric" "request_shed" {
  project     = var.project_id
  name        = "${local.name}-request-shed"
  description = "Requests rejected with 429 by admission control (overload) on the router."
  filter      = "${local.log_filter} AND jsonPayload.msg=\"request_shed\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

resource "google_monitoring_alert_policy" "overload" {
  project               = var.project_id
  display_name          = "${local.name}: router shedding load (429 overloaded)"
  combiner              = "OR"
  severity              = "WARNING"
  notification_channels = local.channels

  conditions {
    display_name = "request_shed > ${var.shed_alert_per_min}/min for 10 min"
    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.request_shed.name}\" AND resource.type=\"cloud_run_revision\""
      comparison      = "COMPARISON_GT"
      threshold_value = var.shed_alert_per_min / 60
      duration        = "600s"
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  documentation {
    content = "Admission control is rejecting requests (MAX_INFLIGHT/MAX_QUEUE per instance). Check the saturation alert; raise router_max_instances or vLLM capacity. ${local.runbook}"
  }
}

resource "google_logging_metric" "classified" {
  project     = var.project_id
  name        = "${local.name}-classified"
  description = "Classified requests by serving source (student | teacher | student_degraded)."
  filter      = "${local.log_filter} AND jsonPayload.msg=\"classified\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
    labels {
      key        = "source"
      value_type = "STRING"
    }
  }

  label_extractors = {
    source = "EXTRACT(jsonPayload.source)"
  }
}

resource "google_monitoring_alert_policy" "escalation_rate" {
  project               = var.project_id
  display_name          = "${local.name}: escalation rate above ${var.escalation_alert_ratio * 100}%"
  combiner              = "OR"
  severity              = "WARNING"
  notification_channels = local.channels

  conditions {
    display_name = "teacher share of classified requests (1 h)"
    condition_threshold {
      filter             = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.classified.name}\" AND resource.type=\"cloud_run_revision\" AND metric.labels.source=\"teacher\""
      denominator_filter = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.classified.name}\" AND resource.type=\"cloud_run_revision\""
      comparison         = "COMPARISON_GT"
      threshold_value    = var.escalation_alert_ratio
      duration           = "3600s"
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }
      denominator_aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  documentation {
    content = "More traffic than expected goes to the paid teacher (validation-tuned rate ~29%). Input drift or a bad model upload; compare with results/router_eval.json. ${local.runbook}"
  }
}
