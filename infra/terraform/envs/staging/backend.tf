# Remote state in GCS, configured at init time:
#   terraform init -backend-config="bucket=<tf-state-bucket>" -backend-config="prefix=distill/staging"
terraform {
  backend "gcs" {}
}
