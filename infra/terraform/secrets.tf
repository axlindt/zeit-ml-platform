# Credentials werden nie im Code gespeichert.
# Für lokale Entwicklung koennen defaults genutzt werden.
# In Production: defaults entfernen und via Umgebungsvariablen uebergeben:
#
#   export TF_VAR_minio_access_key="dein-access-key"
#   export TF_VAR_minio_secret_key="dein-secret-key"
#   terraform apply
#
# Auf GCP wuerde man stattdessen Workload Identity nutzen --
# Pods bekommen Zugriff auf GCS via GCP IAM ohne Credentials.

variable "minio_access_key" {
  description = "MinIO Access Key. In Production via TF_VAR_minio_access_key setzen."
  type        = string
  sensitive   = true
  default     = "minioadmin" # REMOVE IN PRODUCTION
}

variable "minio_secret_key" {
  description = "MinIO Secret Key. In Production via TF_VAR_minio_secret_key setzen."
  type        = string
  sensitive   = true
  default     = "minioadmin" # REMOVE IN PRODUCTION
}

resource "kubernetes_secret" "minio_credentials" {
  metadata {
    name      = "minio-credentials"
    namespace = "mlflow"
  }

  data = {
    access-key = var.minio_access_key
    secret-key = var.minio_secret_key
  }

  depends_on = [helm_release.minio]
}
