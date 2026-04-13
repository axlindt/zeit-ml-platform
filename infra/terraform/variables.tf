variable "bucket_name" {
  description = "Name des MinIO Buckets für MLflow Artefakte"
  type        = string
  default     = "mlflow-artifacts"
}

variable "data_bucket_name" {
  description = "Name des MinIO Buckets für Trainingsdaten"
  type        = string
  default     = "ml-data"
}

variable "train_data_path" {
  description = "Lokaler Pfad zur Trainings-CSV"
  type        = string
  default     = "../../data/train_data.csv"
}

variable "score_data_path" {
  description = "Lokaler Pfad zur Scoring-CSV"
  type        = string
  default     = "../../data/scoring_data.csv"
}
