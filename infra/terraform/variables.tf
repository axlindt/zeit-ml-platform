variable "bucket_name" {
  description = "Name des MinIO Buckets fuer MLflow Artefakte. Muss ein gueltiger S3 Bucket-Name sein."
  type        = string
  default     = "mlflow-artifacts"

  validation {
    condition     = length(var.bucket_name) > 0
    error_message = "bucket_name darf nicht leer sein."
  }

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.bucket_name))
    error_message = "bucket_name darf nur Kleinbuchstaben, Zahlen und Bindestriche enthalten."
  }

  validation {
    condition     = length(var.bucket_name) <= 63
    error_message = "bucket_name darf maximal 63 Zeichen lang sein (S3 Limit)."
  }

  validation {
    condition     = !startswith(var.bucket_name, "-") && !endswith(var.bucket_name, "-")
    error_message = "bucket_name darf nicht mit einem Bindestrich beginnen oder enden."
  }
}

variable "data_bucket_name" {
  description = "Name des MinIO Buckets fuer Trainingsdaten. Muss sich von bucket_name unterscheiden."
  type        = string
  default     = "ml-data"

  validation {
    condition     = length(var.data_bucket_name) > 0
    error_message = "data_bucket_name darf nicht leer sein."
  }

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.data_bucket_name))
    error_message = "data_bucket_name darf nur Kleinbuchstaben, Zahlen und Bindestriche enthalten."
  }

  validation {
    condition     = length(var.data_bucket_name) <= 63
    error_message = "data_bucket_name darf maximal 63 Zeichen lang sein (S3 Limit)."
  }

  validation {
    condition     = !startswith(var.data_bucket_name, "-") && !endswith(var.data_bucket_name, "-")
    error_message = "data_bucket_name darf nicht mit einem Bindestrich beginnen oder enden."
  }
}

variable "train_data_path" {
  description = "Lokaler Pfad zur Trainings-CSV."
  type        = string
  default     = "../../data/train_data.csv"

  validation {
    condition     = length(var.train_data_path) > 0
    error_message = "train_data_path darf nicht leer sein."
  }

  validation {
    condition     = can(regex("\\.csv$", var.train_data_path))
    error_message = "train_data_path muss auf .csv enden."
  }
}

variable "score_data_path" {
  description = "Lokaler Pfad zur Scoring-CSV."
  type        = string
  default     = "../../data/scoring_data.csv"

  validation {
    condition     = length(var.score_data_path) > 0
    error_message = "score_data_path darf nicht leer sein."
  }

  validation {
    condition     = can(regex("\\.csv$", var.score_data_path))
    error_message = "score_data_path muss auf .csv enden."
  }
}
