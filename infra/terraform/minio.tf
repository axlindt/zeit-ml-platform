# Deployt MinIO als lokalen GCS-Ersatz via Helm Chart.
# Die Buckets und Trainingsdaten werden beim ersten terraform apply erstellt.
# In Production würde man stattdessen den google Provider mit GCS nutzen.

resource "helm_release" "minio" {
  name             = "minio"
  repository       = "https://charts.min.io/"
  chart            = "minio"
  namespace        = "mlflow"
  create_namespace = true

  set {
    name  = "mode"
    value = "standalone"
  }

  set {
    name  = "replicas"
    value = "1"
  }

  set {
    name  = "rootUser"
    value = "minioadmin"
  }

  set {
    name  = "rootPassword"
    value = "minioadmin"
  }

  set {
    name  = "buckets[0].name"
    value = var.bucket_name
  }

  set {
    name  = "buckets[0].policy"
    value = "none"
  }

  set {
    name  = "buckets[0].purge"
    value = "false"
  }

  set {
    name  = "resources.requests.memory"
    value = "512Mi"
  }
}

# Separater Bucket für Daten
resource "minio_s3_bucket" "data" {
  bucket     = var.data_bucket_name
  depends_on = [helm_release.minio]
}

# Trainings- & Evaluationsdaten beim Setup hochladen
resource "minio_s3_object" "train_data" {
  bucket_name  = minio_s3_bucket.data.bucket
  object_name  = "train_data.csv"
  source       = var.train_data_path
  content_type = "text/csv"
}

resource "minio_s3_object" "score_data" {
  bucket_name  = minio_s3_bucket.data.bucket
  object_name  = "scoring_data.csv"
  source       = var.score_data_path
  content_type = "text/csv"
}
