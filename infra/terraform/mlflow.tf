# Deployt MLflow via eigenem Helm Chart.
resource "helm_release" "mlflow" {
  name             = "mlflow"
  chart            = "${path.module}/../helm/mlflow"
  namespace        = "mlflow"
  create_namespace = true

  depends_on = [helm_release.minio]
}
