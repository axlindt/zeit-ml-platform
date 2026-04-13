# ZEIT ML Platform

Lokale MLOps-Plattform als Lösung der ZEIT ML/AI Platform Bewerbungsaufgabe.

## Voraussetzungen

- Docker Desktop
- kubectl, helm, k3d, terraform (macOS: `brew install terraform kubectl helm k3d`)

## Schritt 1: Kubernetes Cluster aufsetzen

Wir nutzen k3d für einen lokalen Kubernetes Cluster in Docker, kostenlos und identisch zu GKE in der Verwedung. In Production würde das Cluster selbst via Terraform als `google_container_cluster` erstellt werden.

```bash
k3d cluster create zeit-ml-platform
kubectl get nodes
```

## Schritt 2: Storage bereitstellen

Fuer den Storage nutze ich MinIO, einen lokalen Object Store der das S3-Protokoll spricht und damit als GCS-Ersatz funktioniert. Alle Plattform-Komponenten (MinIO, MLflow) laufen im Namespace `mlflow`.

Terraform erstellt zwei Buckets: `mlflow-artifacts` für MLflow Artefakte und `ml-data` für Trainings- und Scoring-Daten.

Da der MinIO Terraform Provider eine direkte Verbindung braucht, muss MinIO erst laufen bevor die Buckets erstellt werden koennen:

```bash
cd infra/terraform
terraform init
terraform apply  # deployt MinIO

kubectl port-forward svc/minio 9000:9000 -n mlflow &
terraform apply  # erstellt Buckets und laedt Daten hoch
```
