# ZEIT ML Platform

Lokale MLOps-Plattform als Lösung der ZEIT ML/AI Platform Bewerbungsaufgabe.

## Voraussetzungen

- Docker Desktop
- kubectl, helm, k3d, terraform (macOS: `brew install terraform kubectl helm k3d`)

## Schritt 1: Kubernetes Cluster aufsetzen

Ich nutze k3d für einen lokalen Kubernetes Cluster in Docker. Das vermeidet GCP Billing-Setup und funktioniert für dieses Beispiel genauso wie GKE. In Production würde das Cluster selbst via Terraform als `google_container_cluster` erstellt werden.

```bash
k3d cluster create zeit-ml-platform
kubectl get nodes
```

## Schritt 2: Storage bereitstellen

Für den Storage nutze ich MinIO, einen lokalen Object Store der das S3-Protokoll spricht und damit als GCS-Ersatz funktioniert. Alle Plattform-Komponenten (MinIO, MLflow) laufen im Namespace `mlflow`.

Terraform erstellt zwei Buckets: `mlflow-artifacts` für MLflow Artefakte und `ml-data` für Trainings- und Scoring-Daten.

Da der MinIO Terraform Provider eine direkte Verbindung braucht, muss MinIO erst laufen bevor die Buckets erstellt werden können:

```bash
cd infra/terraform
terraform init
terraform apply  # deployt MinIO

kubectl port-forward svc/minio 9000:9000 -n mlflow &
terraform apply  # erstellt Buckets und lädt Daten hoch
```

## Schritt 3: MLflow deployen
 
MLflow ist der Tracking Server für Experimente, Metriken und Modell-Artefakte. Ich deploye ihn via einem eigenen Helm Chart (einem Kubernetes Deployment-Paket) statt einem fertigen Chart wie Bitnami, weil der Bitnami Chart seine eigene Datenbank und seinen eigenen Storage mitbringt was mit unserem MinIO kollidiert.
 
MLflow wird so konfiguriert dass er Artefakte in dem von Terraform erstellten Bucket `mlflow-artifacts` speichert. Mit `--serve-artifacts` fungiert MLflow als Proxy zwischen Browser und MinIO, so dass der Browser nie direkt auf MinIO zugreifen muss.
 
Da das offizielle MLflow Image kein `boto3` hat (das S3 SDK das MLflow braucht um mit MinIO zu reden), bauen wir ein eigenes Image:
 
```bash
docker build -t zeit-mlflow -f infra/docker/Dockerfile.mlflow .
k3d image import zeit-mlflow -c zeit-ml-platform
```
 
Dann MLflow via Terraform deployen:
 
```bash
cd infra/terraform
terraform apply
 
# MLflow UI erreichbar unter
kubectl port-forward svc/mlflow 5001:5000 -n mlflow
```

## Schritt 4: Modell trainieren
 
Das Training Script wird als Docker Container gebaut und als Kubernetes Job ausgeführt. Es liest die Trainingsdaten direkt aus MinIO und loggt Parameter, Metriken und das Modell-Artefakt in MLflow.
 
```bash
docker build -t zeit-churn-model .
k3d image import zeit-churn-model -c zeit-ml-platform
 
kubectl apply -f infra/k8s/training-job.yaml
kubectl logs -n mlflow job/churn-training -f
```
 
## Schritt 5: Scoring
 
Der Scoring Job lädt automatisch den letzten Training Run aus MLflow. Optional kann eine spezifische Run ID via `MODEL_RUN_ID` angegeben werden (in `scoring-job.yaml` auskommentiert).
 
```bash
kubectl apply -f infra/k8s/scoring-job.yaml
kubectl logs -n mlflow job/churn-scoring -f
```
 
Die berechneten Scores sind dann in zwei Formen verfügbar:
- Als CSV in MinIO (`s3://ml-data/scored_output.csv`)
- Als downloadbares Artefakt im MLflow Scoring Run